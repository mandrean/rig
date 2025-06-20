//! Haystack PromptHub adapter that conforms to the unified `PromptRepository`
//! abstraction used by the orchestration framework.
//!
//! The Haystack endpoint always returns a **single prompt string**.  We wrap this into a
//! one‑segment `PromptTemplate` with `Role::User`.
//!
//! Versioning is provided by a semver `version` field in the payload; we surface that in
//! `PromptTemplate::version` (when parseable) and keep the rest of the metadata hidden
//! in the `metadata` map.

use super::lib::{
    PromptError, PromptRepository, PromptSegment, PromptTemplate, Role, TemplateSyntax,
};
use async_trait::async_trait;
use reqwest::{self, Client as HttpClient};
use rig::{prompt_repository::lib::PromptQuery, prompt_repository::Version};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

const HAYSTACK_PROMPTHUB_API_BASE_URL: &str = "https://api.prompthub.deepset.ai";

/// Adapter for the Haystack PromptHub REST API.
#[derive(Clone)]
pub struct HaystackPromptHub {
    base_url: String,
    http_client: HttpClient,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptPayload {
    name: String,
    tags: Vec<String>,
    meta: Value,
    version: String,
    text: Option<String>,
    description: String,
}

impl HaystackPromptHub {
    /// Construct with the hosted service endpoint.
    pub fn new() -> Self {
        Self::from_url(HAYSTACK_PROMPTHUB_API_BASE_URL)
    }

    /// Construct with a custom/self‑hosted endpoint (useful for tests).
    pub fn from_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            http_client: HttpClient::builder()
                .build()
                .expect("Haystack PromptHub reqwest client should build"),
        }
    }

    /// Replace the internally‑owned `reqwest::Client` (for custom TLS pools, etc.).
    pub fn with_custom_client(mut self, client: HttpClient) -> Self {
        self.http_client = client;
        self
    }

    /// Internal helper to fetch a prompt by its path.
    async fn _fetch(&self, path: &str) -> Result<PromptPayload, PromptError> {
        let url = format!("{}/prompts/{}", self.base_url.trim_end_matches('/'), path);

        let response = self
            .http_client
            .get(url)
            .send()
            .await
            .map_err(|e| PromptError::Network(e.into()))?;

        if !response.status().is_success() {
            return Err(PromptError::Api(format!(
                "Haystack PromptHub returned status {}",
                response.status()
            )));
        }

        response
            .json::<PromptPayload>()
            .await
            .map_err(|e| PromptError::Deserialization(e.to_string()))
    }
}

#[async_trait]
impl PromptRepository for HaystackPromptHub {
    type Query<'a> = PromptQuery<'a>;
    type Prompt = PromptTemplate;

    async fn query<'a>(&self, q: PromptQuery<'a>) -> Result<PromptTemplate, PromptError> {
        self.fetch(q.id, q.version).await
    }

    async fn fetch<'a>(
        &self,
        id: &str,
        version: Version<'a>,
    ) -> Result<PromptTemplate, PromptError> {
        let payload = self._fetch(id).await?;

        match version {
            // Haystack doesn't (yet?) support fetching prompts by version. If the caller supplied a version,
            // we optimistically check for equality with the returned one and error otherwise so that
            // users can rely on deterministic behavior.
            Version::Ref(req) if req != payload.version => {
                return Err(PromptError::Unsupported(format!(
                    "Haystack PromptHub does not expose the requested version '{}'. Returned version '{}'.",
                    req, payload.version
                )));
            }
            _ => {}
        }

        let prompt_str = payload.text.ok_or(PromptError::PromptExtraction(
            "`text` field missing in Haystack response".into(),
        ))?;

        let segment = PromptSegment {
            role: Role::User,
            template: prompt_str,
            syntax: TemplateSyntax::Moustache,
        };

        let version = Option::from(payload.version);

        let mut metadata = HashMap::new();
        metadata.insert(
            "description".to_string(),
            Value::String(payload.description),
        ); // keep the long description
        metadata.insert(
            "tags".to_string(),
            Value::Array(payload.tags.into_iter().map(Value::String).collect()),
        );
        metadata.insert("meta".to_string(), payload.meta);

        Ok(PromptTemplate {
            id: format!("haystack:{}", payload.name),
            version,
            segments: vec![segment],
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito;

    #[tokio::test]
    async fn test_fetch_success() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/prompts/test-prompt")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "name": "test-prompt",
                "tags": ["test", "example"],
                "meta": {"author": "test-author"},
                "version": "1.0.0",
                "text": "This is a test prompt with {{variable}}",
                "description": "A test prompt for unit testing"
            }"#,
            )
            .create();

        let haystack = HaystackPromptHub::from_url(server.url());
        let result = haystack.fetch("test-prompt", Version::Latest).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.id, "haystack:test-prompt");
        assert_eq!(template.version, Some("1.0.0".to_string()));
        assert_eq!(template.segments.len(), 1);
        assert_eq!(template.segments[0].role, Role::User);
        assert_eq!(
            template.segments[0].template,
            "This is a test prompt with {{variable}}"
        );
        assert_eq!(template.segments[0].syntax, TemplateSyntax::Moustache);
        assert!(template.metadata.contains_key("description"));
        assert_eq!(
            template.metadata["description"],
            "A test prompt for unit testing"
        );

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_with_version() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/prompts/test-prompt")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "name": "test-prompt",
                "tags": ["test", "example"],
                "meta": {"author": "test-author"},
                "version": "2.0.0",
                "text": "This is version 2.0.0 of the test prompt",
                "description": "A test prompt for unit testing"
            }"#,
            )
            .create();

        let haystack = HaystackPromptHub::from_url(server.url());
        // Note: Haystack doesn't support version-specific fetching, so this should return
        // the latest version and check if it matches the requested one
        let result = haystack.fetch("test-prompt", Version::Ref("2.0.0")).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.version, Some("2.0.0".to_string()));

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_version_mismatch() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/prompts/test-prompt")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "name": "test-prompt",
                "tags": ["test", "example"],
                "meta": {"author": "test-author"},
                "version": "1.0.0",
                "text": "This is version 1.0.0 of the test prompt",
                "description": "A test prompt for unit testing"
            }"#,
            )
            .create();

        let haystack = HaystackPromptHub::from_url(server.url());
        let result = haystack.fetch("test-prompt", Version::Ref("2.0.0")).await;

        assert!(result.is_err());
        match result {
            Err(PromptError::Unsupported(msg)) => {
                assert!(msg.contains("does not expose the requested version"));
            }
            _ => panic!("Expected PromptError::Unsupported"),
        }

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_missing_text() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/prompts/test-prompt")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "name": "test-prompt",
                "tags": ["test", "example"],
                "meta": {"author": "test-author"},
                "version": "1.0.0",
                "description": "A test prompt for unit testing"
            }"#,
            )
            .create();

        let haystack = HaystackPromptHub::from_url(server.url());
        let result = haystack.fetch("test-prompt", Version::Latest).await;

        assert!(result.is_err());
        match result {
            Err(PromptError::PromptExtraction(msg)) => {
                assert!(msg.contains("`text` field missing"));
            }
            _ => panic!("Expected PromptError::PromptExtraction"),
        }

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_api_error() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/prompts/test-prompt")
            .with_status(404)
            .with_body("Not found")
            .create();

        let haystack = HaystackPromptHub::from_url(server.url());
        let result = haystack.fetch("test-prompt", Version::Latest).await;

        assert!(result.is_err());
        match result {
            Err(PromptError::Api(msg)) => {
                assert!(msg.contains("404"));
            }
            _ => panic!("Expected PromptError::Api"),
        }

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_query_method() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/prompts/test-prompt")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "name": "test-prompt",
                "tags": ["test", "example"],
                "meta": {"author": "test-author"},
                "version": "1.0.0",
                "text": "This is a test prompt with {{variable}}",
                "description": "A test prompt for unit testing"
            }"#,
            )
            .create();

        let haystack = HaystackPromptHub::from_url(server.url());
        let query = PromptQuery {
            id: "test-prompt",
            version: Version::Latest,
        };
        let result = haystack.query(query).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.id, "haystack:test-prompt");

        mock.assert_async().await
    }
}
