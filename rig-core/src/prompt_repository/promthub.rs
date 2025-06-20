//! PromptHub adapter rewritten to conform to the unified `PromptRepository`
//! abstraction proposed earlier.
//!
//! All PromptHub–specific quirks (two‑segment template, commit‑hash “versioning”,
//! server‑side variable ignorance) are localised to this file.  Nothing leaks into
//! the rest of the framework.

use super::lib::{PromptError, PromptRepository, PromptSegment, PromptTemplate, Role};
use crate::prompt_repository::Version;
use async_trait::async_trait;
use reqwest::{self, Client as HttpClient};
use rig::prompt_repository::lib::TemplateSyntax;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;

const PROMPTHUB_API_BASE_URL: &str = "https://app.prompthub.us/api/v1";

/// Adapter for the PromptHub REST API.
#[derive(Clone)]
pub struct PromptHub {
    base_url: String,
    api_key: String,
    http_client: HttpClient,
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    data: PromptResponse,
}

#[derive(Debug, Deserialize)]
struct PromptResponse {
    project_id: u32,
    hash: String,
    prompt: Option<String>,
    system_message: Option<String>,
}

pub struct PromptHubQuery<'a> {
    pub id: &'a str,
    pub branch: Option<&'a str>,
}

impl PromptHub {
    /// Construct a new adapter with the default cloud endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::from_url(api_key, PROMPTHUB_API_BASE_URL)
    }

    /// Construct a new adapter with a custom endpoint (useful for testing).
    pub fn from_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            http_client: HttpClient::builder()
                .build()
                .expect("PromptHub reqwest client should build"),
        }
    }

    /// From `PROMPT_HUB_API_KEY` env var.
    pub fn from_env() -> Self {
        let api_key = env::var("PROMPT_HUB_API_KEY").expect("PROMPT_HUB_API_KEY not set");
        Self::new(api_key)
    }

    /// Replace the internally‑owned `reqwest::Client`.
    pub fn with_custom_client(mut self, client: HttpClient) -> Self {
        self.http_client = client;
        self
    }

    /// Internal helper
    async fn _query<'a>(&self, q: PromptHubQuery<'a>) -> Result<PromptResponse, PromptError> {
        let params = q
            .branch
            .map(|branch| format!("?branch={}", branch))
            .unwrap_or_default();
        let url = format!("{}/projects/{}/head{}", self.base_url, q.id, params);

        let response = self
            .http_client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| PromptError::Network(e.into()))?;

        if !response.status().is_success() {
            return Err(PromptError::Api(format!(
                "PromptHub returned status {}",
                response.status()
            )));
        }

        response
            .json::<ApiResponse>()
            .await
            .map_err(|e| PromptError::Deserialization(e.to_string()))
            .map(|res| res.data)
    }
}

#[async_trait]
impl PromptRepository for PromptHub {
    type Query<'a> = PromptHubQuery<'a>;
    type Prompt = PromptTemplate;

    async fn query<'a>(&self, q: PromptHubQuery<'a>) -> Result<PromptTemplate, PromptError> {
        let res = self._query(q).await?;

        // Build segments in the order: system, then user.
        let mut segments = Vec::with_capacity(2);

        if let Some(sys) = res.system_message {
            segments.push(PromptSegment {
                role: Role::System,
                template: sys,
                syntax: TemplateSyntax::Moustache,
            });
        }

        if let Some(user) = res.prompt {
            segments.push(PromptSegment {
                role: Role::User,
                template: user,
                syntax: TemplateSyntax::Moustache,
            });
        }

        Ok(PromptTemplate {
            id: format!("prompthub:{}", res.project_id),
            version: Option::from(res.hash),
            segments,
            metadata: HashMap::new(),
        })
    }

    async fn fetch<'a>(
        &self,
        id: &str,
        version: Version<'a>,
    ) -> Result<PromptTemplate, PromptError> {
        let res = self.query(PromptHubQuery { id, branch: None }).await;

        match res {
            // PromptHub doesn't (yet?) support fetching prompts by version. If the caller supplied a version,
            // we optimistically check for equality with the returned one and error otherwise so that
            // users can rely on deterministic behavior.
            Ok(PromptTemplate {
                version: Some(v), ..
            }) if version != Version::Latest && version.to_string() != v => {
                Err(PromptError::Unsupported(format!(
                    "PromptHub does not expose the requested version '{}'. Returned version '{}'.",
                    version, v
                )))
            }
            _ => res,
        }
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
            .mock("GET", "/projects/test-project/head")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "data": {
                    "project_id": "test-project",
                    "hash": "1874502c",
                    "prompt": "This is a test prompt with {{variable}}",
                    "system_message": "You are a test assistant."
                }
            }"#,
            )
            .create();

        let prompthub = PromptHub::from_url("test-api-key", server.url());
        let result = prompthub.fetch("test-project", Version::Latest).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.id, "prompthub:test-project");
        assert_eq!(template.version, Some("1874502c".to_string()));
        assert_eq!(template.segments.len(), 2);
        assert_eq!(template.segments[0].role, Role::System);
        assert_eq!(template.segments[0].template, "You are a test assistant.");
        assert_eq!(template.segments[1].role, Role::User);
        assert_eq!(
            template.segments[1].template,
            "This is a test prompt with {{variable}}"
        );

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_only_user_prompt() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/projects/test-project/head")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "data": {
                    "project_id": "test-project",
                    "hash": "1874502c",
                    "prompt": "This is a test prompt with no system message."
                }
            }"#,
            )
            .create();

        let prompthub = PromptHub::from_url("test-api-key", server.url());
        let result = prompthub.fetch("test-project", Version::Latest).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.segments.len(), 1);
        assert_eq!(template.segments[0].role, Role::User);
        assert_eq!(
            template.segments[0].template,
            "This is a test prompt with no system message."
        );

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_only_system_message() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/projects/test-project/head")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "data": {
                    "project_id": "test-project",
                    "hash": "1874502c",
                    "system_message": "This is only a system message with no user prompt."
                }
            }"#,
            )
            .create();

        let prompthub = PromptHub::from_url("test-api-key", server.url());
        let result = prompthub.fetch("test-project", Version::Latest).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.segments.len(), 1);
        assert_eq!(template.segments[0].role, Role::System);
        assert_eq!(
            template.segments[0].template,
            "This is only a system message with no user prompt."
        );

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_api_error() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/projects/test-project/head")
            .with_status(404)
            .with_body("Not found")
            .create();

        let prompthub = PromptHub::from_url("test-api-key", server.url());
        let result = prompthub.fetch("test-project", Version::Latest).await;

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
            .mock("GET", "/projects/test-project/head?branch=staging")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "data": {
                    "project_id": "test-project",
                    "hash": "1874502c",
                    "prompt": "This is a test prompt for query method.",
                    "system_message": "System message for query test."
                }
            }"#,
            )
            .create();

        let prompthub = PromptHub::from_url("test-api-key", server.url());
        let query = PromptHubQuery {
            id: "test-project",
            branch: Some("staging"),
        };
        let result = prompthub.query(query).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.id, "prompthub:test-project");

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_deserialization_error() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/projects/test-project/head")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "invalid_json": {
            }"#,
            )
            .create();

        let prompthub = PromptHub::from_url("test-api-key", server.url());
        let result = prompthub.fetch("test-project", Version::Latest).await;

        assert!(result.is_err());
        match result {
            Err(PromptError::Deserialization(_)) => {
                // Success - we got the expected error type
            }
            _ => panic!("Expected PromptError::Deserialization"),
        }

        mock.assert_async().await
    }
}
