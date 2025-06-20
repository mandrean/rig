//! OpenLIT adapter rewritten to fit the unified `PromptRepository` / `PromptCompiler`
//! abstraction.
//!
//! * The OpenLIT REST endpoint can *optionally* compile a prompt server‑side.  We keep
//!   the transport hidden and expose **either** an un‑compiled `PromptTemplate`
//!   (`fetch`) **or** a compiled prompt via the regular `compile` function (local) or
//!   an extra helper `compile_remote` if you want the server to do it.
//! * A prompt in OpenLIT is always a **single string**; we wrap it into one
//!   `PromptSegment` with role `User`.
//! * Version strings follow semantic‑versioning rules (e.g. "1.0.0").
//! * OpenLIT supports identification by **UUID** (`promptId`) or **name**; callers
//!   pass whatever identifier they have in `id`.

use super::lib::{
    PromptError, PromptRepository, PromptSegment, PromptTemplate, Role, TemplateSyntax,
};
use crate::prompt_repository::Version;
use async_trait::async_trait;
use reqwest::{self, Client as HttpClient};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

const OPENLIT_API_BASE_URL: &str = "http://localhost:3000";

/// Adapter for the OpenLIT REST API.
#[derive(Clone)]
pub struct OpenLit {
    base_url: String,
    api_key: String,
    http_client: HttpClient,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiRequest<'a> {
    #[serde(flatten)]
    query_type: QueryType<'a>,
    #[serde(skip_serializing_if = "Version::is_latest")]
    version: Version<'a>,
    compile: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiResponse {
    #[serde(default)]
    err: Value,
    #[serde(default)]
    res: Option<PromptResponse>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptResponse {
    prompt_id: String,
    version_id: String,
    name: String,
    version: String,
    tags: Vec<Value>,
    meta_properties: Map<String, Value>,
    prompt: Option<String>,
}

#[derive(Serialize)]
pub enum QueryType<'a> {
    #[serde[rename = "id"]]
    ById(&'a str),
    #[serde[rename = "name"]]
    ByName(&'a str),
}

impl OpenLit {
    /// Construct with default cloud/local endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::from_url(api_key, OPENLIT_API_BASE_URL)
    }

    /// Construct with a custom base url (useful for self‑hosted / tests).
    pub fn from_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            http_client: HttpClient::builder()
                .build()
                .expect("OpenLIT reqwest client should build"),
        }
    }

    /// Replace the internal `reqwest::Client` instance (for custom pools, etc.).
    pub fn with_custom_client(mut self, client: HttpClient) -> Self {
        self.http_client = client;
        self
    }

    /// Internal helper
    async fn _fetch(&self, req: ApiRequest<'_>) -> Result<PromptResponse, PromptError> {
        let url = format!("{}/api/prompt/get-compiled", self.base_url);

        let body =
            serde_json::to_string(&req).map_err(|e| PromptError::Serialization(e.to_string()))?;

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await
            .map_err(|e| PromptError::Network(e.into()))?;

        if !response.status().is_success() {
            return Err(PromptError::Api(format!(
                "OpenLIT returned status {}",
                response.status()
            )));
        }

        let api: ApiResponse = response
            .json()
            .await
            .map_err(|e| PromptError::Deserialization(e.to_string()))?;

        match api {
            ApiResponse {
                res: Some(res),
                err: Value::Null,
            } => Ok(res),
            ApiResponse {
                res: None,
                err: Value::Null,
            } => Err(PromptError::Other("Response missing payload".into())),
            ApiResponse {
                err: Value::String(s),
                ..
            } => Err(PromptError::Api(s)),
            ApiResponse { err, .. } => Err(PromptError::Api(format!("{:?}", err))),
        }
    }
}

pub struct OpenLitQuery<'a> {
    pub query_type: QueryType<'a>,
    pub version: Version<'a>,
}

impl<'a> Into<ApiRequest<'a>> for OpenLitQuery<'a> {
    fn into(self) -> ApiRequest<'a> {
        ApiRequest {
            query_type: self.query_type,
            version: self.version,
            compile: false,
        }
    }
}

#[async_trait]
impl PromptRepository for OpenLit {
    type Query<'a> = OpenLitQuery<'a>;
    type Prompt = PromptTemplate;

    async fn query<'a>(&self, q: OpenLitQuery<'a>) -> Result<PromptTemplate, PromptError> {
        let res = self._fetch(q.into()).await?;

        let template_str = res.prompt.ok_or(PromptError::PromptExtraction(
            "`prompt` field missing in OpenLIT response".into(),
        ))?;

        let version = Option::from(res.version);
        let segment = PromptSegment {
            role: Role::User,
            template: template_str,
            syntax: TemplateSyntax::Moustache,
        };

        let mut metadata = HashMap::new();
        metadata.insert("promptId".to_string(), Value::String(res.prompt_id));
        metadata.insert("versionId".to_string(), Value::String(res.version_id));
        metadata.insert("tags".to_string(), Value::Array(res.tags));
        metadata.insert(
            "metaProperties".to_string(),
            Value::Object(res.meta_properties),
        );

        Ok(PromptTemplate {
            id: format!("openlit:{}", res.name),
            version,
            segments: vec![segment],
            metadata,
        })
    }

    async fn fetch<'a>(
        &self,
        name: &str,
        version: Version<'a>,
    ) -> Result<PromptTemplate, PromptError> {
        self.query(OpenLitQuery {
            query_type: QueryType::ByName(name),
            version,
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito;

    #[tokio::test]
    async fn test_fetch_by_name() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/api/prompt/get-compiled")
            .with_status(200)
            .with_header("content-type", "application/json")
            .match_body(r#"{"name":"test-prompt","compile":false}"#)
            .with_body(
                r#"{
                "err": {},
                "res": {
                    "promptId": "12345678-1234-1234-1234-123456789012",
                    "versionId": "87654321-4321-4321-4321-210987654321",
                    "name": "test-prompt",
                    "version": "1.0.0",
                    "tags": ["test", "example"],
                    "metaProperties": {"author": "test-author"},
                    "prompt": "This is a test prompt with {{variable}}"
                }
            }"#,
            )
            .create();

        let openlit = OpenLit::from_url("test-api-key", server.url());
        let result = openlit.fetch("test-prompt", Version::Latest).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.id, "openlit:test-prompt");
        assert_eq!(template.version, Some("1.0.0".to_string()));
        assert_eq!(template.segments.len(), 1);
        assert_eq!(template.segments[0].role, Role::User);
        assert_eq!(
            template.segments[0].template,
            "This is a test prompt with {{variable}}"
        );
        assert_eq!(
            template.metadata["promptId"],
            "12345678-1234-1234-1234-123456789012"
        );
        assert_eq!(
            template.metadata["versionId"],
            "87654321-4321-4321-4321-210987654321"
        );

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_by_name_with_version() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/api/prompt/get-compiled")
            .with_status(200)
            .with_header("content-type", "application/json")
            .match_body(r#"{"name":"test-prompt","version":"2.0.0","compile":false}"#)
            .with_body(
                r#"{
                "err": {},
                "res": {
                    "promptId": "12345678-1234-1234-1234-123456789012",
                    "versionId": "99999999-9999-9999-9999-999999999999",
                    "name": "test-prompt",
                    "version": "2.0.0",
                    "tags": ["test", "example"],
                    "metaProperties": {"author": "test-author"},
                    "prompt": "This is version 2.0.0 of the test prompt"
                }
            }"#,
            )
            .create();

        let openlit = OpenLit::from_url("test-api-key", server.url());
        let result = openlit.fetch("test-prompt", Version::Ref("2.0.0")).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.version, Some("2.0.0".to_string()));
        assert_eq!(
            template.segments[0].template,
            "This is version 2.0.0 of the test prompt"
        );

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_missing_prompt() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/api/prompt/get-compiled")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "err": {},
                "res": {
                    "promptId": "12345678-1234-1234-1234-123456789012",
                    "versionId": "87654321-4321-4321-4321-210987654321",
                    "name": "test-prompt",
                    "version": "1.0.0",
                    "tags": ["test", "example"],
                    "metaProperties": {"author": "test-author"}
                }
            }"#,
            )
            .create();

        let openlit = OpenLit::from_url("test-api-key", server.url());
        let result = openlit.fetch("test-prompt", Version::Latest).await;

        assert!(result.is_err());
        match result {
            Err(PromptError::PromptExtraction(msg)) => {
                assert!(msg.contains("`prompt` field missing"));
            }
            _ => panic!("Expected PromptError::PromptExtraction"),
        }

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_api_error() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/api/prompt/get-compiled")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "err": {"message": "Prompt not found"},
                "res": null
            }"#,
            )
            .create();

        let openlit = OpenLit::from_url("test-api-key", server.url());
        let result = openlit.fetch("test-prompt", Version::Latest).await;

        assert!(result.is_err());
        match result {
            Err(PromptError::Api(_)) => {
                // Success - we got the expected error type
            }
            _ => panic!("Expected PromptError::Api"),
        }

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_http_error() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("POST", "/api/prompt/get-compiled")
            .with_status(404)
            .with_body("Not found")
            .create();

        let openlit = OpenLit::from_url("test-api-key", server.url());
        let result = openlit.fetch("test-prompt", Version::Latest).await;

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
            .mock("POST", "/api/prompt/get-compiled")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "err": {},
                "res": {
                    "promptId": "12345678-1234-1234-1234-123456789012",
                    "versionId": "87654321-4321-4321-4321-210987654321",
                    "name": "test-prompt",
                    "version": "1.0.0",
                    "tags": ["test", "example"],
                    "metaProperties": {"author": "test-author"},
                    "prompt": "This is a test prompt for query method"
                }
            }"#,
            )
            .create();

        let openlit = OpenLit::from_url("test-api-key", server.url());
        let query = OpenLitQuery {
            query_type: QueryType::ByName("test-prompt"),
            version: Version::Latest,
        };
        let result = openlit.query(query).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.id, "openlit:test-prompt");
        assert_eq!(
            template.segments[0].template,
            "This is a test prompt for query method"
        );

        mock.assert_async().await
    }
}
