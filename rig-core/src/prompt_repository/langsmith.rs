//! LangSmith adapter that implements the unified `PromptRepository` / `PromptCompiler`
//! traits.  Handles both **Chat** and **Instruct** prompts returned by the LangSmith
//! `/commits/{path}/{commit}` endpoint.
//!
//! Design notes
//! -------------
//! * `id` ≡ the *path* component used by LangSmith (e.g. `foo/bar`).
//! * `version` is interpreted as the desired *commit hash*; `None` ⇒ "latest".
//! * The commit hash is stored in `metadata["commit_hash"]`; `PromptTemplate::version` is
//!   `None` because the hash isn't semver.
//! * Chat prompts are flattened into a vector of `PromptSegment`s whose `role` is derived
//!   from the LangSmith message type (`System|Human|AI`).
//! * Instruct prompts become a single `Role::User` segment.
//! * Variable placeholders are read from `input_variables` arrays rather than regex
//!   scraping because LangSmith already exposes them.
//!
//! The file is self‑contained—no dependency on the old `prompt.rs` module.

use super::lib::{PromptError, PromptRepository, PromptSegment, PromptTemplate, Role};
use crate::prompt_repository::{TemplateSyntax, Version};
use async_trait::async_trait;
use reqwest::{self, Client as HttpClient};
use rig::prompt_repository::lib::PromptQuery;
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    env,
};

const LANGSMITH_API_BASE_URL: &str = "https://api.smith.langchain.com";

/// Adapter for LangSmith.
#[derive(Clone)]
pub struct LangSmith {
    base_url: String,
    api_key: String,
    http_client: HttpClient,
}

#[derive(Debug, Deserialize)]
struct CommitResponse {
    commit_hash: String,
    manifest: Value,
    #[allow(dead_code)]
    examples: Vec<Value>,
}

/// Determine role via msg/id array (SystemMessagePromptTemplate, Human..., AI...)
fn determine_role(ids: Option<&Vec<Value>>) -> Role {
    match ids {
        Some(vec) if vec.iter().any(|v| v == "SystemMessagePromptTemplate") => Role::System,
        Some(vec) if vec.iter().any(|v| v == "HumanMessagePromptTemplate") => Role::User,
        Some(vec) if vec.iter().any(|v| v == "AIMessagePromptTemplate") => Role::Assistant,
        _ => Role::Custom("unknown".to_string()),
    }
}

/// Determine the syntax of a template based on the `template_format` pointer.
fn determine_syntax(p: Option<&Value>) -> TemplateSyntax {
    p.and_then(Value::as_str).into()
}

/// Extract the `/id` array from the JSON value.
fn extract_ids(v: &Value) -> Result<Vec<&str>, PromptError> {
    let ids = v
        .pointer("/id")
        .and_then(Value::as_array)
        .ok_or(PromptError::PromptExtraction("missing id array".into()))?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    Ok(ids)
}

impl LangSmith {
    /// Construct with default endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::from_url(api_key, LANGSMITH_API_BASE_URL)
    }

    /// Construct with custom base URL.
    pub fn from_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            http_client: HttpClient::builder()
                .build()
                .expect("LangSmith reqwest client should build"),
        }
    }

    /// From `LANGSMITH_API_KEY` env var.
    pub fn from_env() -> Self {
        let api_key = env::var("LANGSMITH_API_KEY").expect("LANGSMITH_API_KEY not set");
        Self::new(api_key)
    }

    /// Inject your own reqwest client.
    pub fn with_custom_client(mut self, client: HttpClient) -> Self {
        self.http_client = client;
        self
    }

    /// Internal helper to fetch a commit by path and commit hash.
    async fn _fetch<'a>(
        &self,
        path: &str,
        version: Version<'a>,
    ) -> Result<CommitResponse, PromptError> {
        let url = format!(
            "{}/commits/{}/{}",
            self.base_url.trim_end_matches('/'),
            path,
            version
        );

        let resp = self
            .http_client
            .get(url)
            .header("x-api-key", &self.api_key)
            .send()
            .await
            .map_err(|e| PromptError::Network(e.into()))?;

        if !resp.status().is_success() {
            return Err(PromptError::Api(format!(
                "LangSmith returned status {}",
                resp.status()
            )));
        }

        resp.json::<CommitResponse>()
            .await
            .map_err(|e| PromptError::Deserialization(e.to_string()))
    }

    /// Transform LangSmith manifest JSON → [`PromptTemplate`].
    fn manifest_to_template(
        path: &str,
        commit_ref: &str,
        manifest: &Value,
    ) -> Result<PromptTemplate, PromptError> {
        let ids = extract_ids(manifest)?;

        let mut segments = Vec::new();
        let mut variables: HashSet<String> = HashSet::new();

        let is_chat = ids.contains(&"ChatPromptTemplate");
        if is_chat {
            Self::extract_chat_messages(manifest, &mut segments, &mut variables)?;
        } else {
            Self::extract_instruct_prompt(manifest, &mut segments, &mut variables)?;
        }

        Ok(PromptTemplate {
            id: format!("langsmith:{}", path),
            version: Option::from(commit_ref.to_string()),
            segments,
            metadata: HashMap::new(),
        })
    }

    fn extract_instruct_prompt(
        manifest: &Value,
        segments: &mut Vec<PromptSegment>,
        variables: &mut HashSet<String>,
    ) -> Result<(), PromptError> {
        // Instruct prompt: template string at /kwargs/template
        let template = manifest
            .pointer("/kwargs/template")
            .and_then(Value::as_str)
            .ok_or(PromptError::PromptExtraction(
                "instruct prompt missing template".into(),
            ))?
            .to_string();

        let syntax = determine_syntax(manifest.pointer("/kwargs/template_format"));

        segments.push(PromptSegment {
            role: Role::User,
            template,
            syntax,
        });

        if let Some(vars) = manifest
            .pointer("/kwargs/input_variables")
            .and_then(|v| v.as_array())
        {
            for var in vars.iter().filter_map(|v| v.as_str()) {
                variables.insert(var.to_string());
            }
        }
        Ok(())
    }

    fn extract_chat_messages(
        manifest: &Value,
        segments: &mut Vec<PromptSegment>,
        variables: &mut HashSet<String>,
    ) -> Result<(), PromptError> {
        // Extract messages under /kwargs/messages
        let messages = manifest
            .pointer("/kwargs/messages")
            .and_then(|v| v.as_array())
            .ok_or(PromptError::PromptExtraction(
                "chat prompt missing messages".into(),
            ))?;

        for msg in messages {
            let role = determine_role(msg.pointer("/id").and_then(Value::as_array));
            let syntax = determine_syntax(msg.pointer("/kwargs/prompt/kwargs/template_format"));

            let template = msg
                .pointer("/kwargs/prompt/kwargs/template")
                .and_then(Value::as_str)
                .ok_or(PromptError::PromptExtraction(
                    "message missing template".into(),
                ))?
                .to_string();

            segments.push(PromptSegment {
                role,
                template,
                syntax,
            });

            // input variables live at /kwargs/prompt/kwargs/input_variables
            if let Some(vars) = msg
                .pointer("/kwargs/prompt/kwargs/input_variables")
                .and_then(|v| v.as_array())
            {
                for var in vars.iter().filter_map(|v| v.as_str()) {
                    variables.insert(var.to_string());
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl PromptRepository for LangSmith {
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
        let res = self._fetch(id, version).await?;
        Self::manifest_to_template(id, &res.commit_hash, &res.manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_fetch_success() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/commits/test-path/latest")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "commit_hash": "abc123",
                "manifest": {
                    "id": ["ChatPromptTemplate"],
                    "kwargs": {
                        "messages": [
                            {
                                "id": ["SystemMessagePromptTemplate"],
                                "kwargs": {
                                    "prompt": {
                                        "kwargs": {
                                            "template": "You are a helpful assistant.",
                                            "input_variables": []
                                        }
                                    }
                                }
                            },
                            {
                                "id": ["HumanMessagePromptTemplate"],
                                "kwargs": {
                                    "prompt": {
                                        "kwargs": {
                                            "template": "Hello, {{name}}!",
                                            "input_variables": ["name"]
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "examples": []
            }"#,
            )
            .create();

        let langsmith = LangSmith::from_url("test-api-key", &server.url());
        let result = langsmith.fetch("test-path", Version::Latest).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.id, "langsmith:test-path");
        assert_eq!(template.version, Some("abc123".to_string()));
        assert_eq!(template.segments.len(), 2);
        assert_eq!(template.segments[0].role, Role::System);
        assert_eq!(
            template.segments[0].template,
            "You are a helpful assistant."
        );
        assert_eq!(template.segments[1].role, Role::User);
        assert_eq!(template.segments[1].template, "Hello, {{name}}!");

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_instruct_prompt() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/commits/test-path/latest")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "commit_hash": "def456",
                "manifest": {
                    "id": ["PromptTemplate"],
                    "kwargs": {
                        "template": "Answer the following question: {{question}}",
                        "template_format": "f-string",
                        "input_variables": ["question"]
                    }
                },
                "examples": []
            }"#,
            )
            .create();

        let langsmith = LangSmith::from_url("test-api-key", &server.url());
        let result = langsmith.fetch("test-path", Version::Latest).await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.id, "langsmith:test-path");
        assert_eq!(template.version, Some("def456".to_string()));
        assert_eq!(template.segments.len(), 1);
        assert_eq!(template.segments[0].role, Role::User);

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_with_version() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/commits/test-path/specific-version")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "commit_hash": "specific-version",
                "manifest": {
                    "id": ["ChatPromptTemplate"],
                    "kwargs": {
                        "messages": [
                            {
                                "id": ["HumanMessagePromptTemplate"],
                                "kwargs": {
                                    "prompt": {
                                        "kwargs": {
                                            "template": "Version specific prompt",
                                            "input_variables": []
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "examples": []
            }"#,
            )
            .create();

        let langsmith = LangSmith::from_url("test-api-key", &server.url());
        let result = langsmith
            .fetch("test-path", Version::Ref("specific-version"))
            .await;

        assert!(result.is_ok());
        let template = result.unwrap();
        assert_eq!(template.version, Some("specific-version".to_string()));
        assert_eq!(template.segments[0].template, "Version specific prompt");

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_fetch_error_handling() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/commits/test-path/latest")
            .with_status(404)
            .with_body("Not found")
            .create();

        let langsmith = LangSmith::from_url("test-api-key", &server.url());
        let result = langsmith.fetch("test-path", Version::Latest).await;

        assert!(result.is_err());
        if let Err(PromptError::Api(msg)) = result {
            assert!(msg.contains("404"));
        } else {
            panic!("Expected PromptError::Api");
        }

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_query_method() {
        let mut server = mockito::Server::new_async().await;

        let mock = server
            .mock("GET", "/commits/test-path/latest")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "commit_hash": "abc123",
                "manifest": {
                    "id": ["ChatPromptTemplate"],
                    "kwargs": {
                        "messages": [
                            {
                                "id": ["HumanMessagePromptTemplate"],
                                "kwargs": {
                                    "prompt": {
                                        "kwargs": {
                                            "template": "Query test",
                                            "input_variables": []
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "examples": []
            }"#,
            )
            .create();

        let langsmith = LangSmith::from_url("test-api-key", &server.url());
        let query = PromptQuery {
            id: "test-path",
            version: Version::Latest,
        };
        let result = langsmith.query(query).await;

        assert!(result.is_ok());

        mock.assert_async().await
    }

    #[tokio::test]
    async fn test_determine_role() {
        let system_ids = vec![json!("SystemMessagePromptTemplate")];
        assert_eq!(determine_role(Some(&system_ids)), Role::System);

        let human_ids = vec![json!("HumanMessagePromptTemplate")];
        assert_eq!(determine_role(Some(&human_ids)), Role::User);

        let ai_ids = vec![json!("AIMessagePromptTemplate")];
        assert_eq!(determine_role(Some(&ai_ids)), Role::Assistant);

        let unknown_ids = vec![json!("UnknownTemplate")];
        if let Role::Custom(role) = determine_role(Some(&unknown_ids)) {
            assert_eq!(role, "unknown");
        } else {
            panic!("Expected Role::Custom");
        }
    }
}
