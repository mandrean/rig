use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TemplateSyntax {
    #[serde(rename = "f-string")]
    FString, // `{foo}`
    #[serde(rename = "moustache")]
    Moustache, // `{{ foo }}`
}

impl From<Option<&str>> for TemplateSyntax {
    fn from(s: Option<&str>) -> Self {
        match s {
            Some("f-string") => TemplateSyntax::FString,
            Some("moustache") => TemplateSyntax::Moustache,
            _ => TemplateSyntax::Moustache,
        }
    }
}

/// A single message inside a multi-turn prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PromptSegment {
    pub role: Role,
    pub template: String,
    pub syntax: TemplateSyntax,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// Vendor-specific / arbitrary role (`"tool"`, `"function"`, etc.).
    Custom(String),
}

/// A prompt as returned by *any* repository.
#[derive(Debug, Clone, Serialize)]
pub struct PromptTemplate {
    pub id: String,
    pub version: Option<String>,
    pub segments: Vec<PromptSegment>,
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct PromptQuery<'a> {
    pub id: &'a str,
    pub version: Version<'a>,
}

#[derive(Debug, derive_more::Display, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum Version<'a> {
    /// Latest version.
    #[display("latest")]
    Latest,
    /// Version reference. Could be a commit hash, tag, or some other identifier supported by the repository.
    /// Examples: `head`, `latest`, `v1.0.0`, `1234567890abcdef`, etc.
    Ref(&'a str),
}

impl<'a> Version<'a> {
    pub fn is_latest(&self) -> bool {
        matches!(self, Version::Latest)
    }
}

/// Something that *stores* prompts.
#[async_trait]
pub trait PromptRepository: Send + Sync + 'static {
    /// Repository-specific query request type.
    type Query<'a>: Send;

    type Prompt;

    /// Query the repository for a prompt template. Like `fetch()` but with more parameters.
    async fn query<'a>(&self, q: Self::Query<'a>) -> Result<Self::Prompt, PromptError>;

    /// Fetch by id. `version` is optional; adaptor decides what it supports. `None` implies latest.
    async fn fetch<'a>(&self, id: &str, version: Version<'a>) -> Result<Self::Prompt, PromptError>;

    async fn list(&self) -> Result<Vec<PromptSummary>, PromptError> {
        Err(PromptError::Unsupported(
            "repository does not support `list()`".into(),
        ))
    }
}

/// Error type returned by repository adapters
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum PromptError {
    /// The prompt could not be found in the repository.
    #[error("prompt not found")]
    NotFound,

    /// The operation requested is not supported by the repository
    #[error("repository does not implement this feature: {0}")]
    Unsupported(String),

    /// Network‑level failure (DNS, TLS, connection reset, etc.).
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Repository returned a non‑2xx HTTP status or an explicit error object.
    #[error("api error: {0}")]
    Api(String),

    /// Failure while deserialising data returned by the repository.
    #[error("deserialization error: {0}")]
    Deserialization(String),

    /// Failure while serialising data to send to the repository.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// The adapter could not extract a usable prompt from the repository payload.
    #[error("prompt extraction error: {0}")]
    PromptExtraction(String),

    /// Catch‑all for anything else.
    #[error("other: {0}")]
    Other(String),
}

/// Concise metadata about a prompt returned by [`PromptRepository::list`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSummary {
    /// Fully‑qualified identifier understood by [`PromptRepository::fetch`].
    pub id: String,
    /// Latest published version, if the repository supports versioning.
    pub latest_version: Option<String>,
    /// Human‑readable description or title.
    pub description: Option<String>,
    /// Free‑form tags/categories supplied by the repository.
    pub tags: Vec<String>,
}
