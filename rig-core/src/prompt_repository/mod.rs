//! This module contains clients for different prompt providers that Rig supports.
//!
//! Currently, the following providers are supported:
//! - LangSmith
//! - Haystack PromptHub
//! - PromptHub
//! - OpenLIT
//!
//! Each provider has its own module, which contains a `Client` implementation that can
//! be used to retrieve prompts from the respective provider.
//!
//! # Example (LangSmith)
//! ```
//! use rig::prompt_repository::langsmith;
//!
//! // Initialize the LangSmith client
//! let hub = langsmith::Client::new("your-langsmith-api-key");
//!
//! // Retrieve a prompt
//! let prompt = hub.prompt("hardkothari/prompt-maker");
//! ```
//!
//! # Example (Haystack PromptHub)
//! ```
//! use rig::prompt_repository::haystack;
//!
//! // Initialize the Haystack PromptHub client
//! let hub = haystack::Client::new();
//!
//! // Retrieve a prompt
//! let prompt = hub.prompt("deepset/few-shot-hotpot-qa");
//! ```
//!
//! # Example (PromptHub)
//! ```
//! use rig::prompt_repository::promthub;
//!
//! // Initialize the PromptHub client
//! let hub = promthub::Client::new("your-prompthub-api-key");
//!
//! // Retrieve a prompt
//! let prompt = hub.prompt("18531", "1874502c");
//! ```
//!
//! # Example (OpenLIT)
//! ```
//! use rig::prompt_repository::openlit;
//! use rig::prompt_repository::openlit::GetPrompt;
//! use std::collections::HashMap;
//!
//! // Initialize the OpenLIT client
//! let hub = openlit::Client::new("your-openlit-api-key");
//!
//! // Retrieve a prompt by ID
//! let prompt = hub.prompt(
//!     GetPrompt::ById("a7a55e48-1588-44ee-99e2-d9de5c026238".to_string()),
//!     Some("1.0.0".to_string()),
//!     Some(true),
//!     Some(HashMap::new()),
//!     None
//! );
//! ```

pub mod haystack;
pub mod langsmith;
mod lib;
pub mod openlit;
pub mod promthub;

// Re-export the clients for easier access
pub use haystack::HaystackPromptHub;
pub use langsmith::LangSmith;
pub use lib::*;
pub use openlit::OpenLit;
pub use promthub::PromptHub;

#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum PromptRepositoryError {
    #[error("Request error: {0}")]
    RequestError(String),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Failed to serialize object: {0}")]
    SerializationError(String),
    #[error("Failed to deserialize object: {0}")]
    DeserializationError(String),
    #[error("Failed to extract prompt")]
    PromptExtractionError,
}
