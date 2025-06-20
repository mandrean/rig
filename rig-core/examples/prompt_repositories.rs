use rig::prompt_repository::openlit::{OpenLitQuery, QueryType};
use rig::prompt_repository::{
    HaystackPromptHub, LangSmith, OpenLit, PromptHub, PromptRepository, Version,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: LangSmith
    println!("=== LangSmith Example ===");

    // Create a LangSmith client
    // This requires the `LANGSMITH_API_KEY` environment variable to be set
    let langsmith = LangSmith::from_env();

    // Alternatively, you can provide the API key directly
    // let langsmith = LangSmith::new("your-langsmith-api-key");

    // Retrieve a prompt by its path
    let prompt1 = langsmith
        .fetch("hardkothari/prompt-maker", Version::Latest)
        .await?;

    let prompt2 = langsmith
        .fetch("gitmaxd/synthetic-training-data", Version::Latest)
        .await?;

    println!("Retrieved LangSmith prompt template:");
    println!("{}", serde_json::to_string_pretty(&prompt1)?);
    println!("{}", serde_json::to_string_pretty(&prompt2)?);
    println!();

    // Example 2: Haystack PromptHub
    println!("=== Haystack PromptHub Example ===");

    // Create a Haystack PromptHub client
    let haystack = HaystackPromptHub::new();

    // Retrieve a prompt by its ID
    let prompt = haystack
        .fetch("deepset/few-shot-hotpot-qa", Version::Latest)
        .await?;

    println!("Retrieved Haystack PromptHub prompt:");
    println!("{}", serde_json::to_string_pretty(&prompt)?);
    println!();

    // Example 3: PromptHub
    println!("=== PromptHub Example ===");

    // Create a PromptHub client
    // This requires the `PROMPT_HUB_API_KEY` environment variable to be set
    let prompthub = PromptHub::from_env();

    // Alternatively, you can provide the API key directly
    // let prompthub = prompthub::new("your-prompthub-api-key");

    // Retrieve a prompt by its project ID
    let prompt = prompthub.fetch("18531", Version::Latest).await?;

    // You can also use the query method with a PromptQuery
    // let query = PromptQuery {
    //     id: "18531",
    //     version: Some("head"),
    // };
    // let prompt = prompthub.query(query).await?;

    println!("Retrieved PromptHub prompt:");
    println!("{}", serde_json::to_string_pretty(&prompt)?);
    println!();

    // Example 4: OpenLIT
    println!("=== OpenLIT Example ===");

    // Create an OpenLIT client
    let openlit = OpenLit::from_url("your-openlit-api-key", "http://localhost:3000");

    // Retrieve a prompt by its name using fetch
    let prompt_name = "testprompt";
    let prompt1 = openlit.fetch(prompt_name, Version::Latest).await?;

    // Alternatively, you can use the query method with OpenLitQuery
    let query = OpenLitQuery {
        query_type: QueryType::ById("22843c90-936c-4534-9b9f-69a6e26f1504"),
        version: Version::Ref("0.1.0"),
    };
    let prompt2 = openlit.query(query).await?;

    println!("Retrieved OpenLIT prompt:");
    println!("{}", serde_json::to_string_pretty(&prompt1)?);
    println!("{}", serde_json::to_string_pretty(&prompt2)?);

    Ok(())
}
