#![allow(dead_code)]
mod api;
mod fft;
mod real_time;

use tracing_subscriber::EnvFilter;

use crate::api::serve;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    tracing::info!("Initializing server");
    serve().await
}
