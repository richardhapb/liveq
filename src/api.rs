use anyhow::bail;
use axum::{
    Form, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use ringbuf::Arc;
use serde::{Deserialize, Serialize};
use std::{sync::Mutex, time::Duration};
use tera::{Context, Tera};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info, trace};

use crate::real_time::{EQController, init_eq, run_system_eq};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Band {
    id: usize,
    frequency: String,
    gain: f32,
}

#[derive(Clone)]
struct AppState {
    tera: Tera,
    bands: Arc<Mutex<Vec<Band>>>,
    controller: Arc<EQController>,
}

fn init_bands() -> Vec<Band> {
    [
        "32Hz", "64Hz", "125Hz", "250Hz", "500Hz", "1kHz", "2kHz", "4kHz", "8kHz", "16kHz",
    ]
    .iter()
    .enumerate()
    .map(|(id, frequency)| Band {
        id,
        frequency: frequency.to_string(),
        gain: 0.0,
    })
    .collect()
}

async fn index(State(state): State<AppState>) -> impl IntoResponse {
    let bands = state.bands.lock().unwrap().clone();
    let mut context = Context::new();
    context.insert("bands", &bands);
    context.insert("status", "");

    trace!(?context);

    match state.tera.render("index.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            error!(%e, "Template error"),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct UpdateGain {
    gain: f32,
}

async fn update_band(
    Path(id): Path<usize>,
    State(state): State<AppState>,
    Form(params): Form<UpdateGain>,
) -> impl IntoResponse {
    debug!(%id, "updating");
    let mut bands = state.bands.lock().unwrap();

    if let Some(band) = bands.get_mut(id) {
        band.gain = params.gain.clamp(-12.0, 12.0);
        state.controller.update_band(id, band.gain, 1.0);
        info!(
            "Updated band {} ({}) to {} dB",
            id, band.frequency, band.gain
        );

        let mut context = Context::new();
        context.insert("band", band);

        match state.tera.render("band.html", &context) {
            Ok(html) => Html(html).into_response(),
            Err(e) => {
                error!(%e, "rendering equalizer");
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {e}")).into_response()
            }
        }
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

async fn reset_equalizer(State(state): State<AppState>) -> impl IntoResponse {
    let mut bands = state.bands.lock().unwrap();
    *bands = init_bands();

    let mut context = Context::new();
    context.insert("bands", &*bands);

    state.controller.reset_all();

    trace!(?context);

    match state.tera.render("equalizer.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(e) => {
            error!(%e, "reseting equalizer");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {e}")).into_response()
        }
    }
}

#[derive(Deserialize)]
struct PresetForm {
    preset: String,
}

async fn apply_preset(
    State(state): State<AppState>,
    Form(form): Form<PresetForm>,
) -> impl IntoResponse {
    let mut bands = state.bands.lock().unwrap();

    let gains: Vec<f32> = match form.preset.as_str() {
        "rock" => vec![5.0, 3.0, -1.0, -2.0, -1.0, 1.0, 3.0, 4.0, 5.0, 5.0],
        "jazz" => vec![4.0, 3.0, 1.0, 2.0, -1.0, -1.0, 0.0, 1.0, 3.0, 4.0],
        "electronic" => vec![6.0, 5.0, 0.0, -2.0, -2.0, 0.0, 2.0, 4.0, 5.0, 6.0],
        _ => vec![0.0; 10], // flat
    };

    for (band, &gain) in bands.iter_mut().zip(gains.iter()) {
        band.gain = gain;
    }

    let mut context = Context::new();
    context.insert("bands", &*bands);

    match state.tera.render("equalizer.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e)).into_response(),
    }
}

pub async fn serve() -> Result<(), anyhow::Error> {
    let tera = match Tera::new("templates/**/*") {
        Ok(t) => t,
        Err(e) => bail!("Tera parsing error: {}", e),
    };

    debug!(templates=?tera.get_template_names().collect::<Vec<_>>());

    let handler = run_system_eq().await?;

    let state = AppState {
        tera,
        bands: Arc::new(Mutex::new(init_bands())),
        controller: Arc::new(EQController::new(Arc::clone(&handler.eq))),
    };

    tokio::spawn(async move {
        let handler_ref = Arc::new(handler);
        loop {
            match init_eq(Arc::clone(&handler_ref)).await {
                Ok(_) => {
                    info!("Closing equalizer");
                    break;
                }
                Err(e) => {
                    let seconds = 10;
                    error!(%e, "re-trying in {seconds} seconds");
                    tokio::time::sleep(Duration::from_secs(seconds)).await;
                }
            }
        }
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/equalizer/update/{id}", post(update_band))
        .route("/equalizer/reset", post(reset_equalizer))
        .route("/equalizer/preset", post(apply_preset))
        .nest_service("/static", ServeDir::new("static"))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:9000").await?;

    info!("Server running on http://127.0.0.1:3000");
    axum::serve(listener, app).await?;

    Ok(())
}
