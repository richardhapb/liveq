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
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info, trace};

use crate::real_time::{
    AudioHandler, EQController, get_ouput_device_names, init_eq, run_system_eq,
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Band {
    id: usize,
    frequency: String,
    gain: f32,
}

#[derive(Debug, Clone)]
struct AppState {
    tera: Tera,
    bands: Arc<Mutex<Vec<Band>>>,
    devices: Vec<Device>,
    session: Arc<Mutex<ActiveSession>>,
}

#[derive(Debug)]
struct ActiveSession {
    controller: Arc<EQController>,
    eq_job: JoinHandle<()>,
    cancel_token: Arc<CancellationToken>,
    selected_device: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpdateGain {
    gain: f32,
}

#[derive(Debug, Deserialize)]
struct PresetForm {
    preset: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Device {
    name: String,
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
    context.insert("devices", &state.devices);
    context.insert(
        "selected_device",
        &state.session.lock().unwrap().selected_device,
    );

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

/// Update the gain of the band
async fn update_band(
    Path(id): Path<usize>,
    State(state): State<AppState>,
    Form(params): Form<UpdateGain>,
) -> impl IntoResponse {
    debug!(%id, "updating");
    let mut bands = state.bands.lock().unwrap();

    if let Some(band) = bands.get_mut(id) {
        band.gain = params.gain.clamp(-12.0, 12.0);
        {
            let controller = &state.session.lock().unwrap().controller;

            update_band_eq(id, band.gain, controller);
        }

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

/// Reset the equalizer to a flat mode
async fn reset_equalizer(State(state): State<AppState>) -> impl IntoResponse {
    let mut bands = state.bands.lock().unwrap();
    *bands = init_bands();

    let mut context = Context::new();
    context.insert("bands", &*bands);

    state.session.lock().unwrap().controller.reset_all();

    trace!(?context);

    match state.tera.render("equalizer.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(e) => {
            error!(%e, "reseting equalizer");
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {e}")).into_response()
        }
    }
}

/// Apply a selected preset
async fn apply_preset(
    State(state): State<AppState>,
    Form(form): Form<PresetForm>,
) -> impl IntoResponse {
    let mut bands = state.bands.lock().unwrap();
    let preset_name = form.preset.as_str();

    let gains: Vec<f32> = match preset_name {
        "rock" => vec![
            5.0,  // ~31 Hz  -> sub-bass weight (kick / low-end impact)
            3.0,  // ~63 Hz  -> bass punch
            -1.0, // ~125 Hz -> reduce mud in guitars
            -2.0, // ~250 Hz -> cut boxiness
            -1.0, // ~500 Hz -> clean low-mids
            1.0,  // ~1 kHz  -> mid presence (guitar/vocal body)
            3.0,  // ~2 kHz  -> attack and clarity (guitars, snare)
            4.0,  // ~4 kHz  -> edge and aggression
            5.0,  // ~8 kHz  -> brightness
            5.0,  // ~16 kHz -> air / excitement
        ],
        "jazz" => vec![
            4.0,  // ~31 Hz  -> subtle low-end warmth (upright bass)
            3.0,  // ~63 Hz  -> bass body
            1.0,  // ~125 Hz -> natural low-mid fullness
            2.0,  // ~250 Hz -> instrument body (piano, horns)
            -1.0, // ~500 Hz -> reduce muddiness
            -1.0, // ~1 kHz  -> soften forward mids
            0.0,  // ~2 kHz  -> neutral presence
            1.0,  // ~4 kHz  -> articulation without harshness
            3.0,  // ~8 kHz  -> cymbal detail
            4.0,  // ~16 kHz -> air and openness
        ],

        "electronic" => vec![
            6.0,  // ~31 Hz  -> sub-bass power (808 / synth subs)
            5.0,  // ~63 Hz  -> bass energy
            0.0,  // ~125 Hz -> avoid low-end congestion
            -2.0, // ~250 Hz -> remove mud
            -2.0, // ~500 Hz -> hollow mids (space for synths)
            0.0,  // ~1 kHz  -> neutral
            2.0,  // ~2 kHz  -> transient clarity
            4.0,  // ~4 kHz  -> sparkle and attack
            5.0,  // ~8 kHz  -> brightness
            6.0,  // ~16 kHz -> air / hype
        ],
        "vocal" => vec![
            -6.0, // ~31 Hz  -> remove rumble
            -4.0, // ~63 Hz  -> reduce boom
            -2.0, // ~125 Hz -> clean low body
            0.0,  // ~250 Hz -> neutral (avoid boxiness)
            2.0,  // ~500 Hz -> clarity
            4.0,  // ~1 kHz  -> intelligibility
            5.0,  // ~2 kHz  -> presence
            4.0,  // ~4 kHz  -> articulation
            2.0,  // ~8 kHz  -> light sibilance lift
            1.0,  // ~16 kHz -> air
        ],
        _ => vec![0.0; 10], // flat
    };

    info!(preset = preset_name, "Applying");

    {
        let controller = &state.session.lock().unwrap().controller;
        for (band, &gain) in bands.iter_mut().zip(gains.iter()) {
            band.gain = gain;
            update_band_eq(band.id, band.gain, controller);
        }
    }

    let mut context = Context::new();
    context.insert("bands", &*bands);

    match state.tera.render("equalizer.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {e}")).into_response(),
    }
}

/// Change the audio device stopping the active device
async fn select_device(
    State(state): State<AppState>,
    Form(device): Form<Device>,
) -> impl IntoResponse {
    debug!(?device, "selecting device");

    // Initialize the new hardware
    let (handler, controller) = match initialize_eq(Some(&device.name)).await {
        Ok(res) => res,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("Init error: {e}")).into_response(),
    };

    let new_cancel_token = Arc::new(CancellationToken::new());
    let new_job = match spawn_eq(handler, Arc::clone(&new_cancel_token)).await {
        Ok(job) => job,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Spawn error: {e}"),
            )
                .into_response();
        }
    };

    // Lock the session and swap them
    {
        let mut session = state.session.lock().unwrap();

        // Cancel the old task
        session.cancel_token.cancel();
        // Abort the old handle to be sure
        session.eq_job.abort();

        // Update the global state with new values
        *session = ActiveSession {
            controller,
            eq_job: new_job,
            cancel_token: new_cancel_token,
            selected_device: Some(device.name.to_string()),
        };
    }

    StatusCode::OK.into_response()
}

async fn rescan(State(mut state): State<AppState>) -> impl IntoResponse {
    debug!("re-scanning devices");

    match get_output_devices().await {
        Ok(devices) => state.devices = devices,
        Err(e) => {
            error!(%e);
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    let selected_device = state
        .session
        .lock()
        .unwrap()
        .selected_device
        .as_ref()
        .and_then(|sd| state.devices.iter().find(|d| d.name == *sd))
        .as_ref()
        .map(|sel_device| sel_device.name.clone());

    let mut context = Context::new();
    context.insert("devices", &state.devices);
    context.insert("selected_device", &selected_device.unwrap_or("".into()));

    match state.tera.render("devices.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {e}")).into_response(),
    }
}

/// Update a single band using the [`EQController`]
fn update_band_eq(id: usize, gain: f32, controller: &EQController) {
    controller.update_band(id, gain, 1.0);
    info!("Updated band {} to {} dB", id, gain);
}

/// Iniitialize all the config of the EQ
async fn initialize_eq(
    name: Option<&str>,
) -> Result<(Arc<AudioHandler>, Arc<EQController>), anyhow::Error> {
    let handler = Arc::new(run_system_eq(name).await?);
    let controller = Arc::new(EQController::new(Arc::clone(&handler.eq)));

    Ok((handler, controller))
}

/// Spawn a task that keeps the audio playing on the selected device
async fn spawn_eq(
    handler: Arc<AudioHandler>,
    cancel_token: Arc<CancellationToken>,
) -> Result<JoinHandle<()>, anyhow::Error> {
    Ok(tokio::spawn(async move {
        match init_eq(Arc::clone(&handler)).await {
            Ok((input_stream, output_stream)) => {
                debug!("EQ initialized, waiting for cancel trigger");
                cancel_token.cancelled().await;

                // Clean shutdown
                drop(input_stream);
                drop(output_stream);
                info!("Streams stopped cleanly");
            }
            Err(e) => {
                let seconds = 10;
                error!(%e, "re-trying in {seconds} seconds");
                tokio::time::sleep(Duration::from_secs(seconds)).await;
            }
        }
    }))
}

/// Get a list of available output devices
async fn get_output_devices() -> Result<Vec<Device>, anyhow::Error> {
    let device_names: Vec<String> = get_ouput_device_names().await?;
    Ok(device_names
        .into_iter()
        .map(|name| Device { name })
        .collect())
}

pub async fn serve() -> Result<(), anyhow::Error> {
    let tera = match Tera::new("templates/**/*") {
        Ok(t) => t,
        Err(e) => bail!("Tera parsing error: {e}"),
    };

    debug!(templates=?tera.get_template_names().collect::<Vec<_>>());

    let (handler, controller) = initialize_eq(None).await?;
    let cancel_token = Arc::new(CancellationToken::new());
    let eq_job = spawn_eq(handler, Arc::clone(&cancel_token)).await?;
    let devices = get_output_devices().await?;

    let session = Arc::new(Mutex::new(ActiveSession {
        controller,
        eq_job,
        cancel_token,
        selected_device: None,
    }));

    let state = AppState {
        tera,
        bands: Arc::new(Mutex::new(init_bands())),
        devices,
        session,
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/equalizer/update/{id}", post(update_band))
        .route("/equalizer/reset", post(reset_equalizer))
        .route("/equalizer/preset", post(apply_preset))
        .route("/device", post(select_device))
        .route("/rescan", post(rescan))
        .nest_service("/static", ServeDir::new("static"))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:9000").await?;

    info!("Server running on http://127.0.0.1:3000");
    axum::serve(listener, app).await?;

    Ok(())
}
