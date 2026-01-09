//! ## AUDIO DSP THEORY: PARAMETRIC EQUALIZATION & BIQUAD FILTERS
//!
//! This module implements a Real-Time Parametric Equalizer. Below is the
//! underlying theory required to understand the signal flow and mathematics.
//!
//! ---
//!
//! ### 1. THE BIQUAD FILTER (The Building Block)
//! Most digital EQs are built using "Biquads" (short for Bi-Quadratic). It is a
//! second-order IIR (Infinite Impulse Response) filter. We use biquads because
//! higher-order filters are numerically unstable; instead, we chain multiple
//! biquads in series to create complex EQ curves.
//!
//! The mathematical Transfer Function in the Z-domain is:
//! $$H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$$
//!
//! * **b coefficients:** Control the "Zeros" (primarily the boost/cut logic).
//! * **a coefficients:** Control the "Poles" (primarily the resonance/feedback).
//! * **z^-1:** Represents a single sample delay.
//!
//! ---
//!
//! ### 2. TOPOLOGY: DIRECT FORM II TRANSPOSED (DF2T)
//! The code uses the DF2T structure. In DSP, "Topology" refers to how we arrange
//! the additions and delays.
//!
//! **Why DF2T?**
//! 1.  **Memory Efficient:** It only requires two state variables (z1, z2) for storage.
//! 2.  **Numerical Stability:** It is less prone to "quantization noise" or
//!     clipping in fixed-point math than the standard Direct Form I.
//! 3.  **Low Latency:** It processes sample-by-sample with zero algorithmic delay.
//!
//! ---
//!
//! ### 3. PEAKING EQ PARAMETERS (The "Bell" Curve)
//! To calculate the coefficients ($a_n, b_n$), we use three primary user inputs:
//!
//! 1.  **Center Frequency ($f_0$):** The frequency where the boost or cut is strongest.
//! 2.  **Gain (dB):** The amplitude of the peak or dip.
//!     * $Gain > 0$: Boost.
//!     * $Gain < 0$: Cut.
//! 3.  **Q-Factor (Quality):** Defines the "width" of the bell.
//!     * High Q = Narrow, surgical notch.
//!     * Low Q = Wide, musical transparency.
//!
//! ---
//!
//! ### 4. SERIAL PROCESSING (The EQ Chain)
//! This module implements a "Serial" EQ. When we have 10 bands, the audio
//! travels through them like a literal chain:
//! `Input -> Band 1 -> Band 2 -> ... -> Band 10 -> Output`
//!
//! This means the total frequency response is the *product* of all individual
//! transfer functions. If Band 1 boosts 100Hz and Band 2 boosts 100Hz, the
//! result is a cumulative boost.
//!
//! ---
//!
//! ### 5. FREQUENCY ANALYSIS & STATE SANITIZATION
//! To visualize the EQ, we use the "Impulse Response" method, which follows these steps:
//!
//! 1. **State Reset (Sanitization):** Before measurement, we must zero out the filter's
//!    internal delay lines ($z_1, z_2$). If we don't, "energy" left over from previous
//!    audio samples will bleed into the measurement, creating artifacts or "broken" plots.
//!
//! 2. **The Dirac Impulse:** We feed a "Dirac Delta" signal (a single sample of 1.0
//!    followed by a string of 0.0s) into the clean EQ.
//!
//! 3. **Spectral Theory:** Mathematically, a perfect impulse has a "flat" spectrum,
//!    meaning it contains all frequencies at equal amplitude. By passing this through
//!    the EQ, the output becomes the "Impulse Response."
//!
//! 4. **FFT Analysis:** We apply a **Fast Fourier Transform (FFT)** to the Impulse
//!    Response. This algorithm moves the data from the **Time Domain** (amplitude vs. time)
//!    into the **Frequency Domain** (amplitude vs. frequency).
//!
//! 5. **Logarithmic Mapping:** Since human hearing perceives frequency doubled by
//!    octaves (20Hz, 40Hz, 80Hz...) rather than linearly (20Hz, 30Hz, 40Hz...), we
//!    map the linear FFT bins to a logarithmic scale for a "musical" visualization.
//!
use biquad::*;
use cpal::{
    Device, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use ringbuf::traits::{Consumer, Producer, Split};
use rustfft::{
    FftPlanner,
    num_complex::{Complex, ComplexFloat},
};
use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use tracing::{debug, error, info};

/// A single EQ band using the Direct Form 2 Transposed (DF2T) structure.
/// DF2T is preferred in audio because it is more numerically stable than DF1.
#[derive(Clone, Debug)]
struct ParametricBand {
    filter: DirectForm2Transposed<f32>,
    coeffs: Coefficients<f32>,
    frequency: f32,
}

impl ParametricBand {
    /// Creates a new Peaking EQ band.
    /// Peaking EQ allows you to boost or cut a specific frequency range.
    fn new(sample_rate: f32, freq: f32, q: f32, gain_db: f32) -> Self {
        let fs = sample_rate.hz();
        let f0 = freq.hz();

        // Calculate coefficients using the biquad crate.
        // Q_BUTTERWORTH_F32 (0.707) is a common starting point for a "musical" width.
        let coeffs =
            Coefficients::from_params(Type::PeakingEQ(gain_db), fs, f0, Q_BUTTERWORTH_F32 * q)
                .expect("Failed to calculate filter coefficients");

        Self {
            filter: DirectForm2Transposed::new(coeffs),
            coeffs,
            frequency: freq,
        }
    }

    /// Feeds a single sample through the filter logic.
    fn process(&mut self, sample: f32) -> f32 {
        self.filter.run(sample)
    }

    /// Updates filter parameters (Gain, Freq, Q) without reallocating memory.
    fn update(&mut self, sample_rate: f32, freq: f32, q: f32, gain_db: f32) {
        let fs = sample_rate.hz();
        let f0 = freq.hz();
        self.frequency = freq;

        let coeffs =
            Coefficients::from_params(Type::PeakingEQ(gain_db), fs, f0, Q_BUTTERWORTH_F32 * q)
                .expect("Failed to update filter coefficients");

        self.coeffs = coeffs; // Update our stored copy
        self.filter.update_coefficients(coeffs);
    }

    /// It creates a fresh filter with the same math but zeroed-out memory.
    fn reset_state(&mut self) {
        self.filter = DirectForm2Transposed::new(self.coeffs);
    }
}

/// A collection of filters arranged in series.
/// The output of Band 1 is the input of Band 2.
#[derive(Clone, Debug)]
pub struct ParametricEQ {
    bands: Vec<ParametricBand>,
    sample_rate: f32,
}

impl ParametricEQ {
    /// Initializes a 10-band Graphic EQ at standard ISO octave frequencies.
    pub fn new(sample_rate: f32) -> Self {
        let frequencies = [
            31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
        ];
        let q = 1.0;

        let bands = frequencies
            .iter()
            .map(|&freq| ParametricBand::new(sample_rate, freq, q, 0.0))
            .collect();

        Self { bands, sample_rate }
    }

    /// Serial processing: Input -> Band 1 -> Band 2 -> ... -> Output
    pub fn process(&mut self, sample: f32) -> f32 {
        self.bands
            .iter_mut()
            .fold(sample, |acc, band| band.process(acc))
    }

    /// Safely modifies the gain of a specific band.
    fn set_band_gain(&mut self, band_idx: usize, gain_db: f32) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            let freq = band.frequency;
            info!("Setting band {} ({} Hz) to {} dB", band_idx, freq, gain_db);
            band.update(self.sample_rate, freq, 1.0, gain_db);
        }
    }
}

/// The main entry point for real-time audio playback with EQ processing.
///
/// ### Theory of Operation:
/// 1. **Hardware Discovery:** Uses the `cpal` crate to connect to the system's
///    default audio output device (e.g., your headphones or speakers).
///
/// 2. **Clock Domain Matching:** Digital audio relies on a steady "clock."
///    The function identifies the hardware's sample rate. If it differs
///    from the WAV file (e.g., 44.1kHz vs 48kHz), the audio will play
///    at the wrong speed unless a resampler is used.
///
/// 3. **Thread Safety & Concurrency:** Audio hardware runs on its own
///    high-priority system thread. Because the EQ parameters and audio data
///    live on the main thread, we use `Arc` (Atomic Reference Counting) and
///    `Mutex` (Mutual Exclusion) to share data safely across threads.
///
/// 4. **The Audio Callback (The Engine):** This is a high-frequency closure
///    that the hardware calls whenever it needs more audio data.
///    Inside this loop, we:
///    - Pull a sample from the source.
///    - Push it through the EQ filter chain.
///    - Write the result to the hardware buffer.
///
/// ### Parameters:
/// * `samples`: A vector of raw floating-point audio samples (`f32`).
/// * `wav_sample_rate`: The original sample rate of the input file.
///
/// ### Returns:
/// * `Ok(())` if playback completes or is stopped by the user.
/// * `Err` if the audio device is unavailable or the stream fails.
pub fn run_realtime_eq(samples: Vec<f32>, wav_sample_rate: u32) -> Result<(), anyhow::Error> {
    // 1. Initialize the audio host (Windows WASAPI, macOS CoreAudio, or Linux ALSA)
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("No output device found");
    let config = device.default_output_config()?;
    let device_sample_rate = config.sample_rate() as f32;

    info!(
        "WAV rate: {}Hz | Device rate: {}Hz",
        wav_sample_rate, device_sample_rate
    );

    // 2. Instantiate the EQ specifically for the HARDWARE's sample rate.
    // This ensures the math (cutoff frequencies) is accurate to what you hear.
    let mut eq = ParametricEQ::new(device_sample_rate);

    // Initial EQ Settings
    eq.set_band_gain(0, -6.0); // Cut 500Hz
    eq.set_band_gain(5, 6.0); // Boost 8kHz
    eq.set_band_gain(10, -6.0); // Boost 8kHz

    // 3. Visualize the filter curve before starting playback
    plot_frequency_response(&eq, 2048);

    // 4. Wrap data in thread-safe containers
    // Arc: Allows multiple owners (Main thread and Audio thread)
    // Mutex: Prevents both threads from changing the EQ or Index at the same time
    let eq = Arc::new(Mutex::new(eq));
    let samples = Arc::new(samples);
    let index = Arc::new(Mutex::new(0usize));

    // Create clones for the move into the audio thread closure
    let eq_cb = Arc::clone(&eq);
    let samples_cb = Arc::clone(&samples);
    let index_cb = Arc::clone(&index);

    // 5. Build the Output Stream
    // This closure is the "Real-Time" part. It must be fast and should
    // never perform "heavy" tasks like file I/O or networking.
    let stream = device.build_output_stream(
        &config.into(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut eq = eq_cb.lock().unwrap();
            let mut idx = index_cb.lock().unwrap();

            for sample_out in data.iter_mut() {
                if *idx < samples_cb.len() {
                    let input = samples_cb[*idx];

                    // 1. Process through the EQ chain
                    let processed = eq.process(input);

                    // 2. SAFETY: Hard Clipping Limiter
                    // This prevents values > 1.0 or < -1.0 from reaching my speakers
                    *sample_out = processed.clamp(-1.0, 1.0);

                    *idx += 1;
                } else {
                    *sample_out = 0.0;
                }
            }
        },
        |err| error!("Stream error: {}", err),
        None,
    )?;

    // 6. Start the clock
    stream.play()?;

    let controller = EQController::new(Arc::clone(&eq));
    controller.main_loop()?;

    Ok(())
}

/// Analyzes and visualizes the frequency response of the EQ using an Impulse Response and FFT.
///
/// ### How it works:
/// 1. **Impulse Response (Probing):** The function creates a "Dirac Delta" signal (a single 1.0
///    followed by zeros). In signal processing theory, an impulse contains all frequencies
///    at equal energy.
///
/// 2. **Filtering:** This impulse is passed through the EQ. The resulting output is the
///    "Impulse Response," which characterizes how the EQ changes audio over time.
///
/// 3. **FFT (Fast Fourier Transform):** The function performs an FFT on the impulse response.
///    This translates the data from the **Time Domain** (amplitude over time) to the
///    **Frequency Domain** (amplitude over frequency).
///
/// 4. **Logarithmic Mapping:** Human hearing is logarithmic (we perceive octaves, not linear Hz).
///    This function maps the linear FFT bins to a logarithmic scale (20Hz to 20kHz) so the
///    visualization matches how we actually hear music.
///
/// 5. **Decibel Conversion:** Magnitudes are converted to Decibels (dB) using the formula:
///    $dB = 20 * \log_{10}(magnitude)$.
///
/// ### Parameters:
/// * `eq`: A reference to the `ParametricEQ` to be analyzed.
/// * `fft_size`: The number of samples to use for the FFT. Higher values provide better
///   frequency resolution (especially in the bass), but require more computation.
///   2048 or 4096 is standard for audio analysis.
pub fn plot_frequency_response(eq: &ParametricEQ, fft_size: usize) {
    let mut eq_copy = eq.clone();

    // Clear the "ghosts" of previous audio processing
    for band in eq_copy.bands.iter_mut() {
        band.reset_state();
    }

    let mut impulse = vec![0.0; fft_size];
    impulse[0] = 1.0;

    // Now the response will be pure math, no noise!
    let response: Vec<f32> = impulse.iter().map(|&x| eq_copy.process(x)).collect();

    // ... (FFT logic remains the same) ...
    let mut buffer: Vec<Complex<f32>> = response.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut buffer);

    println!("\n--- LOGARITHMIC FREQUENCY RESPONSE ---");
    println!("(Matches human hearing: Bass is expanded, Treble is compressed)");

    let f_min = 20.0; // Starting frequency (sub-bass)
    let f_max = 20000.0; // Ending frequency (limit of human hearing)
    let steps = 30; // Number of lines to print in the console

    for s in 0..steps {
        // Calculate the frequency for this step using an exponential growth formula
        let fraction = s as f32 / (steps - 1) as f32;
        let freq = f_min * (f_max / f_min).powf(fraction);

        // Map the frequency (Hz) to the correct FFT bin index
        // Formula: bin = freq * total_bins / sample_rate
        let bin_idx = (freq * fft_size as f32 / eq.sample_rate) as usize;

        // Ensure we stay within the Nyquist limit (half the FFT size)
        if bin_idx < fft_size / 2 {
            let magnitude = buffer[bin_idx].norm();

            // Convert magnitude to dB. 1.0 = 0dB.
            // We use 1e-10 as a floor to prevent log10(0) which is infinity.
            let db = if magnitude > 1e-10 {
                20.0 * magnitude.log10()
            } else {
                -100.0
            };

            // Create an ASCII bar. We center it around a "0dB" baseline.
            // the multiplier (3.0) makes the bars longer or shorter.
            let bar_len = ((db + 12.0) * 2.0).max(0.0) as usize;
            let bar = "=".repeat(bar_len);

            println!("{:>8.1} Hz | {:>6.2} dB |{}", freq, db, bar);
        }
    }
    println!("--------------------------------------\n");
}

/// A controller used to modify EQ parameters from the main thread
/// while the audio thread is actively processing.
#[derive(Clone, Debug)]
pub struct EQController {
    // We hold a clone of the Arc pointing to the EQ
    eq: Arc<Mutex<ParametricEQ>>,
}

impl EQController {
    pub fn new(eq: Arc<Mutex<ParametricEQ>>) -> Self {
        Self { eq }
    }

    /// Updates a specific band's parameters dynamically.
    ///
    /// ### Theory: Thread-Safe Parameter Updates
    /// When you call this, the main thread "requests" the Mutex lock.
    /// As soon as the audio thread finishes its current buffer (a few milliseconds),
    /// the lock is granted, coefficients are recalculated, and the next
    /// audio buffer is processed with the new sound settings.
    pub fn update_band(&self, index: usize, gain: f32, q: f32) {
        debug!(index, "Updating band");
        if let Ok(mut eq) = self.eq.lock() {
            let sr = eq.sample_rate;
            if let Some(band) = eq.bands.get_mut(index) {
                let freq = band.frequency;

                info!("Dynamic Update: Band {} -> {}dB, Q: {}", index, gain, q);
                band.update(sr, freq, q, gain);
            }
        }
    }

    /// Resets all bands to 0dB (Unity Gain).
    pub fn reset_all(&self) {
        debug!("Reseting audio");
        if let Ok(mut eq) = self.eq.lock() {
            let sr = eq.sample_rate;
            for band in eq.bands.iter_mut() {
                let freq = band.frequency;
                band.update(sr, freq, 1.0, 0.0);
            }
            info!("EQ Reset to Flat.");
        }
    }

    pub fn main_loop(&self) -> Result<(), anyhow::Error> {
        loop {
            println!("\n--- LIVE EQ CONTROL ---");
            println!("Commands: 'set <band_idx> <gain_db>' or 'reset' or 'quit'");
            println!("Example: 'set 4 10.0' (Boosts 500Hz by 10dB)");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let parts: Vec<&str> = input.split_whitespace().collect();

            match parts.as_slice() {
                ["set", idx_str, gain_str] => {
                    if let (Ok(idx), Ok(gain)) = (idx_str.parse::<usize>(), gain_str.parse::<f32>())
                    {
                        self.update_band(idx, gain, 1.0);
                        // Re-plot to see the change visually!
                        let current_eq = self.eq.lock().unwrap().clone();
                        plot_frequency_response(&current_eq, 2048);
                    }
                }
                ["reset"] => self.reset_all(),
                ["quit"] => return Ok(()),
                _ => println!("Unknown command."),
            }
        }
    }
}

/// A container to keep our background audio threads alive.
#[derive(Clone)]
pub struct AudioHandler {
    pub input_config: StreamConfig,
    pub output_config: StreamConfig,
    pub input_device: Device,
    pub output_device: Device,
    pub eq: Arc<Mutex<ParametricEQ>>,
}

impl Debug for AudioHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioHandler")
            .field("input_config", &self.input_config)
            .field("output_config", &self.output_config)
            .field("eq", &self.eq)
            .finish()
    }
}

#[cfg(target_os = "macos")]
pub async fn run_system_eq(name: Option<&str>) -> Result<AudioHandler, anyhow::Error> {
    info!("Connecting to the audio");
    let host = cpal::default_host();

    // Find BlackHole for Input
    let input_device = host
        .input_devices()?
        .find(|x| {
            x.description()
                .map(|n| n.name().contains("2ch"))
                .unwrap_or(false)
        })
        .expect("BlackHole not found!");

    // Find the device if the name is provided or use the default if not
    let output_device = match name {
        Some(name) => host
            .output_devices()?
            .find(|x| {
                x.description()
                    .map(|n| n.name().contains(name))
                    .unwrap_or(false)
            })
            .ok_or(anyhow::anyhow!("{name} not found"))?,
        None => host
            .default_output_device()
            .ok_or(anyhow::anyhow!("default device not found"))?,
    };

    // Get the Output Device's default config first
    let output_config_req = output_device.default_output_config()?;
    let target_sample_rate = output_config_req.sample_rate();

    info!("Targeting output sample rate: {}Hz", target_sample_rate);

    // Find a matching input config on BlackHole
    let mut input_supported = input_device.supported_input_configs()?;
    let input_config: StreamConfig = input_supported
        .find(|c| {
            c.min_sample_rate() <= target_sample_rate && c.max_sample_rate() >= target_sample_rate
        })
        .map(|c| c.with_sample_rate(target_sample_rate))
        .ok_or_else(|| {
            anyhow::anyhow!("BlackHole does not support the output rate of {target_sample_rate}Hz",)
        })?
        .into();

    let output_config: StreamConfig = output_config_req.into();

    // Double-check synchronization
    if input_config.sample_rate != output_config.sample_rate {
        anyhow::bail!(
            "Clock Mismatch! Input: {}Hz, Output: {}Hz",
            input_config.sample_rate,
            output_config.sample_rate
        );
    }

    let eq = Arc::new(Mutex::new(ParametricEQ::new(target_sample_rate as f32)));

    Ok(AudioHandler {
        input_device,
        output_device,
        input_config,
        output_config,
        eq,
    })
}

#[cfg(target_os = "linux")]
pub fn run_system_eq() -> Result<AudioHandler, anyhow::Error> {
    let host = cpal::default_host();
    let device = host.default_output_device().unwrap();
    let config: StreamConfig = device.default_output_config()?.into();

    println!("Device: {}", device.description()?,);

    let (mut producer, mut consumer) = ringbuf::HeapRb::<f32>::new(16384).split();
    let eq = Arc::new(Mutex::new(ParametricEQ::new(config.sample_rate as f32)));

    // Input Stream: This will show up as a node in PipeWire
    let input_stream = device.build_input_stream(
        &config,
        move |data: &[f32], _| {
            producer.push_slice(data);
        },
        |e| error!("{e}"),
        None,
    )?;

    let eq_cb = Arc::clone(&eq);
    let output_stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _| {
            let mut eq = eq_cb.lock().unwrap();
            for s in data {
                // Apply EQ here ONCE
                *s = eq.process(consumer.try_pop().unwrap_or(0.0));
            }
        },
        |e| error!("{e}"),
        None,
    )?;

    Ok(AudioHandler {
        input_device,
        output_device,
        input_config,
        output_config,
        eq,
    })
}

pub async fn get_ouput_device_names() -> Result<Vec<String>, anyhow::Error> {
    let host = cpal::default_host();
    let devices = host.output_devices()?;

    Ok(devices
        .filter_map(|d| Some(d.description().ok()?.name().to_string()))
        .collect())
}

pub async fn init_eq(handler: Arc<AudioHandler>) -> Result<(Stream, Stream), anyhow::Error> {
    // Increase buffer size or ensure it's a power of 2
    // 16384 samples provides a larger safety net for OS scheduling jitter
    let ring_buffer = ringbuf::HeapRb::<f32>::new(16384);
    let (mut producer, mut consumer) = ring_buffer.split();

    let input_stream = handler.input_device.build_input_stream(
        &handler.input_config,
        move |data: &[f32], _| {
            // Push as much as possible
            let _ = producer.push_slice(data);
        },
        |e| error!("Input Error: {e}"),
        None,
    )?;

    let eq_cb = Arc::clone(&handler.eq);
    let output_stream = handler.output_device.build_output_stream(
        &handler.output_config,
        move |data: &mut [f32], _| {
            // Try to lock, but if it's contested, we might prefer
            // to use the last known good coefficients rather than blocking.
            if let Ok(mut eq) = eq_cb.try_lock() {
                for sample_out in data.iter_mut() {
                    if let Some(input) = consumer.try_pop() {
                        *sample_out = eq.process(input);
                    } else {
                        // Buffer underrun: output is faster than input
                        *sample_out = 0.0;
                    }
                }
            } else {
                // If the Mutex is locked by the Web UI,
                // skip EQ processing for this buffer to avoid a hang
                for sample_out in data.iter_mut() {
                    *sample_out = consumer.try_pop().unwrap_or(0.0);
                }
            }
        },
        |e| error!("Output Error: {e}"),
        None,
    )?;

    input_stream.play()?;
    output_stream.play()?;
    Ok((input_stream, output_stream))
}
