use hound::{WavReader, WavSpec};
use rustfft::{FftPlanner, num_complex::Complex};
use std::{f32::consts::PI, fs::File, io::BufReader};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

fn play_song(samples: Vec<f32>, spec: WavSpec) -> Result<(), anyhow::Error> {
    println!(
        "WAV spec: channels={}, sample_rate={}, bits={}",
        spec.channels, spec.sample_rate, spec.bits_per_sample
    );

    println!("Loaded {} samples", samples.len());
    println!(
        "Sample range: min={:.4}, max={:.4}",
        samples.iter().cloned().fold(f32::INFINITY, f32::min),
        samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    let samples = std::sync::Arc::new(std::sync::Mutex::new(samples));
    let sample_index = std::sync::Arc::new(std::sync::Mutex::new(0usize));

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("No output device available");

    let config = device.default_output_config()?;
    println!(
        "Device config: sample_rate={}, channels={}, format={:?}",
        config.sample_rate(),
        config.channels(),
        config.sample_format()
    );

    // CRITICAL: Match channel count
    let wav_channels = spec.channels as usize;
    let device_channels = config.channels() as usize;

    let samples_clone = samples.clone();
    let index_clone = sample_index.clone();

    let stream = device.build_output_stream(
        &config.into(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let samples = samples_clone.lock().unwrap();
            let mut index = index_clone.lock().unwrap();

            for frame in data.chunks_mut(device_channels) {
                if *index < samples.len() {
                    let sample = samples[*index];

                    // Write same sample to all output channels
                    for out_channel in frame.iter_mut() {
                        *out_channel = sample;
                    }

                    // Advance by WAV channel count (skip interleaved channels if stereo WAV)
                    *index += wav_channels;
                } else {
                    for out_channel in frame.iter_mut() {
                        *out_channel = 0.0;
                    }
                }
            }
        },
        |err| eprintln!("Stream error: {}", err),
        None,
    )?;

    stream.play()?;

    println!("Playing without EQ. Press Enter to stop...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    Ok(())
}

pub fn analyze_audio() -> Result<(), anyhow::Error> {
    // Open a WAV file
    let mut reader = WavReader::open("sweet-life-luxury-chill-438146.wav")?;
    let spec = reader.spec();

    // Read audio samples
    let samples = extract_samples(&mut reader)?;

    // Take a segment of the audio for analysis
    let segment_size = 4096;
    let segment = if samples.len() > segment_size {
        // Take a segment from the middle of the audio
        let start = (samples.len() - segment_size) / 2;
        &samples[start..start + segment_size]
    } else {
        &samples
    };

    // Apply a Hann window to reduce spectral leakage
    let mut windowed_segment: Vec<Complex<f32>> = segment
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f32 / segment.len() as f32).cos());
            Complex::new(sample * window, 0.0)
        })
        .collect();

    // Perform FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(windowed_segment.len());
    fft.process(&mut windowed_segment);

    // Calculate magnitude spectrum
    let spectrum: Vec<f32> = windowed_segment
        .iter()
        .map(|c| (c.norm() / segment_size as f32).log10() * 20.0) // Convert to dB
        .collect();

    // Find the peak frequency
    let mut max_magnitude = -f32::INFINITY;
    let mut peak_bin = 0;

    for (i, spectrum) in spectrum.iter().enumerate().take(segment_size / 2).skip(1) {
        if *spectrum > max_magnitude {
            max_magnitude = *spectrum;
            peak_bin = i;
        }
    }

    let peak_frequency = peak_bin as f32 * spec.sample_rate as f32 / segment_size as f32;
    println!("Peak frequency: {:.2} Hz", peak_frequency);

    play_song(samples.clone(), reader.spec())?;

    Ok(())
}

fn extract_samples(reader: &mut WavReader<BufReader<File>>) -> Result<Vec<f32>, anyhow::Error> {
    let spec = reader.spec();
    println!("Audio format: {:?}", spec);

    // Read audio samples
    let samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
        reader
            .samples::<f32>()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>()
    } else {
        // Convert integer samples to float
        match spec.bits_per_sample {
            16 => {
                let scale = 1.0 / 32768.0;
                reader
                    .samples::<i16>()
                    .map(|s| s.unwrap() as f32 * scale)
                    .collect()
            }
            24 => {
                let scale = 1.0 / 8388608.0;
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap() as f32 * scale)
                    .collect()
            }
            32 => {
                let scale = 1.0 / 2147483648.0;
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap() as f32 * scale)
                    .collect()
            }
            _ => return Err(anyhow::anyhow!("Unsupported bit depth")),
        }
    };

    Ok(samples)
}

pub fn get_samples_and_rate() -> Result<(Vec<f32>, u32), anyhow::Error> {
    // Open a WAV file
    let mut reader = WavReader::open("sweet-life-luxury-chill-438146.wav")?;
    let samples = extract_samples(&mut reader)?;

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("No output device available");

    let config = device.default_output_config()?;
    println!(
        "Device config: sample_rate={}, channels={}, format={:?}",
        config.sample_rate(),
        config.channels(),
        config.sample_format()
    );

    Ok((samples, config.sample_rate()))
}
