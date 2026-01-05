#![allow(dead_code)]
mod fft;
mod real_time;

use crate::fft::{analyze_audio, get_samples_and_rate};
use crate::real_time::{run_realtime_eq, run_system_eq};

fn main() -> Result<(), anyhow::Error> {
    run_system_eq()?;
    let (samples, sample_rate) = get_samples_and_rate()?;
    run_realtime_eq(samples, sample_rate)?;
    analyze_audio()?;
    Ok(())
}
