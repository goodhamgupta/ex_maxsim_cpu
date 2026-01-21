//! Core MaxSim algorithm - no Rustler/NIF dependencies here.
//!
//! This module contains the core MaxSim computation logic:
//! - BLAS GEMM for similarity matrix computation
//! - SIMD-optimized max reduction
//! - Rayon parallel processing
//! - libxsmm acceleration (feature-gated)

pub mod algorithm;
pub mod simd;

#[cfg(feature = "use-libxsmm")]
pub mod libxsmm_bindings;

#[cfg(feature = "use-libxsmm")]
pub mod libxsmm;

pub use algorithm::{maxsim_scores_fixed, maxsim_scores_variable};
