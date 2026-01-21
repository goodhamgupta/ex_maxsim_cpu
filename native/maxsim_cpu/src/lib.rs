//! MaxSim CPU - Rustler NIF bindings for Elixir.
//!
//! High-performance MaxSim scoring using BLAS GEMM + SIMD max reduction.
//!
//! # Platform Requirements
//! - x86_64 with AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)
//! - ARM64/AArch64 (Apple Silicon, AWS Graviton, etc.)

mod core;

use rustler::{Binary, Env, NewBinary, NifResult, Term};
use std::borrow::Cow;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Initialize the NIF module.
fn load(_env: Env, _info: Term) -> bool {
    true
}

rustler::init!("Elixir.ExMaxsimCpu.Nif", load = load);

/// Convert a binary to &[f32], handling alignment.
/// Returns Cow::Borrowed if aligned, Cow::Owned if copy was needed.
fn binary_to_f32<'a>(bin: &'a Binary) -> Result<Cow<'a, [f32]>, &'static str> {
    let bytes = bin.as_slice();
    if !bytes.len().is_multiple_of(4) {
        return Err("Binary length must be a multiple of 4");
    }

    let ptr = bytes.as_ptr() as usize;
    let aligned = ptr.is_multiple_of(std::mem::align_of::<f32>());

    if aligned {
        // SAFETY: alignment checked + length multiple of 4
        let len = bytes.len() / 4;
        let fptr = bytes.as_ptr() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(fptr, len) };
        Ok(Cow::Borrowed(slice))
    } else {
        // Safe, portable fallback (handles unaligned data)
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_ne_bytes(chunk.try_into().unwrap()));
        }
        Ok(Cow::Owned(out))
    }
}

/// Write f32 slice to a new binary.
fn f32_to_binary<'a>(env: Env<'a>, data: &[f32]) -> Binary<'a> {
    let byte_len = data.len() * 4;
    let mut binary = NewBinary::new(env, byte_len);
    let bytes = binary.as_mut_slice();

    for (i, &val) in data.iter().enumerate() {
        let val_bytes = val.to_ne_bytes();
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&val_bytes);
    }

    binary.into()
}

/// Compute MaxSim scores for fixed-length documents.
///
/// # Arguments
/// - `query_bin`: Query vectors as f32 binary [q_len * dim]
/// - `q_len`: Number of query tokens
/// - `dim`: Embedding dimension
/// - `docs_bin`: Document vectors as f32 binary [n_docs * d_len * dim]
/// - `n_docs`: Number of documents
/// - `d_len`: Document length (tokens per doc)
///
/// # Returns
/// Binary containing n_docs f32 scores.
#[rustler::nif(schedule = "DirtyCpu")]
fn maxsim_scores_nif<'a>(
    env: Env<'a>,
    query_bin: Binary,
    q_len: usize,
    dim: usize,
    docs_bin: Binary,
    n_docs: usize,
    d_len: usize,
) -> NifResult<Binary<'a>> {
    // Wrap in catch_unwind to prevent panics from crashing BEAM
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate sizes
        let expected_query_size = q_len * dim * 4;
        let expected_docs_size = n_docs * d_len * dim * 4;

        if query_bin.len() != expected_query_size {
            return Err(format!(
                "Query size mismatch: expected {}, got {}",
                expected_query_size,
                query_bin.len()
            ));
        }

        if docs_bin.len() != expected_docs_size {
            return Err(format!(
                "Docs size mismatch: expected {}, got {}",
                expected_docs_size,
                docs_bin.len()
            ));
        }

        // Convert binaries to f32 slices (handling alignment)
        let query = binary_to_f32(&query_bin).map_err(|e| e.to_string())?;
        let docs = binary_to_f32(&docs_bin).map_err(|e| e.to_string())?;

        // Compute scores
        let scores = core::maxsim_scores_fixed(&query, &docs, q_len, d_len, dim);

        Ok(scores)
    }));

    match result {
        Ok(Ok(scores)) => Ok(f32_to_binary(env, &scores)),
        Ok(Err(msg)) => Err(rustler::Error::Term(Box::new(msg))),
        Err(_) => Err(rustler::Error::Term(Box::new(
            "Internal error: panic in NIF".to_string(),
        ))),
    }
}

/// Compute MaxSim scores for variable-length documents.
///
/// # Arguments
/// - `query_bin`: Query vectors as f32 binary [q_len * dim]
/// - `q_len`: Number of query tokens
/// - `dim`: Embedding dimension
/// - `doc_bins`: List of document binaries, each [doc_len_i * dim]
/// - `doc_lens`: List of document lengths (tokens per doc)
///
/// # Returns
/// Binary containing n_docs f32 scores.
#[rustler::nif(schedule = "DirtyCpu")]
fn maxsim_scores_variable_nif<'a>(
    env: Env<'a>,
    query_bin: Binary,
    q_len: usize,
    dim: usize,
    doc_bins: Vec<Binary>,
    doc_lens: Vec<usize>,
) -> NifResult<Binary<'a>> {
    // Wrap in catch_unwind to prevent panics from crashing BEAM
    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate query size
        let expected_query_size = q_len * dim * 4;
        if query_bin.len() != expected_query_size {
            return Err(format!(
                "Query size mismatch: expected {}, got {}",
                expected_query_size,
                query_bin.len()
            ));
        }

        if doc_bins.len() != doc_lens.len() {
            return Err("doc_bins and doc_lens must have the same length".to_string());
        }

        // Convert query
        let query = binary_to_f32(&query_bin).map_err(|e| e.to_string())?;

        // Convert document binaries and validate sizes
        let doc_data: Result<Vec<(usize, Cow<[f32]>)>, String> = doc_bins
            .iter()
            .zip(doc_lens.iter())
            .enumerate()
            .map(|(idx, (bin, &len))| {
                let expected_size = len * dim * 4;
                if bin.len() != expected_size {
                    return Err(format!(
                        "Doc {} size mismatch: expected {}, got {}",
                        idx,
                        expected_size,
                        bin.len()
                    ));
                }

                let data = binary_to_f32(bin).map_err(|e| e.to_string())?;
                Ok((len, data))
            })
            .collect();

        let doc_data = doc_data?;

        // Create slice references for the algorithm
        let doc_slices: Vec<(usize, &[f32])> = doc_data
            .iter()
            .map(|(len, data)| (*len, data.as_ref()))
            .collect();

        // Compute scores
        let scores = core::maxsim_scores_variable(&query, doc_slices, q_len, dim);

        Ok(scores)
    }));

    match result {
        Ok(Ok(scores)) => Ok(f32_to_binary(env, &scores)),
        Ok(Err(msg)) => Err(rustler::Error::Term(Box::new(msg))),
        Err(_) => Err(rustler::Error::Term(Box::new(
            "Internal error: panic in NIF".to_string(),
        ))),
    }
}
