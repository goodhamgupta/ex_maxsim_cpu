//! MaxSim algorithm implementations.
//!
//! Core algorithms for computing MaxSim scores using BLAS GEMM.

use super::simd::simd_max;
use blas::sgemm;
use rayon::prelude::*;
use std::cell::RefCell;

// Ensure blas-src is linked (needed on macOS for Accelerate)
#[cfg(target_os = "macos")]
extern crate accelerate_src;

#[cfg(target_os = "macos")]
extern crate blas_src;

// Thread-local buffers to avoid repeated allocations
thread_local! {
    static SIMILARITY_BUFFER: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static BATCH_BUFFER: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Compute MaxSim scores for fixed-length documents.
///
/// # Arguments
/// * `q` - Query vectors as flat slice [q_len * dim]
/// * `d` - Document vectors as flat slice [n_docs * d_len * dim]
/// * `q_len` - Number of query tokens
/// * `d_len` - Number of document tokens (uniform)
/// * `dim` - Embedding dimension
///
/// # Returns
/// Vector of scores, one per document.
pub fn maxsim_scores_fixed(
    q: &[f32],
    d: &[f32],
    q_len: usize,
    d_len: usize,
    dim: usize,
) -> Vec<f32> {
    #[cfg(feature = "use-libxsmm")]
    {
        super::libxsmm::maxsim_libxsmm_clean(q, d, q_len, d_len, dim)
    }

    #[cfg(not(feature = "use-libxsmm"))]
    {
        maxsim_fused_doc_tiles(q, d, q_len, d_len, dim)
    }
}

/// Compute MaxSim scores for variable-length documents.
///
/// # Arguments
/// * `q` - Query vectors as flat slice [q_len * dim]
/// * `doc_data` - Vector of (doc_len, doc_slice) tuples
/// * `q_len` - Number of query tokens
/// * `dim` - Embedding dimension
///
/// # Returns
/// Vector of scores, one per document.
pub fn maxsim_scores_variable(
    q: &[f32],
    doc_data: Vec<(usize, &[f32])>,
    q_len: usize,
    dim: usize,
) -> Vec<f32> {
    #[cfg(feature = "use-libxsmm")]
    {
        let doc_infos: Vec<(usize, usize, &[f32])> = doc_data
            .into_iter()
            .enumerate()
            .map(|(idx, (len, data))| (idx, len, data))
            .collect();
        super::libxsmm::maxsim_libxsmm_variable(q, doc_infos, q_len, dim)
    }

    #[cfg(not(feature = "use-libxsmm"))]
    {
        maxsim_variable_length_impl(q, doc_data, q_len, dim)
    }
}

/// Process a single variable-length document directly
fn process_single_doc(q: &[f32], doc: &[f32], q_len: usize, doc_len: usize, dim: usize) -> f32 {
    // Use thread-local buffer to avoid allocations
    SIMILARITY_BUFFER.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        buffer.resize(q_len * doc_len, 0.0);

        // Compute Q × D^T
        unsafe {
            sgemm(
                b'T',
                b'N',
                doc_len as i32,
                q_len as i32,
                dim as i32,
                1.0,
                doc,
                dim as i32,
                q,
                dim as i32,
                0.0,
                buffer.as_mut_slice(),
                doc_len as i32,
            );
        }

        // Find max for each query and sum
        let mut score = 0.0f32;
        for qi in 0..q_len {
            let start = qi * doc_len;
            let query_sims = &buffer[start..start + doc_len];
            score += simd_max(query_sims);
        }

        score
    })
}

/// Fused GEMM+reduction with document tiling
fn maxsim_fused_doc_tiles(
    q: &[f32],
    d: &[f32],
    q_len: usize,
    d_len: usize,
    dim: usize,
) -> Vec<f32> {
    let n_docs = d.len() / (d_len * dim);

    // For macOS/ARM, use more efficient processing strategy
    #[cfg(target_arch = "aarch64")]
    {
        // Process documents in parallel without excessive tiling
        // ARM has unified memory architecture, so tiling is not anywhere near as important.
        (0..n_docs)
            .into_par_iter()
            .map(|doc_idx| {
                let doc_offset = doc_idx * d_len * dim;
                let doc_data = &d[doc_offset..doc_offset + d_len * dim];

                // Process in smaller blocks to fit in L2 cache
                let block_size = 64;
                let mut max_vals = vec![f32::NEG_INFINITY; q_len];

                for block_start in (0..d_len).step_by(block_size) {
                    let block_end = (block_start + block_size).min(d_len);
                    let actual_block_size = block_end - block_start;

                    // Compute similarities for this block
                    let mut block_sims = vec![0.0f32; q_len * actual_block_size];
                    let block_data = &doc_data[block_start * dim..block_end * dim];

                    unsafe {
                        sgemm(
                            b'T',
                            b'N',
                            actual_block_size as i32,
                            q_len as i32,
                            dim as i32,
                            1.0,
                            block_data,
                            dim as i32,
                            q,
                            dim as i32,
                            0.0,
                            &mut block_sims,
                            actual_block_size as i32,
                        );
                    }

                    // Update max values using NEON
                    for (qi, max_val_ref) in max_vals.iter_mut().enumerate() {
                        let base_idx = qi * actual_block_size;
                        let query_sims = &block_sims[base_idx..base_idx + actual_block_size];
                        let max_val = simd_max(query_sims);
                        *max_val_ref = max_val_ref.max(max_val);
                    }
                }

                // Sum max values
                max_vals.iter().sum()
            })
            .collect()
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut results = vec![0.0f32; n_docs];

        // x86 tiling strategy
        let doc_tile_size = match d_len {
            512 => 128,
            1024 => 64,
            2048 => 32,
            4096 => 16,
            _ => 32,
        };

        for doc_tile_start in (0..n_docs).step_by(doc_tile_size) {
            let doc_tile_end = (doc_tile_start + doc_tile_size).min(n_docs);
            let tile_docs = doc_tile_end - doc_tile_start;
            let tile_tokens = tile_docs * d_len;

            let mut tile_sims = vec![0.0f32; q_len * tile_tokens];
            let tile_d_start = doc_tile_start * d_len * dim;
            let tile_d_end = doc_tile_end * d_len * dim;
            let tile_d = &d[tile_d_start..tile_d_end];

            unsafe {
                sgemm(
                    b'T',
                    b'N',
                    tile_tokens as i32,
                    q_len as i32,
                    dim as i32,
                    1.0,
                    tile_d,
                    dim as i32,
                    q,
                    dim as i32,
                    0.0,
                    &mut tile_sims,
                    tile_tokens as i32,
                );
            }

            let tile_results: Vec<f32> = (0..tile_docs)
                .into_par_iter()
                .map(|tile_doc_idx| {
                    let doc_start = tile_doc_idx * d_len;
                    let mut score = 0.0f32;

                    for qi in 0..q_len {
                        let base_idx = doc_start + qi * tile_tokens;
                        let doc_sims = &tile_sims[base_idx..base_idx + d_len];
                        let max_val = simd_max(doc_sims);
                        score += max_val;
                    }

                    score
                })
                .collect();

            for (i, &score) in tile_results.iter().enumerate() {
                results[doc_tile_start + i] = score;
            }
        }

        results
    }
}

/// Process variable-length documents with optimized batching
fn maxsim_variable_length_impl(
    q: &[f32],
    doc_data: Vec<(usize, &[f32])>,
    q_len: usize,
    dim: usize,
) -> Vec<f32> {
    let n_docs = doc_data.len();
    let mut results = vec![0.0f32; n_docs];

    // Fast path: if all documents have similar lengths, process in one batch
    let (min_len, max_len) = doc_data
        .iter()
        .map(|(len, _)| *len)
        .fold((usize::MAX, 0), |(min, max), len| {
            (min.min(len), max.max(len))
        });

    if max_len as f32 / min_len as f32 <= 1.2 && n_docs >= 50 {
        // All documents have similar lengths - process in single batch
        return BATCH_BUFFER.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            let required_size = n_docs * max_len * dim;
            buffer.resize(required_size, 0.0);
            buffer.fill(0.0);

            // Fill all documents
            for (idx, (doc_len, doc_slice)) in doc_data.iter().enumerate() {
                let src_size = doc_len * dim;
                let dst_offset = idx * max_len * dim;
                buffer[dst_offset..dst_offset + src_size].copy_from_slice(&doc_slice[..src_size]);
            }

            // Process all at once
            maxsim_fused_doc_tiles(q, &buffer[..required_size], q_len, max_len, dim)
        });
    }

    // Sort documents by length for better batching
    let mut sorted_indices: Vec<usize> = (0..n_docs).collect();
    sorted_indices.sort_by_key(|&i| doc_data[i].0);

    // Process in larger batches with adaptive sizing
    let target_batch_size = 128;
    let mut i = 0;

    while i < n_docs {
        // Find batch end - include docs within 20% length difference
        let base_len = doc_data[sorted_indices[i]].0;
        let max_acceptable_len = (base_len as f32 * 1.2) as usize;

        let mut batch_end = i + 1;
        while batch_end < n_docs && batch_end < i + target_batch_size {
            if doc_data[sorted_indices[batch_end]].0 > max_acceptable_len {
                break;
            }
            batch_end += 1;
        }

        let batch_size = batch_end - i;

        if batch_size == 1 {
            // Single document
            let idx = sorted_indices[i];
            let (doc_len, doc_slice) = doc_data[idx];
            results[idx] = process_single_doc(q, doc_slice, q_len, doc_len, dim);
        } else if batch_size >= 32 {
            // Large batch - worth the overhead of batched processing
            let first_len = doc_data[sorted_indices[i]].0;
            let all_same_length = sorted_indices[i..batch_end]
                .iter()
                .all(|&idx| doc_data[idx].0 == first_len);

            if all_same_length {
                // Super optimized path - no padding needed!
                let batch_results = BATCH_BUFFER.with(|buffer| {
                    let mut buffer = buffer.borrow_mut();
                    let required_size = batch_size * first_len * dim;
                    buffer.resize(required_size, 0.0);

                    // Copy documents contiguously
                    for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate()
                    {
                        let (_, doc_slice) = doc_data[sorted_idx];
                        let dst_offset = batch_idx * first_len * dim;
                        buffer[dst_offset..dst_offset + first_len * dim]
                            .copy_from_slice(&doc_slice[..first_len * dim]);
                    }

                    // Process with no wasted computation
                    maxsim_fused_doc_tiles(q, &buffer[..required_size], q_len, first_len, dim)
                });

                // Copy results back
                for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate() {
                    results[sorted_idx] = batch_results[batch_idx];
                }
            } else {
                // Batch processing with padding
                let max_len_batch = sorted_indices[i..batch_end]
                    .iter()
                    .map(|&idx| doc_data[idx].0)
                    .max()
                    .unwrap();

                let batch_results = BATCH_BUFFER.with(|buffer| {
                    let mut buffer = buffer.borrow_mut();
                    let required_size = batch_size * max_len_batch * dim;

                    buffer.resize(required_size, 0.0);

                    // Fill batch - only clear padding areas
                    for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate()
                    {
                        let (doc_len, doc_slice) = doc_data[sorted_idx];
                        let src_size = doc_len * dim;
                        let dst_offset = batch_idx * max_len_batch * dim;

                        // Copy actual data
                        buffer[dst_offset..dst_offset + src_size]
                            .copy_from_slice(&doc_slice[..src_size]);

                        // Clear only the padding area
                        if doc_len < max_len_batch {
                            let padding_start = dst_offset + src_size;
                            let padding_end = dst_offset + max_len_batch * dim;
                            buffer[padding_start..padding_end].fill(0.0);
                        }
                    }

                    maxsim_fused_doc_tiles(q, &buffer[..required_size], q_len, max_len_batch, dim)
                });

                // Copy results back to original positions
                for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate() {
                    results[sorted_idx] = batch_results[batch_idx];
                }
            }
        } else {
            // Small batch - process with standard approach
            let max_len_batch = sorted_indices[i..batch_end]
                .iter()
                .map(|&idx| doc_data[idx].0)
                .max()
                .unwrap();

            let batch_results = BATCH_BUFFER.with(|buffer| {
                let mut buffer = buffer.borrow_mut();
                let required_size = batch_size * max_len_batch * dim;

                buffer.resize(required_size, 0.0);

                // Fill batch
                for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate() {
                    let (doc_len, doc_slice) = doc_data[sorted_idx];
                    let src_size = doc_len * dim;
                    let dst_offset = batch_idx * max_len_batch * dim;

                    buffer[dst_offset..dst_offset + src_size]
                        .copy_from_slice(&doc_slice[..src_size]);

                    if doc_len < max_len_batch {
                        let padding_start = dst_offset + src_size;
                        let padding_end = dst_offset + max_len_batch * dim;
                        buffer[padding_start..padding_end].fill(0.0);
                    }
                }

                maxsim_fused_doc_tiles(q, &buffer[..required_size], q_len, max_len_batch, dim)
            });

            for (batch_idx, &sorted_idx) in sorted_indices[i..batch_end].iter().enumerate() {
                results[sorted_idx] = batch_results[batch_idx];
            }
        }

        i = batch_end;
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsim_scores_fixed_basic() {
        // Query: 2 tokens, dim 2
        let q = vec![1.0f32, 0.0, 0.0, 1.0];
        // 1 doc: 2 tokens, dim 2
        let d = vec![1.0f32, 0.0, 0.0, 1.0];

        let scores = maxsim_scores_fixed(&q, &d, 2, 2, 2);
        assert_eq!(scores.len(), 1);
        // Each query token's max is 1.0, sum = 2.0
        assert!((scores[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim_scores_variable_basic() {
        let q = vec![1.0f32, 0.0, 0.0, 1.0];
        let doc1 = vec![1.0f32, 0.0, 0.0, 1.0];
        let doc2 = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];

        let doc_data = vec![(2, doc1.as_slice()), (3, doc2.as_slice())];
        let scores = maxsim_scores_variable(&q, doc_data, 2, 2);

        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 2.0).abs() < 1e-5);
    }
}
