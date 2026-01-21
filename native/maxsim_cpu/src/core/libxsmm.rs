//! libxsmm-accelerated MaxSim implementations.

use super::libxsmm_bindings;
use rayon::prelude::*;
use std::sync::Once;

static LIBXSMM_INIT: Once = Once::new();

/// Clean libxsmm implementation for fixed-length documents.
pub fn maxsim_libxsmm_clean(
    q: &[f32],
    d: &[f32],
    q_len: usize,
    d_len: usize,
    dim: usize,
) -> Vec<f32> {
    // Initialize libxsmm
    LIBXSMM_INIT.call_once(|| unsafe {
        libxsmm_bindings::libxsmm_init();
    });

    let n_docs = d.len() / (d_len * dim);

    // Try to keep tiles in L2 cache
    let block_size = 64;

    // Process documents in parallel
    (0..n_docs)
        .into_par_iter()
        .map(|doc_idx| {
            let doc_offset = doc_idx * d_len * dim;
            let doc_data = &d[doc_offset..doc_offset + d_len * dim];

            // max values for each query
            let mut max_vals = vec![f32::NEG_INFINITY; q_len];

            // Process document tokens in blocks
            for t in (0..d_len).step_by(block_size) {
                let actual_block_size = block_size.min(d_len - t);

                // Workspace for GEMM output
                let mut c = vec![0.0f32; q_len * actual_block_size];

                unsafe {
                    libxsmm_bindings::xsmm_sgemm(
                        b'T',
                        b'N',
                        actual_block_size as i32,
                        q_len as i32,
                        dim as i32,
                        1.0,
                        doc_data.as_ptr().add(t * dim),
                        dim as i32,
                        q.as_ptr(),
                        dim as i32,
                        0.0,
                        c.as_mut_ptr(),
                        actual_block_size as i32,
                    );
                }

                // Update max values
                for qi in 0..q_len {
                    for ti in 0..actual_block_size {
                        let idx = qi * actual_block_size + ti;
                        max_vals[qi] = max_vals[qi].max(c[idx]);
                    }
                }
            }

            // Sum all max values
            max_vals.iter().sum()
        })
        .collect()
}

/// Process variable-length documents with libxsmm.
pub fn maxsim_libxsmm_variable(
    q: &[f32],
    doc_infos: Vec<(usize, usize, &[f32])>,
    q_len: usize,
    dim: usize,
) -> Vec<f32> {
    // Initialize libxsmm
    LIBXSMM_INIT.call_once(|| unsafe {
        libxsmm_bindings::libxsmm_init();
    });

    let n_docs = doc_infos.len();

    // Process documents in parallel, each with its actual length
    let mut results = vec![0.0f32; n_docs];
    let results_vec: Vec<(usize, f32)> = doc_infos
        .into_par_iter()
        .map(|(doc_idx, doc_len, doc_data)| {
            // max values for each query
            let mut max_vals = vec![f32::NEG_INFINITY; q_len];

            // Try to keep tiles in L2 cache
            let block_size = 64;

            // Process document tokens in blocks
            for t in (0..doc_len).step_by(block_size) {
                let actual_block_size = block_size.min(doc_len - t);

                // Workspace for GEMM output
                let mut c = vec![0.0f32; q_len * actual_block_size];

                unsafe {
                    libxsmm_bindings::xsmm_sgemm(
                        b'T',
                        b'N',
                        actual_block_size as i32,
                        q_len as i32,
                        dim as i32,
                        1.0,
                        doc_data.as_ptr().add(t * dim),
                        dim as i32,
                        q.as_ptr(),
                        dim as i32,
                        0.0,
                        c.as_mut_ptr(),
                        actual_block_size as i32,
                    );
                }

                // Update max values
                for qi in 0..q_len {
                    for ti in 0..actual_block_size {
                        let idx = qi * actual_block_size + ti;
                        max_vals[qi] = max_vals[qi].max(c[idx]);
                    }
                }
            }

            // Sum all max values
            (doc_idx, max_vals.iter().sum())
        })
        .collect();

    // Place results in correct order
    for (doc_idx, score) in results_vec {
        results[doc_idx] = score;
    }

    results
}
