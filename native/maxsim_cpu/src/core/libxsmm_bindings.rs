//! FFI bindings for libxsmm.

use libc::{c_char, c_float, c_int};

// Type aliases matching libxsmm
type LibxsmmBlasint = c_int; // LP64: 32-bit int

// Function bindings
#[link(name = "xsmm")]
extern "C" {
    // Initialize and finalize
    pub fn libxsmm_init();
    pub fn libxsmm_finalize();

    // Direct GEMM call
    pub fn libxsmm_sgemm(
        transa: *const c_char,
        transb: *const c_char,
        m: *const LibxsmmBlasint,
        n: *const LibxsmmBlasint,
        k: *const LibxsmmBlasint,
        alpha: *const c_float,
        a: *const c_float,
        lda: *const LibxsmmBlasint,
        b: *const c_float,
        ldb: *const LibxsmmBlasint,
        beta: *const c_float,
        c: *mut c_float,
        ldc: *const LibxsmmBlasint,
    );
}

/// Wrapper for libxsmm_sgemm with Rust-friendly types.
#[allow(clippy::too_many_arguments)]
pub unsafe fn xsmm_sgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    let transa_char = transa as c_char;
    let transb_char = transb as c_char;
    let m_blasint = m as LibxsmmBlasint;
    let n_blasint = n as LibxsmmBlasint;
    let k_blasint = k as LibxsmmBlasint;
    let lda_blasint = lda as LibxsmmBlasint;
    let ldb_blasint = ldb as LibxsmmBlasint;
    let ldc_blasint = ldc as LibxsmmBlasint;

    libxsmm_sgemm(
        &transa_char,
        &transb_char,
        &m_blasint,
        &n_blasint,
        &k_blasint,
        &alpha,
        a,
        &lda_blasint,
        b,
        &ldb_blasint,
        &beta,
        c,
        &ldc_blasint,
    );
}
