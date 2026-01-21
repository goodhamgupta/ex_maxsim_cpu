//! SIMD-optimized operations for MaxSim.
//!
//! Provides platform-specific SIMD implementations:
//! - AVX2 for x86_64
//! - NEON for aarch64
//! - Scalar fallback for other platforms

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Find maximum value in a slice using AVX2 SIMD with prefetching.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn simd_max(slice: &[f32]) -> f32 {
    if slice.len() < 8 {
        return slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }

    unsafe {
        // Use 4 vectors for better ILP (Instruction Level Parallelism)
        let mut max_vec0 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut max_vec1 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut max_vec2 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut max_vec3 = _mm256_set1_ps(f32::NEG_INFINITY);

        let mut i = 0;

        // Process 32 elements at a time (4x8) for better ILP
        while i + 32 <= slice.len() {
            // Prefetch next cache line
            _mm_prefetch(slice.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);

            let data0 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let data1 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let data2 = _mm256_loadu_ps(slice.as_ptr().add(i + 16));
            let data3 = _mm256_loadu_ps(slice.as_ptr().add(i + 24));

            max_vec0 = _mm256_max_ps(max_vec0, data0);
            max_vec1 = _mm256_max_ps(max_vec1, data1);
            max_vec2 = _mm256_max_ps(max_vec2, data2);
            max_vec3 = _mm256_max_ps(max_vec3, data3);

            i += 32;
        }

        // Process remaining groups of 8
        while i + 8 <= slice.len() {
            let data = _mm256_loadu_ps(slice.as_ptr().add(i));
            max_vec0 = _mm256_max_ps(max_vec0, data);
            i += 8;
        }

        // Combine the 4 vectors
        max_vec0 = _mm256_max_ps(max_vec0, max_vec1);
        max_vec2 = _mm256_max_ps(max_vec2, max_vec3);
        max_vec0 = _mm256_max_ps(max_vec0, max_vec2);

        // Horizontal max within the final vector
        let high = _mm256_extractf128_ps(max_vec0, 1);
        let low = _mm256_castps256_ps128(max_vec0);
        let max128 = _mm_max_ps(high, low);

        let shuffled = _mm_shuffle_ps(max128, max128, 0b01001110);
        let max64 = _mm_max_ps(max128, shuffled);
        let shuffled2 = _mm_shuffle_ps(max64, max64, 0b00000001);
        let final_max = _mm_max_ps(max64, shuffled2);

        let mut result = _mm_cvtss_f32(final_max);

        // Handle remaining elements
        for j in i..slice.len() {
            result = result.max(slice[j]);
        }

        result
    }
}

/// Find maximum value in a slice using ARM NEON SIMD.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn simd_max(slice: &[f32]) -> f32 {
    if slice.len() < 4 {
        return slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }

    unsafe {
        // Initialize 4 vectors for better ILP
        let mut max_vec0 = vdupq_n_f32(f32::NEG_INFINITY);
        let mut max_vec1 = vdupq_n_f32(f32::NEG_INFINITY);
        let mut max_vec2 = vdupq_n_f32(f32::NEG_INFINITY);
        let mut max_vec3 = vdupq_n_f32(f32::NEG_INFINITY);

        let mut i = 0;

        // Process 16 elements at a time (4x4)
        while i + 16 <= slice.len() {
            let data0 = vld1q_f32(slice.as_ptr().add(i));
            let data1 = vld1q_f32(slice.as_ptr().add(i + 4));
            let data2 = vld1q_f32(slice.as_ptr().add(i + 8));
            let data3 = vld1q_f32(slice.as_ptr().add(i + 12));

            max_vec0 = vmaxq_f32(max_vec0, data0);
            max_vec1 = vmaxq_f32(max_vec1, data1);
            max_vec2 = vmaxq_f32(max_vec2, data2);
            max_vec3 = vmaxq_f32(max_vec3, data3);

            i += 16;
        }

        // Process remaining groups of 4
        while i + 4 <= slice.len() {
            let data = vld1q_f32(slice.as_ptr().add(i));
            max_vec0 = vmaxq_f32(max_vec0, data);
            i += 4;
        }

        // Combine the 4 vectors
        max_vec0 = vmaxq_f32(max_vec0, max_vec1);
        max_vec2 = vmaxq_f32(max_vec2, max_vec3);
        max_vec0 = vmaxq_f32(max_vec0, max_vec2);

        // Horizontal max within the final vector
        let max_pair = vmaxq_f32(max_vec0, vextq_f32(max_vec0, max_vec0, 2));
        let max_val = vmaxq_f32(max_pair, vextq_f32(max_pair, max_pair, 1));
        let mut result = vgetq_lane_f32(max_val, 0);

        // Handle remaining elements
        for j in i..slice.len() {
            result = result.max(slice[j]);
        }

        result
    }
}

/// Scalar fallback for platforms without SIMD support.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub fn simd_max(slice: &[f32]) -> f32 {
    slice.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_max_basic() {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(simd_max(&data), 5.0);
    }

    #[test]
    fn test_simd_max_negative() {
        let data = vec![-5.0f32, -1.0, -3.0, -2.0];
        assert_eq!(simd_max(&data), -1.0);
    }

    #[test]
    fn test_simd_max_large() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(simd_max(&data), 999.0);
    }

    #[test]
    fn test_simd_max_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(simd_max(&data), f32::NEG_INFINITY);
    }
}
