#pragma once
// ARM64 NEON Intrinsics Header
#include <arm_neon.h>
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class NeonFFT {
public:
    using Complex = std::complex<float>;
    
    NeonFFT(int order) : order(order), fftSize(1 << order) {
        if (order < 3) {
            throw std::invalid_argument("NEON FFT requires order >= 3 (size >= 8) for vectorization");
        }
        
        // Twiddle table MUST be size fftSize to prevent out-of-bounds 
        // when calculating W3 (3 * j * step) in the final stages.
        twiddles.resize(fftSize);  
        for (int i = 0; i < fftSize; ++i) {
            double phase = -2.0 * M_PI * i / fftSize;
            twiddles[i] = Complex((float)std::cos(phase), (float)std::sin(phase));
        }

        // =================================================================
        // TWIDDLE PRE-PACKING (Eliminates the SIMD Memory Wall)
        // ================================================================= 
        packed_twiddles.resize(order + 1);
        
        for (int s = 2; s <= order; s += 2) {
            int m = 1 << s;
            int m_4 = m >> 2;
            int step = fftSize / m;

            if (m_4 >= 4) {
                int num_j_blocks = m_4 / 4;
                // 24 floats per block: W1(8 floats), W2(8 floats), W3(8 floats)
                packed_twiddles[s].resize(num_j_blocks * 24);
                float* ptr = packed_twiddles[s].data();

                for (int j = 0; j < m_4; j += 4) {
                    // Pack W1
                    for (int i = 0; i < 4; ++i) {
                        Complex w = twiddles[(j + i) * step];
                        ptr[i] = w.real();
                        ptr[4 + i] = w.imag();
                    }
                    ptr += 8;
                    // Pack W2
                    for (int i = 0; i < 4; ++i) {
                        Complex w = twiddles[2 * (j + i) * step];
                        ptr[i] = w.real();
                        ptr[4 + i] = w.imag();
                    }
                    ptr += 8;
                    // Pack W3
                    for (int i = 0; i < 4; ++i) {
                        Complex w = twiddles[3 * (j + i) * step];
                        ptr[i] = w.real();
                        ptr[4 + i] = w.imag();
                    }
                    ptr += 8;
                }
            }
        }

        // =================================================================
        // BIT-REVERSAL PRECOMPUTATION
        // =================================================================
        bitReverseLookup.resize(fftSize);
        for (int i = 0; i < fftSize; ++i) {
            int rev = 0;
            int x = i;
            int shift = order;
            
            if (order % 2 != 0)  {
                shift -= 1;
                rev |= (x & 1) << shift;
                x >>= 1;
            }
            
            while (shift > 0) {
                shift -= 2;
                rev |= (x & 3) << shift;
                x >>= 2;
            }
            bitReverseLookup[i] = rev;
        }
    }

    // =====================================================================
    // OUT-OF-PLACE FFT (With __restrict__ and Manual Loop Unrolling)
    // =====================================================================
    void performForward(const Complex* __restrict__ input, Complex* __restrict__ output) const {
        if (input == output) {
            // Fallback for strict in-place calls
            bitReverse(output, fftSize);
        } else {
            // Random Read + SEQUENTIAL WRITE.
            // Manually unrolled by 4 to allow the ARM64 CPU to issue multiple random 
            // loads in parallel, hiding memory latency while the store buffer 
            // handles the sequential writes effortlessly.
            int i = 0;
            for (; i <= fftSize - 4; i += 4) {
                Complex c0 = input[bitReverseLookup[i]];
                Complex c1 = input[bitReverseLookup[i+1]];
                Complex c2 = input[bitReverseLookup[i+2]];
                Complex c3 = input[bitReverseLookup[i+3]];
                
                output[i]   = c0;
                output[i+1] = c1;
                output[i+2] = c2;
                output[i+3] = c3;
            }
            for (; i < fftSize; ++i) {
                output[i] = input[bitReverseLookup[i]];
            }
        }
        
        // Run the SIMD butterflies in-place on the output buffer
        performForward(output); 
    }

    // =====================================================================
    // IN-PLACE SIMD BUTTERFLY MATH
    // =====================================================================
    void performForward(Complex* __restrict__ data) const {
        float* __restrict__ fdata = reinterpret_cast<float*>(data); 
        
        int s = 2;
        
        // =====================================================================
        // RADIX-2 STAGE FOR ODD ORDERS (VECTORIZED)
        // =====================================================================
        if (order % 2 != 0) {
            // For Radix-2, m=2, half_m=1. The twiddle factor is always twiddles[0] = (1.0, 0.0).
            // Thus, no complex multiplication is needed, just addition and subtraction.
            
            int k = 0;
            // Vectorized path: process 8 complex numbers (4 butterflies) per iteration
            for (; k <= fftSize - 8; k += 8) {
                // 1. LOAD 8 COMPLEX NUMBERS (16 floats)
                float32x4x2_t A = vld2q_f32(&fdata[2 * k]);
                float32x4x2_t B = vld2q_f32(&fdata[2 * k + 8]);
                
                // 2. DE-INTERLEAVE to separate even (U) and odd (T) elements
                // vuzp1 extracts even indices (0, 2, 4, 6), vuzp2 extracts odd indices (1, 3, 5, 7)
                float32x4_t U_re = vuzp1q_f32(A.val[0], B.val[0]);
                float32x4_t T_re = vuzp2q_f32(A.val[0], B.val[0]);
                float32x4_t U_im = vuzp1q_f32(A.val[1], B.val[1]);
                float32x4_t T_im = vuzp2q_f32(A.val[1], B.val[1]);
                
                // 3. BUTTERFLY MATH (Twiddle is 1+0i, so no FMA needed)
                float32x4_t O0_re = vaddq_f32(U_re, T_re);
                float32x4_t O0_im = vaddq_f32(U_im, T_im);
                float32x4_t O1_re = vsubq_f32(U_re, T_re);
                float32x4_t O1_im = vsubq_f32(U_im, T_im);
                
                // 4. INTERLEAVE BACK
                // vzip1 takes the lower half, vzip2 takes the upper half
                float32x4x2_t R0, R1;
                R0.val[0] = vzip1q_f32(O0_re, O1_re);
                R0.val[1] = vzip1q_f32(O0_im, O1_im);
                R1.val[0] = vzip2q_f32(O0_re, O1_re);
                R1.val[1] = vzip2q_f32(O0_im, O1_im);
                
                // 5. STORE
                vst2q_f32(&fdata[2 * k], R0);
                vst2q_f32(&fdata[2 * k + 8], R1);
            }
            
            // Scalar fallback (Theoretically unreachable since fftSize is a power of 2 >= 8)
            for (; k < fftSize; k += 2) {
                Complex u = data[k];
                Complex t = data[k + 1];
                data[k] = u + t;
                data[k + 1] = u - t;
            }
        }

        // Radix-4 stages
        for (; s <= order; s += 2) {
            int m = 1 << s;
            int m_4 = m >> 2; 
            int step = fftSize / m;

            if (m_4 >= 4) {
                // =========================================================
                // NEON VECTORIZED PATH
                // =========================================================
                const float* __restrict__ tw_ptr_base = packed_twiddles[s].data();
                
                for (int k = 0; k < fftSize; k += m) {
                    const float* __restrict__ tw_ptr = tw_ptr_base;
                    
                    for (int j = 0; j < m_4; j += 4) {
                        // Note: Software prefetching (__builtin_prefetch) is intentionally 
                        // omitted here. For N=1024 (8KB), the entire dataset fits in L1 cache,
                        // and Apple Silicon's hardware prefetcher is vastly superior.
                        // Adding software prefetches here would waste instruction decode bandwidth.

                        // 1. LOAD & DE-INTERLEAVE DATA
                        float32x4x2_t A0 = vld2q_f32(&fdata[2 * (k + j)]);
                        float32x4x2_t A1 = vld2q_f32(&fdata[2 * (k + j + m_4)]);
                        float32x4x2_t A2 = vld2q_f32(&fdata[2 * (k + j + 2 * m_4)]);
                        float32x4x2_t A3 = vld2q_f32(&fdata[2 * (k + j + 3 * m_4)]);
                        
                        // 2. LOAD PRE-PACKED TWIDDLES
                        float32x4_t W1_re = vld1q_f32(tw_ptr +  0);
                        float32x4_t W1_im = vld1q_f32(tw_ptr +  4);
                        float32x4_t W2_re = vld1q_f32(tw_ptr +  8);
                        float32x4_t W2_im = vld1q_f32(tw_ptr + 12);
                        float32x4_t W3_re = vld1q_f32(tw_ptr + 16);
                        float32x4_t W3_im = vld1q_f32(tw_ptr + 20);
                        tw_ptr  += 24;

                        // 3. COMPLEX MULTIPLICATION (FMA)
                        float32x4_t a1_re = vmulq_f32(W1_re, A1.val[0]);
                        a1_re = vfmsq_f32(a1_re, W1_im, A1.val[1]);
                        float32x4_t a1_im = vmulq_f32(W1_re, A1.val[1]);
                        a1_im = vfmaq_f32(a1_im, W1_im, A1.val[0]);
                        A1.val[0] = a1_re; A1.val[1] = a1_im;

                        float32x4_t a2_re = vmulq_f32(W2_re, A2.val[0]);
                        a2_re = vfmsq_f32(a2_re, W2_im, A2.val[1]);
                        float32x4_t a2_im = vmulq_f32(W2_re, A2.val[1]);
                        a2_im = vfmaq_f32(a2_im, W2_im, A2.val[0]);
                        A2.val[0] = a2_re; A2.val[1] = a2_im;

                        float32x4_t a3_re = vmulq_f32(W3_re, A3.val[0]);
                        a3_re = vfmsq_f32(a3_re, W3_im, A3.val[1]);
                        float32x4_t a3_im = vmulq_f32(W3_re, A3.val[1]);
                        a3_im = vfmaq_f32(a3_im, W3_im, A3.val[0]);
                        A3.val[0] = a3_re; A3.val[1] = a3_im;

                        // 4. RADIX-4 BUTTERFLY MATH
                        float32x4_t t0_re = vaddq_f32(A0.val[0], A2.val[0]);
                        float32x4_t t0_im = vaddq_f32(A0.val[1], A2.val[1]);
                        float32x4_t t1_re = vsubq_f32(A0.val[0], A2.val[0]);
                        float32x4_t t1_im = vsubq_f32(A0.val[1], A2.val[1]);

                        float32x4_t t2_re = vaddq_f32(A1.val[0], A3.val[0]);
                        float32x4_t t2_im = vaddq_f32(A1.val[1], A3.val[1]);
                        float32x4_t t3_re = vsubq_f32(A1.val[0], A3.val[0]);
                        float32x4_t t3_im = vsubq_f32(A1.val[1], A3.val[1]);

                        float32x4x2_t O0, O2;
                        O0.val[0] = vaddq_f32(t0_re, t2_re);
                        O0.val[1] = vaddq_f32(t0_im, t2_im);
                        O2.val[0] = vsubq_f32(t0_re, t2_re);
                        O2.val[1] = vsubq_f32(t0_im, t2_im);

                        float32x4x2_t O1, O3;
                        O1.val[0] = vaddq_f32(t1_re, t3_im);
                        O1.val[1] = vsubq_f32(t1_im, t3_re);
                        O3.val[0] = vsubq_f32(t1_re, t3_im);
                        O3.val[1] = vaddq_f32(t1_im, t3_re);

                        // 5. INTERLEAVE & STORE
                        vst2q_f32(&fdata[2 * (k + j)], O0);
                        vst2q_f32(&fdata[2 * (k + j + m_4)], O1);
                        vst2q_f32(&fdata[2 * (k + j + 2 * m_4)], O2);
                        vst2q_f32(&fdata[2 * (k + j + 3 * m_4)], O3);
                    }
                }
            } else {
                // =========================================================
                // SCALAR FALLBACK (For m_4 < 4)
                // =========================================================
                for (int k = 0; k < fftSize; k += m) {
                    for (int j = 0; j < m_4; ++j) {
                        Complex w1 = twiddles[j * step];
                        Complex w2 = twiddles[2 * j * step];
                        Complex w3 = twiddles[3 * j * step];   

                        Complex a0 = data[k + j];
                        Complex a1 = data[k + j + m_4] * w1;
                        Complex a2 = data[k + j + 2 * m_4] * w2;
                        Complex a3 = data[k + j + 3 * m_4] * w3;

                        Complex t0 = a0 + a2;
                        Complex t1 = a0 - a2;
                        Complex t2 = a1 + a3;
                        Complex t3 = a1 - a3;

                        data[k + j] = t0 + t2;
                        data[k + j + 2 * m_4] = t0 - t2;
                        
                        data[k + j + m_4] = Complex(t1.real() + t3.imag(), t1.imag() - t3.real());
                        data[k + j + 3 * m_4] = Complex(t1.real() - t3.imag(), t1.imag() + t3.real());
                    }
                }
            }
        }
    }

private:
    int order;
    int fftSize;
    std::vector<Complex> twiddles;
    std::vector<std::vector<float>> packed_twiddles;
    std::vector<int> bitReverseLookup;

    // Kept for safety, though the Out-of-Place path bypasses it
    void bitReverse(Complex* data, int n) const {
        for (int i = 0; i < n; ++i) {
            int j = bitReverseLookup[i];
            if (i < j) std::swap(data[i], data[j]); 
        }
    }
};
