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
        // We reorganize twiddles into contiguous SoA (Structure of Arrays) blocks
        // so the inner loop can use fast vld1q_f32 instead of scalar gathers.
        packed_twiddles.resize(order + 1);
        
        for (int s = 2; s <= order; s += 2) {
            int m = 1 << s;
            int m_4 = m >> 2;
            int step = fftSize / m;

            if (m_4 >= 4) {
                int num_j_blocks = m_4 / 4;
                // 24 floats per block: W1(8 floats), W2(8 floats), W3(8 floats)
                // Layout: [W1_re(4), W1_im(4), W2_re(4), W2_im(4), W3_re(4), W3_im(4)]
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
        // BIT-REVERSAL PRECOMPUTATION (Eliminates allocation & shift math)
        // =================================================================
        bitReverseLookup.resize(fftSize);
        for (int i = 0; i < fftSize; ++i) {
            int rev = 0;
            int x = i;
            int shift = order;
            
            // If order is odd, the innermost stage is Radix-2
            if (order % 2 != 0) {
                shift -= 1;
                rev |= (x & 1) << shift;
                x >>= 1;
            }
            
            // Remaining stages are Radix-4
            while (shift > 0) {
                shift -= 2;
                rev |= (x & 3) << shift;
                x >>= 2;
            }
            bitReverseLookup[i] = rev;
        }
    
    }

    void performForward(const Complex* input, Complex* output) const {
        std::copy(input, input + fftSize, output);
        performForward(output); 
    }
    
    
    void performForward(Complex* data) const {
        // 1. Bit-reversal permutation
        bitReverse(data, fftSize);
        //float* fdata = static_cast<float*>(__builtin_assume_aligned(data, 16));
        float* fdata = reinterpret_cast<float*>(data); 
        // 2. Cooley-Tukey Iterative Mixed-Radix DIT Butterfly
        int s = 2;
        
        // Radix-2 stage for odd orders
        if (order % 2 != 0) {
            int m = 2;
            int half_m = 1;
            int step = fftSize / 2;
            
            for (int k = 0; k < fftSize; k += m) {
                for (int j = 0; j < half_m; ++j) {
                    Complex w = twiddles[j * step];
                    Complex t = w * data[k + j + half_m];
                    Complex u = data[k + j];
                    
                    data[k + j] = u + t;
                    data[k + j + half_m] = u - t;
                }
            }
            s = 2;
        }

        // Radix-4 stages
        for (; s <= order; s += 2) {
            int m = 1 << s;
            int m_4 = m >> 2; 
            int step = fftSize / m;

            if (m_4 >= 4) {
                // =========================================================
                // NEON VECTORIZED PATH (With Pre-Packed Twiddles)
                // =========================================================
                const float* tw_ptr_base = packed_twiddles[s].data();
                
                for (int k = 0; k < fftSize; k += m) {
                    const float* tw_ptr = tw_ptr_base; // Reset twiddle pointer for each block
                    
                    for (int j = 0; j < m_4; j += 4) {
                        
                        // 1. LOAD & DE-INTERLEAVE DATA
                        float32x4x2_t A0 = vld2q_f32(&fdata[2 * (k + j)]);
                        float32x4x2_t A1 = vld2q_f32(&fdata[2 * (k + j + m_4)]);
                        float32x4x2_t A2 = vld2q_f32(&fdata[2 * (k + j + 2 * m_4)]);
                        float32x4x2_t A3 = vld2q_f32(&fdata[2 * (k + j + 3 * m_4)]);
                        
                        // 2. LOAD PRE-PACKED TWIDDLES (Continuous SIMD loads!)
                        float32x4_t W1_re = vld1q_f32(tw_ptr +  0);
                        float32x4_t W1_im = vld1q_f32(tw_ptr +  4);
                        float32x4_t W2_re = vld1q_f32(tw_ptr +  8);
                        float32x4_t W2_im = vld1q_f32(tw_ptr + 12);
                        float32x4_t W3_re = vld1q_f32(tw_ptr + 16);
                        float32x4_t W3_im = vld1q_f32(tw_ptr + 20);
                        tw_ptr += 24; // Advance to next j-block

                        // 3. COMPLEX MULTIPLICATION (FMA)
                        // A1 * W1
                        float32x4_t a1_re = vmulq_f32(W1_re, A1.val[0]);
                        a1_re = vfmsq_f32(a1_re, W1_im, A1.val[1]);
                        float32x4_t a1_im = vmulq_f32(W1_re, A1.val[1]);
                        a1_im = vfmaq_f32(a1_im, W1_im, A1.val[0]);
                        A1.val[0] = a1_re; A1.val[1] = a1_im;

                        // A2 * W2
                        float32x4_t a2_re = vmulq_f32(W2_re, A2.val[0]);
                        a2_re = vfmsq_f32(a2_re, W2_im, A2.val[1]);
                        float32x4_t a2_im = vmulq_f32(W2_re, A2.val[1]);
                        a2_im = vfmaq_f32(a2_im, W2_im, A2.val[0]);
                        A2.val[0] = a2_re; A2.val[1] = a2_im;

                        // A3 * W3
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
    std::vector<Complex> twiddles;               // Used for scalar fallbacks and bit-reversal
    std::vector<std::vector<float>> packed_twiddles; // SoA layout for SIMD inner loops
    
    std::vector<int> bitReverseLookup;           // Precomputed bit-reversal indices

    void bitReverse(Complex* data, int n) const {
    for (int i = 0; i < n; ++i) {
        int j = bitReverseLookup[i];
        if (i < j) std::swap(data[i], data[j]); 
    }
}
};