/*
v1.0.0
intrinsic optimizd radix-4
v1.0.1
intrinsic optimized radix-2 and radix-4(better)

v1.0.2
out of place fft using __restrict and manual loop unrolling 
bit reversal precompute switched to bit reversal done with __builtin_bitreverse32

v1.1.1
Switched to split-complex (SoA) and avoid vld2/vst2 in the hot loop.

v1.1.2
removed dead vars
alignedAllocator
std::assume_aligned
==============================================================================
*/
#pragma once
// ARM64 NEON Intrinsics Header
#include <arm_neon.h>
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <limits>
#include <memory>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// =============================================================================
// 64-BYTE ALIGNED ALLOCATOR (Cache-Line Optimization)
// Prevents cache-line splits during SIMD loads/stores in the hot loop.
// =============================================================================
template <typename T, size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;
    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) throw std::bad_alloc();
        size_type size = n * sizeof(T);
        void* ptr = nullptr;
    #if defined(_MSC_VER)
    ptr = _aligned_malloc(size, Alignment);
    if (!ptr) throw std::bad_alloc();
    #else
    if (posix_memalign(&ptr, Alignment, size) != 0) throw std::bad_alloc();
    #endif
    return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr, size_type) noexcept {
    #if defined(_MSC_VER)
    _aligned_free(ptr);
    #else
    free(ptr);
    #endif
    }
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

class NeonFFT {
public:
    using Complex = std::complex<float>;
    // Helper struct to guarantee 64B aligned input/output buffers for the user
    struct AlignedBuffer {
        std::vector<float, AlignedAllocator<float>> re;
        std::vector<float, AlignedAllocator<float>> im;
        AlignedBuffer(size_t size) : re(size), im(size) {}
    };

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

        // =================== ==============================================  
        // TWIDDLE PRE-PACKING (Eliminates the SIMD Memory Wall)
        // Packed in Split-Complex (SoA) format: [Re0, Re1, Re2, Re3, Im0,  Im1, Im2, Im3]
        // v1.1.2 Now using AlignedAllocator to guarantee 64B cache-line alignment
        // 
        // =================================================================  
        packed_twiddles.resize(order + 1);
        
        // FIX 2: Start at 3 if the order is odd, otherwise start at 2
        int start_s = (order % 2 != 0) ? 3 : 2;
        
        for (int s = start_s; s <= order; s += 2) {
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
    }

    // =================================================================
    // BIT-REVERSAL PRECOMPUTATION
    // ======================== =========================================
    // v1.0.2 现在替换为  __builtin_bitreverse32

    // =====================================================================
    // AOS WRAPPERS 
    //       (For compatibility with std::complex<float> benchmarks)
    // Uses mutable temp buffers + NEON vld2/vst2 降低 conversion overhead
    // =====================================================================
    void performForward(const Complex* __restrict__ input, Complex* __restrict__ output) const {
        if (temp_in_re.size() != static_cast<size_t>(fftSize)) {
            temp_in_re.resize(fftSize);
            temp_in_im.resize(fftSize);
            temp_out_re.resize(fftSize);
            temp_out_im.resize(fftSize);
        }
        
        const float* in_ptr = reinterpret_cast<const float*>(input);
        float* out_re_ptr = temp_in_re.data();
        float* out_im_ptr = temp_in_im.data();
        
        // AoS -> SoA using NEON vld2
        for (int i = 0; i < fftSize; i += 4) {
            float32x4x2_t A = vld2q_f32(in_ptr + i * 2);
            vst1q_f32(out_re_ptr + i, A.val[0]);
            vst1q_f32(out_im_ptr + i, A.val[1]);
        }
        
        performForward(temp_in_re.data(), temp_in_im.data(), temp_out_re.data(), temp_out_im.data());
        
        float* out_ptr = reinterpret_cast<float*>(output);
        const float* in_re_ptr = temp_out_re.data();
        const float* in_im_ptr = temp_out_im.data();
        
        // SoA -> AoS using NEON vst2
        for (int i = 0; i < fftSize; i += 4) {
            float32x4_t re = vld1q_f32(in_re_ptr + i);
            float32x4_t im = vld1q_f32(in_im_ptr + i);
            float32x4x2_t A;
            A.val[0] = re;
            A.val[1] = im;
            vst2q_f32(out_ptr + i * 2, A);
        }
    }

    void performForward(Complex* __restrict__ data) const {
        if (temp_in_re.size() != static_cast<size_t>(fftSize)) {
            temp_in_re.resize(fftSize);
            temp_in_im.resize(fftSize);
        }
        
        float* data_ptr = reinterpret_cast<float*>(data);
        float* out_re_ptr = temp_in_re.data();
        float* out_im_ptr = temp_in_im.data();
        
        for (int i = 0; i < fftSize; i += 4) {
            float32x4x2_t A = vld2q_f32(data_ptr + i * 2);
            vst1q_f32(out_re_ptr + i, A.val[0]);
            vst1q_f32(out_im_ptr + i, A.val[1]);
        }
        
        performForward(temp_in_re.data(), temp_in_im.data());
        
        for (int i = 0; i < fftSize; i += 4) {
            float32x4_t re = vld1q_f32(out_re_ptr + i);
            float32x4_t im = vld1q_f32(out_im_ptr + i);
            float32x4x2_t A;
            A.val[0] = re;
            A.val[1] = im;
            vst2q_f32(data_ptr + i * 2, A);
        }
    }

    // =====================================================================
    // OUT-OF-PLACE FFT 
    // v1.0.2  (新增 __restrict__ and Manual Loop Unrolling)
    // v1.1.0  (Switched to Split-Complex / SoA Interface)
    // =====================================================================
    void performForward(const float* __restrict__ in_re, const float* __restrict__ in_im,
                        float* __restrict__ out_re, float* __restrict__ out_im) const {
    #if defined(__GNUC__) || defined(__clang__)
    in_re  = (const float*)__builtin_assume_aligned(in_re, 64);
    in_im  = (const float*)__builtin_assume_aligned(in_im, 64);
    out_re = (float*)__builtin_assume_aligned(out_re, 64);
    out_im = (float*)__builtin_assume_aligned(out_im, 64);
    #endif
        if (in_re == out_re && in_im == out_im) {
            // Fallback for strict in-place calls
            bitReverse(out_re, out_im, fftSize);
        } else {
            // Out-of-place Bit-Reversal using ARM64 `rbit`  intrinsic
            int i = 0;
            for (; i <= fftSize - 4; i += 4) {
                uint32_t rev0 = __builtin_bitreverse32(i) >> (32 - order);
                uint32_t rev1 = __builtin_bitreverse32(i+1) >> (32 - order);
                uint32_t rev2 = __builtin_bitreverse32(i+2) >> (32 - order);
                uint32_t rev3 = __builtin_bitreverse32(i+3) >> (32 - order);
                
                out_re[i]   = in_re[rev0];
                out_re[i+1] = in_re[rev1];
                out_re[i+2] = in_re[rev2];
                out_re[i+3] = in_re[rev3];
                
                out_im[i]   = in_im[rev0];
                out_im[i+1] = in_im[rev1];
                out_im[i+2] = in_im[rev2];
                out_im[i+3] = in_im[rev3];
            }
            for (; i < fftSize; ++i) {
                uint32_t rev = __builtin_bitreverse32(i) >> (32 - order);
                out_re[i] = in_re[rev];
                out_im[i] = in_im[rev];
            }
        }
        
        // Run the SIMD butterflies in-place on the output buffer
        performForward(out_re, out_im); 
    }

    // =====================================================================
    // IN-PLACE SIMD BUTTERFLY MATH
    // ========================================= ============================
    void performForward(float* __restrict__ re, float* __restrict__ im) const {
    #if defined(__GNUC__) || defined(__clang__)
    re = (float*)__builtin_assume_aligned(re, 64);
    im = (float*)__builtin_assume_aligned(im, 64);
    #endif
        int s = 2;
        
        // =====================================================================
        // RADIX-2 STAGE FOR ODD ORDERS (v1.0.2 now vectorized)
        // ================ =====================================================
        if (order % 2 != 0) {
            // Radix-2时, m=2, half_m=1. The twiddle factor is always twiddles[0] = (1.0, 0.0).
            // 无须复数乘法
            
            int k = 0;
            // Vectorized path: process 8 complex numbers (4 butterflies) per iteration
            for (; k <= fftSize - 8; k += 8) {
                // 1. LOAD 8 COMPLEX NUMBERS (Separated into Re/Im arrays)
                float32x4_t re0 = vld1q_f32(&re[k]);
                float32x4_t re1 = vld1q_f32(&re[k + 4]);
                float32x4_t im0 = vld1q_f32(&im[k]);
                float32x4_t im1 = vld1q_f32(&im[k + 4]);
                
                // 2. DE-INTERLEAVE to separate even (U) and odd (T) elements
                // vuzp1 extracts even indices (0, 2, 4, 6), vuzp2 extracts odd indices (1, 3, 5, 7)
                float32x4_t U_re = vuzp1q_f32(re0, re1);
                float32x4_t T_re = vuzp2q_f32(re0, re1);
                float32x4_t U_im = vuzp1q_f32(im0, im1);
                float32x4_t T_im = vuzp2q_f32(im0, im1);
                
                // 3. BUTTERFLY MATH (Twiddle is 1+0i, so no FMA needed)
                float32x4_t O0_re = vaddq_f32(U_re, T_re);
                float32x4_t O0_im = vaddq_f32(U_im, T_im);
                float32x4_t O1_re = vsubq_f32(U_re, T_re);
                float32x4_t O1_im = vsubq_f32(U_im, T_im);
                
                // 4. INTERLEAVE BACK
                // vzip1 takes the lower half, vzip2 takes the upper half
                float32x4_t R0_re = vzip1q_f32(O0_re, O1_re);
                float32x4_t R1_re = vzip2q_f32(O0_re, O1_re);
                float32x4_t R0_im = vzip1q_f32(O0_im, O1_im);
                float32x4_t R1_im = vzip2q_f32(O0_im, O1_im);
                
                // 5. STORE
                vst1q_f32(&re[k], R0_re);
                vst1q_f32(&re[k + 4], R1_re);
                vst1q_f32(&im[k], R0_im);
                vst1q_f32(&im[k + 4], R1_im);
            }
            
            // 后备scalar fft 用不到
            for (; k < fftSize; k += 2) {
                float u_re = re[k], u_im = im[k];
                float t_re = re[k+1], t_im = im[k+1];
                re[k] = u_re + t_re; im[k] = u_im + t_im;
                re[k+1] = u_re - t_re; im[k+1] = u_im - t_im;
            }
            
            // FIX 1: Advance to the correct starting stage for Radix-4
            s = 3; 
        }
        // =======================================================================
        // Radix-4 stages
        //======== ===========================================
        for (; s <= order; s += 2) {
            int m = 1 << s;
            int m_4 = m >> 2; 

            if (m_4 >= 4) {
                // =========================================================
                // NEON VECTORIZED PATH
                // ================================================ =========
                const float* __restrict__ tw_ptr_base = packed_twiddles[s].data();
    #if __cplusplus >= 202002L && defined(__cpp_lib_assume_aligned)
    tw_ptr_base = std::assume_aligned<64>(tw_ptr_base);
    #elif defined(__GNUC__) || defined(__clang__)
    tw_ptr_base = static_cast<const float*>(__builtin_assume_aligned(tw_ptr_base, 64));
    #endif
                for (int k = 0; k < fftSize; k += m) {
                    const float* __restrict__ tw_ptr = tw_ptr_base;
                    
                    for (int j = 0; j < m_4; j += 4) {
                        // Note: Software prefetching (__builtin_prefetch) 故意omitted
                        // 在后续large fft测试中加上
                        // 目前n=1024, 整体能放在L1  cache中

                        // 1. LOAD DATA (SoA layout avoids vld2 completely)
                        float32x4_t A0_re = vld1q_f32(&re[k + j]);
                        float32x4_t A0_im = vld1q_f32(&im[k + j]);
                        
                            // [FIX]: Swap A1 and A2 loads to compensate for Radix-2 bit reversal!
                        float32x4_t A1_re = vld1q_f32(&re[k + j + 2 * m_4]); // Load from 2*m_4
                        float32x4_t A1_im = vld1q_f32(&im[k + j + 2 * m_4]); 

                        float32x4_t A2_re = vld1q_f32(&re[k + j + m_4]);     // Load from m_4
                        float32x4_t A2_im = vld1q_f32(&im[k + j + m_4]);     

                        float32x4_t A3_re = vld1q_f32(&re[k + j + 3 * m_4]);
                        float32x4_t A3_im = vld1q_f32(&im[k + j + 3 * m_4]);
                        
                        // 2. LOAD PRE-PACKED TWIDDLES
                        float32x4_t W1_re = vld1q_f32(tw_ptr +  0);
                        float32x4_t W1_im = vld1q_f32(tw_ptr +  4);
                        float32x4_t W2_re = vld1q_f32(tw_ptr +  8);
                        float32x4_t W2_im = vld1q_f32(tw_ptr + 12);
                        float32x4_t W3_re = vld1q_f32(tw_ptr + 16);
                        float32x4_t W3_im = vld1q_f32(tw_ptr + 20);
                        tw_ptr += 24;

                        // 3.  COMPLEX MULTIPLICATION (FMA)
                        float32x4_t a1_re = vmulq_f32(W1_re, A1_re);
                        a1_re = vfmsq_f32(a1_re, W1_im, A1_im);
                        float32x4_t a1_im = vmulq_f32(W1_re, A1_im);
                        a1_im = vfmaq_f32(a1_im, W1_im, A1_re);
                        
                        float32x4_t a2_re = vmulq_f32(W2_re, A2_re);
                        a2_re = vfmsq_f32(a2_re, W2_im, A2_im);
                        float32x4_t a2_im = vmulq_f32(W2_re, A2_im);
                        a2_im = vfmaq_f32(a2_im, W2_im, A2_re);

                        float32x4_t a3_re = vmulq_f32(W3_re, A3_re);
                        a3_re = vfmsq_f32(a3_re, W3_im, A3_im);
                        float32x4_t a3_im = vmulq_f32(W3_re, A3_im);
                        a3_im = vfmaq_f32(a3_im, W3_im, A3_re);

                        // 4. RADIX-4 BUTTERFLY MATH
                        float32x4_t t0_re = vaddq_f32(A0_re, a2_re);
                        float32x4_t t0_im = vaddq_f32(A0_im, a2_im);
                        float32x4_t t1_re = vsubq_f32(A0_re, a2_re);
                        float32x4_t t1_im = vsubq_f32(A0_im, a2_im);

                        float32x4_t t2_re = vaddq_f32(a1_re, a3_re);
                        float32x4_t t2_im = vaddq_f32(a1_im, a3_im);
                        float32x4_t t3_re = vsubq_f32(a1_re, a3_re);
                        float32x4_t t3_im = vsubq_f32(a1_im, a3_im);

                        float32x4_t O0_re = vaddq_f32(t0_re, t2_re);
                        float32x4_t O0_im = vaddq_f32(t0_im, t2_im);
                        float32x4_t O2_re = vsubq_f32(t0_re, t2_re);
                        float32x4_t O2_im = vsubq_f32(t0_im, t2_im);

                        float32x4_t O1_re = vaddq_f32(t1_re, t3_im);
                        float32x4_t O1_im = vsubq_f32(t1_im, t3_re);
                        float32x4_t O3_re = vsubq_f32(t1_re, t3_im);
                        float32x4_t O3_im = vaddq_f32(t1_im, t3_re);

                        // 5. STORE
                        vst1q_f32(&re[k + j], O0_re);
                        vst1q_f32(&im[k + j], O0_im);
                        vst1q_f32(&re[k + j + m_4], O1_re);
                        vst1q_f32(&im[k + j + m_4], O1_im);
                        vst1q_f32(&re[k + j + 2 * m_4], O2_re);
                        vst1q_f32(&im[k + j + 2 * m_4], O2_im);
                        vst1q_f32(&re[k + j + 3 * m_4], O3_re);
                        vst1q_f32(&im[k + j + 3 * m_4], O3_im);
                    }
                }
            } else {
                // =========================================================
                // 后备scalar fft  (For m_4 < 4) 用不到
                // =========================================================
                int step = fftSize / m;     // 不放在hot loop中
                for (int k = 0; k < fftSize; k += m) {
                    for (int j = 0; j < m_4; ++j) {
                        Complex w1 = twiddles[j * step];
                        Complex w2 = twiddles[2 * j * step];
                        Complex w3 = twiddles[3 * j * step];     

                        Complex a0(re[k + j], im[k + j]);
                        // [FIX]: Swap a1 and a2 loads here as well!
                        Complex a1(re[k + j + 2 * m_4], im[k + j + 2 * m_4]); a1 *= w1;
                        Complex a2(re[k + j + m_4], im[k + j + m_4]); a2 *= w2;

                        Complex a3(re[k + j + 3 * m_4], im[k + j + 3 * m_4]); a3 *= w3;

                        Complex t0 = a0 + a2;
                        Complex t1 = a0 - a2;
                        Complex t2 = a1 + a3;
                        Complex t3 = a1 - a3;

                        Complex o0 = t0 + t2;
                        Complex o2 = t0 - t2;
                        Complex o1(t1.real() + t3.imag(), t1.imag() - t3.real());
                        Complex o3(t1.real() - t3.imag(), t1.imag() + t3.real());

                        re[k + j] = o0.real(); im[k + j] = o0.imag();
                        re[k + j + m_4] = o1.real(); im[k + j + m_4] = o1.imag();
                        re[k + j + 2 * m_4] = o2.real(); im[k + j + 2 * m_4] = o2.imag();
                        re[k + j + 3 * m_4] = o3.real(); im[k + j + 3 * m_4] = o3.imag();
                    }
                }
            }
        }
    }
private:
    int order;
    int fftSize;
    std::vector<Complex> twiddles;
    // Applied AlignedAllocator to twiddle tables to prevent cache-line splits
    std::vector<std::vector<float, AlignedAllocator<float>>> packed_twiddles;
    // Mutable temp buffers for AoS <-> SoA conversion wrappers
    // Avoids allocation overhead in the benchmark hot loop
    mutable std::vector<float, AlignedAllocator<float>> temp_in_re;
    mutable std::vector<float, AlignedAllocator<float>> temp_in_im;
    mutable std::vector<float, AlignedAllocator<float>> temp_out_re;
    mutable std::vector<float, AlignedAllocator<float>> temp_out_im;

    // 后备bit reverse, though the Out-of-Place path bypasses it
    void bitReverse(float* re, float* im, int n) const {
        for (int i = 0; i < n; ++i) {
            int j = __builtin_bitreverse32(i) >> (32 - order);
            if (i < j) {
                std::swap(re[i], re[j]); 
                std::swap(im[i], im[j]);
            }
        }
    }
};