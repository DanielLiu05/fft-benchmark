/*
made by Daniel
v1.0.0 intrinsic optimizd radix-4
v1.0.1 intrinsic optimized radix-2 and radix-4(better)
v1.0.2 out of place fft using __restrict and manual loop unrolling
bit reversal precompute switched to bit reversal done with __builtin_bitreverse32
v1.1.1 Switched to split-complex (SoA) and avoid vld2/vst2 in the hot loop.
v1.1.2 removed dead vars, alignedAllocator, std::assume_aligned
v1.2.0 changed alignment from 64 to architecture specific
v2.0.0 Rewritten core SIMD math to raw AArch64 Inline Assembly
*/
#pragma once

// ARM64 NEON Intrinsics Header (Still needed for types and vld2/vst2)
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
// ARCHITECTURE-SPECIFIC CACHE-LINE ALIGNMENT DETECTION
// Apple Silicon (M1/M2/M3/M4) uses 128-byte L1 Cache Lines.
// Standard x86-64 and most other ARM cores use 64-byte Cache Lines.
// =================================================================================
#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
#define NEONFFT_CACHELINE_ALIGNMENT 128
#else
#define NEONFFT_CACHELINE_ALIGNMENT 64
#endif

// =============================================================================
// 64/128-BYTE ALIGNED ALLOCATOR (Cache-Line Optimization)
// Prevents cache-line splits during SIMD loads/stores in the hot loop.
// =============================================================================
template <typename T, size_t Alignment = NEONFFT_CACHELINE_ALIGNMENT>
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

// =============================================================================
// INLINE ASSEMBLY WRAPPERS FOR AARCH64 SIMD INSTRUCTIONS
// Replaces NEON Intrinsics with raw assembly for maximum compiler transparency.
// =============================================================================

static inline float32x4_t asm_ld1q_f32(const float* ptr) {
    float32x4_t res;
    __asm__ volatile (
        "ld1 {%0.4s}, [%1] \n\t" /* LD1 (Load single 1-element structure): Loads 4 contiguous 32-bit floats from memory address %1 into vector register %0 */
        : "=w" (res) 
        : "r" (ptr)
    );
    return res;
}

static inline void asm_st1q_f32(float* ptr, float32x4_t vec) {
    __asm__ volatile (
        "st1 {%0.4s}, [%1] \n\t" /* ST1 (Store single 1-element structure): Stores 4 contiguous 32-bit floats from vector register %0 to memory address %1 */
        : 
        : "w" (vec), "r" (ptr) 
        : "memory" /* Clobber memory to prevent compiler reordering stores */
    );
}

static inline float32x4_t asm_uzp1q_f32(float32x4_t a, float32x4_t b) {
    float32x4_t res;
    __asm__ (
        "uzp1 %0.4s, %1.4s, %2.4s \n\t" /* UZP1 (Unzip vectors): Extracts even-indexed elements (0, 2 from %1 and 0, 2 from %2) and packs them into %0 */
        : "=w" (res) 
        : "w" (a), "w" (b)
    );
    return res;
}

static inline float32x4_t asm_uzp2q_f32(float32x4_t a, float32x4_t b) {
    float32x4_t res;
    __asm__ (
        "uzp2 %0.4s, %1.4s, %2.4s \n\t" /* UZP2 (Unzip vectors): Extracts odd-indexed elements (1, 3 from %1 and 1, 3 from %2) and packs them into %0 */
        : "=w" (res) 
        : "w" (a), "w" (b)
    );
    return res;
}

static inline float32x4_t asm_zip1q_f32(float32x4_t a, float32x4_t b) {
    float32x4_t res;
    __asm__ (
        "zip1 %0.4s, %1.4s, %2.4s \n\t" /* ZIP1 (Zip vectors): Interleaves the lower halves (elements 0, 1 from %1 and 0, 1 from %2) into %0 */
        : "=w" (res) 
        : "w" (a), "w" (b)
    );
    return res;
}

static inline float32x4_t asm_zip2q_f32(float32x4_t a, float32x4_t b) {
    float32x4_t res;
    __asm__ (
        "zip2 %0.4s, %1.4s, %2.4s \n\t" /* ZIP2 (Zip vectors): Interleaves the upper halves (elements 2, 3 from %1 and 2, 3 from %2) into %0 */
        : "=w" (res) 
        : "w" (a), "w" (b)
    );
    return res;
}

static inline float32x4_t asm_faddq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t res;
    __asm__ (
        "fadd %0.4s, %1.4s, %2.4s \n\t" /* FADD (Floating-point Add): Adds 4 single-precision floats in %1 and %2, stores result in %0 */
        : "=w" (res) 
        : "w" (a), "w" (b)
    );
    return res;
}

static inline float32x4_t asm_fsubq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t res;
    __asm__ (
        "fsub %0.4s, %1.4s, %2.4s \n\t" /* FSUB (Floating-point Subtract): Subtracts 4 single-precision floats in %2 from %1, stores result in %0 */
        : "=w" (res) 
        : "w" (a), "w" (b)
    );
    return res;
}

static inline float32x4_t asm_fmulq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t res;
    __asm__ (
        "fmul %0.4s, %1.4s, %2.4s \n\t" /* FMUL (Floating-point Multiply): Multiplies 4 single-precision floats in %1 and %2, stores result in %0 */
        : "=w" (res) 
        : "w" (a), "w" (b)
    );
    return res;
}

static inline float32x4_t asm_fmlaq_f32(float32x4_t acc, float32x4_t a, float32x4_t b) {
    __asm__ (
        "fmla %0.4s, %1.4s, %2.4s \n\t" /* FMLA (Floating-point fused Multiply-Add): Multiplies %1 and %2, adds to %0, stores back in %0 */
        : "+w" (acc) 
        : "w" (a), "w" (b)
    );
    return acc;
}

static inline float32x4_t asm_fmlsq_f32(float32x4_t acc, float32x4_t a, float32x4_t b) {
    __asm__ (
        "fmls %0.4s, %1.4s, %2.4s \n\t" /* FMLS (Floating-point fused Multiply-Subtract): Multiplies %1 and %2, subtracts from %0, stores back in %0 */
        : "+w" (acc) 
        : "w" (a), "w" (b)
    );
    return acc;
}

class AArch64FFT {
public:
    using Complex = std::complex<float>;
    
    // Helper struct to guarantee 64B/128B aligned input/output buffers for the user
    struct AlignedBuffer {
        std::vector<float, AlignedAllocator<float>> re;
        std::vector<float, AlignedAllocator<float>> im;
        AlignedBuffer(size_t size) : re(size), im(size) {}
    };

    AArch64FFT(int order) : order(order), fftSize(1 << order) {
        if (order < 3) {
            throw std::invalid_argument("AArch64 FFT requires order >= 3 (size >= 8) for vectorization");
        }
        
        // Twiddle table MUST be size fftSize to prevent out-of-bounds 
        // when calculating W3 (3 * j * step) in the final stages.
        twiddles.resize(fftSize);  
        for (int i = 0; i < fftSize; ++i) {
            double phase = -2.0 * M_PI * i / fftSize;
            twiddles[i] = Complex((float)std::cos(phase), (float)std::sin(phase));
        }

        // =====================================================================
        // TWIDDLE PRE-PACKING (Eliminates the SIMD Memory Wall)
        // Packed in Split-Complex (SoA) format: [Re0, Re1, Re2, Re3, Im0, Im1, Im2, Im3]
        // =====================================================================  
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

    // =========================================================================
    // AOS WRAPPERS 
    // (For compatibility with std::complex<float> benchmarks)
    // =========================================================================
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
        
        // AoS -> SoA using NEON vld2 (Kept as intrinsic due to consecutive register constraints)
        for (int i = 0; i < fftSize; i += 4) {
            float32x4x2_t A = vld2q_f32(in_ptr + i * 2);
            asm_st1q_f32(out_re_ptr + i, A.val[0]);
            asm_st1q_f32(out_im_ptr + i, A.val[1]);
        }
        
        performForward(temp_in_re.data(), temp_in_im.data(), temp_out_re.data(), temp_out_im.data());
        
        float* out_ptr = reinterpret_cast<float*>(output);
        const float* in_re_ptr = temp_out_re.data();
        const float* in_im_ptr = temp_out_im.data();
        
        // SoA -> AoS using NEON vst2 (Kept as intrinsic due to consecutive register constraints)
        for (int i = 0; i < fftSize; i += 4) {
            float32x4_t re = asm_ld1q_f32(in_re_ptr + i);
            float32x4_t im = asm_ld1q_f32(in_im_ptr + i);
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
            asm_st1q_f32(out_re_ptr + i, A.val[0]);
            asm_st1q_f32(out_im_ptr + i, A.val[1]);
        }
        
        performForward(temp_in_re.data(), temp_in_im.data());
        
        for (int i = 0; i < fftSize; i += 4) {
            float32x4_t re = asm_ld1q_f32(out_re_ptr + i);
            float32x4_t im = asm_ld1q_f32(out_im_ptr + i);
            float32x4x2_t A;
            A.val[0] = re;
            A.val[1] = im;
            vst2q_f32(data_ptr + i * 2, A);
        }
    }

    // =====================================================================
    // OUT-OF-PLACE FFT 
    // =====================================================================
    void performForward(const float* __restrict__ in_re, const float* __restrict__ in_im,
                        float* __restrict__ out_re, float* __restrict__ out_im) const {
#if defined(__GNUC__) || defined(__clang__)
        in_re  = (const float*)__builtin_assume_aligned(in_re, NEONFFT_CACHELINE_ALIGNMENT);
        in_im  = (const float*)__builtin_assume_aligned(in_im, NEONFFT_CACHELINE_ALIGNMENT);
        out_re = (float*)__builtin_assume_aligned(out_re, NEONFFT_CACHELINE_ALIGNMENT);
        out_im = (float*)__builtin_assume_aligned(out_im, NEONFFT_CACHELINE_ALIGNMENT);
#endif

        if (in_re == out_re && in_im == out_im) {
            // Fallback for strict in-place calls
            bitReverse(out_re, out_im, fftSize);
        } else {
            // Out-of-place Bit-Reversal using ARM64 `rbit` intrinsic
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
    // IN-PLACE SIMD BUTTERFLY MATH (AArch64 Assembly)
    // =====================================================================
    void performForward(float* __restrict__ re, float* __restrict__ im) const {
#if defined(__GNUC__) || defined(__clang__)
        re = (float*)__builtin_assume_aligned(re, NEONFFT_CACHELINE_ALIGNMENT);
        im = (float*)__builtin_assume_aligned(im, NEONFFT_CACHELINE_ALIGNMENT);
#endif

        int s = 2;
        // =====================================================================
        // RADIX-2 STAGE FOR ODD ORDERS
        // =====================================================================
        if (order % 2 != 0) {
            int k = 0;
            // Vectorized path: process 8 complex numbers (4 butterflies) per iteration
            for (; k <= fftSize - 8; k += 8) {
                // 1. LOAD 8 COMPLEX NUMBERS (Separated into Re/Im arrays)
                float32x4_t re0 = asm_ld1q_f32(&re[k]);
                float32x4_t re1 = asm_ld1q_f32(&re[k + 4]);
                float32x4_t im0 = asm_ld1q_f32(&im[k]);
                float32x4_t im1 = asm_ld1q_f32(&im[k + 4]);
                
                // 2. DE-INTERLEAVE to separate even (U) and odd (T) elements
                float32x4_t U_re = asm_uzp1q_f32(re0, re1);
                float32x4_t T_re = asm_uzp2q_f32(re0, re1);
                float32x4_t U_im = asm_uzp1q_f32(im0, im1);
                float32x4_t T_im = asm_uzp2q_f32(im0, im1);
                
                // 3. BUTTERFLY MATH (Twiddle is 1+0i, so no FMA needed)
                float32x4_t O0_re = asm_faddq_f32(U_re, T_re);
                float32x4_t O0_im = asm_faddq_f32(U_im, T_im);
                float32x4_t O1_re = asm_fsubq_f32(U_re, T_re);
                float32x4_t O1_im = asm_fsubq_f32(U_im, T_im);
                 
                // 4. INTERLEAVE BACK
                float32x4_t R0_re = asm_zip1q_f32(O0_re, O1_re);
                float32x4_t R1_re = asm_zip2q_f32(O0_re, O1_re);
                float32x4_t R0_im = asm_zip1q_f32(O0_im, O1_im);
                float32x4_t R1_im = asm_zip2q_f32(O0_im, O1_im);
                
                // 5. STORE
                asm_st1q_f32(&re[k], R0_re);
                asm_st1q_f32(&re[k + 4], R1_re);
                asm_st1q_f32(&im[k], R0_im);
                asm_st1q_f32(&im[k + 4], R1_im);
            }
            
            for (; k < fftSize; k += 2) {
                float u_re = re[k], u_im = im[k];
                float t_re = re[k+1], t_im = im[k+1];
                re[k] = u_re + t_re; im[k] = u_im + t_im;
                re[k+1] = u_re - t_re; im[k+1] = u_im - t_im;
            }
            
            s = 3; 
        }
        
        // =====================================================================
        // Radix-4 stages
        // =====================================================================
        for (; s <= order; s += 2) {
            int m = 1 << s;
            int m_4 = m >> 2; 

            if (m_4 >= 4) {
                const float* __restrict__ tw_ptr_base = packed_twiddles[s].data();
#if __cplusplus >= 202002L && defined(__cpp_lib_assume_aligned)
                tw_ptr_base = std::assume_aligned<NEONFFT_CACHELINE_ALIGNMENT>(tw_ptr_base);
#elif defined(__GNUC__) || defined(__clang__)
                tw_ptr_base = static_cast<const float*>(__builtin_assume_aligned(tw_ptr_base, NEONFFT_CACHELINE_ALIGNMENT));
#endif
                for (int k = 0; k < fftSize; k += m) {
                    const float* __restrict__ tw_ptr = tw_ptr_base;
                    for (int j = 0; j < m_4; j += 4) {
                        if (m_4 >= 128 && (j + 32) < m_4) {
                            __builtin_prefetch(tw_ptr + 24 * 8, 0, 3);
                            __builtin_prefetch(&re[k+j+32], 0, 1);
                            __builtin_prefetch(&im[k+j+32], 0, 1);
                            __builtin_prefetch(&re[k+j+32+m_4], 0, 1);
                            __builtin_prefetch(&im[k+j+32+m_4], 0, 1);
                        }

                        // 1. LOAD DATA
                        float32x4_t A0_re = asm_ld1q_f32(&re[k + j]);
                        float32x4_t A0_im = asm_ld1q_f32(&im[k + j]);
                        
                        float32x4_t A1_re = asm_ld1q_f32(&re[k + j + 2 * m_4]); 
                        float32x4_t A1_im = asm_ld1q_f32(&im[k + j + 2 * m_4]); 

                        float32x4_t A2_re = asm_ld1q_f32(&re[k + j + m_4]);     
                        float32x4_t A2_im = asm_ld1q_f32(&im[k + j + m_4]);     

                        float32x4_t A3_re = asm_ld1q_f32(&re[k + j + 3 * m_4]);
                        float32x4_t A3_im = asm_ld1q_f32(&im[k + j + 3 * m_4]);
                        
                        // 2. LOAD PRE-PACKED TWIDDLES
                        float32x4_t W1_re = asm_ld1q_f32(tw_ptr +  0);
                        float32x4_t W1_im = asm_ld1q_f32(tw_ptr +  4);
                        float32x4_t W2_re = asm_ld1q_f32(tw_ptr +  8);
                        float32x4_t W2_im = asm_ld1q_f32(tw_ptr + 12);
                        float32x4_t W3_re = asm_ld1q_f32(tw_ptr + 16);
                        float32x4_t W3_im = asm_ld1q_f32(tw_ptr + 20);
                        tw_ptr += 24;

                        // 3. COMPLEX MULTIPLICATION (FMA)
                        float32x4_t a1_re = asm_fmulq_f32(W1_re, A1_re);
                        a1_re = asm_fmlsq_f32(a1_re, W1_im, A1_im);
                        float32x4_t a1_im = asm_fmulq_f32(W1_re, A1_im);
                        a1_im = asm_fmlaq_f32(a1_im, W1_im, A1_re);
                        
                        float32x4_t a2_re = asm_fmulq_f32(W2_re, A2_re);
                        a2_re = asm_fmlsq_f32(a2_re, W2_im, A2_im);
                        float32x4_t a2_im = asm_fmulq_f32(W2_re, A2_im);
                        a2_im = asm_fmlaq_f32(a2_im, W2_im, A2_re);

                        float32x4_t a3_re = asm_fmulq_f32(W3_re, A3_re);
                        a3_re = asm_fmlsq_f32(a3_re, W3_im, A3_im);
                        float32x4_t a3_im = asm_fmulq_f32(W3_re, A3_im);
                        a3_im = asm_fmlaq_f32(a3_im, W3_im, A3_re);

                        // 4. RADIX-4 BUTTERFLY MATH
                        float32x4_t t0_re = asm_faddq_f32(A0_re, a2_re);
                        float32x4_t t0_im = asm_faddq_f32(A0_im, a2_im);
                        float32x4_t t1_re = asm_fsubq_f32(A0_re, a2_re);
                        float32x4_t t1_im = asm_fsubq_f32(A0_im, a2_im);

                        float32x4_t t2_re = asm_faddq_f32(a1_re, a3_re);
                        float32x4_t t2_im = asm_faddq_f32(a1_im, a3_im);
                        float32x4_t t3_re = asm_fsubq_f32(a1_re, a3_re);
                        float32x4_t t3_im = asm_fsubq_f32(a1_im, a3_im);

                        float32x4_t O0_re = asm_faddq_f32(t0_re, t2_re);
                        float32x4_t O0_im = asm_faddq_f32(t0_im, t2_im);
                        float32x4_t O2_re = asm_fsubq_f32(t0_re, t2_re);
                        float32x4_t O2_im = asm_fsubq_f32(t0_im, t2_im);

                        float32x4_t O1_re = asm_faddq_f32(t1_re, t3_im);
                        float32x4_t O1_im = asm_fsubq_f32(t1_im, t3_re);
                        float32x4_t O3_re = asm_fsubq_f32(t1_re, t3_im);
                        float32x4_t O3_im = asm_faddq_f32(t1_im, t3_re);

                        // 5. STORE
                        asm_st1q_f32(&re[k + j], O0_re);
                        asm_st1q_f32(&im[k + j], O0_im);
                        asm_st1q_f32(&re[k + j + m_4], O1_re);
                        asm_st1q_f32(&im[k + j + m_4], O1_im);
                        asm_st1q_f32(&re[k + j + 2 * m_4], O2_re);
                        asm_st1q_f32(&im[k + j + 2 * m_4], O2_im);
                        asm_st1q_f32(&re[k + j + 3 * m_4], O3_re);
                        asm_st1q_f32(&im[k + j + 3 * m_4], O3_im);
                    }
                }
            } else {
                // Scalar fallback
                int step = fftSize / m;     
                for (int k = 0; k < fftSize; k += m) {
                    for (int j = 0; j < m_4; ++j) {
                        Complex w1 = twiddles[j * step];
                        Complex w2 = twiddles[2 * j * step];
                        Complex w3 = twiddles[3 * j * step];       

                        Complex a0(re[k + j], im[k + j]);
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
    std::vector<std::vector<float, AlignedAllocator<float>>> packed_twiddles;

    mutable std::vector<float, AlignedAllocator<float>> temp_in_re;
    mutable std::vector<float, AlignedAllocator<float>> temp_in_im;
    mutable std::vector<float, AlignedAllocator<float>> temp_out_re;
    mutable std::vector<float, AlignedAllocator<float>> temp_out_im;

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


