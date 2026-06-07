/*
v1.0.1 step1 step6 融合解包打包，避免两次全量内存搬运

v1.0.2 Step 4 的 NEON 4x4 块转置 
        Step 1 & Step 6 的“对称访存”设计
        Step 1：外层循环 c，内层 r。实现 连续读取 (Sequential Read) 输入数组，跨步写入 2D 矩阵。M1 的硬件预取器 (Hardware Prefetcher) 会疯狂预取输入数据。
        Step 6：外层循环 r，内层 c。实现 跨步读取 2D 矩阵，连续写入 (Sequential Write) 输出数组。CPU 的 Write-Combining 机制会将连续写入合并，直接打入主存。
*/
#pragma once
#include "NeonFFT.h" // Inherits AlignedAllocator and the thread-safe SoA NeonFFT engine
#include <dispatch/dispatch.h> // Apple GCD Native Multi-Threading
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <arm_neon.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SixStepFFT_MT {
private:
    int order1, order2;
    int N1, N2, N;
    int stride_N1, stride_N2;
    NeonFFT engine1; // Engine for N1 (Columns)
    NeonFFT engine2; // Engine for N2 (Rows)

    // Ping-Pong Buffers (Padded for 64-byte Cache-Line Alignment)
    std::vector<float, AlignedAllocator<float>> mat1_re, mat1_im;
    std::vector<float, AlignedAllocator<float>> mat2_re, mat2_im;
    std::vector<float, AlignedAllocator<float>> matT1_re, matT1_im;
    std::vector<float, AlignedAllocator<float>> matT2_re, matT2_im;

    // 2D Twiddle Factors
    std::vector<float, AlignedAllocator<float>> tw2d_re, tw2d_im;

    void generate2DTwiddles() {
        tw2d_re.resize(N1 * stride_N2);
        tw2d_im.resize(N1 * stride_N2);
        for (int r = 0; r < N1; ++r) {
            for (int c = 0; c < N2; ++c) {
                double phase = -2.0 * M_PI * (double)r * (double)c / (double)N;
                tw2d_re[r * stride_N2 + c] = (float)std::cos(phase);
                tw2d_im[r * stride_N2 + c] = (float)std::sin(phase);
            }
        }
    }

public:
    // 1. Single-argument constructor compatibility wrapper
    SixStepFFT_MT(int total_order)
        : SixStepFFT_MT(total_order / 2, total_order - (total_order / 2)) {}

    SixStepFFT_MT(int o1, int o2) 
        : order1(o1), order2(o2), 
          N1(1 << o1), N2(1 << o2), N(N1 * N2),
          engine1(o1), engine2(o2) 
    {
        if (order1 < 3 || order2 < 3) {
            throw std::invalid_argument("SixStepFFT requires both orders to be >= 3");
        }

        // Pad strides to multiples of 16 floats (64 bytes) to prevent False Sharing & Cache-Line Splits
        stride_N1 = (N1 + 15) & ~15;
        stride_N2 = (N2 + 15) & ~15;

        mat1_re.resize(N1 * stride_N2); mat1_im.resize(N1 * stride_N2);
        mat2_re.resize(N1 * stride_N2); mat2_im.resize(N1 * stride_N2);
        matT1_re.resize(N2 * stride_N1); matT1_im.resize(N2 * stride_N1);
        matT2_re.resize(N2 * stride_N1); matT2_im.resize(N2 * stride_N1);

        generate2DTwiddles();
    }

    // =====================================================================
    // IN-PLACE AOS WRAPPER
    // =====================================================================
    void performForward(std::complex<float>* data) {
        performForward(data, data);
    }

    // =====================================================================
    // OUT-OF-PLACE AOS WRAPPER (Fused with Step 1 & Step 6)
    // Eliminates redundant intermediate SoA buffers and memory copies.
    // =====================================================================
    void performForward(const std::complex<float>* input, std::complex<float>* output) {
        const float* in_ptr = reinterpret_cast<const float*>(input);
        float* out_ptr = reinterpret_cast<float*>(output);

        // ---------------------------------------------------------
        // STEP 1: Map 1D AoS Input to 2D Matrix + Unpack
        // FIX: Sequential Read (Hardware Prefetcher friendly)
        // ---------------------------------------------------------
        /*dispatch_apply(N1, DISPATCH_APPLY_AUTO, ^(size_t r) {
            for (int c = 0; c < N2; ++c) {
                int idx = (r * N2 + c) * 2;
                mat1_re[r * stride_N2 + c] = in_ptr[idx];
                mat1_im[r * stride_N2 + c] = in_ptr[idx + 1];
            }
        });*/
        dispatch_apply(N2, DISPATCH_APPLY_AUTO, ^(size_t c) {
            int base_n = c * N1;
            for (int r = 0; r < N1; ++r) {
                int idx = (base_n + r) * 2;
                mat1_re[r * stride_N2 + c] = in_ptr[idx];
                mat1_im[r * stride_N2 + c] = in_ptr[idx + 1];
            }
        });

        // ---------------------------------------------------------
        // STEP 2: 1D FFTs on Rows (Multi-Threaded via GCD)
        // ---------------------------------------------------------
        {
            float* m1_re = mat1_re.data(); float* m1_im = mat1_im.data();
            float* m2_re = mat2_re.data(); float* m2_im = mat2_im.data();
            int stride = stride_N2;
            const NeonFFT* eng = &engine2;
            
            dispatch_apply(N1, DISPATCH_APPLY_AUTO, ^(size_t r) {
                eng->performForward(&m1_re[r * stride], &m1_im[r * stride],
                                    &m2_re[r * stride], &m2_im[r * stride]);
            });
        }

        // ---------------------------------------------------------
        // STEP 3: 2D Twiddle Factor Multiplication (Multi-Threaded)
        // ---------------------------------------------------------
        {
            float* src_re = mat2_re.data(); float* src_im = mat2_im.data();
            float* dst_re = mat1_re.data(); float* dst_im = mat1_im.data();
            float* w_re = tw2d_re.data();   float* w_im = tw2d_im.data();
            int stride = stride_N2;

            dispatch_apply(N1, DISPATCH_APPLY_AUTO, ^(size_t r) {
                int offset = r * stride;
                int c = 0;
                for (; c <= N2 - 4; c += 4) {
                    float32x4_t in_re = vld1q_f32(&src_re[offset + c]);
                    float32x4_t in_im = vld1q_f32(&src_im[offset + c]);
                    float32x4_t tw_re = vld1q_f32(&w_re[offset + c]);
                    float32x4_t tw_im = vld1q_f32(&w_im[offset + c]);
                    
                    float32x4_t out_re = vmulq_f32(in_re, tw_re);
                    out_re = vfmsq_f32(out_re, in_im, tw_im);
                    float32x4_t out_im = vmulq_f32(in_re, tw_im);
                    out_im = vfmaq_f32(out_im, in_im, tw_re);
                    
                    vst1q_f32(&dst_re[offset + c], out_re);
                    vst1q_f32(&dst_im[offset + c], out_im);
                }
                for (; c < N2; ++c) {
                    float re = src_re[offset + c], im = src_im[offset + c];
                    float wr = w_re[offset + c], wi = w_im[offset + c];
                    dst_re[offset + c] = re * wr - im * wi;
                    dst_im[offset + c] = re * wi + im * wr;
                }
            });
        }

        // ---------------------------------------------------------
        // STEP 4: Cache-Oblivious Tiled Transpose (NEON 4x4 Block)
        // Bypasses Store Buffer bottlenecks via register-level transposition
        // ---------------------------------------------------------
        {
            const int TILE = 32; 
            int tiles_r = (N1 + TILE - 1) / TILE;
            int tiles_c = (N2 + TILE - 1) / TILE;
            int total_tiles = tiles_r * tiles_c;

            float* s_re = mat1_re.data(); float* s_im = mat1_im.data();
            float* d_re = matT1_re.data(); float* d_im = matT1_im.data();
            int s_stride = stride_N2; int d_stride = stride_N1;

            dispatch_apply(total_tiles, DISPATCH_APPLY_AUTO, ^(size_t idx) {
                int tr = idx / tiles_c;
                int tc = idx % tiles_c;
                int r_start = tr * TILE; int c_start = tc * TILE;
                int r_end = std::min(r_start + TILE, N1);
                int c_end = std::min(c_start + TILE, N2);
                
                for (int i = r_start; i < r_end; i += 4) {
                    for (int j = c_start; j < c_end; j += 4) {
                        // --- Transpose Real Part ---
                        float32x4_t row0_re = vld1q_f32(&s_re[i * s_stride + j]);
                        float32x4_t row1_re = vld1q_f32(&s_re[(i+1) * s_stride + j]);
                        float32x4_t row2_re = vld1q_f32(&s_re[(i+2) * s_stride + j]);
                        float32x4_t row3_re = vld1q_f32(&s_re[(i+3) * s_stride + j]);

                        float32x4x2_t t01_re = vtrnq_f32(row0_re, row1_re); // val[0]=[a0,b0,a2,b2], val[1]=[a1,b1,a3,b3]
                        float32x4x2_t t23_re = vtrnq_f32(row2_re, row3_re); // val[0]=[c0,d0,c2,d2], val[1]=[c1,d1,c3,d3]

                        vst1q_f32(&d_re[j * d_stride + i],       vcombine_f32(vget_low_f32(t01_re.val[0]),  vget_low_f32(t23_re.val[0])));  // [a0, b0, c0, d0]
                        vst1q_f32(&d_re[(j+1) * d_stride + i],   vcombine_f32(vget_low_f32(t01_re.val[1]),  vget_low_f32(t23_re.val[1])));  // [a1, b1, c1, d1]
                        vst1q_f32(&d_re[(j+2) * d_stride + i],   vcombine_f32(vget_high_f32(t01_re.val[0]), vget_high_f32(t23_re.val[0]))); // [a2, b2, c2, d2]
                        vst1q_f32(&d_re[(j+3) * d_stride + i],   vcombine_f32(vget_high_f32(t01_re.val[1]), vget_high_f32(t23_re.val[1]))); // [a3, b3, c3, d3]

                        // --- Transpose Imaginary Part ---
                        float32x4_t row0_im = vld1q_f32(&s_im[i * s_stride + j]);
                        float32x4_t row1_im = vld1q_f32(&s_im[(i+1) * s_stride + j]);
                        float32x4_t row2_im = vld1q_f32(&s_im[(i+2) * s_stride + j]);
                        float32x4_t row3_im = vld1q_f32(&s_im[(i+3) * s_stride + j]);

                        float32x4x2_t t01_im = vtrnq_f32(row0_im, row1_im);
                        float32x4x2_t t23_im = vtrnq_f32(row2_im, row3_im);

                        vst1q_f32(&d_im[j * d_stride + i],       vcombine_f32(vget_low_f32(t01_im.val[0]),  vget_low_f32(t23_im.val[0])));
                        vst1q_f32(&d_im[(j+1) * d_stride + i],   vcombine_f32(vget_low_f32(t01_im.val[1]),  vget_low_f32(t23_im.val[1])));
                        vst1q_f32(&d_im[(j+2) * d_stride + i],   vcombine_f32(vget_high_f32(t01_im.val[0]), vget_high_f32(t23_im.val[0])));
                        vst1q_f32(&d_im[(j+3) * d_stride + i],   vcombine_f32(vget_high_f32(t01_im.val[1]), vget_high_f32(t23_im.val[1])));
                    }
                }
            });
        }

        // ---------------------------------------------------------
        // STEP 5: 1D FFTs on Columns (Multi-Threaded via GCD)
        // ---------------------------------------------------------
        {
            float* mT1_re = matT1_re.data(); float* mT1_im = matT1_im.data();
            float* mT2_re = matT2_re.data(); float* mT2_im = matT2_im.data();
            int stride = stride_N1;
            const NeonFFT* eng = &engine1;
            
            dispatch_apply(N2, DISPATCH_APPLY_AUTO, ^(size_t c) {
                eng->performForward(&mT1_re[c * stride], &mT1_im[c * stride],
                                    &mT2_re[c * stride], &mT2_im[c * stride]);
            });
        }

        // ---------------------------------------------------------
        // STEP 6: Extract 1D Output + Pack to AoS
        // FIX: Sequential Write (Write-Combining friendly)
        // ---------------------------------------------------------
        dispatch_apply(N1, DISPATCH_APPLY_AUTO, ^(size_t r) {
            int base_k = r * N2;
            for (int c = 0; c < N2; ++c) {
                int k = base_k + c;
                int idx = k * 2;
                out_ptr[idx]     = matT2_re[c * stride_N1 + r];
                out_ptr[idx + 1] = matT2_im[c * stride_N1 + r];
            }
        });
    }

    // =====================================================================
    // OUT-OF-PLACE SOA PIPELINE (For pure SoA inputs)
    // =====================================================================
    void performForward(const float* in_re, const float* in_im, 
                        float* out_re, float* out_im) {
        
        // STEP 1: Sequential Read
        /*dispatch_apply(N1, DISPATCH_APPLY_AUTO, ^(size_t r) {
            for (int c = 0; c < N2; ++c) {
                mat1_re[r * stride_N2 + c] = in_re[r * N2 + c];
                mat1_im[r * stride_N2 + c] = in_im[r * N2 + c];
            }
        });*/
        dispatch_apply(N2, DISPATCH_APPLY_AUTO, ^(size_t c) {
            int base_n = c * N1;
            for (int r = 0; r < N1; ++r) {
                mat1_re[r * stride_N2 + c] = in_re[base_n + r];
                mat1_im[r * stride_N2 + c] = in_im[base_n + r];
            }
        });

        // STEP 2
        {
            float* m1_re = mat1_re.data(); float* m1_im = mat1_im.data();
            float* m2_re = mat2_re.data(); float* m2_im = mat2_im.data();
            int stride = stride_N2;
            const NeonFFT* eng = &engine2;
            dispatch_apply(N1, DISPATCH_APPLY_AUTO, ^(size_t r) {
                eng->performForward(&m1_re[r * stride], &m1_im[r * stride],
                                    &m2_re[r * stride], &m2_im[r * stride]);
            });
        }

        // STEP 3
        {
            float* src_re = mat2_re.data(); float* src_im = mat2_im.data();
            float* dst_re = mat1_re.data(); float* dst_im = mat1_im.data();
            float* w_re = tw2d_re.data();   float* w_im = tw2d_im.data();
            int stride = stride_N2;
            dispatch_apply(N1, DISPATCH_APPLY_AUTO, ^(size_t r) {
                int offset = r * stride;
                int c = 0;
                for (; c <= N2 - 4; c += 4) {
                    float32x4_t in_re = vld1q_f32(&src_re[offset + c]);
                    float32x4_t in_im = vld1q_f32(&src_im[offset + c]);
                    float32x4_t tw_re = vld1q_f32(&w_re[offset + c]);
                    float32x4_t tw_im = vld1q_f32(&w_im[offset + c]);
                    float32x4_t out_re = vmulq_f32(in_re, tw_re);
                    out_re = vfmsq_f32(out_re, in_im, tw_im);
                    float32x4_t out_im = vmulq_f32(in_re, tw_im);
                    out_im = vfmaq_f32(out_im, in_im, tw_re);
                    vst1q_f32(&dst_re[offset + c], out_re);
                    vst1q_f32(&dst_im[offset + c], out_im);
                }
                for (; c < N2; ++c) {
                    float re = src_re[offset + c], im = src_im[offset + c];
                    float wr = w_re[offset + c], wi = w_im[offset + c];
                    dst_re[offset + c] = re * wr - im * wi;
                    dst_im[offset + c] = re * wi + im * wr;
                }
            });
        }

        // STEP 4: NEON 4x4 Block Transpose
        {
            const int TILE = 32; 
            int tiles_r = (N1 + TILE - 1) / TILE;
            int tiles_c = (N2 + TILE - 1) / TILE;
            int total_tiles = tiles_r * tiles_c;
            float* s_re = mat1_re.data(); float* s_im = mat1_im.data();
            float* d_re = matT1_re.data(); float* d_im = matT1_im.data();
            int s_stride = stride_N2; int d_stride = stride_N1;
            dispatch_apply(total_tiles, DISPATCH_APPLY_AUTO, ^(size_t idx) {
                int tr = idx / tiles_c;
                int tc = idx % tiles_c;
                int r_start = tr * TILE; int c_start = tc * TILE;
                int r_end = std::min(r_start + TILE, N1);
                int c_end = std::min(c_start + TILE, N2);
                for (int i = r_start; i < r_end; i += 4) {
                    for (int j = c_start; j < c_end; j += 4) {
                        // --- Transpose Real Part ---
                        float32x4_t row0_re = vld1q_f32(&s_re[i * s_stride + j]);
                        float32x4_t row1_re = vld1q_f32(&s_re[(i+1) * s_stride + j]);
                        float32x4_t row2_re = vld1q_f32(&s_re[(i+2) * s_stride + j]);
                        float32x4_t row3_re = vld1q_f32(&s_re[(i+3) * s_stride + j]);

                        float32x4x2_t t01_re = vtrnq_f32(row0_re, row1_re); // val[0]=[a0,b0,a2,b2], val[1]=[a1,b1,a3,b3]
                        float32x4x2_t t23_re = vtrnq_f32(row2_re, row3_re); // val[0]=[c0,d0,c2,d2], val[1]=[c1,d1,c3,d3]

                        vst1q_f32(&d_re[j * d_stride + i],       vcombine_f32(vget_low_f32(t01_re.val[0]),  vget_low_f32(t23_re.val[0])));  // [a0, b0, c0, d0]
                        vst1q_f32(&d_re[(j+1) * d_stride + i],   vcombine_f32(vget_low_f32(t01_re.val[1]),  vget_low_f32(t23_re.val[1])));  // [a1, b1, c1, d1]
                        vst1q_f32(&d_re[(j+2) * d_stride + i],   vcombine_f32(vget_high_f32(t01_re.val[0]), vget_high_f32(t23_re.val[0]))); // [a2, b2, c2, d2]
                        vst1q_f32(&d_re[(j+3) * d_stride + i],   vcombine_f32(vget_high_f32(t01_re.val[1]), vget_high_f32(t23_re.val[1]))); // [a3, b3, c3, d3]

                        // --- Transpose Imaginary Part ---
                        float32x4_t row0_im = vld1q_f32(&s_im[i * s_stride + j]);
                        float32x4_t row1_im = vld1q_f32(&s_im[(i+1) * s_stride + j]);
                        float32x4_t row2_im = vld1q_f32(&s_im[(i+2) * s_stride + j]);
                        float32x4_t row3_im = vld1q_f32(&s_im[(i+3) * s_stride + j]);

                        float32x4x2_t t01_im = vtrnq_f32(row0_im, row1_im);
                        float32x4x2_t t23_im = vtrnq_f32(row2_im, row3_im);

                        vst1q_f32(&d_im[j * d_stride + i],       vcombine_f32(vget_low_f32(t01_im.val[0]),  vget_low_f32(t23_im.val[0])));
                        vst1q_f32(&d_im[(j+1) * d_stride + i],   vcombine_f32(vget_low_f32(t01_im.val[1]),  vget_low_f32(t23_im.val[1])));
                        vst1q_f32(&d_im[(j+2) * d_stride + i],   vcombine_f32(vget_high_f32(t01_im.val[0]), vget_high_f32(t23_im.val[0])));
                        vst1q_f32(&d_im[(j+3) * d_stride + i],   vcombine_f32(vget_high_f32(t01_im.val[1]), vget_high_f32(t23_im.val[1])));
                    }
                }
            });
        }

        // STEP 5
        {
            float* mT1_re = matT1_re.data(); float* mT1_im = matT1_im.data();
            float* mT2_re = matT2_re.data(); float* mT2_im = matT2_im.data();
            int stride = stride_N1;
            const NeonFFT* eng = &engine1;
            dispatch_apply(N2, DISPATCH_APPLY_AUTO, ^(size_t c) {
                eng->performForward(&mT1_re[c * stride], &mT1_im[c * stride],
                                    &mT2_re[c * stride], &mT2_im[c * stride]);
            });
        }

        // STEP 6: Sequential Write
        dispatch_apply(N1, DISPATCH_APPLY_AUTO, ^(size_t r) {
            int base_k = r * N2;
            for (int c = 0; c < N2; ++c) {
                int k = base_k + c; 
                out_re[k] = matT2_re[c * stride_N1 + r];
                out_im[k] = matT2_im[c * stride_N1 + r];
            }
        });
    }
};