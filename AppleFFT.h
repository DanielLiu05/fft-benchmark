#pragma once
#include <Accelerate/Accelerate.h>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

class AppleFFT {
public:
    using Complex = std::complex<float>;

    AppleFFT(int order) : order(order), fftSize(1 << order) {
        // Create fft setup 用的radix-2
        fftSetup = vDSP_create_fftsetup(order, FFT_RADIX2);
        if (!fftSetup) {
            throw std::runtime_error("Failed to create vDSP FFT Setup");
        }

        // Pre-allocate contiguous split-complex scratch buffers.
        // Apple Silicon's vDSP requires contiguous memory to use its fastest NEON/AMX paths.
        // make them mutable so performForward() can remain const for the benchmark harness.
        scratchReal.resize(fftSize);
        scratchImag.resize(fftSize);
        outReal.resize(fftSize);
        outImag.resize(fftSize);
    }

    ~AppleFFT() {
        if (fftSetup) {
            vDSP_destroy_fftsetup(fftSetup);
        }
    }

    // Matches ScalarFFT 控制变量 
    void performForward(const Complex* input, Complex* output) const {
        // 1. DE-INTERLEAVE: Convert std::complex (interleaved) to SplitComplex (contiguous)
        // 创建最优memory layout
        DSPSplitComplex splitIn = { scratchReal.data(), scratchImag.data() };
        vDSP_ctoz((const DSPComplex*)input, 2, &splitIn, 1, fftSize);

        // 2. PERFORM FFT on contiguous split-complex arrays (Stride 1)
        DSPSplitComplex splitOut = { outReal.data(), outImag.data() };
        vDSP_fft_zop(fftSetup, 
                     &splitIn, 1, 
                     &splitOut, 1, 
                     order, 
                     kFFTDirection_Forward);
                     
        // 3. INTERLEAVE: Convert SplitComplex back to std::complex format
        vDSP_ztoc(&splitOut, 1, (DSPComplex*)output, 2, fftSize);

        // NOTE ON NORMALIZATION: 
        // Standard FFTW and KissFFT DO NOT scale the forward transform.
        // ========= omit vDSP_vsmul scaling 保证对比公平性

    }

    std::string getName() const { return "Apple vDSP (Accelerate)"; }

private:
    vDSP_Length order;
    size_t fftSize;
    FFTSetup fftSetup;
    
    // Mutable scratch buffers to avoid allocation overhead inside the timing loop
    mutable std::vector<float> scratchReal;
    mutable std::vector<float> scratchImag;
    mutable std::vector<float> outReal;
    mutable std::vector<float> outImag;
};