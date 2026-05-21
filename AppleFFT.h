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
        // Create the FFT setup. FFT_RADIX2 is the standard Cooley-Tukey algorithm.
        fftSetup = vDSP_create_fftsetup(order, FFT_RADIX2);
        if (!fftSetup) {
            throw std::runtime_error("Failed to create vDSP FFT Setup");
        }

        // Pre-allocate contiguous split-complex scratch buffers.
        // Apple Silicon's vDSP requires contiguous memory to use its fastest NEON/AMX paths.
        // We make them mutable so performForward() can remain const for the benchmark harness.
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

    // Matches the ScalarFFT interface for a 1:1 benchmark
    void performForward(const Complex* input, Complex* output) const {
        // 1. DE-INTERLEAVE: Convert std::complex (interleaved) to SplitComplex (contiguous)
        // This gives vDSP the memory layout it needs to unleash its fastest SIMD instructions.
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
        // We omit vDSP_vsmul scaling here to ensure a mathematically fair 
        // 1:1 comparison against your NEON code and the Scalar Fallback.
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