#include <iostream>
#include <chrono>
#include <complex>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <stdexcept>

//engines
#include "ScalarFFT.h"
#include "NeonFFT.h"
#include "AppleFFT.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Benchmark Harness
// ============================================================================

// We use a template to avoid virtual function overhead in the timing loop, 
// ensuring the compiler optimizes the call as much as possible.
template <typename FFT_Engine>
void runBenchmark(const std::string& engineName, int order, int iterations) {
    size_t fftSize = 1 << order;
    
    // 1. Allocate ALIGNED memory. 
    // NEON and vDSP require/perform best with 16-byte or 32-byte aligned memory.
    // aligned_alloc requires the size to be a multiple of the alignment.
    size_t alignment = 32;
    size_t bytes = fftSize * sizeof(std::complex<float>);
    if (bytes % alignment != 0) {
        bytes += alignment - (bytes % alignment);
    }

    void* rawIn = nullptr;
    void* rawOut = nullptr;
    if (posix_memalign(&rawIn, alignment, bytes) != 0 || 
        posix_memalign(&rawOut, alignment, bytes) != 0) {
        std::cerr << "Memory allocation failed!\n";
        return;
    }
    auto* inputBuffer  = static_cast<std::complex<float>*>(rawIn);
    auto* outputBuffer = static_cast<std::complex<float>*>(rawOut);
    if (!inputBuffer || !outputBuffer) {
        std::cerr << "Memory allocation failed!\n";
        return;
    }

    // 2. Fill with dummy data (a simple sine wave) so it's not all zeros.
    // Zeroed memory can sometimes trigger edge-case optimizations in FFTs.
    for (size_t i = 0; i < fftSize; ++i) {
        float val = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
        inputBuffer[i] = {val, 0.0f};
    }

    try {
        // 3. Initialize the Engine
        FFT_Engine engine(order);

        // 4. WARM-UP PHASE
        // Brings the CPU out of idle/low-power states and warms up the L1/L2 cache.
        for (int i = 0; i < 1000; ++i) {
            engine.performForward(inputBuffer, outputBuffer);
        }

        // 5. TIMING PHASE
        // 
        // read output prevent compiler deleting the loop
        float preventionSink = 0.0f; 
        
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            engine.performForward(inputBuffer, outputBuffer);
            // Accumulate a tiny bit of the output to force the compiler to execute the FFT
            preventionSink += outputBuffer[0].real(); 
        }
        // Prevent DCE without penalizing the loop
        if (preventionSink == -99999.0f) std::cout << preventionSink;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsedUs = end - start;

        double avgTimeUs = elapsedUs.count() / iterations;
        double avgTimeNs = avgTimeUs * 1000.0;

        std::cout << std::left << std::setw(20) << engineName 
                  << " | Avg Time: " << std::fixed << std::setprecision(2) 
                  << std::setw(10) << avgTimeNs << " ns  (" 
                  << avgTimeUs << " us)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error initializing " << engineName << ": " << e.what() << "\n";
    }

    // 6. Cleanup
    free(inputBuffer);
    free(outputBuffer);
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main() {
    // FFT Order: 10 means 2^10 = 1024 samples 
    // (11 for 2048)
    int order = 10; 
    int iterations = 10000;

    std::cout << "========================================================\n";
    std::cout << " FFT Benchmark: Size = " << (1 << order) << " samples\n";
    std::cout << " Iterations per engine: " << iterations << "\n";
    std::cout << "========================================================\n";

    // Run the benchmarks back-to-back
    // Note: The order here shouldn't matter due to the warm-up phase, 
    // but we run Scalar first to establish the baseline.
    
    runBenchmark<ScalarFFT>("Scalar Fallback", order, iterations);
    runBenchmark<NeonFFT>("Custom ARM NEON", order, iterations);
    runBenchmark<AppleFFT>("Apple vDSP", order, iterations);

    std::cout << "========================================================\n";
    std::cout << "Note: Ensure this was compiled with -O3 and -march=native!\n";
    std::cout << "========================================================\n";

    return 0;
}


//clang++ -O3 -march=native -ffast-math -std=c++17 main.cpp -o fft_benchmark -framework Accelerate 
//./fft_benchmark