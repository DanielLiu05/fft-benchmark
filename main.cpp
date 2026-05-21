#include <iostream>
#include <chrono>
#include <complex>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <algorithm>

// Engines
#include "ScalarFFT.h"
#include "NeonFFT.h"
#include "AppleFFT.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Advanced Statistical Benchmark Harness
// ============================================================================
template <typename FFT_Engine>
void runBenchmark(const std::string& engineName, int order, int iterations) {
    size_t fftSize = 1 << order;
    
    // 1. Allocate ALIGNED memory (32-byte alignment for optimal SIMD/AMX)
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

    // 2. Fill with dummy data (a simple sine wave)
    for (size_t i = 0; i < fftSize; ++i) {
        float val = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
        inputBuffer[i] = {val, 0.0f};
    }

    try {
        // 3. Initialize the Engine
        FFT_Engine engine(order);

        // 4. WARM-UP PHASE
        // Brings the CPU out of idle/low-power states (DVFS) and warms up the L1/L2 cache.
        for (int i = 0; i < 1000; ++i) {
            engine.performForward(inputBuffer, outputBuffer);
        }

        // 5. STATISTICAL TIMING PHASE
        std::vector<double> samples;
        samples.reserve(iterations);
        float preventionSink = 0.0f;

        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            engine.performForward(inputBuffer, outputBuffer);
            
            auto end = std::chrono::high_resolution_clock::now();
            
            // Accumulate a tiny bit of the output to force the compiler to execute the FFT
            preventionSink += outputBuffer[0].real();
            
            std::chrono::duration<double, std::nano> elapsedNs = end - start;
            samples.push_back(elapsedNs.count());
        }

        // Prevent Dead Code Elimination (DCE) without penalizing the loop
        if (preventionSink == -99999.0f) std::cout << preventionSink;

        // 6. STATISTICAL ANALYSIS
        std::sort(samples.begin(), samples.end());

        double minTime = samples.front();
        double maxTime = samples.back();
        double medianTime = samples[samples.size() / 2];
        
        size_t p95_idx = (size_t)(samples.size() * 0.95);
        if (p95_idx >= samples.size()) p95_idx = samples.size() - 1;
        double p95Time = samples[p95_idx];

        size_t p99_idx = (size_t)(samples.size() * 0.99);
        if (p99_idx >= samples.size()) p99_idx = samples.size() - 1;
        double p99Time = samples[p99_idx];
        
        // Trimmed Average (Discard top 1% of outliers caused by OS context switches)
        double trimmedSum = 0;
        size_t trimmedCount = (size_t)(samples.size() * 0.99);
        for(size_t i = 0; i < trimmedCount; ++i) {
            trimmedSum += samples[i];
        }
        double trimmedAvg = trimmedSum / trimmedCount;

        // Standard Average (The traditional, but flawed, metric)
        double totalSum = 0;
        for(double s : samples) totalSum += s;
        double stdAvg = totalSum / samples.size();

        // 7. PRINT RESULTS
        std::cout << std::left << std::setw(24) << engineName << " |\n"
                  << "  Min:       " << std::fixed << std::setprecision(2) << std::setw(10) << minTime    << " ns  (" << std::setprecision(3) << minTime/1000.0    << " us)\n"
                  << "  Median:    " << std::fixed << std::setprecision(2) << std::setw(10) << medianTime << " ns  (" << std::setprecision(3) << medianTime/1000.0 << " us)\n"
                  << "  TrimAvg:   " << std::fixed << std::setprecision(2) << std::setw(10) << trimmedAvg << " ns  (" << std::setprecision(3) << trimmedAvg/1000.0 << " us)\n"
                  << "  StdAvg:    " << std::fixed << std::setprecision(2) << std::setw(10) << stdAvg     << " ns  (" << std::setprecision(3) << stdAvg/1000.0     << " us)\n"
                  << "  P95:       " << std::fixed << std::setprecision(2) << std::setw(10) << p95Time    << " ns  (" << std::setprecision(3) << p95Time/1000.0    << " us)\n"
                  << "  P99:       " << std::fixed << std::setprecision(2) << std::setw(10) << p99Time    << " ns  (" << std::setprecision(3) << p99Time/1000.0    << " us)\n"
                  << "  Max:       " << std::fixed << std::setprecision(2) << std::setw(10) << maxTime    << " ns  (" << std::setprecision(3) << maxTime/1000.0    << " us)\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error initializing " << engineName << ": " << e.what() << "\n";
    }

    // 8. Cleanup
    free(rawIn);
    free(rawOut);
}

// ============================================================================
// Main Entry Point
// ============================================================================
int main() {
    // FFT Order: 10 means 2^10 = 1024 samples
    int order = 10;
    
    // Increased iterations for better statistical significance
    int iterations = 10000; 

    std::cout << "================================================================\n";
    std::cout << " Advanced FFT Benchmark: Size = " << (1 << order) << " samples\n";
    std::cout << " Iterations per engine: " << iterations << "\n";
    std::cout << "================================================================\n\n";

    // Run the benchmarks back-to-back
    runBenchmark<ScalarFFT>("Scalar Fallback", order, iterations);
    runBenchmark<NeonFFT>("Custom ARM NEON", order, iterations);
    runBenchmark<AppleFFT>("Apple vDSP", order, iterations);

    std::cout << "================================================================\n";
    std::cout << "Note: Ensure compiled with -O3 -march=native -ffast-math\n";
    std::cout << "================================================================\n";
    
    return 0;
}