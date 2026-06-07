#include <iostream>
#include <chrono>
#include <thread>
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
#include "SixStepFFT_MT.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Ground Truth Calculation (Naive O(N^2) DFT in Double Precision)
// ============================================================================
void computeGroundTruth(const std::complex<float>* input, std::complex<float>* output, size_t N) {
    for (size_t k = 0; k < N; ++k) {
        std::complex<double> sum = {0.0, 0.0};
        for (size_t n = 0; n < N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            std::complex<double> w(std::cos(angle), std::sin(angle));
            sum += std::complex<double>(input[n].real(), input[n].imag()) * w;
        }
        // Cast back to single precision to establish the float32 baseline
        output[k] = std::complex<float>((float)sum.real(), (float)sum.imag());
    }
}

// ============================================================================
// Statistical Benchmark Harness + Verification
// ============================================================================
template <typename FFT_Engine>
void runBenchmark(const std::string& engineName, int order, int iterations, const std::complex<float>* trueOutput) {
    size_t fftSize = 1 << order;
    
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

    for (size_t i = 0; i < fftSize; ++i) {
        float val = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
        inputBuffer[i] = {val, 0.0f};
    }

    try {
        FFT_Engine engine(order);

        // Warm-up
        // now dynamic
        int warmup_iters = (order >= 22) ? 10 : (order >= 18) ? 50 : ((order >= 15) ? 200 : 1000);
        for (int i = 0; i < warmup_iters; ++i) {
            engine.performForward(inputBuffer, outputBuffer);
        }

        // Statistical Timing
        std::vector<double> samples;
        samples.reserve(iterations);
        float preventionSink = 0.0f;

        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            engine.performForward(inputBuffer, outputBuffer);
            auto end = std::chrono::high_resolution_clock::now();
            
            preventionSink += outputBuffer[0].real();
            std::chrono::duration<double, std::nano> elapsedNs = end - start;
            samples.push_back(elapsedNs.count());
        }

        if (preventionSink == -99999.0f) std::cout << preventionSink;

        // Verification Pass (Run one last time to ensure outputBuffer is fresh)
        engine.performForward(inputBuffer, outputBuffer);

        double maxAbsErr = 0.0;
        double meanAbsErr = 0.0;
        double maxRelErr = 0.0;

        for (size_t i = 0; i < fftSize; ++i) {
            float err = std::abs(outputBuffer[i] - trueOutput[i]);
            if (err > maxAbsErr) maxAbsErr = err;
            meanAbsErr += err;
            
            float mag = std::abs(trueOutput[i]);
            // Only calculate relative error for bins with actual signal energy 
            // to avoid division-by-zero or NaN explosions in the noise floor.
            if (mag > 1e-5f) { 
                double relErr = (err / mag) * 100.0; // Percentage
                if (relErr > maxRelErr) maxRelErr = relErr;
            }
        }
        meanAbsErr /= fftSize;

        // Stats calculation
        std::sort(samples.begin(), samples.end());
        double minTime = samples.front();
        double maxTime = samples.back();
        double medianTime = samples[samples.size() / 2];
        
        size_t p95_idx = std::min((size_t)(samples.size() * 0.95), samples.size() - 1);
        double p95Time = samples[p95_idx];

        size_t p99_idx = std::min((size_t)(samples.size() * 0.99), samples.size() - 1);
        double p99Time = samples[p99_idx];
        
        double trimmedSum = 0;
        size_t trimmedCount = (size_t)(samples.size() * 0.99);
        for(size_t i = 0; i < trimmedCount; ++i) trimmedSum += samples[i];
        double trimmedAvg = trimmedSum / trimmedCount;

        double totalSum = 0;
        for(double s : samples) totalSum += s;
        double stdAvg = totalSum / samples.size();

        // Print Results
        std::cout << std::left << std::setw(24) << engineName << " |\n"
                  << "  Min:       " << std::fixed << std::setprecision(2) << std::setw(10) << minTime    << " ns  (" << std::setprecision(3) << minTime/1000.0    << " us)\n"
                  << "  Median:    " << std::fixed << std::setprecision(2) << std::setw(10) << medianTime << " ns  (" << std::setprecision(3) << medianTime/1000.0 << " us)\n"
                  << "  TrimAvg:   " << std::fixed << std::setprecision(2) << std::setw(10) << trimmedAvg << " ns  (" << std::setprecision(3) << trimmedAvg/1000.0 << " us)\n"
                  << "  StdAvg:    " << std::fixed << std::setprecision(2) << std::setw(10) << stdAvg     << " ns  (" << std::setprecision(3) << stdAvg/1000.0     << " us)\n"
                  << "  P95:       " << std::fixed << std::setprecision(2) << std::setw(10) << p95Time    << " ns  (" << std::setprecision(3) << p95Time/1000.0    << " us)\n"
                  << "  P99:       " << std::fixed << std::setprecision(2) << std::setw(10) << p99Time    << " ns  (" << std::setprecision(3) << p99Time/1000.0    << " us)\n"
                  << "  Max:       " << std::fixed << std::setprecision(2) << std::setw(10) << maxTime    << " ns  (" << std::setprecision(3) << maxTime/1000.0    << " us)\n"
                  << "  ----------------------------------------\n"
                  << "  Verification (vs Naive DFT):\n"
                  << "    Max Abs Error:  " << std::scientific << std::setprecision(4) << maxAbsErr  << "\n"
                  << "    Mean Abs Error: " << std::scientific << std::setprecision(4) << meanAbsErr << "\n"
                  << "    Max Rel Error:  " << std::fixed      << std::setprecision(4) << maxRelErr  << " %\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error initializing " << engineName << ": " << e.what() << "\n";
    }

    free(rawIn);
    free(rawOut);
}

// ============================================================================
// Main Entry Point
// ============================================================================
int main() {
    // The Scientific Test Matrix
    std::vector<int> test_orders = {10, 16};
    
    for (int order : test_orders) {
        // Dynamic Iteration Scaling
        int iterations = 10000;
        if (order == 16) iterations = 1000;
        if (order == 18) iterations = 500;
        if (order == 20) iterations = 200;
        if (order == 22) iterations = 60;
        if (order == 24) iterations = 15;
        if (order == 26) iterations = 4;

        size_t fftSize = 1 << order;
        
        std::cout << "================================================================\n";
        std::cout << " FFT Benchmark & Verification: Size = " << fftSize << " samples; order=" << order << "\n";
        std::cout << " Iterations per engine: " << iterations << "\n";
        std::cout << "================================================================\n";

        // Allocate and fill input
        std::vector<std::complex<float>> inputBuffer(fftSize);
        std::vector<std::complex<float>> trueOutput(fftSize);
        for (size_t i = 0; i < fftSize; ++i) {
            float val = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
            inputBuffer[i] = {val, 0.0f};
        }

        // TRAP PREVENTION
        if (order <= 12) {
            std::cout << "Computing Ground Truth (Naive O(N^2) DFT)...\n";
            auto startDFT = std::chrono::high_resolution_clock::now();
            computeGroundTruth(inputBuffer.data(), trueOutput.data(), fftSize);
            auto endDFT = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dftMs = endDFT - startDFT;
            std::cout << "Ground Truth computed in " << std::fixed << std::setprecision(2) << dftMs.count() << " ms.\n";
        } else {
            std::cout << "Skipping Naive DFT ,ignore verification errors.\n";
            std::copy(inputBuffer.begin(), inputBuffer.end(), trueOutput.begin()); 
        }

        // Run the engines
        runBenchmark<ScalarFFT>("Scalar Fallback", order, iterations, trueOutput.data());
        runBenchmark<NeonFFT>("my Custom ARM NEON", order, iterations, trueOutput.data());
        runBenchmark<SixStepFFT_MT>("my sixstep algorithm optimization", order, iterations, trueOutput.data()); 
        
        runBenchmark<AppleFFT>("Apple vDSP", order, iterations, trueOutput.data());
        
        std::cout << "\n";
        
        // Thermal cooldown between large sizes
        if (order >= 16 && order <=20) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if(order >=22){
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }

    std::cout << "================================================================\n";
    std::cout << "Note: Ensure compiled with -O3 -march=native -ffast-math\n";
    std::cout << "================================================================\n";
    return 0;
}