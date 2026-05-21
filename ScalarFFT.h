#pragma once

#include <complex>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class ScalarFFT {
public:
    using Complex = std::complex<float>;

    ScalarFFT(int order) : fftSize(1 << order), inverse(false) {
        // Precompute Twiddle Factors
        double inverseFactor = -2.0 * M_PI / (double)fftSize;
        twiddleTable.resize(fftSize);

        if (fftSize <= 4) {
            for (int i = 0; i < fftSize; ++i) {
                double phase = i * inverseFactor;
                twiddleTable[i] = Complex((float)std::cos(phase), (float)std::sin(phase));
            }
        } else {
            for (int i = 0; i < fftSize / 4; ++i) {
                double phase = i * inverseFactor;
                twiddleTable[i] = Complex((float)std::cos(phase), (float)std::sin(phase));
            }
            for (int i = fftSize / 4; i < fftSize / 2; ++i) {
                Complex other = twiddleTable[i - fftSize / 4];
                twiddleTable[i] = Complex(other.imag(), -other.real());
            }
            twiddleTable[fftSize / 2] = Complex(-1.0f, 0.0f);
            for (int i = fftSize / 2; i < fftSize; ++i) {
                int index = fftSize / 2 - (i - fftSize / 2);
                twiddleTable[i] = std::conj(twiddleTable[index]);
            }
        }

        // Compute Factors (Radix 2 and 4)
        int root = (int)std::sqrt((double)fftSize);
        int divisor = 4, n = fftSize;
        for (int i = 0; i < 32; ++i) {
            while ((n % divisor) != 0) {
                if (divisor == 2) divisor = 3;
                else if (divisor == 4) divisor = 2;
                else divisor += 2;
                if (divisor > root) divisor = n;
            }
            n /= divisor;
            assert(divisor == 1 || divisor == 2 || divisor == 4);
            factors[i] = {divisor, n};
            if (divisor == 1) break;
        }
    }

    void performForward(const Complex* input, Complex* output) const {
        perform(input, output, 1, 1, factors);
    }

private:
    struct Factor { int radix, length; };
    
    int fftSize;
    bool inverse;
    std::vector<Complex> twiddleTable;
    Factor factors[32];

    void perform(const Complex* input, Complex* output, int stride, int strideIn, const Factor* facs) const {
        auto factor = *facs++;
        auto* originalOutput = output;
        auto* outputEnd = output + factor.radix * factor.length;

        if (stride == 1 && factor.radix <= 5) {
            for (int i = 0; i < factor.radix; ++i)
                perform(input + stride * strideIn * i, output + i * factor.length, stride * factor.radix, strideIn, facs);
            butterfly(factor, output, stride);
            return;
        }

        if (factor.length == 1) {
            do {
                *output++ = *input;
                input += stride * strideIn;
            } while (output < outputEnd);
        } else {
            do {
                perform(input, output, stride * factor.radix, strideIn, facs);
                input += stride * strideIn;
                output += factor.length;
            } while (output < outputEnd);
        }
        butterfly(factor, originalOutput, stride);
    }

    void butterfly(const Factor factor, Complex* data, int stride) const {
        if (factor.radix == 2) { butterfly2(data, stride, factor.length); return; }
        if (factor.radix == 4) { butterfly4(data, stride, factor.length); return; }
        
        // Fallback for other radixes (rarely used in this config)
        std::vector<Complex> scratch(factor.radix);
        for (int i = 0; i < factor.length; ++i) {
            for (int k = i, q1 = 0; q1 < factor.radix; ++q1) {
                scratch[q1] = data[k];
                k += factor.length;
            }
            for (int k = i, q1 = 0; q1 < factor.radix; ++q1) {
                int twiddleIndex = 0;
                data[k] = scratch[0];
                for (int q = 1; q < factor.radix; ++q) {
                    twiddleIndex += stride * k;
                    if (twiddleIndex >= fftSize) twiddleIndex -= fftSize;
                    data[k] += scratch[q] * twiddleTable[twiddleIndex];
                }
                k += factor.length;
            }
        }
    }

    void butterfly2(Complex* data, const int stride, const int length) const {
        auto* dataEnd = data + length;
        auto* tw = twiddleTable.data();
        for (int i = length; --i >= 0;) {
            auto s = *dataEnd;
            s *= (*tw);
            tw += stride;
            *dataEnd++ = *data - s;
            *data++ += s;
        }
    }

    void butterfly4(Complex* data, const int stride, const int length) const {
        auto lengthX2 = length * 2;
        auto lengthX3 = length * 3;
        auto strideX2 = stride * 2;
        auto strideX3 = stride * 3;

        auto* twiddle1 = twiddleTable.data();
        auto* twiddle2 = twiddle1;
        auto* twiddle3 = twiddle1;

        for (int i = length; --i >= 0;) {
            auto s0 = data[length]   * *twiddle1;
            auto s1 = data[lengthX2] * *twiddle2;
            auto s2 = data[lengthX3] * *twiddle3;
            auto s3 = s0 + s2;
            auto s4 = s0 - s2;
            auto s5 = *data - s1;

            *data += s1;
            data[lengthX2] = *data - s3;
            twiddle1 += stride;
            twiddle2 += strideX2;
            twiddle3 += strideX3;
            *data += s3;

            // Forward transform logic
            data[length]   = Complex(s5.real() + s4.imag(), s5.imag() - s4.real());
            data[lengthX3] = Complex(s5.real() - s4.imag(), s5.imag() + s4.real());
            ++data;
        }
    }
};