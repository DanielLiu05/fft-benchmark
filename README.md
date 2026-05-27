# fft-benchmark
messing around with fft optimizations.
# current optimizations
intrinsic optimized radix-2 and radix-4
out of place fft using __restrict and manual loop unrolling
bit reversal done with __builtin_bitreverse32
# testing
right now the test suite only does size 1024 ffts. this is small enough to just sit entirely in the l1 cache. i plan to improve the benchmarking later to test larger sizes.
<img width="2940" height="1912" alt="微信图片_2026-05-27_162721_039" src="https://github.com/user-attachments/assets/87989d75-5ae5-4c87-974d-72f8e51e5848" />

# build
use Clang with -O3 -march=native -ffast-math
