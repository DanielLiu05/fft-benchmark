# fft-benchmark
messing around with fft optimizations.
# current optimizations
intrinsic optimized radix-2 and radix-4
out of place fft using __restrict and manual loop unrolling
bit reversal done with __builtin_bitreverse32
# testing
right now the test suite only does size 1024 ffts. this is small enough to just sit entirely in the l1 cache. i plan to improve the benchmarking later to test larger sizes.
# build
