#pragma once
#include <cstddef>

// Example kernel: element-wise multiply a float array by a scalar.
// h_data   - input host pointer (read-only)
// h_result - output host pointer (pre-allocated, size n)
// scalar   - multiplier
// n        - number of elements
void cuda_scale(const float* h_data, float* h_result, float scalar, size_t n);
