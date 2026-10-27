#pragma once

#include <cstdint>
#include <vector>

constexpr bool kUseOpenCheetahAvgPoolWrapper = true;

void max_pooling_share(std::vector<std::vector<uint64_t>> &data,
                       int width,
                       int kernel_size,
                       int stride,
                       int padding,
                       int ring_bits = 0);

// Jaguar_52: SCI pooling session ring for the whole run (one session; SCI
// cannot switch rings mid-run). 30 by default, 21 when JAGUAR_POOL_FIRST=1.
int j52_pool_session_bits();

void avg_pooling_share_oc(std::vector<std::vector<uint64_t>> &data,
                          int width,
                          int kernel_size,
                          int stride,
                          int padding);

void relu6_share_quant(const std::vector<std::vector<uint64_t>> &local_share,
                       uint64_t bitwidth,
                       std::vector<std::vector<uint64_t>> &local_relu6_share);
