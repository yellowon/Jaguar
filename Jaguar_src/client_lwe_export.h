#pragma once

#include <cstdint>
#include <vector>

#include "seal/seal.h"
#include "client_lwe_shared.h"

template <int N>
void build_shared_lwe_groups_from_cout_quant(
    const std::vector<seal::Ciphertext> &cout,
    int stride,
    int f_w,
    int w,
    int num_output_channels,
    const uint64_t log2_q,
    std::vector<LweSharedGroup> &groups,
    int &next_w);
