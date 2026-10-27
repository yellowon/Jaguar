#pragma once

#include <cstdint>
#include <vector>
#include "seal/seal.h"

struct ConvExtractPos {
    int src_ct_index;
    int coeff_index;
    int out_channel;
    int out_offset;
};

struct LweSharedGroup {
    int src_ct_index;
    uint64_t a_shared_bit_width;
    uint64_t b_values_bit_width;
    std::vector<uint64_t> a_shared;
    std::vector<uint64_t> b_values;
    std::vector<int> coeff_indices;
    std::vector<int> out_channels;
    std::vector<int> out_offsets;
};
