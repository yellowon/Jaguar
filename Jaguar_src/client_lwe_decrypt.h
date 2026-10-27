#pragma once

#include <cstdint>
#include <vector>

#include "seal/seal.h"
#include "client_lwe_shared.h"

void precompute_secret_key_ntt(
    const seal::Plaintext &sk_client_coeff,
    seal::Plaintext &sk_client_coeff_ntt,
    const seal::SEALContext &context_ntt);

template <int N>
void client_lwe_stride_dec_quant(
    const std::vector<LweSharedGroup> &groups,
    std::vector<std::vector<uint64_t>> &normal_data,
    const seal::Plaintext &secret_key_coeff,
    const int num_out_channels,
    const int next_w,
    const uint64_t log2_q,
    const uint64_t log2_p,
    const int discard_lsb);
