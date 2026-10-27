#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <omp.h>

#include "seal/seal.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "jaguar_profiler.h"
#include "client_lwe_decrypt.h"

using namespace std;
using namespace seal;

void precompute_secret_key_ntt(
    const Plaintext &sk_client_coeff,
    Plaintext &sk_client_coeff_ntt,
    const SEALContext &context_ntt)
{
    auto &context_ntt_data = *context_ntt.first_context_data();
    auto ntt_tables = context_ntt_data.small_ntt_tables();
    auto &parms_ntt = context_ntt_data.parms();
    auto &coeff_modulus_ntt = parms_ntt.coeff_modulus();
    uint64_t q_ntt = coeff_modulus_ntt[0].value();

    int n = static_cast<int>(parms_ntt.poly_modulus_degree());

    sk_client_coeff_ntt = sk_client_coeff;

    for (int i = 0; i < n; i++)
    {
        if ((sk_client_coeff_ntt.data()[i] != 0) && (sk_client_coeff_ntt.data()[i] != 1))
            sk_client_coeff_ntt.data()[i] = q_ntt - 1;
    }

    ntt_negacyclic_harvey_lazy(sk_client_coeff_ntt.data(), ntt_tables[0]);
}

static inline void rlwe_negacyclic_mul_pow2_select_ntt(
    const std::vector<uint64_t> &a_shared,
    const seal::Plaintext &secret_key_coeff_ntt,
    const seal::util::NTTTables *ntt_tables,
    const uint64_t q_ntt,
    const uint64_t log2_q,
    const uint64_t n,
    const std::vector<int> &coeff_indices,
    std::vector<uint64_t> &selected_values,
    seal::Plaintext &ntt_tmp)
{
    using namespace seal::util;

    if (a_shared.size() != static_cast<size_t>(n))
    {
        throw std::runtime_error("rlwe_negacyclic_mul_pow2_select_ntt: invalid a_shared size");
    }

    const uint64_t qM1 = (1ULL << log2_q) - 1;
    const uint64_t sign_bit = (1ULL << (log2_q - 1));

    ntt_tmp.resize(n);
    for (size_t i = 0; i < static_cast<size_t>(n); ++i)
    {
        const uint64_t v = a_shared[i] & qM1;
        if (v & sign_bit)
        {
            const uint64_t neg_mag = ((~v) + 1) & qM1;
            ntt_tmp.data()[i] = neg_mag ? (q_ntt - neg_mag) : 0;
        }
        else
        {
            ntt_tmp.data()[i] = v;
        }
    }

    ntt_negacyclic_harvey_lazy(ntt_tmp.data(), ntt_tables[0]);
    multiply_eval_poly_poly_mod(ntt_tmp.data(), secret_key_coeff_ntt.data(), q_ntt, n);
    inverse_ntt_negacyclic_harvey_lazy(ntt_tmp.data(), ntt_tables[0]);

    selected_values.resize(coeff_indices.size());
    const uint64_t qPrimeHalf = (q_ntt >> 1);
    for (size_t i = 0; i < coeff_indices.size(); ++i)
    {
        const int coeff_index = coeff_indices[i];
        if (coeff_index < 0 || coeff_index >= static_cast<int>(n))
        {
            throw std::runtime_error("rlwe_negacyclic_mul_pow2_select_ntt: coeff_index out of range");
        }
        const uint64_t raw = ntt_tmp.data()[static_cast<size_t>(coeff_index)];
        if (raw > qPrimeHalf)
        {
            selected_values[i] = (raw - q_ntt + (1ULL << log2_q)) & qM1;
        }
        else
        {
            selected_values[i] = raw & qM1;
        }
    }
}


template <int N>
void client_lwe_stride_dec_quant(
    const std::vector<LweSharedGroup> &groups,
    std::vector<std::vector<uint64_t>> &normal_data,
    const seal::Plaintext &secret_key_coeff,
    const int num_out_channels,
    const int next_w,
    const uint64_t log2_q,
    const uint64_t log2_p,
    const int discard_lsb)
{
    using namespace std;
    using namespace seal;
    using namespace seal::util;

    auto time_start = chrono::high_resolution_clock::now();

    normal_data.assign(
        num_out_channels,
        std::vector<uint64_t>(next_w * next_w, 0));

    int max_ch = -1;
    int max_off = -1;
    size_t total_vals = 0;
    for (const auto &group : groups) {
        total_vals += group.b_values.size();
        for (int ch : group.out_channels) if (ch > max_ch) max_ch = ch;
        for (int off : group.out_offsets) if (off > max_off) max_off = off;
    }
    std::cerr << "[Jaguar][client_lwe_stride_dec_quant] groups=" << groups.size()
              << " total_vals=" << total_vals
              << " num_out_channels=" << num_out_channels
              << " next_w=" << next_w
              << " max_ch=" << max_ch
              << " max_off=" << max_off << std::endl;

    // Algorithm 2: two-step divide-and-round
    const uint64_t qM1 = (1ULL << log2_q) - 1;
    const uint64_t p_effM1 = (1ULL << (log2_p - discard_lsb)) - 1;
    const uint64_t q_to_p_shift = log2_q - log2_p;
    const uint64_t delta_div_2 = (1ULL << (log2_q - log2_p - 1 - discard_lsb));

    EncryptionParameters parms_ntt(scheme_type::bfv);
    parms_ntt.set_poly_modulus_degree(N);
    parms_ntt.set_coeff_modulus(CoeffModulus::Create(N, {60}));
    parms_ntt.set_plain_modulus(PlainModulus::Batching(N, 20));
    SEALContext context_ntt(parms_ntt);
    auto &context_ntt_data = *context_ntt.first_context_data();
    auto ntt_tables = context_ntt_data.small_ntt_tables();
    const uint64_t q_ntt = context_ntt_data.parms().coeff_modulus()[0].value();
    Plaintext secret_key_coeff_ntt(N);
    precompute_secret_key_ntt(secret_key_coeff, secret_key_coeff_ntt, context_ntt);
#pragma omp parallel for schedule(dynamic)
    for (int g = 0; g < static_cast<int>(groups.size()); ++g)
    {
        const auto &group = groups[static_cast<size_t>(g)];

        if (group.b_values.size() != group.coeff_indices.size() ||
            group.b_values.size() != group.out_channels.size() ||
            group.b_values.size() != group.out_offsets.size())
        {
            throw std::runtime_error("client_lwe_stride_dec_quant: malformed LweSharedGroup");
        }

        if (group.a_shared.size() != static_cast<size_t>(N))
        {
            throw std::runtime_error("client_lwe_stride_dec_quant: invalid shared a size");
        }

        std::vector<uint64_t> selected_values;
        Plaintext ntt_tmp_local(N);
        rlwe_negacyclic_mul_pow2_select_ntt(
            group.a_shared, secret_key_coeff_ntt, ntt_tables, q_ntt, log2_q, N,
            group.coeff_indices, selected_values, ntt_tmp_local);

        for (size_t i = 0; i < group.b_values.size(); ++i)
        {
            const uint64_t shared_dot = selected_values[i] & qM1;
            const uint64_t acc = (shared_dot + (group.b_values[i] & qM1)) & qM1;
            const uint64_t operand_0 = ((acc + delta_div_2) & qM1) >> q_to_p_shift;
            const uint64_t value = operand_0 & p_effM1;

            const int ch = group.out_channels[i];
            const int off = group.out_offsets[i];

            if (ch < 0 || ch >= num_out_channels)
            {
                throw std::runtime_error("client_lwe_stride_dec_quant: out_channel out of range");
            }
            if (off < 0 || off >= next_w * next_w)
            {
                throw std::runtime_error("client_lwe_stride_dec_quant: out_offset out of range");
            }

            normal_data[ch][off] = value;
        }
    }

    auto time_end = chrono::high_resolution_clock::now();
    auto time_us =
        chrono::duration_cast<chrono::microseconds>(time_end - time_start).count();

    std::cout << "\nLWE decrypt, groups=" << groups.size()
              << ", out_ch=" << num_out_channels
              << ", next_w=" << next_w << "\n";
    std::cout << "Runtime for client LWE decryption (us): "
              << time_us << " [usec]\n";

    jaguar::LayerSummary layer;
    layer.kind = "decrypt";
    layer.name = "client_lwe_dec_quant";
    layer.total_latency_us = static_cast<uint64_t>(time_us);
    layer.compute_latency_us = static_cast<uint64_t>(time_us);
    layer.comm_bytes = 0;
    layer.rounds = 0;
    layer.details =
        "groups=" + std::to_string(groups.size()) +
        ",out_ch=" + std::to_string(num_out_channels) +
        ",next_w=" + std::to_string(next_w);
    jaguar::profiler().record_layer_summary(layer);
}




template void client_lwe_stride_dec_quant<2048>(
    const std::vector<LweSharedGroup> &groups,
    std::vector<std::vector<uint64_t>> &normal_data,
    const seal::Plaintext &secret_key_coeff,
    const int num_out_channels,
    const int next_w,
    const uint64_t log2_q,
    const uint64_t log2_p,
    const int discard_lsb);
