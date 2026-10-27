#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <omp.h>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "seal/seal.h"
#include "seal/util/uintarithsmallmod.h"

#include "client_lwe_shared.h"
#include "client_lwe_decrypt.h"
#include "client_lwe_export.h"
#include "client_layer_helpers.h"
#include "jaguar_transport.h"
#include "jaguar_model_dispatch.h"

#include "cloud.cpp"
#include "../models/filter.cpp"

#include "seal/util/polyarithsmallmod.h"
#include "seal/keygenerator.h"
#include "seal/batchencoder.h"
#include "gemini/cheetah/tensor.h"
#include "cheetah/cheetah-api.h"
#include "globals.h"
#include "library_fixed_uniform.h"
#include "seal/util/scalingvariant.h"
#include "seal/util/rlwe.h"

#include <cmath>
#include "jaguar_relu.h"
#include "jaguar_profiler.h"
#include "../networks/networks.h"
#include "../networks/input_loader.h"

#define con(n) n

using namespace std;
using namespace seal;

namespace seal { namespace util {
void decrypt_lwe_quant_reduce(
    const std::vector<uint64_t> &a,
    uint64_t b,
    seal::util::ConstCoeffIter secret_key,
    const uint64_t log2_q,
    const uint64_t log2_p,
    const int N,
    const int discard_lsb,
    uint64_t &out);
}}

void client_ct_stride_dec_quant(vector<Ciphertext> &cout, vector<vector<uint64_t> > &normal_data, const SEALContext & context, const Plaintext &secret_key_coeff, const int stride, const int f_w, const int w, const uint64_t log2_q, const uint64_t log2_p, const SEALContext &context_ntt, const uint64_t q_ntt, const int discard_lsb);
void client_enc_zero_padding_quant(const SEALContext & context, const int f_w, const Plaintext &sk_client_coeff_ntt, const vector<vector<uint64_t> > &din, vector<Ciphertext> &cout, const uint64_t log2_q, const uint64_t log2_p, const bool past_encoding, const SEALContext & context_ntt);

namespace jaguar_quant {

static inline uint64_t jaguar_refresh_mask_for_bits(const int bits) {
    if (bits >= 64) return ~0ULL;
    return (1ULL << bits) - 1ULL;
}

static std::vector<std::vector<uint64_t>> sample_refresh_mask_like(
    const std::vector<std::vector<uint64_t>> &like,
    const int effective_bits) {
    std::vector<std::vector<uint64_t>> mask(like.size());
    const uint64_t ring_mask = jaguar_refresh_mask_for_bits(effective_bits);
    for (size_t i = 0; i < like.size(); ++i) {
        mask[i].resize(like[i].size());
        for (size_t j = 0; j < like[i].size(); ++j) {
            mask[i][j] = jaguar_secure_random_u64() & ring_mask;
        }
    }
    return mask;
}

// Jaguar_52: after an exact ReLU the shares live in the 30-bit output ring;
// refreshing them modulo 2^21 would reintroduce the carry error.
static int relu_share_bits(const uint64_t log2_p, const int discard_lsb) {
    if (jaguar::exact_relu_enabled()) return static_cast<int>(jaguar::exact_relu_output_bits());
    return std::max<int>(1, static_cast<int>(log2_p) - discard_lsb);
}

// Jaguar_52 opt-in (JAGUAR_POOL_FIRST=1): stem MaxPool before ReLU. ReLU is
// monotone and the padding is zero, so ReLU(MaxPool(x)) == MaxPool(ReLU(x)).
// MaxPool runs on the pre-ReLU ring-21 shares (|x| < 2^19 certified, so the
// signed differences fit 21 bits); exact ReLU then runs on 200,704 values
// instead of 802,816. Default off: the Jaguar_52 baseline is unchanged.
static bool j52_pool_first() {
    static const bool on = [] { const char *v = std::getenv("JAGUAR_POOL_FIRST"); return v && std::string(v) == "1"; }();
    return on;
}

// Jaguar_52 validation hook (off unless JAGUAR_DEBUG_SHARES=<dir>): every
// stage writes this party's output share; offline the two shares are added
// modulo 2^bits and compared with a float model. Never set in timing runs.
static void j52_debug_dump(const char *party, bool relu, int bits,
                           const std::vector<std::vector<uint64_t>> &m) {
    const char *dir = std::getenv("JAGUAR_DEBUG_SHARES");
    if (!dir) return;
    static int index = 0;
    std::ofstream out(std::string(dir) + "/" + party + "_" + std::to_string(index++) + ".txt");
    out << "# rows=" << m.size() << " cols=" << (m.empty() ? 0 : m[0].size())
        << " relu=" << (relu ? 1 : 0) << " bits=" << bits << "\n";
    for (const auto &row : m) for (uint64_t v : row) out << v << "\n";
}

static void add_refresh_mask_mod_inplace(
    std::vector<std::vector<uint64_t>> &dst,
    const std::vector<std::vector<uint64_t>> &mask,
    const int effective_bits) {
    const uint64_t ring_mask = jaguar_refresh_mask_for_bits(effective_bits);
    for (size_t i = 0; i < dst.size(); ++i) {
        for (size_t j = 0; j < dst[i].size(); ++j) {
            dst[i][j] = (dst[i][j] + mask[i][j]) & ring_mask;
        }
    }
}

static void sub_refresh_mask_mod_inplace(
    std::vector<std::vector<uint64_t>> &dst,
    const std::vector<std::vector<uint64_t>> &mask,
    const int effective_bits) {
    const uint64_t ring_mask = jaguar_refresh_mask_for_bits(effective_bits);
    for (size_t i = 0; i < dst.size(); ++i) {
        for (size_t j = 0; j < dst[i].size(); ++j) {
            dst[i][j] = (dst[i][j] - mask[i][j]) & ring_mask;
        }
    }
}

template <int N>
void client_stage_quant(const SEALContext &context,
                        const uint64_t log2_q,
                        const uint64_t log2_p,
                        const Plaintext &secret_key_coeff,
                        std::vector<Ciphertext> &ct_out1,
                        const int conv_stride,
                        const int f_w,
                        int &w,
                        const bool relu,
                        const bool use_relu6,
                        const bool pooling,
                        const int pooling_kernel_size,
                        const int pooling_stride,
                        const int pooling_padding,
                        std::vector<std::vector<uint64_t>> &ss_r_data,
                        const int next_f_w,
                        std::vector<Ciphertext> &ct_relu,
                        const SEALContext &context_ntt,
                        const uint64_t q_ntt,
                        const int discard_lsb,
                        const bool use_lwe_output,
                        const std::vector<LweSharedGroup> *lwe_groups) {
    std::cerr << "[Jaguar][client_stage_quant][entry] w=" << w
              << " conv_stride=" << conv_stride
              << " f_w=" << f_w
              << " relu=" << (relu ? 1 : 0)
              << " pooling=" << (pooling ? 1 : 0)
              << " ct_out1=" << ct_out1.size()
              << std::endl;
    std::vector<std::vector<uint64_t>> normal_data;
    std::vector<std::vector<uint64_t>> relu_data;
    const bool stem_maxpool = relu && pooling && (f_w == 7) && (pooling_kernel_size == 3) && (pooling_stride == 2) && (pooling_padding == 1);
    std::cerr << "[Jaguar][client_stage_quant][before_decrypt]" << std::endl;
    if (use_lwe_output) {
        if (lwe_groups == nullptr) throw std::runtime_error("client_stage_quant: lwe_groups is null");
        const int p_l = (f_w - 1) / 2;
        const int p_r = (f_w - 1) / 2 + ((f_w - 1) % 2 ? 1 : 0);
        const int wPp = w + p_l + p_r;
        const int new_w = 1 + (wPp - f_w) / conv_stride;
        int num_out_channels = 0;
        for (const auto &group : *lwe_groups) for (int ch : group.out_channels) if (ch + 1 > num_out_channels) num_out_channels = ch + 1;
        ::client_lwe_stride_dec_quant<N>(*lwe_groups, normal_data, secret_key_coeff, num_out_channels, new_w, log2_q, log2_p, discard_lsb);
    } else {
        client_ct_stride_dec_quant(ct_out1, normal_data, context, secret_key_coeff, conv_stride, f_w, w, log2_q, log2_p, context_ntt, q_ntt, discard_lsb);
    }
    std::cerr << "[Jaguar][client_stage_quant][after_decrypt] normal_rows=" << normal_data.size()
              << " normal_cols=" << (normal_data.empty() ? 0 : normal_data[0].size())
              << " w=" << w << std::endl;
    j52_debug_dump("prec", relu, static_cast<int>(log2_p) - discard_lsb, normal_data);
    const bool pool_first = stem_maxpool && j52_pool_first() && !use_relu6;
    if (pool_first) {
        const int ring = static_cast<int>(log2_p) - discard_lsb;
        max_pooling_share(normal_data, w, pooling_kernel_size, pooling_stride, pooling_padding, ring);
        jaguar::relu_share_ring_external(normal_data, log2_p - discard_lsb, relu_data);
    } else if (relu) {
        std::cerr << "[Jaguar][client_stage_quant][before_relu_local] elements="
                  << (normal_data.empty() ? 0 : normal_data.size() * normal_data[0].size())
                  << std::endl;
        if (use_relu6) relu6_share_quant(normal_data, log2_p - discard_lsb, relu_data);
        else jaguar::relu_share_ring_external(normal_data, log2_p - discard_lsb, relu_data);
        std::cerr << "[Jaguar][client_stage_quant][after_relu_local] relu_rows=" << relu_data.size()
                  << " relu_cols=" << (relu_data.empty() ? 0 : relu_data[0].size()) << std::endl;
    } else {
        relu_data = normal_data;
    }
    if (pooling && !pool_first) {
        std::cerr << "[Jaguar][client_stage_quant][before_pool]" << std::endl;
        if (!stem_maxpool && kUseOpenCheetahAvgPoolWrapper) {
            std::vector<std::vector<uint64_t>> refresh_mask;
            jaguar_quant::recv_matrix_u64_fast(refresh_mask);
            add_refresh_mask_mod_inplace(relu_data, refresh_mask, relu_share_bits(log2_p, discard_lsb));
        }
        if (stem_maxpool) max_pooling_share(relu_data, w, pooling_kernel_size, pooling_stride, pooling_padding);
        else if (kUseOpenCheetahAvgPoolWrapper) avg_pooling_share_oc(relu_data, w, pooling_kernel_size, pooling_stride, pooling_padding);
        else avg_pooling(relu_data, pooling_kernel_size, pooling_stride, pooling_padding);
        std::cerr << "[Jaguar][client_stage_quant][after_pool] relu_rows=" << relu_data.size()
                  << " relu_cols=" << (relu_data.empty() ? 0 : relu_data[0].size()) << std::endl;
    }
    w = static_cast<int>(sqrt(relu_data[0].size()));
    j52_debug_dump("client", relu, relu ? relu_share_bits(log2_p, discard_lsb) : static_cast<int>(log2_p) - discard_lsb, relu_data);
    std::cerr << "[Jaguar][client_stage_quant][before_encrypt] next_w=" << w << std::endl;
    client_enc_zero_padding_quant(context, next_f_w, secret_key_coeff, relu_data, ct_relu, log2_q, log2_p, 0, context_ntt);
    std::cerr << "[Jaguar][client_stage_quant][after_encrypt] ct_relu=" << ct_relu.size() << std::endl;
    std::cerr << "[Jaguar][client_stage_quant][after_send]" << std::endl;
}

template <int N>
void cloud_stage_quant(const SEALContext &context,
                       const SEALContext & /* legacy transport_context */,
                       const uint64_t log2_q,
                       const uint64_t log2_p,
                       const int conv_stride,
                       const int f_w,
                       int &w,
                       const bool relu,
                       const bool use_relu6,
                       const bool pooling,
                       const int pooling_kernel_size,
                       const int pooling_stride,
                       const int pooling_padding,
                       std::vector<std::vector<uint64_t>> &ss_r_data,
                       const int next_f_w,
                       std::vector<Ciphertext> &ct_relu,
                       const int discard_lsb) {
    std::cerr << "[Jaguar][cloud_stage_quant][entry] w=" << w
              << " conv_stride=" << conv_stride
              << " f_w=" << f_w
              << " relu=" << (relu ? 1 : 0)
              << " pooling=" << (pooling ? 1 : 0)
              << " ss_r_rows=" << ss_r_data.size()
              << " ct_relu_in=" << ct_relu.size()
              << std::endl;

    std::vector<std::vector<uint64_t>> cloud_relu_share;
    const bool stem_maxpool = relu && pooling && (f_w == 7) && (pooling_kernel_size == 3) && (pooling_stride == 2) && (pooling_padding == 1);
    if (relu) {
        int relu_w = 0;
        auto activation_prelude_start = std::chrono::steady_clock::now();
        stride_extract_packed_quant(ss_r_data,
                                    static_cast<int>(context.key_context_data()->parms().poly_modulus_degree()),
                                    conv_stride,
                                    f_w,
                                    w,
                                    cloud_relu_share,
                                    relu_w);
        auto activation_prelude_end = std::chrono::steady_clock::now();
        jaguar::add_activation_prelude(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(activation_prelude_end - activation_prelude_start).count()),
            "pre=cloud_stride_extract_packed_quant");
        std::cerr << "[Jaguar][cloud_stage_quant][post_stride] relu_w=" << relu_w
                  << " share_rows=" << cloud_relu_share.size()
                  << " share_cols=" << (cloud_relu_share.empty() ? 0 : cloud_relu_share[0].size())
                  << std::endl;
        j52_debug_dump("pres", relu, static_cast<int>(log2_p) - discard_lsb, cloud_relu_share);
        std::vector<std::vector<uint64_t>> new_cloud_relu_share;
        const bool pool_first = stem_maxpool && j52_pool_first() && !use_relu6;
        if (pool_first) {
            // The cloud's additive share of y is -r; the exact ReLU negates its
            // input share itself, so negate around the pre-ReLU MaxPool.
            const int ring = static_cast<int>(log2_p) - discard_lsb;
            const uint64_t rm = (1ULL << ring) - 1;
            for (auto &row : cloud_relu_share) for (auto &v : row) v = (0 - v) & rm;
            max_pooling_share(cloud_relu_share, relu_w, pooling_kernel_size, pooling_stride, pooling_padding, ring);
            for (auto &row : cloud_relu_share) for (auto &v : row) v = (0 - v) & rm;
        }
        if (use_relu6) relu6_share_quant(cloud_relu_share, log2_p - discard_lsb, new_cloud_relu_share);
        else jaguar::relu_share_ring_external(cloud_relu_share, log2_p - discard_lsb, new_cloud_relu_share);
        cloud_relu_share.swap(new_cloud_relu_share);
        if (pooling && !pool_first) {
            if (!stem_maxpool && kUseOpenCheetahAvgPoolWrapper) {
                auto refresh_mask = sample_refresh_mask_like(cloud_relu_share, relu_share_bits(log2_p, discard_lsb));
                jaguar_quant::send_matrix_u64_fast(refresh_mask);
                sub_refresh_mask_mod_inplace(cloud_relu_share, refresh_mask, relu_share_bits(log2_p, discard_lsb));
            }
            if (stem_maxpool) max_pooling_share(cloud_relu_share, relu_w, pooling_kernel_size, pooling_stride, pooling_padding);
            else if (kUseOpenCheetahAvgPoolWrapper) avg_pooling_share_oc(cloud_relu_share, relu_w, pooling_kernel_size, pooling_stride, pooling_padding);
            else avg_pooling(cloud_relu_share, pooling_kernel_size, pooling_stride, pooling_padding);
        }
        std::cerr << "[Jaguar][cloud_stage_quant][post_relu] share_rows=" << cloud_relu_share.size()
                  << " share_cols=" << (cloud_relu_share.empty() ? 0 : cloud_relu_share[0].size())
                  << std::endl;
    }

    j52_debug_dump("cloud", relu, relu ? relu_share_bits(log2_p, discard_lsb) : static_cast<int>(log2_p) - discard_lsb,
                   relu ? cloud_relu_share : std::vector<std::vector<uint64_t>>{});
    std::vector<Ciphertext> ct_client_share;
    recv_ciphertexts(context, ct_client_share);
    std::cerr << "[Jaguar][cloud_stage_quant][recv_client_share] ct_count=" << ct_client_share.size()
              << " w_before_complete=" << w << std::endl;
    if (relu) {
        cloud_complete_relu_share_quant<N>(context, log2_q, log2_p, next_f_w, cloud_relu_share, w, ct_client_share);
    } else {
        cloud_complete_no_relu_quant<N>(context, log2_q, log2_p, conv_stride, f_w, ss_r_data, pooling, pooling_kernel_size, pooling_stride, pooling_padding, next_f_w, ct_client_share, w, ct_client_share);
    }
    std::cerr << "[Jaguar][cloud_stage_quant][exit] w=" << w
              << " ct_relu_out=" << ct_client_share.size()
              << std::endl;
    ct_relu.swap(ct_client_share);
}

template <int N>
void client_padding_stage_quant(const SEALContext &context,
                                const uint64_t log2_q,
                                const uint64_t log2_p,
                                const Plaintext &secret_key_coeff,
                                std::vector<Ciphertext> &ct_out1,
                                const int conv_stride,
                                const int f_w,
                                int &w,
                                const bool pooling,
                                const int pooling_kernel_size,
                                const int pooling_stride,
                                const int pooling_padding,
                                std::vector<std::vector<uint64_t>> &ss_r_data,
                                const int next_f_w,
                                std::vector<Ciphertext> &ct_relu,
                                const SEALContext &context_ntt,
                                const uint64_t q_ntt,
                                const int discard_lsb) {
    std::vector<std::vector<uint64_t>> normal_data;
    client_ct_stride_dec_quant(ct_out1, normal_data, context, secret_key_coeff, conv_stride, f_w, w,
                               log2_q, log2_p, context_ntt, q_ntt, discard_lsb);
    if (pooling) {
        avg_pooling(normal_data, pooling_kernel_size, pooling_stride, pooling_padding);
    }
    std::vector<std::vector<uint64_t>> padding_data(2 * normal_data.size(),
                                                    std::vector<uint64_t>(normal_data[0].size(), 0));
    for (size_t i = 0; i < normal_data.size(); ++i) {
        for (size_t j = 0; j < normal_data[i].size(); ++j) {
            padding_data[i + (normal_data.size() / 2)][j] = normal_data[i][j];
        }
    }
    client_enc_zero_padding_quant(context, next_f_w, secret_key_coeff, padding_data, ct_relu,
                                  log2_q, log2_p, 0, context_ntt);
}

template <int N>
void cloud_padding_stage_quant(const SEALContext &context,
                               const SEALContext & /* legacy transport_context */,
                               const uint64_t log2_q,
                               const uint64_t log2_p,
                               const int conv_stride,
                               const int f_w,
                               int &w,
                               const bool pooling,
                               const int pooling_kernel_size,
                               const int pooling_stride,
                               const int pooling_padding,
                               std::vector<std::vector<uint64_t>> &ss_r_data,
                               const int next_f_w,
                               std::vector<Ciphertext> &ct_relu) {
    std::vector<Ciphertext> ct_client_share;
    recv_ciphertexts(context, ct_client_share);
    cloud_complete_avg_pooling_padding_quant<N>(context, log2_q, log2_p, conv_stride, f_w, ss_r_data,
                                                pooling, pooling_kernel_size, pooling_stride,
                                                pooling_padding, next_f_w, ct_client_share, w,
                                                ct_client_share);
    ct_relu.swap(ct_client_share);
}

}  // namespace jaguar_quant


template <int N>
void stride_avg_pooling_padding(const SEALContext &context, const PublicKey &public_key, const SecretKey &secret_key, vector<Ciphertext> &ct_out1, const int conv_stride, const int f_w, int &w, const bool pooling, const int pooling_kernel_size, const int pooling_stride, const int pooling_padding, vector<vector<uint64_t> > &ss_r_data, const int next_f_w, vector<Ciphertext> &ct_relu);

template <int N>
void residual_path_A_quant(const int order, const int th_count, const int layer, const int flag_sw, const int flag_perm, const int seed_pre, int &seed, int &w, vector<vector<vector<uint64_t> > > &filter1, const int conv_stride, uint64_t f_w, bool pooling, int pooling_k, int pooling_s, int pooling_p, int next_conv_stride, int next_f_w, int next_pooling, int next_pooling_k, int next_pooling_s, int next_pooling_p, const SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, const Plaintext &secret_key_coeff, const SEALContext &context_ntt, const uint64_t q_ntt, const PublicKey &public_key, const SecretKey &secret_key, const vector<Ciphertext> &cin, vector<Ciphertext> &cout, const int discard_lsb_previous, const int discard_lsb);

template <int N>
void residual_path_B_quant(const int order, const int th_count, const int layer, const int flag_sw, const int flag_perm, const int seed_pre, int &seed, int &w, vector<vector<vector<uint64_t> > > &filter1, const int conv_stride, uint64_t f_w, bool pooling, int pooling_k, int pooling_s, int pooling_p, vector<vector<vector<uint64_t> > > &filter2, int next_conv_stride, int next_f_w, int next_pooling, int next_pooling_k, int next_pooling_s, int next_pooling_p, const SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, const Plaintext &secret_key_coeff, const SEALContext &context_ntt, const uint64_t q_ntt,  const PublicKey &public_key, const SecretKey &secret_key, const vector<Ciphertext> &cin, vector<Ciphertext> &cout, const int discard_lsb_previous, const int discard_lsb=0);

template <int N>
void call_conv_quant(const int order, const int th_count, const int layer, const int flag_sw, const int flag_perm, const int seed_pre, int &seed, const int in_h, vector<vector<vector<uint64_t>>> &filter, const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p, const vector<Ciphertext> &cin, vector<Ciphertext> &cout, vector<vector<uint64_t> > &ss_r_data, int extract_stride = 1, bool use_lwe_output = false, std::vector<LweSharedGroup> *lwe_groups = nullptr, const int discard_lsb=0);

void client_enc(const SEALContext &context, const PublicKey &pk_client, const SecretKey &sk_client, const vector<vector<uint64_t> > &din, vector<Ciphertext> &cout);

void client_dec_quant(const SEALContext &context_quant, const Plaintext &secret_key_coeff, const uint64_t log2_q, const uint64_t log2_p, vector<Ciphertext> &cout, vector<vector<uint64_t> > &dout, const SEALContext &context_ntt, const int discard_lsb=0);

void decrypt_quant(const Plaintext &secret_key_coeff_ntt, const uint64_t log2_q, const uint64_t log2_p, const Ciphertext &cout, Plaintext &pout, const NTTTables *&ntt_tables, const uint64_t q_ntt, const uint64_t n, const int discard_lsb=0);

template <int N>
void stride_relu_avg_pooling_quant(const SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, const Plaintext &secret_key_coeff, vector<Ciphertext> &ct_out1, const int conv_stride, const int f_w, int &w, const bool relu, const bool pooling, const int pooling_kernel_size, const int pooling_stride, const int pooling_padding, vector<vector<uint64_t> > &ss_r_data, const int next_f_w, vector<Ciphertext> &ct_relu, const SEALContext &context_ntt, const uint64_t q_ntt, const PublicKey &public_key, const SecretKey &secret_key, const int discard_lsb=0, bool use_lwe_output = false, const std::vector<LweSharedGroup> *lwe_groups = nullptr, bool use_relu6 = false);

void client_ct_stride_dec_quant(vector<Ciphertext> &cout, vector<vector<uint64_t> > &normal_data, const SEALContext & context, const Plaintext &secret_key_coeff, const int stride, const int f_w, const int w, const uint64_t log2_q, const uint64_t log2_p, const SEALContext &context_ntt, const uint64_t q_ntt, const int discard_lsb=0); // for new width

void run_fc_hybrid_ntt_last_layer(
    const SEALContext &context_ntt,
    const PublicKey &public_key,
    const SecretKey &secret_key,
    const vector<uint64_t> &feat,
    const vector<vector<uint64_t>> &W,
    int in_block,
    int out_block,
    vector<uint64_t> &fc_out);



void encrypt_quant_new(const SEALContext & context, const Plaintext & zero_plain, Ciphertext & temp_out, const Plaintext & sk_client_coeff_ntt, const uint64_t log2_q, const SEALContext & context_ntt)
{

    // parameters of context_ntt
    auto &context_ntt_data = *context_ntt.first_context_data();
    auto ntt_tables = context_ntt_data.small_ntt_tables();
    auto &parms_ntt = context_ntt_data.parms();
    auto &coeff_modulus_ntt = parms_ntt.coeff_modulus();
    uint64_t q_ntt = coeff_modulus_ntt[0].value();


    auto &context_data = *context.first_context_data();
    prng_seed_type public_prng_seed;

    auto ciphertext_prng = UniformRandomGeneratorFactory::DefaultFactory()->create(public_prng_seed);
    auto &parms_id = context_data.parms_id();
    
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count = parms.poly_modulus_degree();

    temp_out.resize(context, 2);

    temp_out.is_ntt_form() = 0;
    temp_out.correction_factor() = 1;

    uint64_t *c0 = temp_out.data(0);
    uint64_t *c1 = temp_out.data(1);
    // Sample a uniformly at random
    sample_poly_uniform(ciphertext_prng, parms, c1);

    // Sample e <-- chi
    MemoryPoolHandle pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);
    auto noise(allocate_poly(coeff_count, coeff_modulus_size, pool));
    auto bootstrap_prng = parms.random_generator()->create();
    bootstrap_prng->generate(prng_seed_byte_count, reinterpret_cast<seal_byte *>(public_prng_seed.data()));
    SEAL_NOISE_SAMPLER(bootstrap_prng, parms, noise.get());

    Plaintext temp(coeff_count);
    // ------------------------
    // NTT path only
    // temp = a
    // ------------------------
    int n = static_cast<int>(coeff_count);
    for (int i = 0; i < n; i++)
        temp.data()[i] = temp_out.data(1)[i];

    // NTT(a)
    ntt_negacyclic_harvey_lazy(temp.data(), ntt_tables[0]);

    // pointwise multiply with precomputed NTT(secret_key)
    multiply_eval_poly_poly_mod(temp.data(), sk_client_coeff_ntt.data(), q_ntt, n);

    // inverse NTT
    inverse_ntt_negacyclic_harvey_lazy(temp.data(), ntt_tables[0]);

    // convert from qPrime-domain representation to q=2^log2_q representation
    reduce_quant_ntt_poly(temp.data(), temp.data(), n, q_ntt, log2_q);

    // add noise
    add_plain_plain_quant(temp.data(), noise.get(), log2_q, coeff_count);

    // reduce into quant modulus
    reduce_quant_poly(temp.data(), temp.data(), coeff_count, log2_q);

    // c0 = -(a*s + e)
    negate_poly_coeffmod(temp.data(), coeff_count, coeff_modulus[0], c0);
}



void client_enc_zero_padding_quant(const SEALContext & context, const int f_w, const Plaintext &sk_client_coeff_ntt/*revised*/, const vector<vector<uint64_t> > &din, vector<Ciphertext> &cout, const uint64_t log2_q, const uint64_t log2_p, const bool past_encoding, const SEALContext & context_ntt)
{
    if (jaguar_quant::enabled() && jaguar_quant::is_cloud())
    {
        jaguar_quant::recv_ciphertexts(context, cout);
        return;
    }
    const auto transport_before = jaguar::profiler().transport_snapshot();
    std::cout << "quant_enc";
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_client_sum(0);
	chrono::microseconds time_off_client_sum(0);

	vector<vector<uint64_t> > din_new;
	
	time_start = chrono::high_resolution_clock::now();
	zero_padding_to_N(context, f_w, din, din_new);
	time_end = chrono::high_resolution_clock::now();
	time_client_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

	// For zero-padding
	const int p_l = (f_w - 1)/2; // left
	const int p_r = (f_w - 1)/2 + ((f_w - 1) % 2? 1: 0); // right

	const int din0_sqrt = sqrt(din[0].size()); // image width
	const int padded_size_sqrt = din0_sqrt + p_l + p_r; // image width including zero padding
	const int padded_size = padded_size_sqrt * padded_size_sqrt; // image size including zero padding

    auto &context_data = *context.key_context_data();
	const int n = context_data.parms().poly_modulus_degree(); // Polynomial degree n
	//std::cout << "Poly_modulus_degree: " << n << endl;
	int num_packed = n / padded_size; // # of packed image into n, if padded_size < n
	//int num_packed_inv = (padded_size % n)? (padded_size/n + 1): padded_size / n; // # of n to represent one image, if padded_size > n
	int remain_zero = n%padded_size_sqrt;
	int num_packed_inv;
	int first_col_num;
	int col_num;
	int col_num_num;
	if (n<padded_size)
	{
		first_col_num = (n-remain_zero)/padded_size_sqrt;
		col_num = first_col_num - 2*p_l;
		col_num_num = ((padded_size_sqrt - first_col_num) % col_num)?((padded_size_sqrt - first_col_num)/col_num + 1) : (padded_size_sqrt-first_col_num)/col_num;
		num_packed_inv = 1 + col_num_num;
	}

	int row_new; // # of plaintext for total input.
	if (num_packed > 0)
		row_new = din.size() / num_packed + ((din.size() % num_packed) ? 1: 0);
	else
		row_new = din.size() * num_packed_inv;

	if (static_cast<int>(din_new.size()) != row_new)
	{
		std::cout << "[debug] adjust row_new from " << row_new << " to actual padded rows " << din_new.size() << std::endl;
		row_new = static_cast<int>(din_new.size()); 
	}

	std::cout << "[debug] padded_size_sqrt=" << padded_size_sqrt
          << " padded_size=" << padded_size
          << " n=" << n
          << " row_new=" << row_new << std::endl;

	Plaintext zero_plain(n);
	zero_plain.set_zero(0, n);
	cout.resize(row_new);

	auto &first_context_data = *context.first_context_data();
	auto &parms = context.first_context_data()->parms();
	auto &plain_modulus = parms.plain_modulus();

    uint64_t pM1 = plain_modulus.value() - 1;
    for (int i = 0; i < din_new.size(); i++)
    {
        for (int j = 0; j < din_new[i].size(); j++)
        {
            while (static_cast<int64_t>(din_new[i][j]) < 0)
            {
                din_new[i][j] += plain_modulus.value();
            }
            din_new[i][j] = din_new[i][j] & pM1;
        }
    }

	time_start = chrono::high_resolution_clock::now();

	// Jaguar_52: callers pass the coefficient-form ternary key; encrypt_quant_new
	// multiplies pointwise in the NTT domain, so lift it once here (the reference
	// research/Jaguar does this lift inside encrypt_quant_new). Without it the
	// encryption key differs from the decryption key and every live output is noise.
	Plaintext sk_ntt_local(n);
	precompute_secret_key_ntt(sk_client_coeff_ntt, sk_ntt_local, context_ntt);

	// Quantized zero-padding encryption crashes in the parallel path for later ResNet stages; keep it serial.
	for (int i = 0; i < row_new; i++) // loop for # of plaintext for total input (packed or splited)
	{
        Ciphertext temp_out_local;
        Plaintext new_encoding_local(n);

        // 1. Encrypt zero
		encrypt_quant_new(
            context,
            zero_plain,
            temp_out_local,
            sk_ntt_local,   // NTT-form secret key
            log2_q,
            context_ntt
        );

        // 2. Encode plaintext block for this row
        const size_t coeff_count_local = din_new[i].size();
        if (coeff_count_local > static_cast<size_t>(n))
        {
            std::cerr << "[quant_enc][fatal] padded row " << i << " has " << coeff_count_local
                      << " coefficients, larger than poly degree " << n << std::endl;
            std::abort();
        }
        for (int j = 0; j < n; ++j)
        {
            new_encoding_local.data()[j] = 0;
        }
        for (size_t j = 0; j < coeff_count_local; ++j)
        {
            new_encoding_local.data()[j] = din_new[i][j];
        }

        // 3. Multiply by Delta = 2^(log2_q - log2_p)
        left_shift_plain(new_encoding_local.data(), (log2_q - log2_p), n);

        // 4. Add plaintext into c0 of Enc(0)
        add_plain_plain_quant(temp_out_local.data(0), new_encoding_local.data(0), log2_q, n);

        // 5. Reduce c0 into quant modulus domain
        reduce_quant_poly(temp_out_local.data(0), temp_out_local.data(0), n, log2_q);

        // 6. Store result
        cout[i] = temp_out_local;
	}

	time_end = chrono::high_resolution_clock::now();
	time_client_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

	if (num_packed>0) // If n >= padded_size
		std::cout << "\nEncrypt w/ padding=" << p_l << ", width=" << din0_sqrt << " >> " << padded_size_sqrt << ", input ch=" << din.size() << ", N=" << n << ", ct_# = " << row_new << " (" << num_packed << " ch in one ct)\n";
	else
		std::cout << "\nEncrypt w/ padding=" << p_l << ", width=" << din0_sqrt << " >> " << padded_size_sqrt << ", input ch=" << din.size() << ", N=" << n << ", ct_# = " << row_new << " (" << num_packed_inv << " cts per one ch)\n";

	if (past_encoding)
		std::cout << "Runtime for client encrypt w/ padding & past encoding (us): " << time_client_sum.count() << " [usec]\n";
	else
		std::cout << "Runtime for client encrypt w/ padding & new encoding (us): " << time_client_sum.count() << " [usec]\n";

	//std::cout << "Message size: " << 16 * n / 1024 * cout.size() << " [KB]\n" ;
	std::cout << "Message size: " << 2 * (log2_q+1) * n / 1024 / 8 * cout.size() << " [KB]\n" ;

    if (jaguar_quant::enabled() && jaguar_quant::is_client())
    {
        jaguar_quant::send_ciphertexts(cout);
    }

    const auto transport_after = jaguar::profiler().transport_snapshot();
    const uint64_t encrypt_sent_bytes = (transport_after.sent_bytes - transport_before.sent_bytes);
    const uint64_t encrypt_recv_bytes = (transport_after.recv_bytes - transport_before.recv_bytes);
    if (jaguar_quant::enabled() && jaguar_quant::is_client()) {
        jaguar_quant::stash_conv_input_transport(encrypt_sent_bytes, encrypt_sent_bytes + encrypt_recv_bytes);
    }
    jaguar::LayerSummary layer;
    layer.kind = "encrypt";
    layer.name = "client_enc_zero_padding_quant";
    layer.total_latency_us = static_cast<uint64_t>(time_client_sum.count());
    layer.compute_latency_us = static_cast<uint64_t>(time_client_sum.count());
    layer.comm_bytes = 0;
    layer.comm_bytes_compat = 0;
    layer.comm_bytes_actual = 0;
    layer.rounds = 0;
    layer.details = "f_w=" + std::to_string(f_w) +
                    ",input_ch=" + std::to_string(din.size()) +
                    ",ct_count=" + std::to_string(cout.size()) +
                    ",width=" + std::to_string(din0_sqrt) +
                    ",padded_w=" + std::to_string(padded_size_sqrt) +
                    ",sent_to_next_conv=" + std::to_string(encrypt_sent_bytes) +
                    ",recv_bytes=" + std::to_string(encrypt_recv_bytes);
    jaguar::profiler().record_layer_summary(layer);

}

inline std::string uint64_to_hex_string(std::uint64_t value)
{
	return seal::util::uint_to_hex_string(&value, std::size_t(1));
}

inline void print_parameters(const seal::SEALContext &context)
{
    auto &context_data = *context.key_context_data();

    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data.parms().scheme())
    {
    case seal::scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case seal::scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    case seal::scheme_type::bgv:
        scheme_name = "BGV";
        break;
    default:
        throw std::invalid_argument("unsupported scheme");
    }
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
    {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == seal::scheme_type::bfv)
    {
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    }

    std::cout << "\\" << std::endl;
}



void init(int dbc_perm=22, int dbc_sw=30, size_t n=2048, Modulus q=0xffffffff07c3001, int p=307201, string fname_sw="sw", string fname_gal1="gal1", string fname_gal2="gal2", string fname_parms="parms", string fname_pk1="pk1", string fname_pk2="pk2", string fname_sk1="sk1", string fname_sk2="sk2") {
   
    chrono::high_resolution_clock::time_point time_start, time_end;
       
    string poly = "1x^" + to_string(n) + " + 1";
    vector<Modulus> cipher_modulus = {q};

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(n);
    parms.set_coeff_modulus(cipher_modulus);
    parms.set_plain_modulus(p);
    SEALContext context(parms);
	print_parameters(context);
	std::cout << "Parameter validation (success): " << context.parameter_error_message() << endl;

    // Secret, Public Key Generation
    cout << "Generating secret/public keys: ";
    time_start = chrono::high_resolution_clock::now();
    KeyGenerator keygen(context);
    time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "Done [" << time_diff.count() << " us]" << endl;
    KeyGenerator keygen2(context);
	PublicKey public_key;
	keygen.create_public_key(public_key);
    auto secret_key = keygen.secret_key();
	PublicKey pk;
	keygen2.create_public_key(pk);
    auto secret_key_proxy = keygen2.secret_key();
	auto &key_context_data = *context.key_context_data();
    auto &key_parms = key_context_data.parms();

	auto &key_modulus = key_parms.coeff_modulus();
	size_t key_modulus_size = key_modulus.size();	

    std::ofstream f0(fname_parms.c_str(), ios::binary), f1(fname_sw.c_str()), f2(fname_gal1.c_str()), f3(fname_gal2.c_str()), f4(fname_pk1.c_str(), ios::binary), f5(fname_pk2.c_str()), f6(fname_sk1.c_str(), ios::binary), f7(fname_sk2.c_str());
    parms.save(f0);
    f0.close();
    public_key.save(f4);
    f4.close();
    pk.save(f5);
    f5.close();
    secret_key.save(f6);
    f6.close();
    secret_key_proxy.save(f7);
    f7.close();
}

void load_key(EncryptionParameters &parms, PublicKey &public_key, SecretKey &secret_key, string fname_parms="parms", string fname_pk="pk1", string fname_sk="sk1") {
    std::ifstream f0(fname_parms.c_str(), ios::binary), f1(fname_pk.c_str(), ios::binary), f2(fname_sk.c_str(), ios::binary);
    //const SEALContext context(parms);
    parms.load(f0);
    f0.close();
    const SEALContext context(parms);
	public_key.load(context, f1);
    f1.close();
	secret_key.load(context, f2);
    f2.close(); 
}

void init_quant(size_t n=2048, Modulus q=0xffffffff07c3001, int p=307201, string fname_parms="parms", string fname_sk1="sk1") {
    chrono::high_resolution_clock::time_point time_start, time_end;
       
    string poly = "1x^" + to_string(n) + " + 1";
    vector<Modulus> cipher_modulus = {q};

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(n);
    parms.set_coeff_modulus(cipher_modulus);
    parms.set_plain_modulus(p);
    SEALContext context(parms);
	print_parameters(context);
	std::cout << "Parameter validation (success): " << context.parameter_error_message() << endl;

    std::ofstream f0(fname_parms.c_str(), ios::binary), f6(fname_sk1.c_str(), ios::binary);
    parms.save(f0);
    f0.close();
}

void load_key_quant(EncryptionParameters &parms, Plaintext &secret_key_coeff, string fname_parms="parms", string fname_sk="sk1") {
    std::ifstream f0(fname_parms.c_str(), ios::binary), f2(fname_sk.c_str(), ios::binary);
    parms.load(f0);
    f0.close();
    const SEALContext context(parms);
    auto &context_data = *context.key_context_data();
    auto &coeff_modulus = parms.coeff_modulus();
    size_t coeff_count = parms.poly_modulus_degree();
    size_t coeff_modulus_size = coeff_modulus.size();
    parms.set_random_generator(UniformRandomGeneratorFactory::DefaultFactory());
    sample_poly_ternary(parms.random_generator()->create(), parms, secret_key_coeff.data());
}


namespace {
constexpr int kFcWrapperNumThreads = 4;
constexpr int kFcWrapperScale = 9;  // Jaguar_52: activation/weight scale 9 (was 12)
constexpr int kFcWrapperBitlength = 37;
constexpr int kFcWrapperPortOffset = 96;
constexpr bool kMatchOpenCheetahFcShape = true;

void ensure_fc_wrapper_session() {
    if (cheetah_linear != nullptr) {
        kIsSharedInput = true;
        return;
    }

    ::party = jaguar_quant::party_id();
    ::address = jaguar_quant::peer_host();
    ::port = jaguar_quant::base_port() + kFcWrapperPortOffset;
    ::num_threads = kFcWrapperNumThreads;
    ::bitlength = kFcWrapperBitlength;
    ::kScale = kFcWrapperScale;
    kIsSharedInput = true;
    StartComputation();
}

void build_fc_weight_tensor(
    const vector<vector<uint64_t>> &W,
    gemini::Tensor<uint64_t> &weight_tensor)
{
    const long out_dim = static_cast<long>(W.size());
    const long in_dim = out_dim ? static_cast<long>(W[0].size()) : 0;
    weight_tensor.Reshape(gemini::TensorShape({out_dim, in_dim}));
    for (long o = 0; o < out_dim; ++o) {
        for (long i = 0; i < in_dim; ++i) {
            weight_tensor(o, i) = W[static_cast<size_t>(o)][static_cast<size_t>(i)];
        }
    }
}
}  // namespace

// Convenience wrapper for last-layer FC.
void run_fc_hybrid_ntt_last_layer(
    const SEALContext &context_ntt,
    const PublicKey &public_key,
    const SecretKey &secret_key,
    const vector<uint64_t> &feat,
    const vector<vector<uint64_t>> &W,
    int in_block,
    int out_block,
    vector<uint64_t> &fc_out)
{
    (void)context_ntt;
    (void)public_key;
    (void)secret_key;
    (void)in_block;
    (void)out_block;

    using clock = std::chrono::high_resolution_clock;

    if (W.empty()) {
        fc_out.clear();
        return;
    }
    if (feat.size() != W[0].size()) {
        throw std::runtime_error("[FC] feature/weight dimension mismatch");
    }

    if (!jaguar_quant::enabled()) {
        fc_out.assign(W.size(), 0);
        for (size_t o = 0; o < W.size(); ++o) {
            uint64_t acc = 0;
            for (size_t i = 0; i < feat.size(); ++i) {
                acc += feat[i] * W[o][i];
            }
            fc_out[o] = acc;
        }
        return;
    }

    ensure_fc_wrapper_session();

    auto total_start = clock::now();
    const uint64_t comm_before = cheetah_linear ? cheetah_linear->io_counter() : 0;

    std::vector<std::vector<uint64_t>> padded_W;
    const std::vector<std::vector<uint64_t>> *fc_W = &W;
    if (kMatchOpenCheetahFcShape && feat.size() == 2048 && W.size() == 1000) {
        padded_W = W;
        padded_W.emplace_back(feat.size(), 0ULL);
        fc_W = &padded_W;
    }

    gemini::CheetahLinear::FCMeta meta;
    meta.input_shape = gemini::TensorShape({static_cast<long>(feat.size())});
    meta.weight_shape = gemini::TensorShape({static_cast<long>(fc_W->size()), static_cast<long>(feat.size())});
    meta.is_shared_input = true;
    kIsSharedInput = true;

    auto input_tensor = gemini::Tensor<uint64_t>::Wrap(
        const_cast<uint64_t *>(feat.data()), meta.input_shape);

    gemini::Tensor<uint64_t> weight_tensor;
    if (cheetah_linear->party() == SERVER) {
        build_fc_weight_tensor(*fc_W, weight_tensor);
    }

    gemini::Tensor<uint64_t> out_tensor;
    cheetah_linear->fc(input_tensor, weight_tensor, meta, out_tensor);
    fc_out.assign(out_tensor.data(), out_tensor.data() + out_tensor.shape().num_elements());

    const uint64_t comm_after = cheetah_linear ? cheetah_linear->io_counter() : comm_before;
    const auto total_end = clock::now();

    if (jaguar_quant::is_cloud() && weight_tensor.NumElements() > 0) {
        cheetah_linear->safe_erase(weight_tensor.data(), weight_tensor.NumElements());
    }

    jaguar::LayerSummary layer;
    layer.kind = "fc_total";
    layer.name = jaguar_quant::is_client() ? "fc_cheetah_client" : "fc_cheetah_cloud";
    layer.total_latency_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count());
    layer.comm_bytes_compat = comm_after - comm_before;
    layer.comm_bytes_actual = comm_after - comm_before;
    layer.rounds = 0;
    layer.details = "backend=cheetah_fc,input_dim=" + std::to_string(feat.size()) +
                    ",output_dim=" + std::to_string(fc_W->size()) +
                    ",shared_input=1,num_threads=" + std::to_string(kFcWrapperNumThreads) +
                    ",bitwidth=" + std::to_string(kFcWrapperBitlength) +
                    ",kScale=" + std::to_string(kFcWrapperScale);
    jaguar::profiler().record_layer_summary(layer);
}


static int run_argmax_quant(const std::vector<uint64_t> &logits)
{
    if (logits.empty()) {
        return -1;
    }

    if (!jaguar_quant::enabled()) {
        return static_cast<int>(std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    const int32_t bw_logits = kFcWrapperBitlength;
    const uint64_t mask = (bw_logits == 64) ? ~0ULL : ((1ULL << bw_logits) - 1);
    const uint64_t comm_before = ArgMaxCommSent;
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<intType> argmax_in(logits.size(), 0);
    if (jaguar_quant::is_client()) {
        for (size_t i = 0; i < logits.size(); ++i) {
            argmax_in[i] = static_cast<intType>(logits[i] & mask);
        }
    }

    intType pred_share = 0;
    // We exculded the ArgMax layer in evaluation BOTH Jaguar and Baselines (OpenCheetah, Rhombus).
    //ArgMax(1, static_cast<int32_t>(logits.size()), argmax_in.data(), &pred_share);

    //signedIntType pred_reconstructed = funcReconstruct2PCCons(pred_share, CLIENT);

    auto end = std::chrono::high_resolution_clock::now();

    jaguar::LayerSummary layer;
    layer.kind = "argmax";
    layer.name = jaguar_quant::is_client() ? "argmax_client" : "argmax_cloud";
    layer.total_latency_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    layer.comm_bytes_compat = ArgMaxCommSent - comm_before;
    layer.comm_bytes_actual = layer.comm_bytes_compat;
    layer.details = "num_classes=" + std::to_string(logits.size()) +
                    ",bw_logits=" + std::to_string(bw_logits);
    jaguar::profiler().record_layer_summary(layer);

    if (jaguar_quant::is_client()) {
        //return static_cast<int>(pred_reconstructed);
        return 0;
    }
    return -1;
}



int main(int argc, char** argv) {

	std::ios_base::sync_with_stdio(false); std::cin.tie(NULL); std::cout.tie(NULL);

	std::string jaguar_party_arg = "";
	std::string jaguar_peer_arg = "127.0.0.1";
	int jaguar_port_arg = 0;
	std::string jaguar_bn_mode_arg = "fused";
	int jaguar_use_fused_bn_arg = 1;
	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		if (arg == "--jaguar-party" && i + 1 < argc)
		{
			jaguar_party_arg = argv[++i];
		}
		else if (arg == "--jaguar-peer" && i + 1 < argc)
		{
			jaguar_peer_arg = argv[++i];
		}
		else if (arg == "--jaguar-port" && i + 1 < argc)
		{
			jaguar_port_arg = stoi(argv[++i]);
		}
		else if (arg == "--jaguar-bn-mode" && i + 1 < argc)
		{
			jaguar_bn_mode_arg = argv[++i];
		}
		else if (arg == "--USE_FUSED_BN" && i + 1 < argc)
		{
			jaguar_use_fused_bn_arg = stoi(argv[++i]);
		}
	}

    int n = stoi(argv[1]);
    int dbc_perm = stoi(argv[2]);
	uint64_t q , p, delta;
    if (n == 2048)
    {
        init(dbc_perm, 30, 2048, 0xffffffff07c3001, 307201);
		q = 0xffffffff07c3001;
		p = 307201;
		delta = q/p;
    }

    EncryptionParameters parms(scheme_type::bfv);
    PublicKey public_key;
    SecretKey secret_key;

    load_key(parms, public_key, secret_key);
    SEALContext context(parms);

    vector<Ciphertext> ct;
    vector<Ciphertext> ct_out(1); 
    vector<vector<uint64_t> > dout;

    int w = stoi(argv[3]);
    int f_w = stoi(argv[4]);
    int c_i = stoi(argv[5]);
    int c_o = stoi(argv[6]);
    vector<vector<uint64_t> > image(c_i, vector<uint64_t> (w*w, 0));

    bool model_input = false;
    std::string model_name = "";
    bool isMultiThread = false;

	// model parameter
    int relu = 1;
	int conv_stride = 1;
	bool pooling = 0;
	int pooling_kernel_size = 0;
	int pooling_stride = 0;
	int pooling_padding = 0;
    int next_f_w = 3;

    if (argc > 7)
    {
        std::string a7 = argv[7];

        model_input = true;
        model_name = a7;
    }
    if (argc > 8)
    {
        if (std::string(argv[8])=="mt")
            isMultiThread = true;
    }
    if (argc > 15)
    {
        relu = stoi(argv[9]);
        conv_stride = stoi(argv[10]);
        pooling = stoi(argv[11]);
        pooling_kernel_size = stoi(argv[12]);
        pooling_stride = stoi(argv[13]);
        pooling_padding = stoi(argv[14]);
        next_f_w = stoi(argv[15]);
    }

	if (!jaguar_party_arg.empty())
	{
		int jaguar_party = 0;
		if (jaguar_party_arg == "client" || jaguar_party_arg == "alice" || jaguar_party_arg == "1")
			jaguar_party = 1;
		else if (jaguar_party_arg == "cloud" || jaguar_party_arg == "bob" || jaguar_party_arg == "2")
			jaguar_party = 2;
		else
		{
			std::cerr << "[Jaguar] invalid --jaguar-party: " << jaguar_party_arg << std::endl;
			return 1;
		}
		if (jaguar_port_arg <= 0)
		{
			std::cerr << "[Jaguar] --jaguar-port must be provided for external ReLU runtime" << std::endl;
			return 1;
		}
		jaguar::configure_runtime(true, jaguar_party, jaguar_peer_arg.c_str(), jaguar_port_arg);
        jaguar_quant::configure(jaguar_party == 1 ? jaguar_quant::kClient : jaguar_quant::kCloud,
                               jaguar_peer_arg,
                               jaguar_port_arg);
		std::cout << "[Jaguar] external ReLU runtime enabled: party=" << jaguar_party_arg
		          << " peer=" << jaguar_peer_arg << " port=" << jaguar_port_arg << std::endl;
	}
	else
	{
		jaguar::configure_runtime(false, 0, "127.0.0.1", 0);
        jaguar_quant::configure(jaguar_quant::kMonolithic, "127.0.0.1", 0);
	}
	jaguar_bn_mode_arg = jaguar_use_fused_bn_arg ? "fused" : "standalone";
	jaguar_networks::set_bn_mode_from_string(jaguar_bn_mode_arg);
	std::cout << "[Jaguar] USE_FUSED_BN=" << (jaguar_networks::runtime_config().USE_FUSED_BN ? 1 : 0)
	          << " bn_mode=" << jaguar_networks::bn_mode_name(jaguar_networks::runtime_config().bn_mode)
	          << std::endl;

	init_all_fc_weights();


    vector<Ciphertext> ct_out1;	
    
    if (model_input && jaguar_networks::load_default_model_input(model_name, c_i, w, image))
    {
        std::cout << "[Jaguar] using OpenCheetah-style pretrained input for model " << model_name << "\n";
    }
    else
    {
        jaguar_networks::fill_debug_input(image);
    }

	// HE parameter (See section 5.1.)
    // Jaguar_52: team spec (N,P,Q,f) = (2048,30,52,9). Was 54.
    uint64_t log2_q = 52;
    if (const char *q_env = std::getenv("JAGUAR_LOG2_Q")) log2_q = std::stoull(q_env);  // diagnostics only
    uint64_t log2_p = 30; 
    uint64_t q_quant = (1ULL<<log2_q);
    uint64_t p_quant = (1ULL<<log2_p);
    int discard_lsb_model_output = 0;
    EncryptionParameters parms_quant(scheme_type::bfv);
    Plaintext secret_key_coeff(n);
    init_quant(2048, q_quant, p_quant);
    load_key_quant(parms_quant, secret_key_coeff);

    SEALContext context_quant(parms_quant);
    const size_t rerand_budget = model_input
        ? jaguar::output_rerandomization_budget(model_name, n, w,
            jaguar_networks::runtime_config().USE_FUSED_BN)
        : 0;

    // The cloud needs the client's public encryption key to rerandomize outputs;
    // its independently generated local key must never be used for this purpose.
    const auto rerand_start = std::chrono::steady_clock::now();
    const auto rerand_transport_before = jaguar::profiler().transport_snapshot();
    std::vector<Ciphertext> rerand_public_key;
    if (!jaguar_quant::is_cloud()) {
        rerand_public_key.push_back(
            jaguar::OutputRerandomizer::create_public_key(context_quant, secret_key_coeff));
        if (jaguar_quant::is_client())
            jaguar_quant::send_ciphertexts(rerand_public_key);
    } else {
        jaguar_quant::recv_ciphertexts(context_quant, rerand_public_key);
    }
    if (rerand_public_key.size() != 1)
        throw std::runtime_error("expected one output-rerandomization public key");
    if (!jaguar_quant::is_client())
        jaguar::configure_output_rerandomization(context_quant, rerand_public_key[0]);
    const auto rerand_transport_after = jaguar::profiler().transport_snapshot();
    jaguar::LayerSummary rerand_setup;
    rerand_setup.kind = "rerandomization_setup";
    rerand_setup.name = "pow2_output_public_key";
    rerand_setup.total_latency_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - rerand_start).count());
    rerand_setup.comm_bytes = rerand_transport_after.sent_bytes - rerand_transport_before.sent_bytes;
    rerand_setup.comm_bytes_compat = rerand_setup.comm_bytes;
    rerand_setup.comm_bytes_actual = rerand_setup.comm_bytes +
        rerand_transport_after.recv_bytes - rerand_transport_before.recv_bytes;
    rerand_setup.details = "N=" + std::to_string(n) + ",Q=" + std::to_string(log2_q) +
        ",one_time_setup=1";
    jaguar::profiler().record_layer_summary(rerand_setup);

    

    SEALContext context_ntt(parms);
    auto &context_ntt_data = *context_ntt.first_context_data();
    auto ntt_tables = context_ntt_data.small_ntt_tables();

	Plaintext secret_key_coeff_ntt(n);
	precompute_secret_key_ntt(secret_key_coeff, secret_key_coeff_ntt, context_ntt);

    // Offline preprocessing is complete on BOTH parties before the timed ROI.
    // The pool is local to this process/key, never sent to the client or reused.
    if (model_input) {
        const auto precompute_start = std::chrono::steady_clock::now();
        const auto before = jaguar::profiler().transport_snapshot();
        uint64_t compute_us = 0;
        if (!jaguar_quant::is_client()) {
            jaguar::precompute_output_rerandomization(rerand_budget, isMultiThread ? 4 : 1);
            compute_us = static_cast<uint64_t>(std::chrono::duration_cast<
                std::chrono::microseconds>(std::chrono::steady_clock::now() - precompute_start).count());
        }
        // An explicit readiness barrier prevents the client from charging
        // cloud preprocessing to its online inference interval.
        if (jaguar_quant::is_cloud())
            jaguar_quant::send_matrix_u64({{static_cast<uint64_t>(rerand_budget)}});
        else if (jaguar_quant::is_client()) {
            std::vector<std::vector<uint64_t>> ready;
            jaguar_quant::recv_matrix_u64(ready);
            if (ready.size()!=1 || ready[0].size()!=1 || ready[0][0]!=rerand_budget)
                throw std::runtime_error("rerandomization precompute budget mismatch");
        }
        const auto after = jaguar::profiler().transport_snapshot();
        jaguar::LayerSummary prep;
        prep.kind = "rerandomization_precompute";
        prep.name = "pow2_one_use_zero_pool";
        prep.total_latency_us = static_cast<uint64_t>(std::chrono::duration_cast<
            std::chrono::microseconds>(std::chrono::steady_clock::now() - precompute_start).count());
        prep.compute_latency_us = compute_us;
        prep.comm_bytes = after.sent_bytes - before.sent_bytes;
        prep.comm_bytes_compat = prep.comm_bytes;
        prep.comm_bytes_actual = prep.comm_bytes + after.recv_bytes - before.recv_bytes;
        prep.details = "pool_count=" + std::to_string(rerand_budget) +
            ",pool_bytes=" + std::to_string(jaguar::output_rerandomization_pool_bytes()) +
            ",offline=1,one_use=1,online_refill=0";
        jaguar::profiler().record_layer_summary(prep);
        std::cout << "RERANDOMIZATION_PRECOMPUTE_US " << compute_us << "\n"
                  << "RERANDOMIZATION_POOL_SIZE " << jaguar::output_rerandomization_prepared_count() << "\n"
                  << "RERANDOMIZATION_POOL_BYTES " << jaguar::output_rerandomization_pool_bytes()
                  << std::endl;
    }

	if (n==2048)
	{
        discard_lsb_model_output = 20;
        std::cout <<  "Model input: " << model_input << "\n";
  	
        // Model
        if (model_input)
        {
            // Jaguar_52: one-time two-party sessions (exact Ferret ReLU and the
            // SCI pooling session) are created before the timed inference ROI.
            if (jaguar_quant::enabled()) {
                jaguar::exact_relu_init();
                if (jaguar::exact_relu_enabled())
                    jaguar::ensure_pool_runtime(static_cast<uint64_t>(j52_pool_session_bits()));
            }
            const auto j52_start = std::chrono::steady_clock::now();
            { jaguar::LayerSummary m; m.kind = "cnn_begin"; m.name = model_name; jaguar::profiler().record_layer_summary(m); }
            dispatch_model_2048(model_name,
                                isMultiThread,
                                context_quant,
                                log2_q,
                                log2_p,
                                secret_key_coeff,
                                image,
                                w,
                                context_ntt,
                                public_key,
                                secret_key,
                                ct_out1,
                                discard_lsb_model_output);
            { jaguar::LayerSummary m; m.kind = "cnn_end"; m.name = model_name; jaguar::profiler().record_layer_summary(m); }
            std::cout << "J52_E2E_US " << std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - j52_start).count() << std::endl;
        }        
	}   
    
    if (model_input && !jaguar_quant::is_client() &&
        jaguar::output_rerandomization_remaining_count() != 0)
        throw std::runtime_error("rerandomization budget exceeds graph consumption; update the manifest");
    std::cout << "RERANDOMIZATION_POOL_REMAINING "
              << jaguar::output_rerandomization_remaining_count() << std::endl;
    std::cout << "RERANDOMIZED_OUTPUT_CIPHERTEXTS " << jaguar::rerandomized_output_count() << std::endl;

    // Decryption
	print_title("Remove this latency");
    std::cout << "discard_lsb_model_output: " << discard_lsb_model_output << "\n";
    client_dec_quant(context_quant, secret_key_coeff, log2_q, log2_p, ct_out1, dout, context_ntt, discard_lsb_model_output);

    // Jaguar_52: with the exact-ReLU DSO loaded, static destructors start a new
    // BaseOT against a peer that has already exited and the process returns 1.
    // All logs/CSV rows are written per record, so flush and skip teardown.
    if (jaguar_quant::enabled()) jaguar::exact_relu_save_trace();
    std::cout << std::flush; std::cerr << std::flush; std::fflush(nullptr);
    std::_Exit(0);
}



template <int N>
void stride_relu_avg_pooling_quant(const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p, const Plaintext &secret_key_coeff, vector<Ciphertext> &ct_out1, const int conv_stride, const int f_w, int &w, const bool relu, const bool pooling, const int pooling_kernel_size, const int pooling_stride, const int pooling_padding, vector<vector<uint64_t> > &ss_r_data, const int next_f_w, vector<Ciphertext> &ct_relu, const SEALContext &context_ntt, const uint64_t q_ntt, const PublicKey &public_key, const SecretKey &secret_key, const int discard_lsb, bool use_lwe_output, const std::vector<LweSharedGroup> *lwe_groups, bool use_relu6)
{
    if (jaguar_quant::enabled())
    {
        if (jaguar_quant::is_client())
        {
            jaguar_quant::client_stage_quant<N>(context, log2_q, log2_p, secret_key_coeff, ct_out1,
                                               conv_stride, f_w, w, relu, use_relu6, pooling, pooling_kernel_size,
                                               pooling_stride, pooling_padding, ss_r_data, next_f_w,
                                               ct_relu, context_ntt, q_ntt, discard_lsb,
                                               use_lwe_output, lwe_groups);
            return;
        }
        jaguar_quant::cloud_stage_quant<N>(context, context_ntt, log2_q, log2_p, conv_stride, f_w, w, relu, use_relu6, pooling,
                                          pooling_kernel_size, pooling_stride, pooling_padding, ss_r_data,
                                          next_f_w, ct_relu, discard_lsb);
        return;
    }
    const int input_w_before_stride = w;
    vector<vector<uint64_t> > normal_data;
    vector<vector<uint64_t> > relu_data;
    vector<vector<uint64_t> > cloud_relu_share;
    vector<vector<uint64_t> > relu_cloud_input;
    vector<Ciphertext> ct_normal2;
    if (pooling)
        print_title("ReLU + Avg Pooling (Cloud has result)");
    else
        print_title("ReLU w/ no pooling (Cloud has result)");

    const int p_l = (f_w - 1) / 2;
    const int p_r = (f_w - 1) / 2 + ((f_w - 1) % 2 ? 1 : 0);
    const int wPp = w + p_l + p_r;
    const int new_w = 1 + (wPp - f_w) / conv_stride;

    if (use_lwe_output) {
        if (lwe_groups == nullptr) {
            throw std::runtime_error("stride_relu_avg_pooling_quant: lwe_groups is null");
        }
        client_lwe_stride_dec_quant<N>(
            *lwe_groups,
            normal_data,
            secret_key_coeff,
            static_cast<int>(ss_r_data.size()),
            new_w,
            log2_q,
            log2_p,
            discard_lsb);
    } else {
        client_ct_stride_dec_quant(ct_out1, normal_data, context, secret_key_coeff, conv_stride, f_w, w, log2_q, log2_p, context_ntt, q_ntt, discard_lsb);
    }
    if (relu)
    {
        int relu_w = 0;
        auto activation_prelude_start = std::chrono::steady_clock::now();
        stride_extract_packed_quant(ss_r_data,
                                    static_cast<int>(context.key_context_data()->parms().poly_modulus_degree()),
                                    conv_stride,
                                    f_w,
                                    input_w_before_stride,
                                    relu_cloud_input,
                                    relu_w);
        auto activation_prelude_end = std::chrono::steady_clock::now();
        jaguar::add_activation_prelude(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(activation_prelude_end - activation_prelude_start).count()),
            "pre=stride_extract_packed_quant");
        if (relu_cloud_input.size() != normal_data.size() || (!normal_data.empty() && relu_cloud_input[0].size() != normal_data[0].size()))
        {
            std::cerr << "[Jaguar] relu input shape mismatch client_rows=" << normal_data.size()
                      << " client_cols=" << (normal_data.empty() ? 0 : normal_data[0].size())
                      << " cloud_rows=" << relu_cloud_input.size()
                      << " cloud_cols=" << (relu_cloud_input.empty() ? 0 : relu_cloud_input[0].size())
                      << " relu_w=" << relu_w << std::endl;
            throw std::runtime_error("relu packed share shape mismatch");
        }
        relu_share_ring_ideal(normal_data, relu_cloud_input, log2_p - discard_lsb, relu_data, cloud_relu_share);
    }
    else
    {
        relu_data = normal_data;
    }
    if (pooling)
    {
        avg_pooling(relu_data, pooling_kernel_size, pooling_stride, pooling_padding);
        if (relu)
        {
            avg_pooling(cloud_relu_share, pooling_kernel_size, pooling_stride, pooling_padding);
        }
    }
    w = static_cast<int>(sqrt(relu_data[0].size()));
    if (relu == 0)
    {
        int cloud_w = input_w_before_stride;
        client_enc_zero_padding_quant(context, next_f_w, secret_key_coeff, relu_data, ct_normal2, log2_q, log2_p, 0, context_ntt);
        cloud_complete_no_relu_quant<N>(context, log2_q, log2_p, conv_stride, f_w, ss_r_data, pooling, pooling_kernel_size, pooling_stride, pooling_padding, next_f_w, ct_normal2, cloud_w, ct_relu);
        w = cloud_w;
        return;
    }
    client_enc_zero_padding_quant(context, next_f_w, secret_key_coeff, relu_data, ct_relu, log2_q, log2_p, 0, context_ntt);
    cloud_complete_relu_share_quant<N>(context, log2_q, log2_p, next_f_w, cloud_relu_share, w, ct_relu);
}
 

template <int N>
void stride_avg_pooling_padding_quant(const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p, const Plaintext &secret_key_coeff, vector<Ciphertext> &ct_out1, const int conv_stride, const int f_w, int &w, const bool pooling, const int pooling_kernel_size, const int pooling_stride, const int pooling_padding, vector<vector<uint64_t> > &ss_r_data, const int next_f_w, vector<Ciphertext> &ct_relu, const SEALContext &context_ntt, const uint64_t q_ntt, const PublicKey &public_key, const SecretKey &secret_key, const int discard_lsb)
{
	if (jaguar_quant::enabled())
	{
		if (jaguar_quant::is_client())
		{
			jaguar_quant::client_padding_stage_quant<N>(
				context, log2_q, log2_p, secret_key_coeff, ct_out1, conv_stride, f_w, w,
				pooling, pooling_kernel_size, pooling_stride, pooling_padding,
				ss_r_data, next_f_w, ct_relu, context_ntt, q_ntt, discard_lsb);
			return;
		}
		if (jaguar_quant::is_cloud())
		{
			jaguar_quant::cloud_padding_stage_quant<N>(
				context, context_ntt, log2_q, log2_p, conv_stride, f_w, w,
				pooling, pooling_kernel_size, pooling_stride, pooling_padding,
				ss_r_data, next_f_w, ct_relu);
			return;
		}
	}
	vector<vector<uint64_t> > normal_data;
	vector<Ciphertext> ct_normal;
	vector<Ciphertext> ct_normal2;
	vector<vector<uint64_t> > ss_v_data;

	if (pooling)
		print_title("Avg Pooling + padding for ResNet (Cloud has result)");
	else
		print_title("Padding for ResNet (Cloud has result)");
	client_ct_stride_dec_quant(ct_out1, normal_data, context, secret_key_coeff, conv_stride, f_w, w, log2_q, log2_p, context_ntt, q_ntt, discard_lsb);

	if (pooling)
	{
		avg_pooling(normal_data, pooling_kernel_size, pooling_stride, pooling_padding);
	}

	vector<vector<uint64_t> > padding_data(2*normal_data.size(), vector<uint64_t> (normal_data[0].size(), 0));
	for (int i=0; i<normal_data.size(); i++)
	{
		for (int j=0; j<normal_data[i].size(); j++)
		{
			padding_data[i+(normal_data.size()/2)][j] = normal_data[i][j];
		}
	}
	client_enc_zero_padding_quant(context, next_f_w, secret_key_coeff, padding_data, ct_normal2, log2_q, log2_p, 0, context_ntt);
	cloud_complete_avg_pooling_padding_quant<N>(context, log2_q, log2_p, conv_stride, f_w, ss_r_data, pooling, pooling_kernel_size, pooling_stride, pooling_padding, next_f_w, ct_normal2, w, ct_relu);
}



void client_ct_stride_dec_quant(vector<Ciphertext> &cout, vector<vector<uint64_t> > &normal_data, const SEALContext & context, const Plaintext &secret_key_coeff, const int stride, const int f_w, const int w, const uint64_t log2_q, const uint64_t log2_p, const SEALContext &context_ntt, const uint64_t q_ntt, const int discard_lsb) // for new width
{
    const int past_decoding = 0;
	if (stride == 0)
		throw invalid_argument("Convolution stride can't be 0");
	if (f_w == 0)
		throw invalid_argument("Convolution filter's width can't be 0");
	if (w == 0)
		throw invalid_argument("Convolution input's width can't be 0");

	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_relu_stride_sum(0);

	auto &parms = context.first_context_data() -> parms();
	const int N = parms.poly_modulus_degree();
	auto &plain_modulus = parms.plain_modulus();
	Plaintext temp(N);

	const int p_l = (f_w - 1)/ 2;
	const int p_r = (f_w - 1)/ 2 + ((f_w - 1)%2?1:0);
	const int wPp = w + p_l + p_r;
	const int new_w = 1 + (wPp - f_w) / stride;

	normal_data.assign(1, vector<uint64_t> (new_w*new_w, 0));

	int cnt, idx;
	uint64_t coeff_temp;

    auto &context_data = *context.key_context_data();
    auto &context_ntt_data = *context_ntt.first_context_data();
    auto &coeff_modulus = parms.coeff_modulus();
    auto ntt_tables = context_ntt_data.small_ntt_tables();
    size_t coeff_modulus_size = coeff_modulus.size();
    Plaintext secret_key_coeff_ntt;
    secret_key_coeff_ntt = secret_key_coeff;

    for (int i=0; i<N; i++)
    {
        if ((secret_key_coeff_ntt.data()[i] != 0) && (secret_key_coeff_ntt.data()[i] != 1)) 
            secret_key_coeff_ntt.data()[i] = q_ntt - 1;
    }
    ntt_negacyclic_harvey_lazy(secret_key_coeff_ntt.data(), ntt_tables[0]);

	time_start = chrono::high_resolution_clock::now();
	if (past_decoding == 0)
	{
		if (wPp*wPp <= N)
		{
			cnt = 0; idx = 0;
			for (int i=0; i<cout.size(); i++)
			{
                decrypt_quant(secret_key_coeff_ntt, log2_q, log2_p, cout[i], temp, ntt_tables, q_ntt, N, discard_lsb);
				for (int j=0; j<new_w; j++)
				{
					for (int k=0; k<new_w; k++)
					{
						int start_j = j*stride + p_l;
						int start_k = k*stride + p_l;
						coeff_temp = temp.data(0)[start_j*wPp + start_k];
						normal_data[idx][cnt] = coeff_temp;
						cnt++;
					}
				}
				cnt = 0;
				idx++;
				normal_data.emplace_back(vector<uint64_t> (new_w * new_w, 0));
			}
			normal_data.erase(normal_data.end()-1);
		}
		else
		{
			int first_col_num = (N - (N%wPp))/wPp;
			int col_num = first_col_num - 2*p_l;

			int col_num_num = ((wPp - first_col_num)%col_num)?((wPp - first_col_num)/col_num + 1) : (wPp - first_col_num)/col_num;
			int num_packed_inv = 1 + col_num_num;
			cnt = 0; idx = 0;
			int idx2 = 0;
			int start_j, start_k;
	
			for (int i=0; i<cout.size(); i+=num_packed_inv)
			{
					idx2 = 0;
					start_j = p_l;

                    decrypt_quant(secret_key_coeff_ntt, log2_q, log2_p, cout[i], temp, ntt_tables, q_ntt, N, discard_lsb);
					for (int j=0; j<new_w; j++)
					{
						for (int k=0; k<new_w; k++)
						{
	
							if ((j*stride) / col_num > idx2)
							{
                                decrypt_quant(secret_key_coeff_ntt, log2_q, log2_p, cout[i + idx2 + 1], temp, ntt_tables, q_ntt, N, discard_lsb);
								idx2 ++;
								start_j = p_l + (j*stride)%col_num;
							}
							start_k = (k*stride + p_l);
							coeff_temp = temp.data(0)[start_j*wPp + start_k];
							normal_data[idx][cnt] = coeff_temp;
							cnt++;
	
							// Jaguar_52: advance after the row's last column only; the original
							// test (start_k > w - stride) also fired one column early for the
							// 7x7/s2 stem, so column 111 read the wrong row.
							if (k == new_w - 1)
								start_j += stride;
	
						}
					}
					if (cnt == new_w * new_w)
					{
						cnt = 0;
						idx++;
						
						normal_data.emplace_back(vector<uint64_t> (new_w * new_w, 0));
					}
			}
			normal_data.erase(normal_data.end()-1);
		}
	}
	time_end = chrono::high_resolution_clock::now();
	time_relu_stride_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

	if (past_decoding)
		std::cout << "\nDecrypt & past decoding w/ stride " << stride << ", width: " << wPp <<" >> " << new_w << "\n";
	else
		std::cout << "\nDecrypt w/ stride " << stride << ", width: " << wPp <<" >> " << new_w << "\n";
	std::cout << "Runtime for Decrypt (client): " << time_relu_stride_sum.count() << " [usec]\n";
}

void decrypt_quant(const Plaintext &secret_key_coeff_ntt, const uint64_t log2_q, const uint64_t log2_p, const Ciphertext &cout, Plaintext &pout, const NTTTables *&ntt_tables, const uint64_t q_ntt, const uint64_t n, const int discard_lsb)
{
    Ciphertext ctemp = cout;
    CoeffIter cout_0_iter = ctemp.data(0);
    CoeffIter cout_1_iter = ctemp.data(1);
    ntt_negacyclic_harvey_lazy(cout_1_iter, ntt_tables[0]);
    multiply_eval_poly_poly_mod(cout_1_iter, secret_key_coeff_ntt.data(0), q_ntt, n);
    inverse_ntt_negacyclic_harvey_lazy(cout_1_iter, ntt_tables[0]);

    reduce_quant_ntt_poly(cout_1_iter, cout_1_iter, n, q_ntt, log2_q);
    add_plain_plain_quant(cout_0_iter, cout_1_iter, log2_q, n);

    reduce_quant_poly(cout_0_iter, cout_0_iter, n, log2_q);

    decrypt_quant_reduce(cout_0_iter, pout.data(0), log2_q, log2_p, n, discard_lsb);

    reduce_quant_poly(pout.data(0), pout.data(0), n, log2_p-discard_lsb);
}


void client_dec_quant(const SEALContext &context_quant, const Plaintext &secret_key_coeff, const uint64_t log2_q, const uint64_t log2_p, vector<Ciphertext> &cout, vector<vector<uint64_t> > &dout, const SEALContext &context_ntt, const int discard_lsb)
{
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_client_sum(0);

	print_title("Client Decryption");
    auto &context_data = *context_quant.key_context_data();
	const int n = context_data.parms().poly_modulus_degree(); // Polynomial degree n
    bool past_decoding = 0; // Never use a batch encoding
	int num_cout = cout.size();
	dout.assign(num_cout, vector<uint64_t>(n, 0));

    // parameters of context_ntt
    auto &context_ntt_data = *context_ntt.first_context_data();
    auto ntt_tables = context_ntt_data.small_ntt_tables();
    auto &parms_ntt = context_ntt_data.parms();
    auto &coeff_modulus_ntt = parms_ntt.coeff_modulus();
    uint64_t q_ntt = coeff_modulus_ntt[0].value();

    // parameters of context_quant
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    Plaintext temp(n);
    //size_t coeff_count = parms.poly_modulus_degree();
    size_t coeff_modulus_size = coeff_modulus.size();

    vector<CoeffIter> cout_1_iter(num_cout);
    for (int i=0; i<num_cout; i++)
    {
        cout_1_iter[i] = cout[i].data(1);
    }

    Plaintext secret_key_coeff_ntt;
    secret_key_coeff_ntt = secret_key_coeff;

    //Evaluator evaluator(context_ntt);
    for (int i=0; i<n; i++)
    {
        if ((secret_key_coeff_ntt.data()[i] != 0) && (secret_key_coeff_ntt.data()[i] != 1)) 
            secret_key_coeff_ntt.data()[i] = q_ntt - 1;
    }

    ntt_negacyclic_harvey_lazy(secret_key_coeff_ntt.data(), ntt_tables[0]);
	time_start = chrono::high_resolution_clock::now();
    for (int i=0; i<num_cout; i++)
    {
        decrypt_quant(secret_key_coeff_ntt, log2_q, log2_p, cout[i], temp, ntt_tables, q_ntt, n, discard_lsb);
		set_vector_to_plain(dout[i], temp.data(0), 0, n);
    }

	time_end = chrono::high_resolution_clock::now();
	time_client_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

	std::cout << "\nDecrypt N=" << n << ", ct_# = " << num_cout << "\n";
	if (past_decoding)
		std::cout << "Runtime for client decryption w/ past decoding (us): " << time_client_sum.count() << " [usec]\n";
	else
		std::cout << "Runtime for client decryption (us): " << time_client_sum.count() << " [usec]\n";

    jaguar::LayerSummary layer;
    layer.kind = "decrypt";
    layer.name = "client_dec_quant";
    layer.total_latency_us = static_cast<uint64_t>(time_client_sum.count());
    layer.compute_latency_us = static_cast<uint64_t>(time_client_sum.count());
    layer.comm_bytes = 0;
    layer.rounds = 0;
    layer.details = "N=" + std::to_string(n) +
                    ",ct_count=" + std::to_string(num_cout) +
                    ",past_decoding=" + std::to_string(static_cast<int>(past_decoding));
    jaguar::profiler().record_layer_summary(layer);
}


static void extract_fc_input_from_final_ct_quant(
    const SEALContext &context_quant,
    const Plaintext &secret_key_coeff,
    const uint64_t log2_q,
    const uint64_t log2_p,
    const SEALContext &context_ntt,
    vector<Ciphertext> &ct,
    vector<uint64_t> &feat,
    int expected_feat_dim,
    int discard_lsb = 0)
{
    vector<vector<uint64_t>> dout;

    client_dec_quant(
        context_quant,
        secret_key_coeff,
        log2_q,
        log2_p,
        ct,
        dout,
        context_ntt,
        discard_lsb
    );

    feat.clear();
    feat.reserve(expected_feat_dim);

    if (ct.size() == 1) {
        if (dout.empty()) {
            throw std::runtime_error("[FC] decrypted dout is empty");
        }
        if ((int)dout[0].size() < expected_feat_dim) {
            std::cerr << "[FC] dout[0].size() = " << dout[0].size()
                      << ", expected_feat_dim = " << expected_feat_dim << "\n";
            throw std::runtime_error("[FC] not enough coeffs in final packed ct");
        }

        for (int i = 0; i < expected_feat_dim; ++i) {
            feat.push_back(dout[0][i]);
        }
        return;
    }

    // fallback: flatten multiple ciphertexts if needed
    for (size_t ct_idx = 0; ct_idx < dout.size() && (int)feat.size() < expected_feat_dim; ++ct_idx) {
        for (size_t j = 0; j < dout[ct_idx].size() && (int)feat.size() < expected_feat_dim; ++j) {
            feat.push_back(dout[ct_idx][j]);
        }
    }

    if ((int)feat.size() != expected_feat_dim) {
        std::cerr << "[FC] extracted feature dim = " << feat.size()
                  << ", expected = " << expected_feat_dim << "\n";
        throw std::runtime_error("[FC] feature extraction size mismatch");
    }
}

static void run_model_fc_quant(
    const string &model_name,
    const SEALContext &context_quant,
    const uint64_t log2_q,
    const uint64_t log2_p,
    const Plaintext &secret_key_coeff,
    const SEALContext &context_ntt,
    const PublicKey &public_key,
    const SecretKey &secret_key,
    vector<Ciphertext> &ct_final,
	int discard_lsb_model_output)
{
	int expected_feat_dim = 512;
	const std::vector<int64_t> *fc_bias = nullptr;

	vector<uint64_t> feat;
	if (model_name == "resnet50") {
		expected_feat_dim = 2048;
	}
	if (model_name == "mobilenet_v2") {
		expected_feat_dim = 1280;
	}
	extract_fc_input_from_final_ct_quant(
		context_quant,
		secret_key_coeff,
		log2_q,
		log2_p,
		context_ntt,
		ct_final,
		feat,
		expected_feat_dim,
		discard_lsb_model_output
	);

	std::cout << "\n[FC] extracted feature dim = " << feat.size() << "\n";
	// Jaguar_52 validation hook (off unless JAGUAR_DEBUG_FEATURES is set):
	// both parties write the decrypted final vector for an offline label check
	// (this test build gives both parties the key). Never set in timing runs.
	if (const char *debug_path = std::getenv("JAGUAR_DEBUG_FEATURES")) {
		{
			// Client: its own re-encrypted share only. Cloud: the completed
			// ciphertext (client share + server share) = the actual feature.
			std::ofstream dbg(std::string(debug_path) + (jaguar_quant::is_client() ? ".client" : ".cloud"));
			dbg << "# model=" << model_name << " log2_q=" << log2_q << " log2_p=" << log2_p
			    << " discard_lsb=" << discard_lsb_model_output << "\n";
			for (uint64_t v : feat) dbg << v << "\n";
		}
	}

    auto &context_data = *context_ntt.first_context_data();
    int N = static_cast<int>(context_data.parms().poly_modulus_degree());

    int in_block = static_cast<int>(feat.size());         
    std::vector<std::vector<uint64_t>> fc_weight;

    if (model_name == "resnet18") {
        fc_weight = jaguar_networks::resnet18_fc_weight();
        fc_bias = &jaguar_networks::resnet18_fc_bias();
    } else if (model_name == "resnet50") {
        fc_weight = jaguar_networks::resnet50_fc_weight();
        fc_bias = &jaguar_networks::resnet50_fc_bias();
    } else if (model_name == "mobilenet_v2") {
        fc_weight = jaguar_networks::mobilenet_v2_fc_weight();
        fc_bias = &jaguar_networks::mobilenet_v2_fc_bias();
    } else {
        throw std::runtime_error("[FC] unsupported model_name");
    }

    if (fc_weight.empty()) {
        throw std::runtime_error("[FC] fc_weight is empty");
    }

    int out_block = static_cast<int>(fc_weight.size());   
    if (in_block + out_block - 1 > N) {
        out_block = N - in_block + 1;
    }

    std::cout << "[FC] feature dim = " << feat.size()
              << ", weight in_dim = " << fc_weight[0].size() << "\n";

    if ((int)feat.size() != (int)fc_weight[0].size()) {
        throw std::runtime_error("[FC] feature/weight dimension mismatch");
    }

    vector<uint64_t> logits;
    run_fc_hybrid_ntt_last_layer(
        context_ntt,
        public_key,
        secret_key,
        feat,
        fc_weight,
        in_block,
        out_block,
        logits
    );

    // Jaguar_52: logits are additive shares in the Cheetah FC ring 2^37.
    // Add the bias once (client), at the logit scale (feature 9 + weight 9),
    // modulo 2^37. The original added it on both parties modulo the BFV prime.
    if (fc_bias != nullptr && fc_bias->size() == logits.size() && jaguar_quant::is_client()) {
        const uint64_t ring = (kFcWrapperBitlength >= 64) ? ~0ULL : ((1ULL << kFcWrapperBitlength) - 1ULL);
        for (size_t i = 0; i < logits.size(); ++i) {
            const uint64_t addend = static_cast<uint64_t>((*fc_bias)[i] * (int64_t(1) << kFcWrapperScale));
            logits[i] = (logits[i] + addend) & ring;
        }
    }

    std::cout << "[FC] output dim = " << logits.size() << "\n";

    const int predicted_label = run_argmax_quant(logits);
    if (jaguar_quant::is_client() && predicted_label >= 0) {
        std::cout << "[ArgMax] predicted label = " << predicted_label << "\n";
    }
}



template <int N>
void residual_path_B_quant(const int order, const int th_count, const int layer, const int flag_sw, const int flag_perm, const int seed_pre, int &seed, int &w, vector<vector<vector<uint64_t> > > &filter1, const int conv_stride, uint64_t f_w, bool pooling, int pooling_k, int pooling_s, int pooling_p, vector<vector<vector<uint64_t> > > &filter2, int next_conv_stride, int next_f_w, int next_pooling, int next_pooling_k, int next_pooling_s, int next_pooling_p, const SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, const Plaintext &secret_key_coeff, const SEALContext &context_ntt, const uint64_t q_ntt,  const PublicKey &public_key, const SecretKey &secret_key, const vector<Ciphertext> &cin, vector<Ciphertext> &cout, const int discard_lsb_previous, const int discard_lsb)
{
	int branch_w = w;
	vector<Ciphertext> ct_out;
	vector<vector<uint64_t> > ss_r_data;

	// The projection shortcut starts from the current block input directly.
	std::cout<< "Residual path B\n";
	if (jaguar_quant::enabled() && jaguar_quant::is_client())
	{
		jaguar_quant::recv_ciphertexts(context_quant, ct_out);
	}
	else
	{
		cloud_padded_new_CNN_quant_st<N>(layer, flag_sw, flag_perm, seed_pre, seed, branch_w, filter2, context_quant, log2_q, log2_p, cin, ct_out, ss_r_data, discard_lsb);
        // Apply the projection convolution's BN bias before completing its no-ReLU path.
        jaguar_networks::resnet18_apply_post_conv_bias_quant(order, log2_p, ss_r_data);
        jaguar_networks::resnet50_apply_post_conv_bias_quant(order, log2_p, ss_r_data);
		if (jaguar_quant::enabled() && jaguar_quant::is_cloud())
		{
			jaguar_quant::send_ciphertexts(ct_out);
		}
	}
	bool no_relu = 0;
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, next_conv_stride, next_f_w, branch_w, no_relu, next_pooling, next_pooling_k, next_pooling_s, next_pooling_p, ss_r_data, next_f_w , cout, context_ntt, q_ntt, public_key, secret_key, discard_lsb);
}


template <int N>
void residual_path_A_quant(const int order, const int th_count, const int layer, const int flag_sw, const int flag_perm, const int seed_pre, int &seed, int &w, vector<vector<vector<uint64_t> > > &filter1, const int conv_stride, uint64_t f_w, bool pooling, int pooling_k, int pooling_s, int pooling_p, int next_conv_stride, int next_f_w, int next_pooling, int next_pooling_k, int next_pooling_s, int next_pooling_p, const SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, const Plaintext &secret_key_coeff, const SEALContext &context_ntt, const uint64_t q_ntt, const PublicKey &public_key, const SecretKey &secret_key, const vector<Ciphertext> &cin, vector<Ciphertext> &cout, const int discard_lsb_previous, const int discard_lsb)
{
	int branch_w = w;
	vector<Ciphertext> ct_in;
	vector<Ciphertext> ct_out;
	vector<vector<uint64_t> > ss_r_data;

	// Same as previous layer computation
	print_title("Remove this latency");
	if (jaguar_quant::enabled() && jaguar_quant::is_client())
	{
		jaguar_quant::recv_ciphertexts(context_quant, ct_out);
	}
	else
	{
		cloud_padded_new_CNN_quant_st<N>(layer, flag_sw, flag_perm, seed_pre, seed, branch_w, filter1, context_quant, log2_q, log2_p, cin, ct_out, ss_r_data, discard_lsb_previous);
		if (jaguar_quant::enabled() && jaguar_quant::is_cloud())
		{
			jaguar_quant::send_ciphertexts(ct_out);
		}
	}
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, conv_stride, f_w, branch_w, 1, pooling, pooling_k, pooling_s, pooling_p, ss_r_data, 1, ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb_previous);
	print_title("Finished removing");

	// Residual path computation
	std::cout<< "Residual path A\n";
	int num_out_ch = filter1.size();
	vector<vector<uint64_t> > filter2(num_out_ch, vector<uint64_t>(1, 1));
	print_title("Remove this latency");
	if (jaguar_quant::enabled() && jaguar_quant::is_client())
	{
		jaguar_quant::recv_ciphertexts(context_quant, ct_out);
	}
	else
	{
		cloud_depthwise_padded_new_CNN_quant_st<2048>(1, 0, 0, 0, seed, branch_w, filter2, context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, discard_lsb);
		if (jaguar_quant::enabled() && jaguar_quant::is_cloud())
		{
			jaguar_quant::send_ciphertexts(ct_out);
		}
	}

	print_title("Finished removing");
	//call_conv<N>(order, th_count, layer, flag_sw, flag_perm, seed_pre, seed, w, filter2, context, public_key, ct_in, ct_out, ss_r_data);
	stride_avg_pooling_padding_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, next_conv_stride, 1, branch_w, next_pooling, next_pooling_k, next_pooling_s, next_pooling_p, ss_r_data, next_f_w , cout, context_ntt, q_ntt, public_key, secret_key, discard_lsb);
}


template <int N>
void call_conv_quant(const int order, const int th_count, const int layer, const int flag_sw, const int flag_perm, const int seed_pre, int &seed, const int in_h, vector<vector<vector<uint64_t>>> &filter, const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p, const vector<Ciphertext> &cin, vector<Ciphertext> &cout, vector<vector<uint64_t> > &ss_r_data, int extract_stride, bool use_lwe_output, std::vector<LweSharedGroup> *lwe_groups, const int discard_lsb)
{
    const auto conv_transport_before = jaguar::profiler().transport_snapshot();
    auto conv_start = std::chrono::high_resolution_clock::now();
    std::cerr << "[Jaguar][call_conv_quant][entry] order=" << order
              << " in_h=" << in_h
              << " cin=" << cin.size()
              << std::endl;
    if (jaguar_quant::enabled())
    {
        if (jaguar_quant::is_client())
        {
            if (use_lwe_output) {
                jaguar_quant::recv_lwe_groups(*lwe_groups);
            } else {
                jaguar_quant::recv_ciphertexts(context, cout);
            }
           
            std::cerr << "[Jaguar][call_conv_quant][client_recv] order=" << order
                      << " cout=" << cout.size()
                      << std::endl;
            auto conv_end = std::chrono::high_resolution_clock::now();
            const auto conv_transport_after = jaguar::profiler().transport_snapshot();
            const uint64_t pending_conv_sent_bytes = jaguar_quant::consume_pending_conv_input_sent_bytes();
            const uint64_t pending_conv_actual_bytes = jaguar_quant::consume_pending_conv_input_actual_bytes();
            jaguar::LayerSummary layer_summary;
            layer_summary.kind = "conv_total";
            layer_summary.name = "conv_quant_" + std::to_string(order);
            layer_summary.total_latency_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(conv_end - conv_start).count());
            layer_summary.compute_latency_us = 0;
            layer_summary.comm_bytes = (conv_transport_after.sent_bytes - conv_transport_before.sent_bytes) + pending_conv_sent_bytes;
            layer_summary.comm_bytes_compat = layer_summary.comm_bytes;
            const uint64_t conv_recv_bytes = (conv_transport_after.recv_bytes - conv_transport_before.recv_bytes);
            layer_summary.comm_bytes_actual = pending_conv_actual_bytes + layer_summary.comm_bytes + conv_recv_bytes;
            layer_summary.rounds = 0;
            layer_summary.details = "order=" + std::to_string(order) + ",party=client,ct_count=" + std::to_string(cout.size()) + ",pending_input_sent=" + std::to_string(pending_conv_sent_bytes) + ",recv_bytes=" + std::to_string(conv_recv_bytes);
            jaguar::profiler().record_layer_summary(layer_summary);
            return;
        }
    }
    int f_size = filter[0][0].size();
    int f_h = sqrt(f_size);

    auto &context_data = *context.key_context_data();

    int p_l = (f_h - 1) / 2;
    int p_r = (f_h - 1) / 2 + ((f_h - 1) % 2 ? 1 : 0);
    int wPp = in_h + p_l + p_r;
    int in_size = wPp * wPp;
    int n = context_data.parms().poly_modulus_degree();
 
    if (order == 29) std::cerr << "[Jaguar][debug29][before_conv] filter_out=" << filter.size() << " filter_in=" << (filter.empty() ? 0 : filter[0].size()) << " cin=" << cin.size() << " in_h=" << in_h << " discard_lsb=" << discard_lsb << std::endl;
	if (th_count == 0)
	{
		print_title(to_string(order) + "-th convolution");
		cloud_padded_new_CNN_quant_st<N>(layer, flag_sw, flag_perm, seed_pre, seed, in_h, filter, context, log2_q, log2_p, cin, cout, ss_r_data, discard_lsb);
	}
	else
	{
		print_title(to_string(order)+"-th " + to_string(th_count) + " multi-threading convolution");
		cloud_padded_new_CNN_quant_mt<N>(th_count, layer, flag_sw, flag_perm, seed_pre, seed, in_h, filter, context, log2_q, log2_p, cin, cout, ss_r_data, discard_lsb);
    if (order == 29) std::cerr << "[Jaguar][debug29][after_conv] cout=" << cout.size() << " ss_r_rows=" << ss_r_data.size() << std::endl;
	}

    // The active model updates its mask once per learned convolution. MobileNetV2
    // applies its bias explicitly in its driver, including its depthwise path.
    jaguar_networks::resnet18_apply_post_conv_bias_quant(order, log2_p, ss_r_data);
    jaguar_networks::resnet50_apply_post_conv_bias_quant(order, log2_p, ss_r_data);

    if (order == 29) std::cerr << "[Jaguar][debug29][after_bn] ss_r_rows=" << ss_r_data.size() << std::endl;

    if (jaguar_quant::enabled() && jaguar_quant::is_cloud())
    {
        if (use_lwe_output) {
            int next_w_dummy = 0;
            std::vector<LweSharedGroup> local_groups;
            build_shared_lwe_groups_from_cout_quant<N>(
                cout, extract_stride, f_h, in_h, static_cast<int>(filter.size()), log2_q, local_groups, next_w_dummy);
            jaguar_quant::send_lwe_groups(local_groups);
            if (lwe_groups) *lwe_groups = std::move(local_groups);
        } else {
            jaguar_quant::send_ciphertexts(cout);
        }
        
        std::cerr << "[Jaguar][call_conv_quant][cloud_send] order=" << order
                  << " cout=" << cout.size()
                  << std::endl;
    }

    auto conv_end = std::chrono::high_resolution_clock::now();
    const auto conv_transport_after = jaguar::profiler().transport_snapshot();
    jaguar::LayerSummary layer_summary;
    layer_summary.kind = "conv_total";
    layer_summary.name = "conv_quant_" + std::to_string(order);
    layer_summary.total_latency_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(conv_end - conv_start).count());
    layer_summary.compute_latency_us = layer_summary.total_latency_us;
    layer_summary.comm_bytes = (conv_transport_after.sent_bytes - conv_transport_before.sent_bytes);
    layer_summary.comm_bytes_compat = layer_summary.comm_bytes;
    const uint64_t conv_recv_bytes = (conv_transport_after.recv_bytes - conv_transport_before.recv_bytes);
    layer_summary.comm_bytes_actual = layer_summary.comm_bytes + conv_recv_bytes;
    layer_summary.rounds = 0;
    layer_summary.details = "order=" + std::to_string(order) + ",party=cloud,ct_count=" + std::to_string(cout.size()) + ",ss_r_rows=0" /*+ std::to_string(ss_r_data.size())*/ + ",recv_bytes=" + std::to_string(conv_recv_bytes);
    jaguar::profiler().record_layer_summary(layer_summary);
}

#include "../networks/input_loader.cpp"
#include "../networks/runtime_config.cpp"
#include "../networks/resnet18_support.cpp"
#include "../networks/resnet50_support.cpp"
#include "../networks/mobilenet_v2_support.cpp"
#include "../networks/resnet18_quant.cpp"
#include "../networks/resnet50_quant.cpp"
#include "../networks/mobilenet_v2_quant.cpp"
