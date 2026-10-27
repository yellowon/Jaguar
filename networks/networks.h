#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "seal/seal.h"

namespace jaguar_networks {

enum class BNMode {
    kFused = 0,
    kStandalone = 1,
};

struct RuntimeConfig {
    BNMode bn_mode = BNMode::kFused;
    bool USE_FUSED_BN = true;
};

RuntimeConfig &runtime_config();
void set_bn_mode_from_string(const std::string &mode);
const char *bn_mode_name(BNMode mode);

struct BatchNormLayerParams {
    std::vector<int64_t> mul;
    std::vector<int64_t> add;
    bool identity = true;
    bool loaded = false;
};

struct LinearLayerParams {
    std::vector<std::vector<uint64_t>> weight;
    std::vector<int64_t> bias;
    bool loaded = false;
};

struct ResNet18ParameterBundle {
    std::vector<BatchNormLayerParams> bn_layers;
    LinearLayerParams fc;
    bool loaded = false;
    bool fused_filters_ready = false;
    std::string source;
};

void configure_resnet18_runtime_params(
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> &filters);
void resnet18_log_model_summary();
void resnet18_set_quant_active(bool active);
void resnet18_apply_post_conv_bias_quant(
    int layer_index,
    uint64_t log2_p,
    std::vector<std::vector<uint64_t>> &ss_r_data);
const std::vector<std::vector<uint64_t>> &resnet18_fc_weight();
const std::vector<int64_t> &resnet18_fc_bias();

void configure_resnet50_runtime_params(
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> &filters);
void resnet50_log_model_summary();
void resnet50_set_quant_active(bool active);
void resnet50_apply_post_conv_bias_quant(
    int layer_index,
    uint64_t log2_p,
    std::vector<std::vector<uint64_t>> &ss_r_data);
const std::vector<std::vector<uint64_t>> &resnet50_fc_weight();
const std::vector<int64_t> &resnet50_fc_bias();

void configure_mobilenet_v2_runtime_params(
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> &filters);
void mobilenet_v2_log_model_summary();
void mobilenet_v2_set_quant_active(bool active);
void mobilenet_v2_apply_post_conv_bias_quant(
    int layer_index,
    uint64_t log2_p,
    std::vector<std::vector<uint64_t>> &ss_r_data);
const std::vector<std::vector<uint64_t>> &mobilenet_v2_fc_weight();
const std::vector<int64_t> &mobilenet_v2_fc_bias();

} // namespace jaguar_networks

template <int N>
void resnet18_quant(std::string str, const seal::SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, const seal::Plaintext &secret_key_coeff, std::vector<std::vector<uint64_t>> &image, int &w, const seal::SEALContext &context_ntt, const seal::PublicKey &public_key, const seal::SecretKey &secret_key, std::vector<seal::Ciphertext> &ct, int &discard_lsb_model_output);

template <int N>
void resnet50_quant(std::string str, const seal::SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, const seal::Plaintext &secret_key_coeff, std::vector<std::vector<uint64_t>> &image, int &w, const seal::SEALContext &context_ntt, const seal::PublicKey &public_key, const seal::SecretKey &secret_key, std::vector<seal::Ciphertext> &ct, int &discard_lsb_model_output);

template <int N>
void mobilenet_v2_quant(std::string str, const seal::SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, const seal::Plaintext &secret_key_coeff, std::vector<std::vector<uint64_t>> &image, int &w, const seal::SEALContext &context_ntt, const seal::PublicKey &public_key, const seal::SecretKey &secret_key, std::vector<seal::Ciphertext> &ct, int &discard_lsb_model_output);
