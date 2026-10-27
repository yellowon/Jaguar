#include "networks.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace jaguar_networks {
namespace {

const char *kMobileNetV2Root = "../models/filters_mobilenetv2_224_scale9";
// Jaguar_52: scale-9 weights x scale-12 BN multiplier / 2^12 -> scale-9 fused
// weights, so conv output scale 18 - f(9) keeps activations at scale 9.
constexpr int kMobileNetV2ScaleBits = 12;
constexpr uint64_t kMobileNetV2FilterMod = (1ULL << 27);
constexpr uint64_t kMobileNetV2FcMod = (1ULL << 37);  // Jaguar_52: Cheetah FC ring (bitlength 37), not the BFV prime
constexpr int kMobileNetV2FcInDim = 1280;
constexpr int kMobileNetV2FcOutDim = 1000;

ResNet18ParameterBundle g_mobilenetv2_bundle;
bool g_mobilenetv2_initialized = false;
bool g_mobilenetv2_quant_active = false;
std::vector<std::vector<std::vector<std::vector<uint64_t>>>> g_mobilenetv2_fused_filters;

std::string bn_scale_path(size_t layer_idx) {
    return std::string(kMobileNetV2Root) + "/layer_" + std::to_string(layer_idx) + "_bn_scale.txt";
}

std::string bn_bias_path(size_t layer_idx) {
    return std::string(kMobileNetV2Root) + "/layer_" + std::to_string(layer_idx) + "_bn_bias.txt";
}

bool read_flat_i64_file_mbv2(const std::string &path, std::vector<int64_t> &out) {
    std::ifstream fin(path);
    if (!fin.is_open()) return false;
    out.clear();
    int64_t value = 0;
    while (fin >> value) out.push_back(value);
    return true;
}

uint64_t to_mod_u64_mbv2(int64_t value, uint64_t mod) {
    if (mod == 0) return static_cast<uint64_t>(value);
    int64_t reduced = value % static_cast<int64_t>(mod);
    if (reduced < 0) reduced += static_cast<int64_t>(mod);
    return static_cast<uint64_t>(reduced);
}

int64_t centered_from_mod_u64_mbv2(uint64_t value, uint64_t mod) {
    if (mod == 0) return static_cast<int64_t>(value);
    value %= mod;
    uint64_t half = mod >> 1;
    return (value >= half) ? static_cast<int64_t>(value - mod) : static_cast<int64_t>(value);
}

void ensure_bundle_loaded(std::vector<std::vector<std::vector<std::vector<uint64_t>>>> &filters) {
    if (g_mobilenetv2_initialized) return;

    ResNet18ParameterBundle bundle;
    bundle.source = kMobileNetV2Root;
    bundle.bn_layers.resize(filters.size());

    for (size_t layer_idx = 0; layer_idx < filters.size(); ++layer_idx) {
        const size_t out_ch = filters[layer_idx].size();
        auto &bn = bundle.bn_layers[layer_idx];
        if (!read_flat_i64_file_mbv2(bn_scale_path(layer_idx), bn.mul) || bn.mul.size() != out_ch) {
            throw std::runtime_error("[Jaguar][mobilenet_v2] failed to load BN scale for layer " + std::to_string(layer_idx));
        }
        if (!read_flat_i64_file_mbv2(bn_bias_path(layer_idx), bn.add) || bn.add.size() != out_ch) {
            throw std::runtime_error("[Jaguar][mobilenet_v2] failed to load BN bias for layer " + std::to_string(layer_idx));
        }
        bn.identity = false;
        bn.loaded = true;
    }

    std::vector<int64_t> fc_weight_flat;
    std::vector<int64_t> fc_bias;
    if (!read_flat_i64_file_mbv2(std::string(kMobileNetV2Root) + "/fc_weight.txt", fc_weight_flat) ||
        fc_weight_flat.size() != static_cast<size_t>(kMobileNetV2FcInDim * kMobileNetV2FcOutDim)) {
        throw std::runtime_error("[Jaguar][mobilenet_v2] failed to load fc_weight.txt");
    }
    if (!read_flat_i64_file_mbv2(std::string(kMobileNetV2Root) + "/fc_bias.txt", fc_bias) ||
        fc_bias.size() != static_cast<size_t>(kMobileNetV2FcOutDim)) {
        throw std::runtime_error("[Jaguar][mobilenet_v2] failed to load fc_bias.txt");
    }

    bundle.fc.weight.assign(kMobileNetV2FcOutDim, std::vector<uint64_t>(kMobileNetV2FcInDim, 0));
    for (int in_idx = 0; in_idx < kMobileNetV2FcInDim; ++in_idx) {
        for (int out_idx = 0; out_idx < kMobileNetV2FcOutDim; ++out_idx) {
            const int64_t value = fc_weight_flat[static_cast<size_t>(in_idx) * kMobileNetV2FcOutDim + out_idx];
            bundle.fc.weight[out_idx][in_idx] = to_mod_u64_mbv2(value, kMobileNetV2FcMod);
        }
    }
    bundle.fc.bias = std::move(fc_bias);
    bundle.fc.loaded = true;
    bundle.loaded = true;

    g_mobilenetv2_bundle = std::move(bundle);
    g_mobilenetv2_initialized = true;
}

void fuse_filters_inplace_mbv2(std::vector<std::vector<std::vector<std::vector<uint64_t>>>> &filters) {
    if (g_mobilenetv2_bundle.fused_filters_ready) return;

    const double scale = static_cast<double>(1ULL << kMobileNetV2ScaleBits);
    for (size_t layer_idx = 0; layer_idx < filters.size(); ++layer_idx) {
        auto &layer_filters = filters[layer_idx];
        const auto &bn = g_mobilenetv2_bundle.bn_layers[layer_idx];
        for (size_t oc = 0; oc < layer_filters.size(); ++oc) {
            const double bn_mul = static_cast<double>(bn.mul[oc]);
            for (size_t ic = 0; ic < layer_filters[oc].size(); ++ic) {
                for (size_t k = 0; k < layer_filters[oc][ic].size(); ++k) {
                    const int64_t centered = centered_from_mod_u64_mbv2(layer_filters[oc][ic][k], kMobileNetV2FilterMod);
                    const int64_t fused = static_cast<int64_t>(std::llround((static_cast<double>(centered) * bn_mul) / scale));
                    layer_filters[oc][ic][k] = static_cast<uint64_t>(fused);  // Jaguar_52: signed 64-bit, not mod 2^27
                }
            }
        }
    }
    g_mobilenetv2_bundle.fused_filters_ready = true;
}

} // namespace

void configure_mobilenet_v2_runtime_params(
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> &filters) {
    if (g_mobilenetv2_bundle.fused_filters_ready && !g_mobilenetv2_fused_filters.empty()) {
        filters = g_mobilenetv2_fused_filters;
        return;
    }
    ensure_bundle_loaded(filters);
    if (!runtime_config().USE_FUSED_BN) {
        throw std::runtime_error("[Jaguar][mobilenet_v2] USE_FUSED_BN=0 is not supported");
    }
    fuse_filters_inplace_mbv2(filters);
    g_mobilenetv2_fused_filters = filters;
}

void mobilenet_v2_log_model_summary() {
    if (!g_mobilenetv2_initialized) return;
    std::cerr << "[Jaguar][mobilenet_v2] USE_FUSED_BN=" << (runtime_config().USE_FUSED_BN ? 1 : 0)
              << " pretrained_root=" << g_mobilenetv2_bundle.source
              << " bn_layers=" << g_mobilenetv2_bundle.bn_layers.size()
              << " fc_out_dim=" << g_mobilenetv2_bundle.fc.weight.size()
              << std::endl;
}

void mobilenet_v2_set_quant_active(bool active) {
    g_mobilenetv2_quant_active = active;
}

void mobilenet_v2_apply_post_conv_bias_quant(
    int layer_index,
    uint64_t log2_p,
    std::vector<std::vector<uint64_t>> &ss_r_data) {
    if (!g_mobilenetv2_quant_active) return;
    if (!g_mobilenetv2_initialized) return;
    if (layer_index < 0 || static_cast<size_t>(layer_index) >= g_mobilenetv2_bundle.bn_layers.size()) return;
    if (!runtime_config().USE_FUSED_BN) return;
    if (ss_r_data.empty()) return;

    const auto &bn = g_mobilenetv2_bundle.bn_layers[static_cast<size_t>(layer_index)];
    const size_t out_ch = bn.add.size();
    if (out_ch == 0) return;
    const size_t repeat = ss_r_data.size() / out_ch;
    if (repeat == 0) return;

    const uint64_t mod_p = (1ULL << log2_p);
    const uint64_t mask = mod_p - 1;
    for (size_t oc = 0; oc < out_ch; ++oc) {
        const uint64_t bias_mod = to_mod_u64_mbv2(bn.add[oc], mod_p);
        for (size_t r = 0; r < repeat; ++r) {
            auto &row = ss_r_data[oc * repeat + r];
            for (auto &coeff : row) {
                // Client holds y+r; this is the mask r, so r-b yields y+b.
                coeff = (coeff - bias_mod) & mask;
            }
        }
    }
}

const std::vector<std::vector<uint64_t>> &mobilenet_v2_fc_weight() {
    if (!g_mobilenetv2_initialized) {
        throw std::runtime_error("[Jaguar][mobilenet_v2] FC weights requested before initialization");
    }
    return g_mobilenetv2_bundle.fc.weight;
}

const std::vector<int64_t> &mobilenet_v2_fc_bias() {
    if (!g_mobilenetv2_initialized) {
        throw std::runtime_error("[Jaguar][mobilenet_v2] FC bias requested before initialization");
    }
    return g_mobilenetv2_bundle.fc.bias;
}

} // namespace jaguar_networks
