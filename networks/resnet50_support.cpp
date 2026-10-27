#include "networks.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace jaguar_networks {
namespace {

const char *kResNet50ModelPath = "../pretrained/resnet50_model_scale9.inp";
// Jaguar_52: scale-9 weights x scale-12 BN multiplier / 2^12 -> scale-9 fused
// weights, so conv output scale 18 - f(9) keeps activations at scale 9.
constexpr int kResNet50ScaleBits = 12;
constexpr uint64_t kFilterMod = (1ULL << 27);
constexpr uint64_t kFcMod = (1ULL << 37);  // Jaguar_52: Cheetah FC ring (bitlength 37), not the BFV prime
constexpr int kFcInDim = 2048;
constexpr int kFcOutDim = 1000;

ResNet18ParameterBundle g_resnet50_bundle;
bool g_resnet50_initialized = false;
bool g_resnet50_quant_active = false;
std::vector<std::vector<std::vector<std::vector<uint64_t>>>> g_resnet50_fused_filters;

bool read_next_i64_resnet50(std::ifstream &fin, int64_t &value) {
    long long tmp = 0;
    if (!(fin >> tmp)) return false;
    value = static_cast<int64_t>(tmp);
    return true;
}

uint64_t to_mod_u64_resnet50(int64_t value, uint64_t mod) {
    if (mod == 0) return static_cast<uint64_t>(value);
    int64_t reduced = value % static_cast<int64_t>(mod);
    if (reduced < 0) reduced += static_cast<int64_t>(mod);
    return static_cast<uint64_t>(reduced);
}

int64_t centered_from_mod_u64_resnet50(uint64_t value, uint64_t mod) {
    if (mod == 0) return static_cast<int64_t>(value);
    value %= mod;
    uint64_t half = mod >> 1;
    return (value >= half) ? static_cast<int64_t>(value - mod) : static_cast<int64_t>(value);
}

void ensure_bundle_loaded_resnet50(std::vector<std::vector<std::vector<std::vector<uint64_t>>>> &filters) {
    if (g_resnet50_initialized) return;

    std::ifstream fin(kResNet50ModelPath);
    if (!fin.is_open()) {
        throw std::runtime_error("[Jaguar][resnet50] missing pretrained file: Jaguar/pretrained/resnet50_model_scale9.inp");
    }

    ResNet18ParameterBundle bundle;
    bundle.source = kResNet50ModelPath;
    bundle.bn_layers.resize(filters.size());
    for (size_t layer_idx = 0; layer_idx < filters.size(); ++layer_idx) {
        auto &layer_filters = filters[layer_idx];
        const size_t out_ch = layer_filters.size();
        const size_t in_ch = out_ch ? layer_filters[0].size() : 0;
        const size_t kernel = (in_ch && !layer_filters[0].empty()) ? layer_filters[0][0].size() : 0;

        for (size_t oc = 0; oc < out_ch; ++oc) {
            for (size_t ic = 0; ic < in_ch; ++ic) {
                for (size_t k = 0; k < kernel; ++k) {
                    int64_t value = 0;
                    if (!read_next_i64_resnet50(fin, value)) {
                        throw std::runtime_error("[Jaguar][resnet50] unexpected EOF while reading conv weights from resnet50_model_scale9.inp");
                    }
                    layer_filters[oc][ic][k] = to_mod_u64_resnet50(value, kFilterMod);
                }
            }
        }

        auto &bn = bundle.bn_layers[layer_idx];
        bn.mul.resize(out_ch);
        bn.add.resize(out_ch);
        bn.identity = false;
        bn.loaded = true;
        for (size_t oc = 0; oc < out_ch; ++oc) {
            if (!read_next_i64_resnet50(fin, bn.mul[oc])) {
                throw std::runtime_error("[Jaguar][resnet50] unexpected EOF while reading BN scale parameters");
            }
        }
        for (size_t oc = 0; oc < out_ch; ++oc) {
            if (!read_next_i64_resnet50(fin, bn.add[oc])) {
                throw std::runtime_error("[Jaguar][resnet50] unexpected EOF while reading BN bias parameters");
            }
        }
    }

    bundle.fc.weight.assign(kFcOutDim, std::vector<uint64_t>(kFcInDim, 0));
    bundle.fc.bias.assign(kFcOutDim, 0);
    bundle.fc.loaded = true;

    for (int in_idx = 0; in_idx < kFcInDim; ++in_idx) {
        for (int out_idx = 0; out_idx < kFcOutDim; ++out_idx) {
            int64_t value = 0;
            if (!read_next_i64_resnet50(fin, value)) {
                throw std::runtime_error("[Jaguar][resnet50] unexpected EOF while reading FC weights");
            }
            bundle.fc.weight[out_idx][in_idx] = to_mod_u64_resnet50(value, kFcMod);
        }
    }
    for (int out_idx = 0; out_idx < kFcOutDim; ++out_idx) {
        if (!read_next_i64_resnet50(fin, bundle.fc.bias[out_idx])) {
            throw std::runtime_error("[Jaguar][resnet50] unexpected EOF while reading FC bias");
        }
    }

    bundle.loaded = true;
    g_resnet50_bundle = std::move(bundle);
    g_resnet50_initialized = true;
}

void fuse_filters_inplace_resnet50(std::vector<std::vector<std::vector<std::vector<uint64_t>>>> &filters) {
    if (g_resnet50_bundle.fused_filters_ready) return;

    const double scale = static_cast<double>(1ULL << kResNet50ScaleBits);
    for (size_t layer_idx = 0; layer_idx < filters.size(); ++layer_idx) {
        auto &layer_filters = filters[layer_idx];
        const auto &bn = g_resnet50_bundle.bn_layers[layer_idx];
        for (size_t oc = 0; oc < layer_filters.size(); ++oc) {
            const double bn_mul = static_cast<double>(bn.mul[oc]);
            for (size_t ic = 0; ic < layer_filters[oc].size(); ++ic) {
                for (size_t k = 0; k < layer_filters[oc][ic].size(); ++k) {
                    const int64_t centered = centered_from_mod_u64_resnet50(layer_filters[oc][ic][k], kFilterMod);
                    const int64_t fused = static_cast<int64_t>(std::llround((static_cast<double>(centered) * bn_mul) / scale));
                    layer_filters[oc][ic][k] = static_cast<uint64_t>(fused);  // Jaguar_52: signed 64-bit, not mod 2^27
                }
            }
        }
    }
    g_resnet50_bundle.fused_filters_ready = true;
}

} // namespace

void configure_resnet50_runtime_params(
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> &filters) {
    if (g_resnet50_bundle.fused_filters_ready && !g_resnet50_fused_filters.empty()) {
        filters = g_resnet50_fused_filters;
        return;
    }

    ensure_bundle_loaded_resnet50(filters);
    if (!runtime_config().USE_FUSED_BN) {
        throw std::runtime_error(
            "[Jaguar][resnet50] USE_FUSED_BN=0 is parsed but standalone BN execution is not implemented in the ciphertext path yet");
    }
    fuse_filters_inplace_resnet50(filters);
    g_resnet50_fused_filters = filters;
}
void resnet50_log_model_summary() {
    if (!g_resnet50_initialized) return;
    std::cerr << "[Jaguar][resnet50] USE_FUSED_BN=" << (runtime_config().USE_FUSED_BN ? 1 : 0)
              << " pretrained_root=" << g_resnet50_bundle.source
              << " bn_layers=" << g_resnet50_bundle.bn_layers.size()
              << " fc_out_dim=" << g_resnet50_bundle.fc.weight.size()
              << std::endl;
}

void resnet50_set_quant_active(bool active) {
    g_resnet50_quant_active = active;
}

void resnet50_apply_post_conv_bias_quant(
    int layer_index,
    uint64_t log2_p,
    std::vector<std::vector<uint64_t>> &ss_r_data) {
    if (!g_resnet50_quant_active) return;
    if (!g_resnet50_initialized) return;
    if (layer_index < 0 || static_cast<size_t>(layer_index) >= g_resnet50_bundle.bn_layers.size()) return;
    if (!runtime_config().USE_FUSED_BN) return;
    if (ss_r_data.empty()) return;

    const auto &bn = g_resnet50_bundle.bn_layers[static_cast<size_t>(layer_index)];
    if (bn.add.empty()) return;

    auto bn_start = std::chrono::high_resolution_clock::now();
    const uint64_t mod_p = (1ULL << log2_p);
    const uint64_t mask = mod_p - 1;
    const size_t out_ch = bn.add.size();
    const size_t repeat = ss_r_data.size() / out_ch;
    if (repeat == 0) return;

    for (size_t oc = 0; oc < out_ch; ++oc) {
        const uint64_t bias_mod = to_mod_u64_resnet50(bn.add[oc], mod_p);
        for (size_t r = 0; r < repeat; ++r) {
            auto &row = ss_r_data[oc * repeat + r];
            for (auto &coeff : row) {
                // Client holds y+r; this is the mask r, so r-b yields y+b.
                coeff = (coeff - bias_mod) & mask;
            }
        }
    }
    auto bn_end = std::chrono::high_resolution_clock::now();

    jaguar::LayerSummary layer;
    layer.kind = "batchnorm";
    layer.name = "resnet50_bn_" + std::to_string(layer_index);
    layer.total_latency_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(bn_end - bn_start).count());
    layer.compute_latency_us = layer.total_latency_us;
    layer.comm_bytes = 0;
    layer.rounds = 0;
    layer.details = "layer=" + std::to_string(layer_index) +
                    ",out_ch=" + std::to_string(out_ch) +
                    ",repeat=" + std::to_string(repeat) +
                    ",USE_FUSED_BN=1";
    jaguar::profiler().record_layer_summary(layer);
}

const std::vector<std::vector<uint64_t>> &resnet50_fc_weight() {
    if (!g_resnet50_initialized) {
        throw std::runtime_error("[Jaguar][resnet50] FC weights requested before initialization");
    }
    return g_resnet50_bundle.fc.weight;
}

const std::vector<int64_t> &resnet50_fc_bias() {
    if (!g_resnet50_initialized) {
        throw std::runtime_error("[Jaguar][resnet50] FC bias requested before initialization");
    }
    return g_resnet50_bundle.fc.bias;
}

} // namespace jaguar_networks
