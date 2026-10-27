#include "networks.h"

#include <stdexcept>

namespace jaguar_networks {
namespace {
RuntimeConfig g_runtime_config;
} // namespace

RuntimeConfig &runtime_config() { return g_runtime_config; }

void set_bn_mode_from_string(const std::string &mode) {
    if (mode == "fused") {
        g_runtime_config.bn_mode = BNMode::kFused;
        g_runtime_config.USE_FUSED_BN = true;
        return;
    }
    if (mode == "standalone") {
        g_runtime_config.bn_mode = BNMode::kStandalone;
        g_runtime_config.USE_FUSED_BN = false;
        return;
    }
    throw std::runtime_error("[Jaguar] invalid --jaguar-bn-mode, expected fused|standalone");
}

const char *bn_mode_name(BNMode mode) {
    switch (mode) {
    case BNMode::kFused:
        return "fused";
    case BNMode::kStandalone:
        return "standalone";
    }
    return "unknown";
}

} // namespace jaguar_networks
