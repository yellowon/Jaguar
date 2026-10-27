#pragma once

#include <cstdint>
#include <vector>

namespace jaguar {

void configure_runtime(bool enabled, int party, const char *peer_host, int port);

bool external_runtime_enabled();
int configured_party();

void relu_share_field(const std::vector<std::vector<uint64_t>> &client_share,
                      const std::vector<std::vector<uint64_t>> &cloud_mask_share,
                      uint64_t mod_p,
                      std::vector<std::vector<uint64_t>> &client_relu_share,
                      std::vector<std::vector<uint64_t>> &cloud_relu_share);

void relu_share_ring(const std::vector<std::vector<uint64_t>> &client_share,
                     const std::vector<std::vector<uint64_t>> &cloud_mask_share,
                     uint64_t bitwidth,
                     std::vector<std::vector<uint64_t>> &client_relu_share,
                     std::vector<std::vector<uint64_t>> &cloud_relu_share);

void relu_share_ring_local(const std::vector<std::vector<uint64_t>> &local_share,
                           uint64_t bitwidth,
                           std::vector<std::vector<uint64_t>> &local_relu_share);

void relu_share_ring_external(
    const std::vector<std::vector<uint64_t>> &local_share,
    uint64_t bitwidth,
    std::vector<std::vector<uint64_t>> &local_relu_share);

void relu6_share_ring_external(
    const std::vector<std::vector<uint64_t>> &local_share,
    uint64_t bitwidth,
    std::vector<std::vector<uint64_t>> &local_relu6_share);

// Jaguar_52 exact path (default). JAGUAR_RELU_MODE=approx restores the
// original approximate SCI Relu wrapper for comparison only.
bool exact_relu_enabled();
void exact_relu_init();
uint64_t exact_relu_output_bits();
void exact_relu_save_trace();
void ensure_pool_runtime(uint64_t bitwidth);

void add_activation_prelude(uint64_t latency_us, const char *detail = nullptr);
void clear_activation_prelude();

}  // namespace jaguar
