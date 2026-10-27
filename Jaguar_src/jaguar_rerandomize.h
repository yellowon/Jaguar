#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include "seal/seal.h"

namespace jaguar {

// Public-key Enc(0) in R_{2^Q}; never retains the client's secret key.
// Precompute once before workers start. Concurrent consumption is supported;
// precomputation/reconfiguration must not overlap consumption.
class OutputRerandomizer {
public:
    static seal::Ciphertext create_public_key(
        const seal::SEALContext &context, const seal::Plaintext &secret);
    OutputRerandomizer(const seal::SEALContext &context,
                       const seal::Ciphertext &public_key);
    ~OutputRerandomizer();
    OutputRerandomizer(const OutputRerandomizer &) = delete;
    OutputRerandomizer &operator=(const OutputRerandomizer &) = delete;

    // Offline: generate a private, key-bound, one-use pool. No refills.
    void precompute(size_t count, size_t threads = 1);
    size_t prepared_count() const;
    size_t remaining_count() const;
    size_t pool_bytes() const;

    // Online: consume one prepared Enc(0), add, reduce, and erase its slot.
    // No random sampling, NTT, CRT, allocation, or refill in this method.
    void add_encryption_of_zero(seal::Ciphertext &ciphertext) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Configure once, before starting any convolution workers. Missing setup fails
// closed instead of exporting a deterministic, weight-dependent c1.
void configure_output_rerandomization(const seal::SEALContext &context,
                                     const seal::Ciphertext &public_key);
void precompute_output_rerandomization(size_t count, size_t threads = 1);
size_t output_rerandomization_prepared_count();
size_t output_rerandomization_remaining_count();
size_t output_rerandomization_pool_bytes();
// Exact budgets for the three shipped 224x224, N=2048, fused-BN graphs.
// A graph/packing change must update this manifest and pass end-to-end checks.
size_t output_rerandomization_budget(const std::string &model, size_t n,
                                    size_t input_width, bool fused_bn);
void rerandomize_conv_output(seal::Ciphertext &ciphertext);
uint64_t rerandomized_output_count();

} // namespace jaguar
