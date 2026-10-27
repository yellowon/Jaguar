#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "seal/seal.h"

#include "../networks/networks.h"

inline void dispatch_model_2048(const std::string &model_name,
                                bool isMultiThread,
                                const seal::SEALContext &context_quant,
                                uint64_t log2_q,
                                uint64_t log2_p,
                                const seal::Plaintext &secret_key_coeff,
                                std::vector<std::vector<uint64_t>> &image,
                                int &w,
                                const seal::SEALContext &context_ntt,
                                const seal::PublicKey &public_key,
                                const seal::SecretKey &secret_key,
                                std::vector<seal::Ciphertext> &ct_out1,
                                int discard_lsb_model_output) {
    using namespace seal;

    if (model_name == "resnet18") {
        if (isMultiThread)
            resnet18_quant<2048>("mt", context_quant, log2_q, log2_p, secret_key_coeff, image, w, context_ntt, public_key, secret_key, ct_out1, discard_lsb_model_output);
        else
            resnet18_quant<2048>("st", context_quant, log2_q, log2_p, secret_key_coeff, image, w, context_ntt, public_key, secret_key, ct_out1, discard_lsb_model_output);
        return;
    }
    if (model_name == "resnet50") {
        if (isMultiThread)
            resnet50_quant<2048>("mt", context_quant, log2_q, log2_p, secret_key_coeff, image, w, context_ntt, public_key, secret_key, ct_out1, discard_lsb_model_output);
        else
            resnet50_quant<2048>("st", context_quant, log2_q, log2_p, secret_key_coeff, image, w, context_ntt, public_key, secret_key, ct_out1, discard_lsb_model_output);
        return;
    }
    if (model_name == "mobilenet_v2") {
        if (isMultiThread)
            mobilenet_v2_quant<2048>("mt", context_quant, log2_q, log2_p, secret_key_coeff, image, w, context_ntt, public_key, secret_key, ct_out1, discard_lsb_model_output);
        else
            mobilenet_v2_quant<2048>("st", context_quant, log2_q, log2_p, secret_key_coeff, image, w, context_ntt, public_key, secret_key, ct_out1, discard_lsb_model_output);
        return;
    }
    throw std::runtime_error("unsupported model: " + model_name);
}
