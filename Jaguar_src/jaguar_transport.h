#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "seal/seal.h"
#include "client_lwe_shared.h"

namespace jaguar_quant {

enum Role {
    kMonolithic = 0,
    kClient = 1,
    kCloud = 2,
};

bool enabled();
bool is_client();
bool is_cloud();
int party_id();
const std::string &peer_host();
int base_port();
void configure(Role role, const std::string &peer, int base_port);

void stash_conv_input_transport(uint64_t sent_bytes, uint64_t actual_bytes);
uint64_t consume_pending_conv_input_sent_bytes();
uint64_t consume_pending_conv_input_actual_bytes();

void send_ciphertexts(const std::vector<seal::Ciphertext> &cts);
void recv_ciphertexts(const seal::SEALContext &context, std::vector<seal::Ciphertext> &cts);
void send_lwe_groups(const std::vector<LweSharedGroup> &groups);
void recv_lwe_groups(std::vector<LweSharedGroup> &groups);
void send_matrix_u64(const std::vector<std::vector<uint64_t>> &mat);
void recv_matrix_u64(std::vector<std::vector<uint64_t>> &mat);
void send_matrix_u64_fast(const std::vector<std::vector<uint64_t>> &mat);
void recv_matrix_u64_fast(std::vector<std::vector<uint64_t>> &mat);

} // namespace jaguar_quant
