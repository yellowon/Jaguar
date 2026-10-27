#include "jaguar_transport.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "jaguar_profiler.h"

namespace jaguar_quant {

struct Runtime {
    Role role = kMonolithic;
    std::string peer = "127.0.0.1";
    int base_port = 0;
    int data_fd = -1;
};

static Runtime &runtime() {
    static Runtime rt;
    return rt;
}

static uint64_t &pending_conv_input_sent_bytes() {
    static uint64_t value = 0;
    return value;
}

static uint64_t &pending_conv_input_actual_bytes() {
    static uint64_t value = 0;
    return value;
}

void stash_conv_input_transport(uint64_t sent_bytes, uint64_t actual_bytes) {
    pending_conv_input_sent_bytes() += sent_bytes;
    pending_conv_input_actual_bytes() += actual_bytes;
}

uint64_t consume_pending_conv_input_sent_bytes() {
    uint64_t value = pending_conv_input_sent_bytes();
    pending_conv_input_sent_bytes() = 0;
    return value;
}

uint64_t consume_pending_conv_input_actual_bytes() {
    uint64_t value = pending_conv_input_actual_bytes();
    pending_conv_input_actual_bytes() = 0;
    return value;
}

bool enabled() { return runtime().role != kMonolithic; }
bool is_client() { return runtime().role == kClient; }
bool is_cloud() { return runtime().role == kCloud; }
int party_id() { return runtime().role; }
const std::string &peer_host() { return runtime().peer; }
int base_port() { return runtime().base_port; }

static void write_all(int fd, const void *buf, size_t len) {
    const char *ptr = static_cast<const char *>(buf);
    const size_t original_len = len;
    while (len > 0) {
        ssize_t written = ::send(fd, ptr, len, 0);
        if (written <= 0) {
            throw std::runtime_error("failed to send socket payload");
        }
        ptr += written;
        len -= static_cast<size_t>(written);
    }
    jaguar::profiler().add_transport_bytes(static_cast<uint64_t>(original_len), 0);
}

static void read_all(int fd, void *buf, size_t len) {
    char *ptr = static_cast<char *>(buf);
    const size_t original_len = len;
    while (len > 0) {
        ssize_t read_bytes = ::recv(fd, ptr, len, MSG_WAITALL);
        if (read_bytes <= 0) {
            throw std::runtime_error("failed to recv socket payload");
        }
        ptr += read_bytes;
        len -= static_cast<size_t>(read_bytes);
    }
    jaguar::profiler().add_transport_bytes(0, static_cast<uint64_t>(original_len));
}

static void send_u64(uint64_t value) { write_all(runtime().data_fd, &value, sizeof(value)); }
static uint64_t recv_u64() {
    uint64_t value = 0;
    read_all(runtime().data_fd, &value, sizeof(value));
    return value;
}

static void send_bytes(const std::string &blob) {
    send_u64(static_cast<uint64_t>(blob.size()));
    if (!blob.empty()) write_all(runtime().data_fd, blob.data(), blob.size());
}

static std::string recv_bytes() {
    const uint64_t size = recv_u64();
    std::string blob(size, '\0');
    if (size > 0) read_all(runtime().data_fd, &blob[0], static_cast<size_t>(size));
    return blob;
}

static void ensure_data_channel() {
    if (!enabled() || runtime().data_fd >= 0) return;
    const int port = runtime().base_port + 100;
    if (is_client()) {
        int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0) throw std::runtime_error("failed to create data listen socket");
        int opt = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);
        if (bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
            throw std::runtime_error("failed to bind data socket");
        }
        if (listen(listen_fd, 1) < 0) {
            throw std::runtime_error("failed to listen on data socket");
        }
        runtime().data_fd = accept(listen_fd, nullptr, nullptr);
        ::close(listen_fd);
        if (runtime().data_fd < 0) throw std::runtime_error("failed to accept data socket");
    } else {
        runtime().data_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (runtime().data_fd < 0) throw std::runtime_error("failed to create data socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (::inet_pton(AF_INET, runtime().peer.c_str(), &addr.sin_addr) != 1) {
            throw std::runtime_error("failed to parse peer ip");
        }
        for (int attempt = 0; attempt < 200; ++attempt) {
            if (connect(runtime().data_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        throw std::runtime_error("failed to connect data socket");
    }
}

void configure(Role role, const std::string &peer, int base_port) {
    runtime().role = role;
    runtime().peer = peer;
    runtime().base_port = base_port;
    runtime().data_fd = -1;
}

static uint64_t required_bit_width(uint64_t max_value) {
    if (max_value == 0) {
        return 1;
    }
    return static_cast<uint64_t>(64 - __builtin_clzll(max_value));
}

static std::vector<unsigned char> pack_u64_words(const uint64_t *data, size_t count, uint64_t bit_width) {
    if (count == 0) {
        return {};
    }
    if (bit_width >= 64) {
        const auto *src = reinterpret_cast<const unsigned char *>(data);
        return std::vector<unsigned char>(src, src + count * sizeof(uint64_t));
    }
    const size_t total_bytes = static_cast<size_t>((count * bit_width + 7) / 8);
    std::vector<unsigned char> packed;
    packed.reserve(total_bytes);
    unsigned __int128 acc = 0;
    int acc_bits = 0;
    const uint64_t mask = (1ULL << bit_width) - 1;
    for (size_t i = 0; i < count; ++i) {
        acc |= (static_cast<unsigned __int128>(data[i] & mask) << acc_bits);
        acc_bits += static_cast<int>(bit_width);
        while (acc_bits >= 8) {
            packed.push_back(static_cast<unsigned char>(acc & 0xFF));
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    if (acc_bits > 0) {
        packed.push_back(static_cast<unsigned char>(acc & 0xFF));
    }
    return packed;
}

static void unpack_u64_words(const unsigned char *packed, size_t packed_size, size_t count, uint64_t bit_width, uint64_t *out) {
    if (count == 0) {
        return;
    }
    if (bit_width >= 64) {
        std::memcpy(out, packed, std::min(packed_size, count * sizeof(uint64_t)));
        return;
    }
    unsigned __int128 acc = 0;
    int acc_bits = 0;
    size_t offset = 0;
    const uint64_t mask = (1ULL << bit_width) - 1;
    for (size_t i = 0; i < count; ++i) {
        while (acc_bits < static_cast<int>(bit_width) && offset < packed_size) {
            acc |= (static_cast<unsigned __int128>(packed[offset]) << acc_bits);
            acc_bits += 8;
            ++offset;
        }
        out[i] = static_cast<uint64_t>(acc) & mask;
        acc >>= bit_width;
        acc_bits -= static_cast<int>(bit_width);
    }
}

static uint64_t ciphertext_max_word(const seal::Ciphertext &ct) {
    const uint64_t words = static_cast<uint64_t>(ct.size()) * ct.coeff_modulus_size() * ct.poly_modulus_degree();
    uint64_t max_word = 0;
    for (uint64_t i = 0; i < words; ++i) {
        max_word = std::max(max_word, ct.data()[i]);
    }
    return max_word;
}

void send_ciphertexts(const std::vector<seal::Ciphertext> &cts) {
    ensure_data_channel();
    std::cerr << "[Jaguar][quant-io] send_ciphertexts count=" << cts.size() << std::endl;
    send_u64(static_cast<uint64_t>(cts.size()));
    for (size_t idx = 0; idx < cts.size(); ++idx) {
        const auto &ct = cts[idx];
        const auto data_words =
            static_cast<uint64_t>(ct.size()) * ct.coeff_modulus_size() * ct.poly_modulus_degree();
        send_u64(static_cast<uint64_t>(ct.size()));
        send_u64(static_cast<uint64_t>(ct.coeff_modulus_size()));
        send_u64(static_cast<uint64_t>(ct.poly_modulus_degree()));
        send_u64(static_cast<uint64_t>(ct.is_ntt_form()));
        send_u64(static_cast<uint64_t>(ct.correction_factor()));
        for (size_t limb = 0; limb < ct.parms_id().size(); ++limb) {
            send_u64(ct.parms_id()[limb]);
        }
        uint64_t bit_width = 1;
        if (data_words > 0) {
            bit_width = required_bit_width(ciphertext_max_word(ct));
        }
        send_u64(bit_width);
        std::cerr << "[Jaguar][quant-io] send_ciphertext idx=" << idx
                  << " size=" << ct.size()
                  << " coeff_modulus_size=" << ct.coeff_modulus_size()
                  << " poly_modulus_degree=" << ct.poly_modulus_degree()
                  << " words=" << data_words
                  << " bit_width=" << bit_width << std::endl;
        if (data_words > 0) {
            const auto packed = pack_u64_words(ct.data(), static_cast<size_t>(data_words), bit_width);
            send_u64(static_cast<uint64_t>(packed.size()));
            write_all(runtime().data_fd, packed.data(), packed.size());
        } else {
            send_u64(0);
        }
    }
}

void recv_ciphertexts(const seal::SEALContext &context, std::vector<seal::Ciphertext> &cts) {
    ensure_data_channel();
    const uint64_t count = recv_u64();
    std::cerr << "[Jaguar][quant-io] recv_ciphertexts count=" << count << std::endl;
    cts.assign(count, seal::Ciphertext());
    for (uint64_t i = 0; i < count; ++i) {
        const auto size = static_cast<size_t>(recv_u64());
        const auto coeff_modulus_size = static_cast<size_t>(recv_u64());
        const auto poly_modulus_degree = static_cast<size_t>(recv_u64());
        const bool is_ntt_form = recv_u64() != 0;
        const auto correction_factor = recv_u64();
        seal::parms_id_type parms_id{};
        for (size_t limb = 0; limb < parms_id.size(); ++limb) {
            parms_id[limb] = recv_u64();
        }
        const auto bit_width = recv_u64();
        const auto packed_size = static_cast<size_t>(recv_u64());
        (void)parms_id;
        cts[i].resize(context, context.first_parms_id(), size);
        cts[i].is_ntt_form() = is_ntt_form;
        cts[i].correction_factor() = correction_factor;
        const auto data_words = size * coeff_modulus_size * poly_modulus_degree;
        std::cerr << "[Jaguar][quant-io] recv_ciphertext idx=" << i
                  << " size=" << size
                  << " coeff_modulus_size=" << coeff_modulus_size
                  << " poly_modulus_degree=" << poly_modulus_degree
                  << " words=" << data_words
                  << " bit_width=" << bit_width << std::endl;
        if (data_words > 0) {
            std::vector<unsigned char> packed(packed_size);
            read_all(runtime().data_fd, packed.data(), packed.size());
            unpack_u64_words(packed.data(), packed.size(), static_cast<size_t>(data_words), bit_width, cts[i].data());
        }
    }
}

void send_lwe_groups(const std::vector<LweSharedGroup> &groups) {
    ensure_data_channel();
    send_u64(static_cast<uint64_t>(groups.size()));
    for (const auto &group : groups) {
        send_u64(static_cast<uint64_t>(group.src_ct_index));
        uint64_t max_a = 0;
        for (uint64_t value : group.a_shared) max_a = std::max(max_a, value);
        uint64_t a_bit_width = group.a_shared_bit_width ? group.a_shared_bit_width : required_bit_width(max_a);
        send_u64(a_bit_width);

        send_u64(static_cast<uint64_t>(group.a_shared.size()));
        const auto packed_a = pack_u64_words(group.a_shared.data(), group.a_shared.size(), a_bit_width);
        send_bytes(std::string(reinterpret_cast<const char *>(packed_a.data()), packed_a.size()));

        uint64_t max_b = 0;
        for (uint64_t value : group.b_values) max_b = std::max(max_b, value);
        uint64_t b_bit_width = group.b_values_bit_width ? group.b_values_bit_width : required_bit_width(max_b);
        send_u64(b_bit_width);

        send_u64(static_cast<uint64_t>(group.b_values.size()));
        const auto packed_b = pack_u64_words(group.b_values.data(), group.b_values.size(), b_bit_width);
        send_bytes(std::string(reinterpret_cast<const char *>(packed_b.data()), packed_b.size()));

        send_u64(static_cast<uint64_t>(group.coeff_indices.size()));
        if (!group.coeff_indices.empty()) {
            write_all(runtime().data_fd, group.coeff_indices.data(), group.coeff_indices.size() * sizeof(int));
        }

        send_u64(static_cast<uint64_t>(group.out_channels.size()));
        if (!group.out_channels.empty()) {
            write_all(runtime().data_fd, group.out_channels.data(), group.out_channels.size() * sizeof(int));
        }

        send_u64(static_cast<uint64_t>(group.out_offsets.size()));
        if (!group.out_offsets.empty()) {
            write_all(runtime().data_fd, group.out_offsets.data(), group.out_offsets.size() * sizeof(int));
        }
    }
}

void recv_lwe_groups(std::vector<LweSharedGroup> &groups) {
    ensure_data_channel();
    const uint64_t group_count = recv_u64();
    groups.assign(static_cast<size_t>(group_count), LweSharedGroup{});
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        auto &group = groups[gi];
        group.src_ct_index = static_cast<int>(recv_u64());
        group.a_shared_bit_width = recv_u64();

        const size_t a_size = static_cast<size_t>(recv_u64());
        const auto packed_a = recv_bytes();
        group.a_shared.assign(a_size, 0);
        if (a_size > 0) {
            unpack_u64_words(reinterpret_cast<const unsigned char *>(packed_a.data()), packed_a.size(), a_size, group.a_shared_bit_width, group.a_shared.data());
        }

        group.b_values_bit_width = recv_u64();

        const size_t b_size = static_cast<size_t>(recv_u64());
        const auto packed_b = recv_bytes();
        group.b_values.assign(b_size, 0);
        if (b_size > 0) {
            unpack_u64_words(reinterpret_cast<const unsigned char *>(packed_b.data()), packed_b.size(), b_size, group.b_values_bit_width, group.b_values.data());
        }

        const size_t coeff_indices_size = static_cast<size_t>(recv_u64());
        group.coeff_indices.assign(coeff_indices_size, 0);
        if (coeff_indices_size > 0) {
            read_all(runtime().data_fd, group.coeff_indices.data(), coeff_indices_size * sizeof(int));
        }

        const size_t out_channels_size = static_cast<size_t>(recv_u64());
        group.out_channels.assign(out_channels_size, 0);
        if (out_channels_size > 0) {
            read_all(runtime().data_fd, group.out_channels.data(), out_channels_size * sizeof(int));
        }

        const size_t out_offsets_size = static_cast<size_t>(recv_u64());
        group.out_offsets.assign(out_offsets_size, 0);
        if (out_offsets_size > 0) {
            read_all(runtime().data_fd, group.out_offsets.data(), out_offsets_size * sizeof(int));
        }
    }
}

void send_matrix_u64(const std::vector<std::vector<uint64_t>> &mat) {
    ensure_data_channel();
    uint64_t max_value = 0;
    for (const auto &row : mat) {
        for (uint64_t value : row) {
            max_value = std::max(max_value, value);
        }
    }
    const uint64_t bit_width = required_bit_width(max_value);
    send_u64(static_cast<uint64_t>(mat.size()));
    send_u64(bit_width);
    for (const auto &row : mat) {
        send_u64(static_cast<uint64_t>(row.size()));
        if (row.empty()) {
            send_u64(0);
            continue;
        }
        const auto packed = pack_u64_words(row.data(), row.size(), bit_width);
        send_u64(static_cast<uint64_t>(packed.size()));
        write_all(runtime().data_fd, packed.data(), packed.size());
    }
}

void send_matrix_u64_fast(const std::vector<std::vector<uint64_t>> &mat) {
    ensure_data_channel();
    const uint64_t rows = static_cast<uint64_t>(mat.size());
    const uint64_t cols = rows == 0 ? 0 : static_cast<uint64_t>(mat[0].size());
    uint64_t max_value = 0;
    for (const auto &row : mat) {
        for (uint64_t value : row) {
            max_value = std::max(max_value, value);
        }
    }
    const uint64_t bit_width = required_bit_width(max_value);
    std::vector<uint64_t> flat;
    flat.reserve(static_cast<size_t>(rows * cols));
    for (const auto &row : mat) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    const auto packed = flat.empty() ? std::vector<unsigned char>{} : pack_u64_words(flat.data(), flat.size(), bit_width);
    send_u64(rows);
    send_u64(cols);
    send_u64(bit_width);
    send_u64(static_cast<uint64_t>(packed.size()));
    if (!packed.empty()) {
        write_all(runtime().data_fd, packed.data(), packed.size());
    }
}

void recv_matrix_u64_fast(std::vector<std::vector<uint64_t>> &mat) {
    ensure_data_channel();
    const uint64_t rows = recv_u64();
    const uint64_t cols = recv_u64();
    const uint64_t bit_width = recv_u64();
    const uint64_t packed_size = recv_u64();
    mat.assign(rows, std::vector<uint64_t>(cols, 0));
    if (rows == 0 || cols == 0) return;
    std::vector<unsigned char> packed(static_cast<size_t>(packed_size));
    if (!packed.empty()) {
        read_all(runtime().data_fd, packed.data(), packed.size());
    }
    std::vector<uint64_t> flat(static_cast<size_t>(rows * cols), 0);
    unpack_u64_words(packed.data(), packed.size(), flat.size(), bit_width, flat.data());
    size_t idx = 0;
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) {
            mat[i][j] = flat[idx++];
        }
    }
}

void recv_matrix_u64(std::vector<std::vector<uint64_t>> &mat) {
    ensure_data_channel();
    const uint64_t rows = recv_u64();
    const uint64_t bit_width = recv_u64();
    mat.assign(rows, {});
    for (uint64_t i = 0; i < rows; ++i) {
        const uint64_t cols = recv_u64();
        const uint64_t packed_size = recv_u64();
        mat[i].resize(cols);
        if (cols == 0) continue;
        std::vector<unsigned char> packed(static_cast<size_t>(packed_size));
        read_all(runtime().data_fd, packed.data(), packed.size());
        unpack_u64_words(packed.data(), packed.size(), static_cast<size_t>(cols), bit_width, mat[i].data());
    }
}

} // namespace jaguar_quant
