#include "jaguar_rerandomize.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <stdexcept>
#include <mutex>
#include <thread>
#include <vector>

#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "seal/util/uintarithsmallmod.h"

namespace jaguar {
namespace {
using U128 = unsigned __int128;

seal::EncryptionParameters checked_parameters(const seal::SEALContext &context) {
    const auto data = context.first_context_data();
    if (!data) throw std::invalid_argument("rerandomization: invalid context");
    const auto &p = data->parms();
    if (p.scheme() != seal::scheme_type::bfv || p.coeff_modulus().size() != 1)
        throw std::invalid_argument("rerandomization: expected single-modulus BFV");
    const auto q = p.coeff_modulus()[0].value();
    const auto n = p.poly_modulus_degree();
    if (q < 4 || (q & (q - 1)) || q > (uint64_t(1) << 60) ||
        n < 2 || (n & (n - 1)))
        throw std::invalid_argument("rerandomization: expected power-of-two ring");
    return p;
}

void check_ciphertext(const seal::Ciphertext &ct, const seal::EncryptionParameters &p,
                      const seal::parms_id_type &id, bool require_canonical = true) {
    if (ct.size() != 2 || ct.is_ntt_form() || ct.coeff_modulus_size() != 1 ||
        ct.poly_modulus_degree() != p.poly_modulus_degree() || ct.parms_id() != id)
        throw std::invalid_argument("rerandomization: ciphertext/context mismatch");
    if (!require_canonical) return;
    const uint64_t q = p.coeff_modulus()[0].value();
    for (size_t j = 0; j < 2; ++j)
        for (size_t i = 0; i < p.poly_modulus_degree(); ++i)
            if (ct.data(j)[i] >= q)
                throw std::invalid_argument("rerandomization: noncanonical coefficient");
}

std::unique_ptr<OutputRerandomizer> output_rerandomizer;
std::atomic<uint64_t> output_count{0};
} // namespace

seal::Ciphertext OutputRerandomizer::create_public_key(
    const seal::SEALContext &context, const seal::Plaintext &secret) {
    const auto p = checked_parameters(context);
    const size_t n = p.poly_modulus_degree();
    const uint64_t mask = p.coeff_modulus()[0].value() - 1;
    if (secret.coeff_count() != n || secret.is_ntt_form())
        throw std::invalid_argument("rerandomization: expected coefficient-form secret");
    for (size_t i = 0; i < n; ++i)
        if (secret[i] != 0 && secret[i] != 1 && secret[i] != mask)
            throw std::invalid_argument("rerandomization: expected ternary secret");

    seal::Ciphertext pk;
    pk.resize(context, context.first_parms_id(), 2);
    pk.is_ntt_form() = false;
    pk.scale() = 1.0;
    pk.correction_factor() = 1;
    auto rng = seal::UniformRandomGeneratorFactory::DefaultFactory()->create();
    rng->generate(n * sizeof(uint64_t), reinterpret_cast<seal::seal_byte *>(pk.data(1)));
    for (size_t i = 0; i < n; ++i) pk.data(1)[i] &= mask;
    std::vector<uint64_t> noise(n), product(n, 0);
    seal::util::SEAL_NOISE_SAMPLER(rng, p, noise.data());

    // One-time exact negacyclic a*s in Z_{2^Q}. Unsigned overflow is safe:
    // reduction mod 2^64 followed by a mask equals reduction mod 2^Q.
    for (size_t k = 0; k < n; ++k) {
        if (!secret[k]) continue;
        const bool negative = secret[k] == mask;
        for (size_t i = 0; i < n; ++i) {
            const size_t index = (i + k) & (n - 1);
            const bool subtract = negative != (i + k >= n);
            if (subtract) product[index] -= pk.data(1)[i];
            else product[index] += pk.data(1)[i];
        }
    }
    for (size_t i = 0; i < n; ++i)
        pk.data(0)[i] = (noise[i] - product[i]) & mask;
    return pk;
}

struct OutputRerandomizer::Impl {
    seal::EncryptionParameters parameters;
    seal::parms_id_type parms_id;
    size_t n;
    uint64_t mask;
    std::vector<seal::Modulus> primes;
    seal::util::Pointer<seal::util::NTTTables> tables;
    std::array<std::array<std::vector<uint64_t>, 2>, 2> pk_ntt;
    uint64_t inverse_p0_mod_p1;
    U128 prime_product;
    std::vector<uint64_t> zeros;
    std::atomic<size_t> next{0};
    size_t count = 0;
    bool preparation_started = false;
    bool ready = false;
    void generate_zero(uint64_t *destination) const;

    Impl(const seal::SEALContext &context, const seal::Ciphertext &pk)
        : parameters(checked_parameters(context)), parms_id(context.first_parms_id()),
          n(parameters.poly_modulus_degree()), mask(parameters.coeff_modulus()[0].value() - 1),
          primes(seal::CoeffModulus::Create(n, {60, 60})) {
        check_ciphertext(pk, parameters, parms_id);
        int power = 0;
        for (size_t v = n; v > 1; v >>= 1) ++power;
        seal::util::CreateNTTTables(power, primes, tables, seal::MemoryManager::GetPool());
        prime_product = U128(primes[0].value()) * primes[1].value();
        // pk coefficients are in [0,q), u in {-1,0,1}: |pk*u| <= N*(q-1).
        if (prime_product / 2 <= U128(n) * mask ||
            !seal::util::try_invert_uint_mod(
                primes[0].value() % primes[1].value(), primes[1], inverse_p0_mod_p1))
            throw std::invalid_argument("rerandomization: insufficient CRT range");
        for (size_t r = 0; r < 2; ++r) {
            for (size_t j = 0; j < 2; ++j) {
                auto &v = pk_ntt[r][j];
                v.resize(n);
                for (size_t i = 0; i < n; ++i) v[i] = pk.data(j)[i] % primes[r].value();
                seal::util::ntt_negacyclic_harvey(v.data(), tables[r]);
            }
        }
    }

    uint64_t reconstruct(uint64_t r0, uint64_t r1) const {
        const uint64_t p0 = primes[0].value(), p1 = primes[1].value();
        const uint64_t difference = (r1 + p1 - r0 % p1) % p1;
        const uint64_t t = static_cast<uint64_t>(U128(difference) * inverse_p0_mod_p1 % p1);
        U128 value = r0 + U128(p0) * t;
        // Center before reducing to 2^Q; a nonnegative CRT lift would be wrong
        // for negative integer convolution coefficients.
        if (value > prime_product / 2) value -= prime_product;
        return static_cast<uint64_t>(value) & mask;
    }
};

OutputRerandomizer::OutputRerandomizer(
    const seal::SEALContext &context, const seal::Ciphertext &pk)
    : impl_(new Impl(context, pk)) {}
OutputRerandomizer::~OutputRerandomizer() = default;

void OutputRerandomizer::Impl::generate_zero(uint64_t *destination) const {
    const auto &s = *this;
    // Fresh cryptographic randomness per ciphertext, including under threads.
    auto rng = seal::UniformRandomGeneratorFactory::DefaultFactory()->create();
    std::vector<uint64_t> u(s.n), u_ntt(s.n), work(s.n), noise(s.n);
    std::array<std::vector<uint64_t>, 2> residues;
    for (auto &r : residues) r.resize(s.n);
    seal::util::sample_poly_ternary(rng, s.parameters, u.data());
    std::array<std::vector<uint64_t>, 2> transformed_u;
    for (size_t r = 0; r < 2; ++r) {
        for (size_t i = 0; i < s.n; ++i)
            u_ntt[i] = u[i] == s.mask ? s.primes[r].value() - 1 : u[i];
        seal::util::ntt_negacyclic_harvey(u_ntt.data(), s.tables[r]);
        transformed_u[r] = u_ntt;
    }
    for (size_t j = 0; j < 2; ++j) {
        for (size_t r = 0; r < 2; ++r) {
            seal::util::dyadic_product_coeffmod(transformed_u[r].data(),
                s.pk_ntt[r][j].data(), s.n, s.primes[r], work.data());
            seal::util::inverse_ntt_negacyclic_harvey(work.data(), s.tables[r]);
            residues[r] = work;
        }
        // Keep e0 as well as e1; the plaintext-share mask is not a substitute
        // for encryption noise in this exact-truncation path.
        seal::util::SEAL_NOISE_SAMPLER(rng, s.parameters, noise.data());
        for (size_t i = 0; i < s.n; ++i)
            destination[j*s.n+i] = (s.reconstruct(residues[0][i], residues[1][i]) + noise[i]) & s.mask;
    }
}

void OutputRerandomizer::precompute(size_t count, size_t threads) {
    auto &s = *impl_;
    if (s.preparation_started)
        throw std::logic_error("rerandomization pool cannot be refilled or reset");
    if (!count || !threads || count > s.zeros.max_size() / (2*s.n))
        throw std::invalid_argument("invalid rerandomization pool size/threads");
    s.preparation_started = true;
    // A partial/failed preparation never becomes available for consumption.
    try {
        s.zeros.resize(count * 2*s.n);
        std::atomic<size_t> work{0};
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        std::mutex error_mutex;
        const auto generate = [&] {
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const size_t i = work.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count) break;
                    s.generate_zero(s.zeros.data() + i*2*s.n);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
            }
        };
        std::vector<std::thread> workers;
        threads = std::min(threads, count);
        try {
            for (size_t i=1; i<threads; ++i) workers.emplace_back(generate);
            generate();
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            for (auto &worker : workers) worker.join();
            throw;
        }
        for (auto &worker : workers) worker.join();
        if (error) std::rethrow_exception(error);
        s.count = count;
        s.ready = true; // published before any online worker starts
    } catch (...) {
        std::vector<uint64_t>().swap(s.zeros);
        throw;
    }
}

size_t OutputRerandomizer::prepared_count() const { return impl_->count; }
size_t OutputRerandomizer::remaining_count() const {
    return impl_->count - impl_->next.load(std::memory_order_relaxed);
}
size_t OutputRerandomizer::pool_bytes() const {
    return impl_->zeros.size() * sizeof(uint64_t);
}

void OutputRerandomizer::add_encryption_of_zero(seal::Ciphertext &ct) const {
    auto &s = *impl_;
    // Reject invalid input before reserving a slot. Lazy uint64_t coefficients
    // are valid representatives in Z_{2^Q}.
    check_ciphertext(ct, s.parameters, s.parms_id, false);
    if (!s.ready)
        throw std::logic_error("rerandomization pool has not been precomputed");
    size_t index = s.next.load(std::memory_order_relaxed);
    do {
        if (index >= s.count)
            throw std::runtime_error("rerandomization pool exhausted; online generation is forbidden");
    } while (!s.next.compare_exchange_weak(index, index+1, std::memory_order_relaxed));
    uint64_t *zero = s.zeros.data() + index*2*s.n;
    for (size_t j=0; j<2; ++j)
        for (size_t i=0; i<s.n; ++i) {
            ct.data(j)[i] = (ct.data(j)[i] + zero[j*s.n+i]) & s.mask;
            zero[j*s.n+i] = 0; // burn the exclusively reserved slot
        }
}

void configure_output_rerandomization(
    const seal::SEALContext &context, const seal::Ciphertext &pk) {
    output_rerandomizer.reset(new OutputRerandomizer(context, pk));
    output_count.store(0, std::memory_order_relaxed);
}
void precompute_output_rerandomization(size_t count, size_t threads) {
    if (!output_rerandomizer)
        throw std::logic_error("output rerandomization key is not initialized");
    output_rerandomizer->precompute(count, threads);
}
size_t output_rerandomization_prepared_count() {
    return output_rerandomizer ? output_rerandomizer->prepared_count() : 0;
}
size_t output_rerandomization_remaining_count() {
    return output_rerandomizer ? output_rerandomizer->remaining_count() : 0;
}
size_t output_rerandomization_pool_bytes() {
    return output_rerandomizer ? output_rerandomizer->pool_bytes() : 0;
}
size_t output_rerandomization_budget(const std::string &model, size_t n,
                                    size_t input_width, bool fused_bn) {
    if (n != 2048 || input_width != 224 || !fused_bn)
        throw std::invalid_argument("no rerandomization budget for this graph/packing configuration");
    if (model == "resnet18") return 13312;
    if (model == "resnet50") return 39936;
    if (model == "mobilenet_v2") return 21056;
    throw std::invalid_argument("no rerandomization budget for model: " + model);
}
void rerandomize_conv_output(seal::Ciphertext &ct) {
    if (!output_rerandomizer)
        throw std::logic_error("convolution output rerandomization is not initialized");
    output_rerandomizer->add_encryption_of_zero(ct);
    output_count.fetch_add(1, std::memory_order_relaxed);
}
uint64_t rerandomized_output_count() { return output_count.load(std::memory_order_relaxed); }
} // namespace jaguar
