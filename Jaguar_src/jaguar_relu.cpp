#include "jaguar_relu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "jaguar_profiler.h"
#include "globals.h"
#include "library_fixed_uniform.h"
#include "NonLinear/relu-field.h"
#include "NonLinear/relu-ring.h"
#include "utils/constants.h"
#include "utils/emp-tool.h"

int party = 0;
std::string address = "127.0.0.1";
int port = 32000;
int num_threads = 4;
int32_t bitlength = 41;
int32_t kScale = 9;
extern uint64_t prime_mod;

namespace jaguar {
namespace {

constexpr int kPartyClient = sci::ALICE;
constexpr int kPartyCloud = sci::BOB;
constexpr int kRadixBase = 4;
constexpr int kExternalPortOffset = 32;
std::atomic<int> g_port{43000 + (static_cast<int>(::getpid()) % 32) * 512};

struct RuntimeConfig {
  bool enabled = false;
  int party = 0;
  std::string peer_host = "127.0.0.1";
  int port = 0;
};

RuntimeConfig g_runtime;

struct ExternalRuntimeSession {
  bool initialized = false;
  int party = 0;
  std::string peer_host;
  int port = 0;
  std::unique_ptr<sci::NetIO> io;
  std::unique_ptr<sci::OTPack<sci::NetIO>> otpack;
};

ExternalRuntimeSession &external_session() {
  static ExternalRuntimeSession session;
  return session;
}

void ensure_external_session() {
  auto &session = external_session();
  if (session.initialized &&
      session.party == g_runtime.party &&
      session.peer_host == g_runtime.peer_host &&
      session.port == g_runtime.port) {
    return;
  }

  session.io.reset();
  session.otpack.reset();
  session.initialized = false;
  session.party = g_runtime.party;
  session.peer_host = g_runtime.peer_host;
  session.port = g_runtime.port;

  const bool is_server = (g_runtime.party == kPartyClient);
  std::cerr << "[Jaguar][ReLU] ensure_external_session party=" << g_runtime.party
            << " base_port=" << g_runtime.port
            << " netio_port=" << (g_runtime.port + kExternalPortOffset)
            << " peer=" << g_runtime.peer_host << std::endl;
  session.io = std::make_unique<sci::NetIO>(is_server ? nullptr : g_runtime.peer_host.c_str(), g_runtime.port + kExternalPortOffset, is_server);
  session.otpack = std::make_unique<sci::OTPack<sci::NetIO>>(session.io.get(), g_runtime.party);
  session.initialized = true;
}

constexpr int kWrapperNumThreads = 4;
constexpr int kWrapperScale = 9;
constexpr bool kWrapperDoTruncation = false;
constexpr int kWrapperPortOffset = 64;

struct WrapperRuntimeSession {
  bool initialized = false;
  int party = 0;
  std::string peer_host;
  int port = 0;
  uint64_t bitwidth = 0;
};

struct WrapperSnapshot {
  uint64_t sent_bytes = 0;
  uint64_t rounds = 0;
};

struct ExchangeStats {
  uint64_t sent_bytes = 0;
  uint64_t recv_bytes = 0;
};

struct ActivationPreludeState {
  uint64_t latency_us = 0;
  uint64_t sent_bytes_compat = 0;
  uint64_t sent_bytes_actual = 0;
  uint64_t rounds = 0;
  std::string details;
};

thread_local ActivationPreludeState g_activation_prelude;

void append_activation_detail(std::string &details, const std::string &fragment) {
  if (fragment.empty()) {
    return;
  }
  if (!details.empty()) {
    details += ",";
  }
  details += fragment;
}

void record_activation_total(const std::string &name,
                             uint64_t latency_us,
                             uint64_t sent_bytes_compat,
                             uint64_t sent_bytes_actual,
                             uint64_t rounds,
                             size_t elements,
                             uint64_t bitwidth,
                             bool external,
                             bool local_only,
                             const std::string &extra_details) {
  LayerSummary layer;
  layer.kind = "activation_total";
  layer.name = name;
  layer.total_latency_us = latency_us;
  layer.compute_latency_us = latency_us;
  layer.comm_bytes = sent_bytes_compat;
  layer.comm_bytes_compat = sent_bytes_compat;
  layer.comm_bytes_actual = sent_bytes_actual;
  layer.rounds = rounds;
  layer.details = "elements=" + std::to_string(elements) +
                  ",bitwidth=" + std::to_string(bitwidth) +
                  ",wrapper=1,external=" + std::to_string(external ? 1 : 0) +
                  ",local_only=" + std::to_string(local_only ? 1 : 0) +
                  ",num_threads=" + std::to_string(kWrapperNumThreads) +
                  ",kScale=" + std::to_string(kWrapperScale) +
                  ",do_truncation=" + std::to_string(kWrapperDoTruncation ? 1 : 0);
  append_activation_detail(layer.details, extra_details);
  profiler().record_layer_summary(layer);
}

WrapperRuntimeSession &wrapper_session() {
  static WrapperRuntimeSession session;
  return session;
}

void reset_wrapper_session() {
  auto &session = wrapper_session();
  session.initialized = false;
  session.party = 0;
  session.peer_host.clear();
  session.port = 0;
  session.bitwidth = 0;
}

void shutdown_wrapper_session() {
  auto &session = wrapper_session();
  if (!session.initialized) {
    return;
  }
  EndComputation();
  reset_wrapper_session();
}

void ensure_wrapper_session(uint64_t bitwidth) {
  auto &session = wrapper_session();
  if (session.initialized &&
      session.party == g_runtime.party &&
      session.peer_host == g_runtime.peer_host &&
      session.port == g_runtime.port &&
      session.bitwidth == bitwidth) {
    return;
  }

  shutdown_wrapper_session();

  ::party = g_runtime.party;
  ::address = g_runtime.peer_host;
  ::port = g_runtime.port + kWrapperPortOffset;
  std::cerr << "[Jaguar][ReLU] ensure_wrapper_session party=" << g_runtime.party
            << " base_port=" << g_runtime.port
            << " wrapper_port=" << ::port
            << " bitwidth=" << bitwidth
            << " peer=" << g_runtime.peer_host << std::endl;
  ::num_threads = kWrapperNumThreads;
  ::bitlength = static_cast<int32_t>(bitwidth);
  ::kScale = kWrapperScale;
  StartComputation();

  session.initialized = true;
  session.party = g_runtime.party;
  session.peer_host = g_runtime.peer_host;
  session.port = g_runtime.port;
  session.bitwidth = bitwidth;
}

// ---------------------------------------------------------------------------
// Jaguar_52: exact ReLU backend. Same isolated DSO (libjaguar_exact_v51.so,
// hidden symbols, RTLD_LOCAL) as the ISCA-Jaguar-CiM v51-v59 CPU baselines:
// exact radix-4 Millionaire compare + fused 21->30 carry lift + COT mux on a
// persistent Ferret session. Input: ring-21 shares (cloud share negated).
// Output: ring-30 shares for the P=30 re-encryption. Narrowing each output
// share to 21 bits would lose the secret carry in the next HE encoding.
constexpr int kExactPortOffset = 80;  // uses base+80 and base+81

struct ExactEngine {
  void *handle = nullptr;
  int (*init)(int, const char *, int) = nullptr;
  int (*run)(const uint64_t *, uint64_t *, int) = nullptr;
  uint64_t (*sent)() = nullptr;
  const char *(*error)() = nullptr;
  void (*save)(const char *) = nullptr;  // optional: DSO protocol trace export
  bool ready = false;
};

ExactEngine &exact_engine() {
  static ExactEngine engine;
  return engine;
}

size_t exact_batch_limit() {
  const char *v = std::getenv("JAGUAR_EXACT_BATCH");
  const size_t n = v ? std::stoul(v) : 262144;
  if (n < 8 || n % 8) throw std::runtime_error("JAGUAR_EXACT_BATCH must be a positive multiple of 8");
  return n;
}

void exact_ensure() {
  auto &e = exact_engine();
  if (e.ready) return;
  const char *env = std::getenv("JAGUAR_EXACT_LIBRARY");
  const std::string path = env ? env : "../exact/libjaguar_exact_v51.so";
  e.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!e.handle) throw std::runtime_error(std::string("exact ReLU library: ") + dlerror());
  e.init = reinterpret_cast<int (*)(int, const char *, int)>(dlsym(e.handle, "jaguar_v51_init"));
  e.run = reinterpret_cast<int (*)(const uint64_t *, uint64_t *, int)>(dlsym(e.handle, "jaguar_v51_relu"));
  e.sent = reinterpret_cast<uint64_t (*)()>(dlsym(e.handle, "jaguar_v58_sent_bytes"));
  e.error = reinterpret_cast<const char *(*)()>(dlsym(e.handle, "jaguar_v51_error"));
  e.save = reinterpret_cast<void (*)(const char *)>(dlsym(e.handle, "jaguar_v51_save"));
  if (!e.init || !e.run || !e.sent || !e.error) throw std::runtime_error("exact ReLU library: bad ABI");
  const auto start = std::chrono::steady_clock::now();
  if (e.init(g_runtime.party, g_runtime.peer_host.c_str(), g_runtime.port + kExactPortOffset))
    throw std::runtime_error(std::string("exact ReLU init: ") + e.error());
  LayerSummary layer;
  layer.kind = "exact_init";
  layer.name = "exact_v51_ferret_session";
  layer.total_latency_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start).count());
  layer.compute_latency_us = layer.total_latency_us;
  layer.details = "library=" + path + ",port=" + std::to_string(g_runtime.port + kExactPortOffset);
  profiler().record_layer_summary(layer);
  e.ready = true;
}

// Returns the number of exact protocol invocations (public batches).
size_t exact_apply(const std::vector<uint64_t> &x, std::vector<uint64_t> &y) {
  exact_ensure();
  auto &e = exact_engine();
  const size_t limit = exact_batch_limit();
  y.assign(x.size(), 0);
  size_t calls = 0;
  std::vector<uint64_t> in, out;
  for (size_t off = 0; off < x.size(); off += limit) {
    const size_t n = std::min(limit, x.size() - off);
    const size_t padded = (n + 7) / 8 * 8;  // both parties pad zeros identically
    in.assign(padded, 0);
    out.assign(padded, 0);
    std::copy(x.begin() + off, x.begin() + off + n, in.begin());
    if (e.run(in.data(), out.data(), static_cast<int>(padded)))
      throw std::runtime_error(std::string("exact ReLU: ") + e.error());
    std::copy(out.begin(), out.begin() + n, y.begin() + off);
    ++calls;
  }
  return calls;
}

WrapperSnapshot capture_wrapper_snapshot() {
  WrapperSnapshot snapshot;
  for (int i = 0; i < ::num_threads; ++i) {
    if (ioArr[i] != nullptr) {
      snapshot.sent_bytes += ioArr[i]->counter;
    }
  }
  if (ioArr[0] != nullptr) {
    snapshot.rounds = ioArr[0]->num_rounds;
  }
  return snapshot;
}

ProtocolSummary summarize_wrapper_relu(const std::string &label,
                                       uint64_t latency_us,
                                       const WrapperSnapshot &before,
                                       const WrapperSnapshot &after) {
  ProtocolSummary summary;
  summary.label = label;
  summary.total_latency_us = latency_us;
  summary.total_bytes = after.sent_bytes - before.sent_bytes;
  summary.total_bytes_compat = summary.total_bytes;
  summary.total_bytes_actual = summary.total_bytes;
  summary.total_rounds = after.rounds - before.rounds;
  return summary;
}

struct ThreadResult {
  std::vector<uint64_t> out;
  uint64_t bytes = 0;
  uint64_t rounds = 0;
  std::vector<trace::Event> events;
};

uint64_t mask_for_bits(uint64_t bitwidth) {
  return bitwidth == 64 ? ~0ULL : ((1ULL << bitwidth) - 1);
}

std::vector<uint64_t> flatten(const std::vector<std::vector<uint64_t>> &in) {
  size_t total = 0;
  for (const auto &row : in) {
    total += row.size();
  }
  std::vector<uint64_t> flat;
  flat.reserve(total);
  for (const auto &row : in) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return flat;
}

void reshape_like(const std::vector<std::vector<uint64_t>> &shape_like,
                  const std::vector<uint64_t> &flat,
                  std::vector<std::vector<uint64_t>> &out) {
  std::vector<size_t> row_sizes;
  row_sizes.reserve(shape_like.size());
  for (const auto &row : shape_like) {
    row_sizes.push_back(row.size());
  }
  out.assign(row_sizes.size(), {});
  size_t offset = 0;
  for (size_t i = 0; i < row_sizes.size(); ++i) {
    out[i].assign(flat.begin() + offset, flat.begin() + offset + row_sizes[i]);
    offset += row_sizes[i];
  }
}

bool use_external_runtime() {
  return g_runtime.enabled && g_runtime.party != 0 && g_runtime.port > 0;
}

ExchangeStats exchange_vector(sci::NetIO *io, int party,
                           const std::vector<uint64_t> &local,
                           std::vector<uint64_t> &peer) {
  ExchangeStats stats;
  uint64_t local_size = static_cast<uint64_t>(local.size());
  uint64_t peer_size = 0;
  if (party == kPartyClient) {
    io->send_data(&local_size, sizeof(local_size));
    stats.sent_bytes += sizeof(local_size);
    io->recv_data(&peer_size, sizeof(peer_size));
    stats.recv_bytes += sizeof(peer_size);
    peer.resize(peer_size);
    if (local_size > 0) {
      const uint64_t payload = local_size * sizeof(uint64_t);
      io->send_data(local.data(), static_cast<int>(payload));
      stats.sent_bytes += payload;
    }
    if (peer_size > 0) {
      const uint64_t payload = peer_size * sizeof(uint64_t);
      io->recv_data(peer.data(), static_cast<int>(payload));
      stats.recv_bytes += payload;
    }
  } else {
    io->recv_data(&peer_size, sizeof(peer_size));
    stats.recv_bytes += sizeof(peer_size);
    io->send_data(&local_size, sizeof(local_size));
    stats.sent_bytes += sizeof(local_size);
    peer.resize(peer_size);
    if (peer_size > 0) {
      const uint64_t payload = peer_size * sizeof(uint64_t);
      io->recv_data(peer.data(), static_cast<int>(payload));
      stats.recv_bytes += payload;
    }
    if (local_size > 0) {
      const uint64_t payload = local_size * sizeof(uint64_t);
      io->send_data(local.data(), static_cast<int>(payload));
      stats.sent_bytes += payload;
    }
  }
  return stats;
}

template <typename Runner>
void run_with_netio(Runner runner, ThreadResult &client_result, ThreadResult &cloud_result) {
  const int base_port = g_port.fetch_add(8);
  std::cerr << "[Jaguar][ReLU] run_with_netio pid=" << ::getpid()
            << " base_port=" << base_port << std::endl;
  std::thread server([&]() {
    sci::NetIO io(nullptr, base_port, true);
    sci::OTPack<sci::NetIO> otpack(&io, kPartyClient);
    trace::Collector collector;
    collector.actor = "client";
    trace::Scope scope(&collector);
    runner(kPartyClient, &io, &otpack, client_result.out);
    client_result.bytes = io.counter;
    client_result.rounds = io.num_rounds;
    client_result.events = std::move(collector.events);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  std::thread client([&]() {
    sci::NetIO io("127.0.0.1", base_port, true);
    sci::OTPack<sci::NetIO> otpack(&io, kPartyCloud);
    trace::Collector collector;
    collector.actor = "cloud";
    trace::Scope scope(&collector);
    runner(kPartyCloud, &io, &otpack, cloud_result.out);
    cloud_result.bytes = io.counter;
    cloud_result.rounds = io.num_rounds;
    cloud_result.events = std::move(collector.events);
  });

  server.join();
  client.join();
}

template <typename Runner>
void run_external_netio(Runner runner,
                        ThreadResult &client_result,
                        ThreadResult &cloud_result,
                        bool exchange_peer_output) {
  const int party = g_runtime.party;
  ensure_external_session();
  auto &session = external_session();
  sci::NetIO *io = session.io.get();
  sci::OTPack<sci::NetIO> *otpack = session.otpack.get();
  const uint64_t bytes_before = io->counter;
  const uint64_t rounds_before = io->num_rounds;

  trace::Collector collector;
  collector.actor = (party == kPartyClient) ? "client" : "cloud";
  trace::Scope scope(&collector);

  ThreadResult *local_result = (party == kPartyClient) ? &client_result : &cloud_result;
  runner(party, io, otpack, local_result->out);
  local_result->bytes = io->counter - bytes_before;
  local_result->rounds = io->num_rounds - rounds_before;
  local_result->events = std::move(collector.events);

  if (!exchange_peer_output) {
    return;
  }

  ThreadResult peer_result;
  std::cerr << "[Jaguar][ReLU] party=" << party << " exchange_start local_size=" << local_result->out.size() << std::endl;
  exchange_vector(io, party, local_result->out, peer_result.out);
  std::cerr << "[Jaguar][ReLU] party=" << party << " exchange_done peer_size=" << peer_result.out.size() << std::endl;

  if (party == kPartyClient) {
    cloud_result = std::move(peer_result);
  } else {
    client_result = std::move(peer_result);
  }
}

template <typename Runner>
void run_protocol(Runner runner, ThreadResult &client_result, ThreadResult &cloud_result) {
  if (use_external_runtime()) {
    run_external_netio(runner, client_result, cloud_result, true);
  } else {
    run_with_netio(runner, client_result, cloud_result);
  }
}

}  // namespace

void configure_runtime(bool enabled, int party, const char *peer_host, int port) {
  const std::string next_peer_host = peer_host ? peer_host : "127.0.0.1";
  const auto &session = wrapper_session();
  const bool needs_reset =
      session.initialized &&
      (!enabled ||
       party != g_runtime.party ||
       port != g_runtime.port ||
       next_peer_host != g_runtime.peer_host);
  if (needs_reset) {
    shutdown_wrapper_session();
  }

  g_runtime.enabled = enabled;
  g_runtime.party = party;
  g_runtime.peer_host = next_peer_host;
  g_runtime.port = port;
}

bool external_runtime_enabled() {
  return use_external_runtime();
}

int configured_party() {
  return g_runtime.party;
}

void add_activation_prelude(uint64_t latency_us, const char *detail) {
  g_activation_prelude.latency_us += latency_us;
  if (detail != nullptr && *detail != '\0') {
    append_activation_detail(g_activation_prelude.details, detail);
  }
}

void clear_activation_prelude() {
  g_activation_prelude = ActivationPreludeState{};
}


void relu_share_ring(const std::vector<std::vector<uint64_t>> &client_share,
                     const std::vector<std::vector<uint64_t>> &cloud_mask_share,
                     uint64_t bitwidth,
                     std::vector<std::vector<uint64_t>> &client_relu_share,
                     std::vector<std::vector<uint64_t>> &cloud_relu_share) {
  auto client_flat = flatten(client_share);
  auto cloud_flat = flatten(cloud_mask_share);
  const uint64_t mask = mask_for_bits(bitwidth);

  const size_t original_size = client_flat.size();
  if (cloud_flat.size() != original_size) {
    throw std::runtime_error("relu_share_ring: client/cloud share size mismatch");
  }

  const size_t padded_size = ((original_size + 7) / 8) * 8;
  client_flat.resize(padded_size, 0);
  cloud_flat.resize(padded_size, 0);

  for (auto &value : cloud_flat) {
    value = (-value) & mask;
  }

  ThreadResult client_result;
  ThreadResult cloud_result;
  client_result.out.resize(client_flat.size());
  cloud_result.out.resize(cloud_flat.size());
  ::bitlength = static_cast<int32_t>(bitwidth);

  const ActivationPreludeState activation_prelude = g_activation_prelude;
  g_activation_prelude = ActivationPreludeState{};

  auto activation_start = std::chrono::steady_clock::now();
  auto start = std::chrono::steady_clock::now();

  run_protocol(
      [&](int party, sci::NetIO *io, sci::OTPack<sci::NetIO> *otpack,
          std::vector<uint64_t> &out) {
        std::vector<uint64_t> *share =
            (party == kPartyClient) ? &client_flat : &cloud_flat;

        ReLURingProtocol<sci::NetIO, uint64_t> relu(
            party, RING, io, static_cast<int>(bitwidth), kRadixBase, otpack);

        relu.relu(out.data(),
                  share->data(),
                  static_cast<int>(share->size()),
                  nullptr,
                  false,
                  false,
                  true);

        for (auto &value : out) {
          value &= mask;
        }
      },
      client_result,
      cloud_result);

  auto end = std::chrono::steady_clock::now();
  auto activation_end = std::chrono::steady_clock::now();

  client_result.out.resize(original_size);
  cloud_result.out.resize(original_size);

  reshape_like(client_share, client_result.out, client_relu_share);
  reshape_like(cloud_mask_share, cloud_result.out, cloud_relu_share);

  std::vector<trace::Event> events = client_result.events;
  events.insert(events.end(), cloud_result.events.begin(), cloud_result.events.end());

  const ProtocolSummary summary = summarize_protocol(
      use_external_runtime()
          ? ((g_runtime.party == kPartyClient) ? "relu-ring-client" : "relu-ring-cloud")
          : "relu-ring",
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()),
      client_result.bytes + cloud_result.bytes,
      events);

  profiler().print_protocol_summary(summary);

  LayerSummary layer;
  layer.kind = "relu_ring";
  layer.name = summary.label;
  layer.total_latency_us = summary.total_latency_us;
  layer.compute_latency_us = summary.total_latency_us;
  layer.comm_bytes = summary.total_bytes_compat;
  layer.comm_bytes_compat = summary.total_bytes_compat;
  layer.comm_bytes_actual = summary.total_bytes_actual;
  layer.rounds = summary.total_rounds;
  layer.details = "elements=" + std::to_string(original_size) +
                  ",bitwidth=" + std::to_string(bitwidth) +
                  ",wrapper=0,external=" + std::to_string(use_external_runtime() ? 1 : 0);
  profiler().record_layer_summary(layer);

  record_activation_total(
      use_external_runtime()
          ? ((g_runtime.party == kPartyClient) ? "activation-total-client" : "activation-total-cloud")
          : "activation-total",
      activation_prelude.latency_us +
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  activation_end - activation_start).count()),
      activation_prelude.sent_bytes_compat + summary.total_bytes_compat,
      activation_prelude.sent_bytes_actual + summary.total_bytes_actual,
      activation_prelude.rounds + summary.total_rounds,
      original_size,
      bitwidth,
      false,
      false,
      activation_prelude.details);
}

void relu_share_ring_local(const std::vector<std::vector<uint64_t>> &local_share,
                           uint64_t bitwidth,
                           std::vector<std::vector<uint64_t>> &local_relu_share) {
  if (!use_external_runtime()) {
    throw std::runtime_error("relu_share_ring_local requires external runtime");
  }

  const int party = g_runtime.party;
  auto local_flat = flatten(local_share);
  const uint64_t mask = mask_for_bits(bitwidth);

  ensure_wrapper_session(bitwidth);
  const ActivationPreludeState activation_prelude = g_activation_prelude;
  g_activation_prelude = ActivationPreludeState{};
  auto activation_start = std::chrono::steady_clock::now();
  if (party == kPartyCloud) {
    for (auto &value : local_flat) {
      value = (-value) & mask;
    }
  }

  std::vector<uint64_t> local_out(local_flat.size());
  const WrapperSnapshot before = capture_wrapper_snapshot();
  auto start = std::chrono::steady_clock::now();
  if (!local_flat.empty()) {
    Relu(static_cast<int32_t>(local_flat.size()),
         reinterpret_cast<intType *>(local_flat.data()),
         reinterpret_cast<intType *>(local_out.data()),
         kWrapperScale,
         kWrapperDoTruncation);
  }
  auto end = std::chrono::steady_clock::now();
  const WrapperSnapshot after = capture_wrapper_snapshot();

  for (auto &value : local_out) {
    value &= mask;
  }
  reshape_like(local_share, local_out, local_relu_share);
  auto activation_end = std::chrono::steady_clock::now();

  const ProtocolSummary summary = summarize_wrapper_relu(
      (party == kPartyClient) ? "relu-ring-client-local" : "relu-ring-cloud-local",
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()),
      before,
      after);
  profiler().print_protocol_summary(summary);
  LayerSummary layer;
  layer.kind = "relu_ring";
  layer.name = summary.label;
  layer.total_latency_us = summary.total_latency_us;
  layer.compute_latency_us = summary.total_latency_us;
  layer.comm_bytes = summary.total_bytes_compat;
  layer.comm_bytes_compat = summary.total_bytes_compat;
  layer.comm_bytes_actual = summary.total_bytes_actual;
  layer.rounds = summary.total_rounds;
  layer.details = "elements=" + std::to_string(local_flat.size()) +
                  ",bitwidth=" + std::to_string(bitwidth) +
                  ",local_only=1,wrapper=1,num_threads=" + std::to_string(kWrapperNumThreads) +
                  ",kScale=" + std::to_string(kWrapperScale) +
                  ",do_truncation=" + std::to_string(kWrapperDoTruncation ? 1 : 0);
  profiler().record_layer_summary(layer);

  record_activation_total(
      (party == kPartyClient) ? "activation-total-client-local" : "activation-total-cloud-local",
      activation_prelude.latency_us +
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(activation_end - activation_start).count()),
      activation_prelude.sent_bytes_compat + summary.total_bytes_compat,
      activation_prelude.sent_bytes_actual + summary.total_bytes_actual,
      activation_prelude.rounds + summary.total_rounds,
      local_flat.size(),
      bitwidth,
      false,
      true,
      activation_prelude.details);
}

bool exact_relu_enabled() {
  const char *mode = std::getenv("JAGUAR_RELU_MODE");
  if (mode == nullptr || std::string(mode) == "exact") return true;
  if (std::string(mode) == "approx") return false;
  throw std::runtime_error("JAGUAR_RELU_MODE must be exact or approx");
}

void exact_relu_init() {
  if (exact_relu_enabled() && use_external_runtime()) exact_ensure();
}

uint64_t exact_relu_output_bits() { return 30; }

// Jaguar_52 opt-in (JAGUAR_EXACT_TRACE=<prefix>): write the DSO's own protocol
// trace (<prefix>.exact.csv, .pipeline.csv, .typed.csv and, with V24_WIRE=1,
// the wire log) for the accelerator model. Off by default; timing runs never set it.
void exact_relu_save_trace() {
  const char *prefix = std::getenv("JAGUAR_EXACT_TRACE");
  auto &e = exact_engine();
  if (!prefix || !e.ready || !e.save) return;
  e.save(prefix);
}

void ensure_pool_runtime(uint64_t bitwidth) {
  if (wrapper_session().initialized && wrapper_session().bitwidth == bitwidth) return;
  const auto start = std::chrono::steady_clock::now();
  ensure_wrapper_session(bitwidth);
  LayerSummary summary;
  summary.kind = "pool_setup";
  summary.name = "sci_pool_session_" + std::to_string(bitwidth);
  summary.total_latency_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start).count());
  summary.compute_latency_us = summary.total_latency_us;
  profiler().record_layer_summary(summary);
}

// Exact ReLU (relu6=false) or ReLU6 = ReLU(x) - ReLU(x - 6*2^9) (relu6=true).
// ReLU6 evaluates both comparisons in ONE public batch sequence, so it costs
// two exact ReLUs of work but not twice the number of protocol invocations.
static void exact_relu_share(const std::vector<std::vector<uint64_t>> &local_share,
                             uint64_t bitwidth,
                             std::vector<std::vector<uint64_t>> &local_out_share,
                             bool relu6) {
  if (bitwidth != 21) throw std::runtime_error("exact ReLU backend expects ring-21 shares (P30 - f9)");
  const int party = g_runtime.party;
  const uint64_t mask = mask_for_bits(bitwidth);
  const uint64_t out_mask = mask_for_bits(exact_relu_output_bits());
  auto local_flat = flatten(local_share);
  const size_t n = local_flat.size();
  const ActivationPreludeState activation_prelude = g_activation_prelude;
  g_activation_prelude = ActivationPreludeState{};
  exact_ensure();
  const auto activation_start = std::chrono::steady_clock::now();

  std::vector<uint64_t> in(relu6 ? 2 * n : n);
  const uint64_t six_scaled = (static_cast<uint64_t>(6) << kWrapperScale) & mask;
  for (size_t i = 0; i < n; ++i) {
    uint64_t v = local_flat[i] & mask;
    if (party == kPartyCloud) v = (-v) & mask;
    in[i] = v;
    if (relu6) in[n + i] = (party == kPartyClient) ? ((v - six_scaled) & mask) : v;
  }
  const uint64_t sent_before = exact_engine().sent();
  const auto start = std::chrono::steady_clock::now();
  std::vector<uint64_t> y;
  const size_t calls = n ? exact_apply(in, y) : 0;
  const auto end = std::chrono::steady_clock::now();
  const uint64_t sent = exact_engine().sent() - sent_before;

  std::vector<uint64_t> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = relu6 ? ((y[i] - y[n + i]) & out_mask) : (y[i] & out_mask);
  reshape_like(local_share, out, local_out_share);

  const uint64_t us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
  LayerSummary layer;
  layer.kind = relu6 ? "relu6_ring" : "relu_ring";
  layer.name = std::string(relu6 ? "relu6" : "relu") + ((party == kPartyClient) ? "-exact-client" : "-exact-cloud");
  layer.total_latency_us = us;
  layer.compute_latency_us = us;
  layer.comm_bytes = sent;
  layer.comm_bytes_compat = sent;
  layer.comm_bytes_actual = sent;
  layer.rounds = calls;
  layer.details = "elements=" + std::to_string(n) + ",exact_elements=" + std::to_string(in.size()) +
                  ",bitwidth=" + std::to_string(bitwidth) + ",out_bits=" + std::to_string(exact_relu_output_bits()) +
                  ",backend=exact_v51,batches=" + std::to_string(calls) +
                  ",batch_limit=" + std::to_string(exact_batch_limit()) +
                  ",bytes=sent_including_refill";
  profiler().record_layer_summary(layer);
  record_activation_total(
      (party == kPartyClient) ? "activation-total-client" : "activation-total-cloud",
      activation_prelude.latency_us +
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - activation_start).count()),
      activation_prelude.sent_bytes_compat + sent, activation_prelude.sent_bytes_actual + sent,
      activation_prelude.rounds + calls, n, bitwidth, true, false,
      activation_prelude.details.empty() ? (relu6 ? "exact=1,relu6=1" : "exact=1")
                                         : (activation_prelude.details + (relu6 ? ",exact=1,relu6=1" : ",exact=1")));
}

void relu_share_ring_external(
    const std::vector<std::vector<uint64_t>> &local_share,
    uint64_t bitwidth,
    std::vector<std::vector<uint64_t>> &local_relu_share) {
  if (!use_external_runtime()) {
    throw std::runtime_error("relu_share_ring_external requires external runtime");
  }
  if (exact_relu_enabled()) {
    exact_relu_share(local_share, bitwidth, local_relu_share, false);
    return;
  }

  const int party = g_runtime.party;
  auto local_flat = flatten(local_share);
  const uint64_t mask = mask_for_bits(bitwidth);

  ensure_wrapper_session(bitwidth);
  const ActivationPreludeState activation_prelude = g_activation_prelude;
  g_activation_prelude = ActivationPreludeState{};
  auto activation_start = std::chrono::steady_clock::now();

  if (party == kPartyCloud) {
    for (auto &value : local_flat) {
      value = (-value) & mask;
    }
  }

  const size_t original_size = local_flat.size();
  std::vector<uint64_t> local_out(local_flat.size());
  const WrapperSnapshot before = capture_wrapper_snapshot();
  auto start = std::chrono::steady_clock::now();
  if (!local_flat.empty()) {
    Relu(static_cast<int32_t>(local_flat.size()),
         reinterpret_cast<intType *>(local_flat.data()),
         reinterpret_cast<intType *>(local_out.data()),
         kWrapperScale,
         kWrapperDoTruncation);
  }
  auto end = std::chrono::steady_clock::now();
  const WrapperSnapshot after = capture_wrapper_snapshot();
  auto activation_end = std::chrono::steady_clock::now();

  for (auto &value : local_out) {
    value &= mask;
  }
  reshape_like(local_share, local_out, local_relu_share);

  const ProtocolSummary summary = summarize_wrapper_relu(
      (party == kPartyClient) ? "relu-ring-client" : "relu-ring-cloud",
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()),
      before,
      after);
  profiler().print_protocol_summary(summary);

  LayerSummary layer;
  layer.kind = "relu_ring";
  layer.name = summary.label;
  layer.total_latency_us = summary.total_latency_us;
  layer.compute_latency_us = summary.total_latency_us;
  layer.comm_bytes = summary.total_bytes_compat;
  layer.comm_bytes_compat = summary.total_bytes_compat;
  layer.comm_bytes_actual = summary.total_bytes_actual;
  layer.rounds = summary.total_rounds;
  layer.details = "elements=" + std::to_string(original_size) +
                  ",bitwidth=" + std::to_string(bitwidth) +
                  ",local_only=0,wrapper=1,external=1,num_threads=" +
                  std::to_string(kWrapperNumThreads) +
                  ",kScale=" + std::to_string(kWrapperScale) +
                  ",do_truncation=" +
                  std::to_string(kWrapperDoTruncation ? 1 : 0);
  profiler().record_layer_summary(layer);

  record_activation_total(
      (party == kPartyClient) ? "activation-total-client" : "activation-total-cloud",
      activation_prelude.latency_us +
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  activation_end - activation_start).count()),
      activation_prelude.sent_bytes_compat + summary.total_bytes_compat,
      activation_prelude.sent_bytes_actual + summary.total_bytes_actual,
      activation_prelude.rounds + summary.total_rounds,
      original_size,
      bitwidth,
      true,
      false,
      activation_prelude.details);
}

void relu6_share_ring_external(
    const std::vector<std::vector<uint64_t>> &local_share,
    uint64_t bitwidth,
    std::vector<std::vector<uint64_t>> &local_relu6_share) {
  if (use_external_runtime() == false) {
    throw std::runtime_error("relu6_share_ring_external requires external runtime");
  }
  if (exact_relu_enabled()) {
    exact_relu_share(local_share, bitwidth, local_relu6_share, true);
    return;
  }

  const int party = g_runtime.party;
  auto local_flat = flatten(local_share);
  const uint64_t mask = mask_for_bits(bitwidth);
  const uint64_t six_scaled = (static_cast<uint64_t>(6) << kWrapperScale) & mask;

  ensure_wrapper_session(bitwidth);
  const ActivationPreludeState activation_prelude = g_activation_prelude;
  g_activation_prelude = ActivationPreludeState{};
  auto activation_start = std::chrono::steady_clock::now();

  std::vector<uint64_t> relu_in(local_flat.begin(), local_flat.end());
  std::vector<uint64_t> shifted_in(local_flat.begin(), local_flat.end());
  if (party == kPartyClient) {
    for (auto &value : shifted_in) {
      value = (value - six_scaled) & mask;
    }
  }
  if (party == kPartyCloud) {
    for (auto &value : relu_in) {
      value = (-value) & mask;
    }
    for (auto &value : shifted_in) {
      value = (-value) & mask;
    }
  }

  const size_t original_size = local_flat.size();
  std::vector<uint64_t> relu_out(local_flat.size());
  std::vector<uint64_t> clipped_out(local_flat.size());
  const WrapperSnapshot before = capture_wrapper_snapshot();
  auto start = std::chrono::steady_clock::now();
  if (local_flat.empty() == false) {
    Relu(static_cast<int32_t>(relu_in.size()),
         reinterpret_cast<intType *>(relu_in.data()),
         reinterpret_cast<intType *>(relu_out.data()),
         kWrapperScale,
         kWrapperDoTruncation);
    Relu(static_cast<int32_t>(shifted_in.size()),
         reinterpret_cast<intType *>(shifted_in.data()),
         reinterpret_cast<intType *>(clipped_out.data()),
         kWrapperScale,
         kWrapperDoTruncation);
  }
  auto end = std::chrono::steady_clock::now();
  const WrapperSnapshot after = capture_wrapper_snapshot();
  auto activation_end = std::chrono::steady_clock::now();

  std::vector<uint64_t> local_out(local_flat.size());
  for (size_t i = 0; i < local_out.size(); ++i) {
    local_out[i] = (relu_out[i] - clipped_out[i]) & mask;
  }
  reshape_like(local_share, local_out, local_relu6_share);

  const ProtocolSummary summary = summarize_wrapper_relu(
      (party == kPartyClient) ? "relu6-ring-client" : "relu6-ring-cloud",
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()),
      before,
      after);
  profiler().print_protocol_summary(summary);

  LayerSummary layer;
  layer.kind = "relu6_ring";
  layer.name = summary.label;
  layer.total_latency_us = summary.total_latency_us;
  layer.compute_latency_us = summary.total_latency_us;
  layer.comm_bytes = summary.total_bytes_compat;
  layer.comm_bytes_compat = summary.total_bytes_compat;
  layer.comm_bytes_actual = summary.total_bytes_actual;
  layer.rounds = summary.total_rounds;
  layer.details = "elements=" + std::to_string(original_size) +
                  ",bitwidth=" + std::to_string(bitwidth) +
                  ",local_only=0,wrapper=1,external=1,relu6=1,num_threads=" +
                  std::to_string(kWrapperNumThreads) +
                  ",kScale=" + std::to_string(kWrapperScale) +
                  ",do_truncation=" +
                  std::to_string(kWrapperDoTruncation ? 1 : 0);
  profiler().record_layer_summary(layer);

  std::string activation_detail = activation_prelude.details.empty() ? "relu6=1" : (activation_prelude.details + ",relu6=1");
  record_activation_total(
      (party == kPartyClient) ? "activation-total-client" : "activation-total-cloud",
      activation_prelude.latency_us +
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(activation_end - activation_start).count()),
      activation_prelude.sent_bytes_compat + summary.total_bytes_compat,
      activation_prelude.sent_bytes_actual + summary.total_bytes_actual,
      activation_prelude.rounds + summary.total_rounds,
      original_size,
      bitwidth,
      true,
      false,
      activation_detail);
}


void relu_share_field(const std::vector<std::vector<uint64_t>> &client_share,
                      const std::vector<std::vector<uint64_t>> &cloud_mask_share,
                      uint64_t mod_p,
                      std::vector<std::vector<uint64_t>> &client_relu_share,
                      std::vector<std::vector<uint64_t>> &cloud_relu_share) {
  auto client_flat = flatten(client_share);
  auto cloud_flat = flatten(cloud_mask_share);
  for (auto &value : cloud_flat) {
    value = value == 0 ? 0 : (mod_p - value);
  }

  const size_t original_size = client_flat.size();
  const size_t padded_size = ((original_size + 7) / 8) * 8;
  client_flat.resize(padded_size, 0);
  cloud_flat.resize(padded_size, 0);

  ThreadResult client_result;
  ThreadResult cloud_result;
  client_result.out.resize(client_flat.size());
  cloud_result.out.resize(cloud_flat.size());
  ::prime_mod = mod_p;
  ::bitlength = static_cast<int32_t>(std::ceil(std::log2(static_cast<long double>(mod_p))));

  auto start = std::chrono::steady_clock::now();
  run_protocol(
      [&](int party, sci::NetIO *io, sci::OTPack<sci::NetIO> *otpack, std::vector<uint64_t> &out) {
        std::vector<uint64_t> *share = party == kPartyClient ? &client_flat : &cloud_flat;
        ReLUFieldProtocol<sci::NetIO, uint64_t> relu(
            party, FIELD, io, ::bitlength, kRadixBase, mod_p, otpack);
        relu.relu(out.data(), share->data(), static_cast<int>(share->size()), nullptr, false, false, true);
        for (auto &value : out) {
          value %= mod_p;
        }
      },
      client_result,
      cloud_result);
  auto end = std::chrono::steady_clock::now();

  client_result.out.resize(original_size);
  cloud_result.out.resize(original_size);
  reshape_like(client_share, client_result.out, client_relu_share);
  reshape_like(cloud_mask_share, cloud_result.out, cloud_relu_share);

  std::vector<trace::Event> events = client_result.events;
  events.insert(events.end(), cloud_result.events.begin(), cloud_result.events.end());
  const ProtocolSummary summary = summarize_protocol(
      use_external_runtime() ? ((g_runtime.party == kPartyClient) ? "relu-field-client" : "relu-field-cloud") : "relu-field",
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()),
      client_result.bytes + cloud_result.bytes,
      events);
  profiler().print_protocol_summary(summary);
  LayerSummary layer;
  layer.kind = "relu_field";
  layer.name = summary.label;
  layer.total_latency_us = summary.total_latency_us;
  layer.compute_latency_us = summary.total_latency_us;
  layer.comm_bytes = summary.total_bytes;
  layer.rounds = summary.total_rounds;
  layer.details = "elements=" + std::to_string(original_size) +
                  ",mod_p=" + std::to_string(mod_p);
  profiler().record_layer_summary(layer);
}

}  // namespace jaguar
