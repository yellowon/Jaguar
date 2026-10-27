#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "utils/jaguar_trace.h"

namespace jaguar {

struct RoundSummary {
  uint64_t round_index = 0;
  uint64_t bytes = 0;
  uint64_t bytes_compat = 0;
  uint64_t bytes_actual = 0;
  uint64_t start_us = 0;
  uint64_t duration_us = 0;
};

struct ProtocolSummary {
  std::string label;
  uint64_t total_bytes = 0;
  uint64_t total_bytes_compat = 0;
  uint64_t total_bytes_actual = 0;
  uint64_t total_rounds = 0;
  uint64_t total_latency_us = 0;
  std::vector<RoundSummary> rounds;
};

struct LayerSummary {
  uint64_t layer_index = 0;
  std::string kind;
  std::string name;
  uint64_t total_latency_us = 0;
  uint64_t compute_latency_us = 0;
  uint64_t comm_bytes = 0;
  uint64_t comm_bytes_compat = 0;
  uint64_t comm_bytes_actual = 0;
  uint64_t rounds = 0;
  std::string details;
};

struct TransportSnapshot {
  uint64_t sent_bytes = 0;
  uint64_t recv_bytes = 0;
};

class Profiler {
 public:
  void print_protocol_summary(const ProtocolSummary &summary) const;
  std::string default_log_path() const;
  std::string default_layer_csv_path() const;
  void record_layer_summary(const LayerSummary &summary);
  void add_transport_bytes(uint64_t sent_bytes, uint64_t recv_bytes);
  TransportSnapshot transport_snapshot() const;

 private:
  mutable std::mutex mu_;
  mutable uint64_t next_layer_index_ = 0;
  mutable uint64_t sent_bytes_ = 0;
  mutable uint64_t recv_bytes_ = 0;
};

Profiler &profiler();
ProtocolSummary summarize_protocol(const std::string &label,
                                   uint64_t latency_us,
                                   uint64_t total_bytes,
                                   const std::vector<trace::Event> &events);

}  // namespace jaguar
