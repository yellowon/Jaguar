#include "jaguar_profiler.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <chrono>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace jaguar {

namespace {

std::string dirname_of(const std::string &path) {
  const std::string::size_type pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return ".";
  }
  if (pos == 0) {
    return "/";
  }
  return path.substr(0, pos);
}

void ensure_dir(const std::string &path) {
  if (path.empty() || path == "." || path == "/") {
    return;
  }
  ensure_dir(dirname_of(path));
  mkdir(path.c_str(), 0755);
}

std::string render_summary(const ProtocolSummary &summary) {
  std::ostringstream oss;
  oss << "\n[Jaguar][" << summary.label << "] total_latency_us=" << summary.total_latency_us
      << " total_bytes=" << summary.total_bytes
      << " total_bytes_compat=" << summary.total_bytes_compat
      << " total_bytes_actual=" << summary.total_bytes_actual
      << " total_rounds=" << summary.total_rounds << "\n";
  for (const auto &round : summary.rounds) {
    oss << "  round=" << round.round_index
        << " bytes=" << round.bytes
        << " bytes_compat=" << round.bytes_compat
        << " bytes_actual=" << round.bytes_actual
        << " latency_us=" << round.duration_us
        << " start_us=" << round.start_us << "\n";
  }
  return oss.str();
}

std::string csv_escape(const std::string &value) {
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"') {
      out += "\"\"";
    } else {
      out += ch;
    }
  }
  out += "\"";
  return out;
}

}  // namespace

std::string Profiler::default_log_path() const {
  if (const char *file = std::getenv("JAGUAR_LOG_FILE")) {
    if (*file != '\0') {
      return file;
    }
  }
  std::string dir = "../logs";
  if (const char *env_dir = std::getenv("JAGUAR_LOG_DIR")) {
    if (*env_dir != '\0') {
      dir = env_dir;
    }
  }
  return dir + "/jaguar_protocol_summary.log";
}

std::string Profiler::default_layer_csv_path() const {
  if (const char *file = std::getenv("JAGUAR_LAYER_CSV")) {
    if (*file != '\0') {
      return file;
    }
  }
  std::string dir = "../logs";
  if (const char *env_dir = std::getenv("JAGUAR_LOG_DIR")) {
    if (*env_dir != '\0') {
      dir = env_dir;
    }
  }
  return dir + "/jaguar_layer_summary.csv";
}

void Profiler::print_protocol_summary(const ProtocolSummary &summary) const {
  std::lock_guard<std::mutex> lock(mu_);
  const std::string rendered = render_summary(summary);
  std::cout << rendered;

  const std::string log_path = default_log_path();
  ensure_dir(dirname_of(log_path));
  std::ofstream out(log_path.c_str(), std::ios::app);
  if (out.is_open()) {
    out << rendered;
  }
}

void Profiler::add_transport_bytes(uint64_t sent_bytes, uint64_t recv_bytes) {
  std::lock_guard<std::mutex> lock(mu_);
  sent_bytes_ += sent_bytes;
  recv_bytes_ += recv_bytes;
}

TransportSnapshot Profiler::transport_snapshot() const {
  std::lock_guard<std::mutex> lock(mu_);
  TransportSnapshot snapshot;
  snapshot.sent_bytes = sent_bytes_;
  snapshot.recv_bytes = recv_bytes_;
  return snapshot;
}

void Profiler::record_layer_summary(const LayerSummary &input) {
  std::lock_guard<std::mutex> lock(mu_);

  LayerSummary summary = input;
  if (summary.comm_bytes_compat == 0 && summary.comm_bytes_actual == 0 && summary.comm_bytes != 0) {
    summary.comm_bytes_compat = summary.comm_bytes;
    summary.comm_bytes_actual = summary.comm_bytes;
  }
  if (summary.comm_bytes == 0 && summary.comm_bytes_compat != 0) {
    summary.comm_bytes = summary.comm_bytes_compat;
  }
  if (summary.layer_index == 0) {
    summary.layer_index = ++next_layer_index_;
  } else {
    next_layer_index_ = std::max(next_layer_index_, summary.layer_index);
  }

  std::ostringstream text;
  text << "[Jaguar][Layer] idx=" << summary.layer_index
       << " kind=" << summary.kind
       << " name=" << summary.name
       << " total_latency_us=" << summary.total_latency_us
       << " compute_latency_us=" << summary.compute_latency_us
       << " comm_bytes=" << summary.comm_bytes
       << " comm_bytes_compat=" << summary.comm_bytes_compat
       << " comm_bytes_actual=" << summary.comm_bytes_actual
       << " rounds=" << summary.rounds
       << " details=" << summary.details << "\n";
  std::cout << text.str();

  const std::string log_path = default_log_path();
  ensure_dir(dirname_of(log_path));
  std::ofstream log_out(log_path.c_str(), std::ios::app);
  if (log_out.is_open()) {
    log_out << text.str();
  }

  const std::string csv_path = default_layer_csv_path();
  ensure_dir(dirname_of(csv_path));
  bool write_header = false;
  {
    std::ifstream existing(csv_path.c_str());
    write_header = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
  }
  std::ofstream csv_out(csv_path.c_str(), std::ios::app);
  if (csv_out.is_open()) {
    if (write_header) {
      // Jaguar_52: end_ns (steady clock) and cumulative HE transport bytes per record.
      csv_out << "layer_index,kind,name,total_latency_us,compute_latency_us,comm_bytes,comm_bytes_compat,comm_bytes_actual,rounds,details,end_ns,he_sent,he_recv\n";
    }
    csv_out << summary.layer_index << ","
            << csv_escape(summary.kind) << ","
            << csv_escape(summary.name) << ","
            << summary.total_latency_us << ","
            << summary.compute_latency_us << ","
            << summary.comm_bytes << ","
            << summary.comm_bytes_compat << ","
            << summary.comm_bytes_actual << ","
            << summary.rounds << ","
            << csv_escape(summary.details) << ","
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count() << ","
            << sent_bytes_ << "," << recv_bytes_ << "\n";
  }
}

Profiler &profiler() {
  static Profiler instance;
  return instance;
}

ProtocolSummary summarize_protocol(const std::string &label,
                                   uint64_t latency_us,
                                   uint64_t total_bytes,
                                   const std::vector<trace::Event> &events) {
  ProtocolSummary summary;
  summary.label = label;
  summary.total_latency_us = latency_us;
  summary.total_bytes = total_bytes;
  summary.total_bytes_compat = 0;
  summary.total_bytes_actual = 0;

  std::map<uint64_t, std::vector<trace::Event>> by_round;
  for (const auto &event : events) {
    by_round[event.round_index].push_back(event);
  }

  summary.total_rounds = by_round.size();
  for (auto &[round_index, round_events] : by_round) {
    std::sort(round_events.begin(), round_events.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.timestamp_us < rhs.timestamp_us;
    });
    RoundSummary round;
    round.round_index = round_index;
    round.start_us = round_events.front().timestamp_us;
    uint64_t end_us = round_events.back().timestamp_us;
    for (const auto &event : round_events) {
      if (event.kind == "bytes") {
        round.bytes_actual += event.bytes;
        summary.total_bytes_actual += event.bytes;
        if (event.direction == "send") {
          round.bytes += event.bytes;
          round.bytes_compat += event.bytes;
          summary.total_bytes_compat += event.bytes;
        }
      }
      end_us = std::max(end_us, event.timestamp_us);
    }
    round.duration_us = end_us - round.start_us;
    summary.rounds.push_back(round);
  }
  return summary;
}

}  // namespace jaguar
