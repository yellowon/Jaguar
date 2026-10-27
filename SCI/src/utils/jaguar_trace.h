#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace jaguar::trace {

struct Event {
  uint64_t round_index = 0;
  uint64_t timestamp_us = 0;
  uint64_t bytes = 0;
  std::string actor;
  std::string direction;
  std::string kind;
};

struct Collector {
  std::string actor;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::vector<Event> events;
};

inline thread_local Collector *tls_collector = nullptr;

class Scope {
 public:
  explicit Scope(Collector *collector) : prev_(tls_collector) { tls_collector = collector; }
  ~Scope() { tls_collector = prev_; }

 private:
  Collector *prev_;
};

inline uint64_t now_us(Collector *collector) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - collector->t0).count());
}

inline void note_transition(uint64_t round_index, const char *direction) {
  if (tls_collector == nullptr) {
    return;
  }
  tls_collector->events.push_back(
      Event{round_index, now_us(tls_collector), 0, tls_collector->actor, direction, "transition"});
}

inline void note_bytes(uint64_t round_index, const char *direction, uint64_t bytes) {
  if (tls_collector == nullptr) {
    return;
  }
  tls_collector->events.push_back(
      Event{round_index, now_us(tls_collector), bytes, tls_collector->actor, direction, "bytes"});
}

}  // namespace jaguar::trace
