#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#include "globals.h"
#include "library_fixed_uniform.h"
#include "jaguar_profiler.h"
#include "jaguar_relu.h"
#include "client_layer_helpers.h"

using namespace std;

int j52_pool_session_bits() {
    static const int bits = [] { const char *v = std::getenv("JAGUAR_POOL_FIRST"); return (v && std::string(v) == "1") ? 21 : 30; }();
    return bits;
}

void max_pooling_share(std::vector<std::vector<uint64_t>> &data,
                       int width,
                       int kernel_size,
                       int stride,
                       int padding,
                       int ring_bits) {
    if (data.empty()) return;
    // Jaguar_52: exact ReLU outputs are ring-30 shares; pool in the same ring.
    // ring_bits>0 (pool-first stem) pools pre-ReLU ring-21 shares instead.
    if (ring_bits > 0) jaguar::ensure_pool_runtime(static_cast<uint64_t>(ring_bits));
    else if (jaguar::exact_relu_enabled()) jaguar::ensure_pool_runtime(static_cast<uint64_t>(j52_pool_session_bits()));
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_max_pooling_sum(0);
    INIT_ALL_IO_DATA_SENT;
    int channels = (int)data.size();
    int in_w = (int)std::sqrt((double)data[0].size());
    int out_w = 1 + (in_w + 2 * padding - kernel_size) / stride;
    std::vector<intType> in_arr((size_t)in_w * in_w * channels, 0);
    std::vector<intType> out_arr((size_t)out_w * out_w * channels, 0);
    for (int c = 0; c < channels; ++c)
        for (int idx = 0; idx < in_w * in_w; ++idx)
            in_arr[(size_t)idx * channels + c] = (intType)data[c][idx];
    time_start = chrono::high_resolution_clock::now();
    MaxPool(1, out_w, out_w, channels, kernel_size, kernel_size,
            padding, padding, padding, padding, stride, stride,
            1, in_w, in_w, channels, in_arr.data(), out_arr.data());
    time_end = chrono::high_resolution_clock::now();
    time_max_pooling_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    uint64_t curComm = 0;
    FIND_ALL_IO_TILL_NOW(curComm);
    data.assign(channels, std::vector<uint64_t>((size_t)out_w * out_w, 0));
    for (int c = 0; c < channels; ++c)
        for (int idx = 0; idx < out_w * out_w; ++idx)
            data[c][idx] = (uint64_t)out_arr[(size_t)idx * channels + c];
    jaguar::LayerSummary layer_summary;
    layer_summary.kind = "max_pool";
    layer_summary.name = ring_bits > 0 ? "max_pooling_pre_relu_r" + std::to_string(ring_bits) : "max_pooling";
    layer_summary.total_latency_us = (uint64_t)time_max_pooling_sum.count();
    layer_summary.compute_latency_us = (uint64_t)time_max_pooling_sum.count();
    layer_summary.comm_bytes = curComm;
    layer_summary.comm_bytes_compat = curComm;
    layer_summary.comm_bytes_actual = curComm;
    layer_summary.rounds = 0;
    layer_summary.details = "kernel=" + std::to_string(kernel_size) +
                    ",stride=" + std::to_string(stride) +
                    ",padding=" + std::to_string(padding) +
                    ",in_w=" + std::to_string(in_w) +
                    ",out_w=" + std::to_string(out_w) +
                    ",channels=" + std::to_string(channels);
    jaguar::profiler().record_layer_summary(layer_summary);
}

void avg_pooling_share_oc(std::vector<std::vector<uint64_t>> &data,
                          int width,
                          int kernel_size,
                          int stride,
                          int padding) {
    if (data.empty()) return;
    // Jaguar_52: exact ReLU outputs are ring-30 shares; pool in the same ring.
    if (jaguar::exact_relu_enabled()) jaguar::ensure_pool_runtime(static_cast<uint64_t>(j52_pool_session_bits()));
    // Pool-first runs keep the single 21-bit SCI session: the final AvgPool
    // then runs on shares reduced mod 2^21. Timing-equivalent only (values
    // of this tail are not certified to fit 21 bits); see run.json.
    if (jaguar::exact_relu_enabled() && j52_pool_session_bits() < 30)
        for (auto &row : data) for (auto &v : row) v &= (1ULL << j52_pool_session_bits()) - 1;
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_avg_pooling_sum(0);
    INIT_ALL_IO_DATA_SENT;
    int channels = (int)data.size();
    int in_w = (int)std::sqrt((double)data[0].size());
    int out_w = 1 + (in_w + 2 * padding - kernel_size) / stride;
    std::vector<intType> in_arr((size_t)in_w * in_w * channels, 0);
    std::vector<intType> out_arr((size_t)out_w * out_w * channels, 0);
    for (int c = 0; c < channels; ++c)
        for (int idx = 0; idx < in_w * in_w; ++idx)
            in_arr[(size_t)idx * channels + c] = (intType)data[c][idx];
    time_start = chrono::high_resolution_clock::now();
    AvgPool(1, out_w, out_w, channels, kernel_size, kernel_size,
            padding, padding, padding, padding, stride, stride,
            1, in_w, in_w, channels, in_arr.data(), out_arr.data());
    time_end = chrono::high_resolution_clock::now();
    time_avg_pooling_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    uint64_t curComm = 0;
    FIND_ALL_IO_TILL_NOW(curComm);
    data.assign(channels, std::vector<uint64_t>((size_t)out_w * out_w, 0));
    for (int c = 0; c < channels; ++c)
        for (int idx = 0; idx < out_w * out_w; ++idx)
            data[c][idx] = (uint64_t)out_arr[(size_t)idx * channels + c];
    jaguar::LayerSummary layer_summary;
    layer_summary.kind = "avg_pool";
    layer_summary.name = "avg_pooling_oc";
    layer_summary.total_latency_us = (uint64_t)time_avg_pooling_sum.count();
    layer_summary.compute_latency_us = (uint64_t)time_avg_pooling_sum.count();
    layer_summary.comm_bytes = curComm;
    layer_summary.comm_bytes_compat = curComm;
    layer_summary.comm_bytes_actual = curComm;
    layer_summary.rounds = 0;
    layer_summary.details = "kernel=" + std::to_string(kernel_size) + ",stride=" + std::to_string(stride) + ",padding=" + std::to_string(padding) + ",in_w=" + std::to_string(in_w) + ",out_w=" + std::to_string(out_w) + ",channels=" + std::to_string(channels);
    jaguar::profiler().record_layer_summary(layer_summary);
}

void relu6_share_quant(const std::vector<std::vector<uint64_t>> &local_share,
                       uint64_t bitwidth,
                       std::vector<std::vector<uint64_t>> &local_relu6_share) {
    jaguar::relu6_share_ring_external(local_share, bitwidth, local_relu6_share);
}
