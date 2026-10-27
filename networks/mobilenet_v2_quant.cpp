// MobileNetV2 quant path aligned to OpenCheetah block shapes.
// BN is fully fused into conv weights and post-conv bias share updates.

namespace jaguar_networks {
struct MobileNetV2BlockSpecQuant {
    int expand_channels;
    int out_channels;
    int stride;
    bool use_residual;
    bool has_expand;
};
}

template <int N>
static void call_depthwise_conv_quant_mobilenet_v2(
    int order,
    int &seed,
    int w,
    std::vector<std::vector<std::vector<uint64_t>>> &filter_depthwise_full,
    int extract_stride,
    const seal::SEALContext &context_quant,
    const uint64_t log2_q,
    const uint64_t log2_p,
    const std::vector<seal::Ciphertext> &cin,
    std::vector<seal::Ciphertext> &cout,
    std::vector<std::vector<uint64_t>> &ss_r_data,
    bool use_lwe_output,
    std::vector<LweSharedGroup> *lwe_groups,
    const int discard_lsb)
{
    const auto conv_transport_before = jaguar::profiler().transport_snapshot();
    auto conv_start = std::chrono::high_resolution_clock::now();
    if (jaguar_quant::enabled() && jaguar_quant::is_client()) {
        if (use_lwe_output) {
            if (lwe_groups == nullptr) throw std::runtime_error("call_depthwise_conv_quant_mobilenet_v2: lwe_groups is null on client");
            jaguar_quant::recv_lwe_groups(*lwe_groups);
        } else {
            jaguar_quant::recv_ciphertexts(context_quant, cout);
        }
        auto conv_end = std::chrono::high_resolution_clock::now();
        const auto conv_transport_after = jaguar::profiler().transport_snapshot();
        const uint64_t pending_conv_sent_bytes = jaguar_quant::consume_pending_conv_input_sent_bytes();
        const uint64_t pending_conv_actual_bytes = jaguar_quant::consume_pending_conv_input_actual_bytes();
        jaguar::LayerSummary layer_summary;
        layer_summary.kind = "conv_total";
        layer_summary.name = "conv_quant_" + std::to_string(order);
        layer_summary.total_latency_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(conv_end - conv_start).count());
        layer_summary.compute_latency_us = 0;
        layer_summary.comm_bytes = (conv_transport_after.sent_bytes - conv_transport_before.sent_bytes) + pending_conv_sent_bytes;
        layer_summary.comm_bytes_compat = layer_summary.comm_bytes;
        const uint64_t conv_recv_bytes = (conv_transport_after.recv_bytes - conv_transport_before.recv_bytes);
        layer_summary.comm_bytes_actual = pending_conv_actual_bytes + layer_summary.comm_bytes + conv_recv_bytes;
        layer_summary.rounds = 0;
        layer_summary.details = "order=" + std::to_string(order) + ",party=client,depthwise=1,ct_count=" + std::to_string(cout.size()) + ",pending_input_sent=" + std::to_string(pending_conv_sent_bytes) + ",recv_bytes=" + std::to_string(conv_recv_bytes);
        jaguar::profiler().record_layer_summary(layer_summary);
        return;
    } else {
        constexpr int kMobileNetV2DepthwiseThreads = 4;
        if (kMobileNetV2DepthwiseThreads > 1) {
            cloud_depthwise_padded_new_CNN_quant_mobilenet_v2_mt<N>(
                kMobileNetV2DepthwiseThreads, 1, 0, 0, 0, seed, w, filter_depthwise_full,
                context_quant, log2_q, log2_p, cin, cout, ss_r_data, discard_lsb);
        } else {
            cloud_depthwise_padded_new_CNN_quant_mobilenet_v2_st<N>(
                1, 0, 0, 0, seed, w, filter_depthwise_full,
                context_quant, log2_q, log2_p, cin, cout, ss_r_data, discard_lsb);
        }
        if (jaguar_quant::enabled() && jaguar_quant::is_cloud()) {
            if (use_lwe_output) {
                if (lwe_groups == nullptr) throw std::runtime_error("call_depthwise_conv_quant_mobilenet_v2: lwe_groups is null on cloud");
                int next_w_dummy = 0;
                build_shared_lwe_groups_from_cout_quant<N>(
                    cout, extract_stride, 3, w, static_cast<int>(filter_depthwise_full.size()), log2_q, *lwe_groups, next_w_dummy);
                jaguar_quant::send_lwe_groups(*lwe_groups);
            } else {
                jaguar_quant::send_ciphertexts(cout);
            }
        }
    }
    auto conv_end = std::chrono::high_resolution_clock::now();
    const auto conv_transport_after = jaguar::profiler().transport_snapshot();
    jaguar::LayerSummary layer_summary;
    layer_summary.kind = "conv_total";
    layer_summary.name = "conv_quant_" + std::to_string(order);
    layer_summary.total_latency_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(conv_end - conv_start).count());
    layer_summary.compute_latency_us = layer_summary.total_latency_us;
    layer_summary.comm_bytes = (conv_transport_after.sent_bytes - conv_transport_before.sent_bytes);
    layer_summary.comm_bytes_compat = layer_summary.comm_bytes;
    const uint64_t conv_recv_bytes = (conv_transport_after.recv_bytes - conv_transport_before.recv_bytes);
    layer_summary.comm_bytes_actual = layer_summary.comm_bytes + conv_recv_bytes;
    layer_summary.rounds = 0;
    layer_summary.details = "order=" + std::to_string(order) + ",party=cloud,depthwise=1,ct_count=" + std::to_string(cout.size()) + ",recv_bytes=" + std::to_string(conv_recv_bytes);
    jaguar::profiler().record_layer_summary(layer_summary);
}

template <int N>
static void mobilenet_v2_stage_quant(
    const seal::SEALContext &context_quant,
    const uint64_t log2_q,
    const uint64_t log2_p,
    const seal::Plaintext &secret_key_coeff,
    std::vector<seal::Ciphertext> &ct_out,
    int conv_stride,
    uint64_t f_w,
    int &w,
    bool relu,
    bool pooling,
    int pooling_k,
    int pooling_s,
    int pooling_p,
    std::vector<std::vector<uint64_t>> &ss_r_data,
    uint64_t next_f_w,
    std::vector<seal::Ciphertext> &ct_next,
    const seal::SEALContext &context_ntt,
    const uint64_t q_ntt,
    const seal::PublicKey &public_key,
    const seal::SecretKey &secret_key,
    const int discard_lsb,
    std::vector<LweSharedGroup> &lwe_groups,
    bool use_lwe_output,
    const bool use_relu6)
{
    stride_relu_avg_pooling_quant<N>(
        context_quant, log2_q, log2_p, secret_key_coeff,
        ct_out, conv_stride, f_w, w, relu,
        pooling, pooling_k, pooling_s, pooling_p,
        ss_r_data, next_f_w, ct_next, context_ntt, q_ntt,
        public_key, secret_key, discard_lsb, use_lwe_output, (use_lwe_output ? &lwe_groups : nullptr), use_relu6);
}

template <int N>
void mobilenet_v2_quant(
    std::string str,
    const seal::SEALContext &context_quant,
    const uint64_t log2_q,
    const uint64_t log2_p,
    const seal::Plaintext &secret_key_coeff,
    std::vector<std::vector<uint64_t>> &image,
    int &w,
    const seal::SEALContext &context_ntt,
    const seal::PublicKey &public_key,
    const seal::SecretKey &secret_key,
    std::vector<seal::Ciphertext> &ct,
    int &discard_lsb_model_output)
{
    using jaguar_networks::MobileNetV2BlockSpecQuant;
    static const MobileNetV2BlockSpecQuant blocks[] = {
        {32, 16, 1, false, false}, {96, 24, 2, false, true},
        {144, 24, 1, true, true},  {144, 32, 2, false, true},
        {192, 32, 1, true, true},  {192, 32, 1, true, true},
        {192, 64, 2, false, true}, {384, 64, 1, true, true},
        {384, 64, 1, true, true},  {384, 64, 1, true, true},
        {384, 96, 1, false, true}, {576, 96, 1, true, true},
        {576, 96, 1, true, true},  {576, 160, 2, false, true},
        {960, 160, 1, true, true}, {960, 160, 1, true, true},
        {960, 320, 1, false, true},
    };

    if (str != "st" && str != "mt") {
        throw std::runtime_error("Choose st / mt");
    }
    if (w != 224) {
        throw std::runtime_error("mobilenet_v2_quant currently supports only w=224");
    }

    std::vector<int> th_count(mobilenetv2_in224_ch.size(), 0);
    std::vector<int> discard_lsb(mobilenetv2_in224_ch.size(), 9);
    if (str == "mt") {
        std::fill(th_count.begin(), th_count.end(), 4);
    }
    discard_lsb_model_output = discard_lsb.back();

    auto &context_ntt_data = *context_ntt.first_context_data();
    auto &parms_ntt = context_ntt_data.parms();
    auto &coeff_modulus_ntt = parms_ntt.coeff_modulus();
    uint64_t q_ntt = coeff_modulus_ntt[0].value();

    int seed = 0;
    std::vector<seal::Ciphertext> ct_in;
    std::vector<seal::Ciphertext> ct_out;
    std::vector<seal::Ciphertext> ct_skip;
    std::vector<std::vector<uint64_t>> ss_r_data;
    std::vector<LweSharedGroup> lwe_groups;

    std::string s = "../models/filters_mobilenetv2_224_scale9/";
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> filters;
    std::vector<uint64_t> v_f_w;
    std::vector<uint64_t> v_conv_stride;
    std::vector<bool> v_pooling;
    std::vector<int> v_pooling_k;
    std::vector<int> v_pooling_s;
    std::vector<int> v_pooling_p;

    for (size_t i = 0; i < mobilenetv2_in224_ch.size(); ++i) {
        filters.push_back(read_filters(s + "layer_" + std::to_string(i) + ".txt",
                                       mobilenetv2_in224_ch[i].filter_size,
                                       mobilenetv2_in224_ch[i].filter_size));
        v_f_w.push_back(mobilenetv2_in224_ch[i].filter_size);
        v_conv_stride.push_back(mobilenetv2_in224_ch[i].stride);
        v_pooling.push_back(mobilenetv2_in224_ch[i].pooling);
        v_pooling_k.push_back(mobilenetv2_in224_ch[i].pooling_kernel_size);
        v_pooling_s.push_back(mobilenetv2_in224_ch[i].pooling_stride);
        v_pooling_p.push_back(mobilenetv2_in224_ch[i].pooling_padding);
    }

    jaguar_networks::configure_mobilenet_v2_runtime_params(filters);
    jaguar_networks::mobilenet_v2_log_model_summary();
    jaguar_networks::mobilenet_v2_set_quant_active(true);

    print_title("QUANTIZATION MobileNetV2");
    print_title("Client Input Encryption");
    client_enc_zero_padding_quant(context_quant, v_f_w[0], secret_key_coeff, image,
                                  ct_in, log2_q, log2_p, 0, context_ntt);

    int cfg_idx = 0;
    call_conv_quant<N>(cfg_idx, th_count[cfg_idx], 1, 0, 0, 0, seed, w, filters[cfg_idx],
        context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data,
        v_conv_stride[cfg_idx], true, &lwe_groups, discard_lsb[cfg_idx]);
    jaguar_networks::mobilenet_v2_apply_post_conv_bias_quant(cfg_idx, log2_p, ss_r_data);
    mobilenet_v2_stage_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff,
        ct_out, v_conv_stride[cfg_idx], v_f_w[cfg_idx], w, 1,
        v_pooling[cfg_idx], v_pooling_k[cfg_idx], v_pooling_s[cfg_idx], v_pooling_p[cfg_idx],
        ss_r_data, v_f_w[cfg_idx + 1], ct_in, context_ntt, q_ntt,
        public_key, secret_key, discard_lsb[cfg_idx], lwe_groups, true, true);
    ++cfg_idx;

    for (const auto &block : blocks) {
        ct_skip = ct_in;
        if (block.has_expand) {
            int idx = cfg_idx++;
            call_conv_quant<N>(idx, th_count[idx], 1, 0, 0, 0, seed, w, filters[idx],
                context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data,
                v_conv_stride[idx], true, &lwe_groups, discard_lsb[idx]);
            jaguar_networks::mobilenet_v2_apply_post_conv_bias_quant(idx, log2_p, ss_r_data);
            mobilenet_v2_stage_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff,
                ct_out, v_conv_stride[idx], v_f_w[idx], w, 1,
                v_pooling[idx], v_pooling_k[idx], v_pooling_s[idx], v_pooling_p[idx],
                ss_r_data, v_f_w[idx + 1], ct_in, context_ntt, q_ntt,
                public_key, secret_key, discard_lsb[idx], lwe_groups, true, true);
        }
        {
            int idx = cfg_idx++;
            call_depthwise_conv_quant_mobilenet_v2<N>(
                idx, seed, w, filters[idx], v_conv_stride[idx], context_quant, log2_q, log2_p,
                ct_in, ct_out, ss_r_data, true, &lwe_groups, discard_lsb[idx]);
            jaguar_networks::mobilenet_v2_apply_post_conv_bias_quant(idx, log2_p, ss_r_data);
            mobilenet_v2_stage_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff,
                ct_out, v_conv_stride[idx], v_f_w[idx], w, 1,
                v_pooling[idx], v_pooling_k[idx], v_pooling_s[idx], v_pooling_p[idx],
                ss_r_data, v_f_w[idx + 1], ct_in, context_ntt, q_ntt,
                public_key, secret_key, discard_lsb[idx], lwe_groups, true, true);
        }
        {
            int idx = cfg_idx++;
            call_conv_quant<N>(idx, th_count[idx], 1, 0, 0, 0, seed, w, filters[idx],
                context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data,
                v_conv_stride[idx], true, &lwe_groups, discard_lsb[idx]);
            jaguar_networks::mobilenet_v2_apply_post_conv_bias_quant(idx, log2_p, ss_r_data);
            mobilenet_v2_stage_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff,
                ct_out, v_conv_stride[idx], v_f_w[idx], w, 0,
                v_pooling[idx], v_pooling_k[idx], v_pooling_s[idx], v_pooling_p[idx],
                ss_r_data, (idx + 1 < static_cast<int>(v_f_w.size()) ? v_f_w[idx + 1] : 1),
                ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[idx], lwe_groups, true, false);
            if (block.use_residual) {
                cloud_add_residual_path_quant<N>(
                    context_quant, log2_q, log2_p, seed, w, filters[idx],
                    ct_in, ct_skip, ct_out, ss_r_data, discard_lsb[idx], true, &lwe_groups);
                mobilenet_v2_stage_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff,
                    ct_out, 1, v_f_w[idx], w, 0,
                    false, 0, 0, 0, ss_r_data,
                    (idx + 1 < static_cast<int>(v_f_w.size()) ? v_f_w[idx + 1] : 1),
                    ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[idx], lwe_groups, true, false);
            }
        }
    }

    if (cfg_idx >= static_cast<int>(filters.size())) {
        throw std::runtime_error("mobilenet_v2_quant exhausted filter config before final conv");
    }

    call_conv_quant<N>(cfg_idx, th_count[cfg_idx], 1, 0, 0, 0, seed, w, filters[cfg_idx],
        context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data,
        v_conv_stride[cfg_idx], true, &lwe_groups, discard_lsb[cfg_idx]);
    jaguar_networks::mobilenet_v2_apply_post_conv_bias_quant(cfg_idx, log2_p, ss_r_data);
    mobilenet_v2_stage_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff,
        ct_out, v_conv_stride[cfg_idx], v_f_w[cfg_idx], w, 1,
        v_pooling[cfg_idx], v_pooling_k[cfg_idx], v_pooling_s[cfg_idx], v_pooling_p[cfg_idx],
        ss_r_data, 1, ct, context_ntt, q_ntt, public_key, secret_key, discard_lsb[cfg_idx], lwe_groups, true, true);

    discard_lsb_model_output = discard_lsb[cfg_idx];
    run_model_fc_quant(
        "mobilenet_v2", context_quant, log2_q, log2_p, secret_key_coeff,
        context_ntt, public_key, secret_key, ct, discard_lsb_model_output);
    jaguar_networks::mobilenet_v2_set_quant_active(false);
}
