#include <algorithm>
#include <stdexcept>
#include <vector>

#include "seal/seal.h"
#include "client_lwe_shared.h"

static uint64_t local_required_bit_width(uint64_t max_value) {
    uint64_t bits = 1;
    while (max_value >>= 1) ++bits;
    return bits;
}

#include "client_lwe_export.h"

using namespace std;

static void enumerate_conv_output_positions_quant(
    int poly_degree,
    int stride,
    int f_w,
    int w,
    int num_output_channels,
    int num_ct_out,
    std::vector<ConvExtractPos> &positions,
    int &next_w)
{
    if (poly_degree <= 0)
        throw invalid_argument("poly_degree must be positive");
    if (stride <= 0)
        throw invalid_argument("Convolution stride can't be 0");
    if (f_w <= 0)
        throw invalid_argument("Convolution filter width can't be 0");
    if (w <= 0)
        throw invalid_argument("Convolution input width can't be 0");

    const int p_l = (f_w - 1) / 2;
    const int p_r = (f_w - 1) / 2 + ((f_w - 1) % 2 ? 1 : 0);
    const int wPp = w + p_l + p_r;
    next_w = 1 + (wPp - f_w) / stride;

    positions.clear();
    if (num_output_channels <= 0 || num_ct_out <= 0)
        return;

    if (wPp * wPp <= poly_degree)
    {
        const int channel_count = std::min(num_output_channels, num_ct_out);
        positions.reserve(static_cast<size_t>(channel_count) * next_w * next_w);
        for (int out_channel = 0; out_channel < channel_count; ++out_channel)
        {
            for (int j = 0; j < next_w; ++j)
            {
                for (int k = 0; k < next_w; ++k)
                {
                    const int start_j = j * stride + p_l;
                    const int start_k = k * stride + p_l;
                    positions.push_back({
                        out_channel,
                        start_j * wPp + start_k,
                        out_channel,
                        j * next_w + k
                    });
                }
            }
        }
        return;
    }
    const int first_col_num = (poly_degree - (poly_degree % wPp)) / wPp;
    const int col_num = first_col_num - 2 * p_l;
    const int col_num_num = ((wPp - first_col_num) % col_num)
                              ? ((wPp - first_col_num) / col_num + 1)
                              : ((wPp - first_col_num) / col_num);
    const int num_packed_inv = 1 + col_num_num;
    const int channel_count = std::min(num_output_channels, num_ct_out / num_packed_inv);
    positions.reserve(static_cast<size_t>(channel_count) * next_w * next_w);

    for (int out_channel = 0; out_channel < channel_count; ++out_channel)
    {
        int idx2 = 0;
        int start_j = p_l;
        for (int j = 0; j < next_w; ++j)
        {
            for (int k = 0; k < next_w; ++k)
            {
                if ((j * stride) / col_num > idx2)
                {
                    idx2++;
                    start_j = p_l + (j * stride) % col_num;
                }
                const int start_k = k * stride + p_l;
                positions.push_back({
                    out_channel * num_packed_inv + idx2,
                    start_j * wPp + start_k,
                    out_channel,
                    j * next_w + k
                });
                // Jaguar_52: same row-advance rule as client_ct_stride_dec_quant.
                if (k == next_w - 1)
                    start_j += stride;
            }
        }
    }
}

template <int N>
void build_shared_lwe_groups_from_cout_quant(
    const std::vector<seal::Ciphertext> &cout,
    int stride,
    int f_w,
    int w,
    int num_output_channels,
    const uint64_t log2_q,
    std::vector<LweSharedGroup> &groups,
    int &next_w)
{
    std::vector<ConvExtractPos> positions;
    enumerate_conv_output_positions_quant(
        N, stride, f_w, w, num_output_channels, static_cast<int>(cout.size()), positions, next_w);

    std::vector<std::vector<ConvExtractPos> > grouped_positions(cout.size());
    for (const auto &pos : positions)
    {
        if (pos.src_ct_index < 0 || pos.src_ct_index >= static_cast<int>(cout.size()))
        {
            throw std::out_of_range("src_ct_index");
        }
        grouped_positions[static_cast<size_t>(pos.src_ct_index)].push_back(pos);
    }

    groups.clear();
    groups.reserve(cout.size());
    for (size_t src_ct_index = 0; src_ct_index < grouped_positions.size(); ++src_ct_index)
    {
        const auto &ct_positions = grouped_positions[src_ct_index];
        if (ct_positions.empty())
        {
            continue;
        }

        LweSharedGroup group;
        group.src_ct_index = static_cast<int>(src_ct_index);
        group.a_shared_bit_width = 0;
        group.b_values_bit_width = 0;
        group.a_shared.assign(static_cast<size_t>(N), 0);
        std::copy_n(cout[src_ct_index].data(1), static_cast<size_t>(N), group.a_shared.begin());
        group.b_values.reserve(ct_positions.size());
        group.coeff_indices.reserve(ct_positions.size());
        group.out_channels.reserve(ct_positions.size());
        group.out_offsets.reserve(ct_positions.size());

        for (const auto &pos : ct_positions)
        {
            group.b_values.push_back(cout[src_ct_index].data(0)[pos.coeff_index]);
            group.coeff_indices.push_back(pos.coeff_index);
            group.out_channels.push_back(pos.out_channel);
            group.out_offsets.push_back(pos.out_offset);
        }

        uint64_t max_a = 0;
        for (uint64_t value : group.a_shared) max_a = std::max(max_a, value);
        uint64_t max_b = 0;
        for (uint64_t value : group.b_values) max_b = std::max(max_b, value);
        group.a_shared_bit_width = local_required_bit_width(max_a);
        group.b_values_bit_width = local_required_bit_width(max_b);

        groups.push_back(std::move(group));
    }
}

template void build_shared_lwe_groups_from_cout_quant<2048>(
    const std::vector<seal::Ciphertext> &cout,
    int stride,
    int f_w,
    int w,
    int num_output_channels,
    const uint64_t log2_q,
    std::vector<LweSharedGroup> &groups,
    int &next_w);
