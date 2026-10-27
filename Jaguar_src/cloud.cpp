#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <mutex>
#include <map>
#include <tuple>
#include <numeric> 
#include <algorithm>
#include <cmath>
#include <fstream>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>

#include "seal/seal.h"
#include "seal/util/uintarithsmallmod.h"
//#include "seal/util/polyarith.h"
#include "seal/util/polyarithsmallmod.h"
//#include "seal/util/polyarithsmallmod.cpp"
#include "seal/keygenerator.h"


#include <omp.h>

#include <stdexcept>
#include "seal/util/ntt.h"
#include "jaguar_relu.h"
#include "jaguar_profiler.h"
#include "jaguar_rerandomize.h"
#include "../SCI/src/utils/prg.h"

using namespace std;
using namespace seal;
using namespace seal::util;


#if defined(USE_AVX512)
namespace seal::util {
    void discard_lsb_ct_opt_fast(Ciphertext &ct, const int discard_lsb, const int N);
    void add_plain_plain_quant_opt_fast(CoeffIter poly0, ConstCoeffIter poly1, const uint64_t log2_q, const int N);
    void reduce_quant_poly_opt_fast(CoeffIter operand, CoeffIter poly, const int N, const uint64_t log2_q);
}
#endif

#if defined(USE_AVX512)
#define JAGUAR_MULTIPLY_SCALAR_POLY_QUANT multiply_scalar_poly_quant_opt_fast
#define JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT rotate_multiply_scalar_add_poly_quant_opt_fast
#define JAGUAR_DISCARD_LSB_CT discard_lsb_ct_opt_fast
#define JAGUAR_ADD_PLAIN_PLAIN_QUANT add_plain_plain_quant_opt_fast
#define JAGUAR_REDUCE_QUANT_POLY reduce_quant_poly_opt_fast
#else
#define JAGUAR_MULTIPLY_SCALAR_POLY_QUANT multiply_scalar_poly_quant
#define JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT rotate_multiply_scalar_add_poly_quant
#define JAGUAR_DISCARD_LSB_CT discard_lsb_ct
#define JAGUAR_ADD_PLAIN_PLAIN_QUANT add_plain_plain_quant
#define JAGUAR_REDUCE_QUANT_POLY reduce_quant_poly
#endif



void avg_pooling(vector<vector<uint64_t> > &relu_data, int pooling_kernel_size, int pooling_stride, int pooling_padding);

void zero_padding_to_N(const SEALContext & context, const int f_w, const vector<vector<uint64_t> >  &din, vector<vector<uint64_t> > &din_new);


static void stride_extract_packed_quant(const vector<vector<uint64_t> > &packed_data,
                                       int poly_degree,
                                       int stride,
                                       int f_w,
                                       int w,
                                       vector<vector<uint64_t> > &out,
                                       int &next_w)
{
    if (stride == 0)
        throw invalid_argument("Convolution stride can't be 0");
    if (f_w == 0)
        throw invalid_argument("Convolution filter's width can't be 0");
    if (w == 0)
        throw invalid_argument("Convolution input's width can't be 0");
    if (packed_data.empty())
        throw invalid_argument("packed_data can't be empty");

    const int p_l = (f_w - 1) / 2;
    const int p_r = (f_w - 1) / 2 + ((f_w - 1) % 2 ? 1 : 0);
    const int wPp = w + p_l + p_r;
    const int new_w = 1 + (wPp - f_w) / stride;
    next_w = new_w;

    out.assign(1, vector<uint64_t>(new_w * new_w, 0));
    int cnt = 0;
    int idx = 0;

    if (wPp * wPp <= poly_degree)
    {
        for (int i = 0; i < static_cast<int>(packed_data.size()); i++)
        {
            for (int j = 0; j < new_w; j++)
            {
                for (int k = 0; k < new_w; k++)
                {
                    const int start_j = j * stride + p_l;
                    const int start_k = k * stride + p_l;
                    out[idx][cnt++] = packed_data[i][start_j * wPp + start_k];
                }
            }
            cnt = 0;
            idx++;
            out.emplace_back(vector<uint64_t>(new_w * new_w, 0));
        }
        out.erase(out.end() - 1);
        return;
    }

    const int first_col_num = (poly_degree - (poly_degree % wPp)) / wPp;
    const int col_num = first_col_num - 2 * p_l;
    const int col_num_num = ((wPp - first_col_num) % col_num)
                                ? ((wPp - first_col_num) / col_num + 1)
                                : ((wPp - first_col_num) / col_num);
    const int num_packed_inv = 1 + col_num_num;

    cnt = 0;
    idx = 0;
    int idx2 = 0;
    int start_j = 0;
    int start_k = 0;
    for (int i = 0; i < static_cast<int>(packed_data.size()); i += num_packed_inv)
    {
        idx2 = 0;
        start_j = p_l;
        const vector<uint64_t> *temp = &packed_data[i];
        for (int j = 0; j < new_w; j++)
        {
            for (int k = 0; k < new_w; k++)
            {
                if ((j * stride) / col_num > idx2)
                {
                    temp = &packed_data[i + idx2 + 1];
                    idx2++;
                    start_j = p_l + (j * stride) % col_num;
                }
                start_k = k * stride + p_l;
                out[idx][cnt++] = (*temp)[start_j * wPp + start_k];
                // Jaguar_52: the original never advanced start_j within a
                // ciphertext, so the cloud's ReLU-input share was read from a
                // different row than the client's decrypt in split layers.
                if (k == new_w - 1)
                    start_j += stride;
            }
        }
        cnt = 0;
        idx++;
        out.emplace_back(vector<uint64_t>(new_w * new_w, 0));
    }
    out.erase(out.end() - 1);
}

void relu_share_ring_ideal(const vector<vector<uint64_t> > &client_share, const vector<vector<uint64_t> > &ss_r_data, const uint64_t bitwidth, vector<vector<uint64_t> > &client_relu_share, vector<vector<uint64_t> > &cloud_relu_share)
{
    jaguar::relu_share_ring(client_share, ss_r_data, bitwidth, client_relu_share, cloud_relu_share);
}

void pos_rotate_poly128(Ciphertext &cout, const int n, const int step, Ciphertext &destination)
{
	std::cout << "\ncout.data(0): ";
	for (int i=0; i<10; i++)
	{
		std::cout << cout.data(0)[i] << " ";
	}
	std::cout << "\nRotated cout.data(0): ";
	SEAL_ITERATE(iter(cout.data(0) + step, cout.data(1)), 10, [&](auto I)
	{
		std::cout << get<0>(I) << " ";
	});
	std::cout << "\nRotated cout.data(0): ";
	SEAL_ITERATE(iter(cout.data(0)+ step, cout.data(1)), 10, [&](auto I)
	{
		std::cout << get<0>(I) << " ";
	});	

}


template <int N>
void padded_new_conv_quant(const int in_h, const int num_in_j, const int f_size, const int n_i, const SEALContext &context, vector<uint64_t> &enc_filter, const Ciphertext &cin, Ciphertext &cout)
{
	int f_h = sqrt(f_size);

	auto &context_data = *context.key_context_data();
    int n = context_data.parms().poly_modulus_degree();
    auto &coeff_modulus = context_data.parms().coeff_modulus();
	Modulus coeff_modulus_0 = context_data.parms().coeff_modulus()[0];
    uint64_t mod_q = coeff_modulus_0.value();
    size_t coeff_modulus_size = coeff_modulus.size();
	
    int in_size = in_h * in_h;
	
    JAGUAR_MULTIPLY_SCALAR_POLY_QUANT(cin, n, enc_filter[(f_size-1)/2], cout);

	for (int i=1; i<(f_h-1)/2+1; i++)
	{
		JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, i, mod_q, enc_filter[(f_size-1)/2 + i], cout);
		JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, -i, mod_q, enc_filter[(f_size-1)/2 - i], cout);
		JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, in_h*i, mod_q, enc_filter[(f_size-1)/2 + f_h*i], cout);
		JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, -in_h*i, mod_q, enc_filter[(f_size-1)/2 - f_h*i], cout);

		for (int j=1; j<(f_h-1)/2+1; j++)
		{
			JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, j+ in_h*i, mod_q, enc_filter[(f_size-1)/2 + j+ f_h*i], cout);
			JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, -j + in_h*i, mod_q, enc_filter[(f_size-1)/2 - j + f_h*i], cout);
			JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, j - in_h*i, mod_q, enc_filter[(f_size-1)/2 + j - f_h*i], cout);
			JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, -j - in_h*i, mod_q, enc_filter[(f_size-1)/2 -j - f_h*i], cout);
		}
	}
}



template <int N>
void padded_new_rotate_conv_quant(const int in_h, const int num_in_j, const int f_size, const int n_i, const SEALContext &context, vector<uint64_t> &enc_filter, const Ciphertext &cin, Ciphertext &cout)
{
	int f_h = sqrt(f_size);

	auto &context_data = *context.key_context_data();
    int n = context_data.parms().poly_modulus_degree();
    auto &coeff_modulus = context_data.parms().coeff_modulus();
	Modulus coeff_modulus_0 = context_data.parms().coeff_modulus()[0];
    uint64_t mod_q = coeff_modulus_0.value();
    size_t coeff_modulus_size = coeff_modulus.size();
	
    int in_size = in_h * in_h;
    int step = (num_in_j % n_i) * in_size;
    if (in_size > N)
        step = 0;

	JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, step, mod_q, enc_filter[(f_size-1)/2], cout);

	for (int i=1; i<(f_h-1)/2+1; i++)
	{
		JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, i + step, mod_q, enc_filter[(f_size-1)/2 + i], cout);
		JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, -i + step, mod_q, enc_filter[(f_size-1)/2 - i], cout);
		JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, in_h*i + step, mod_q, enc_filter[(f_size-1)/2 + f_h*i], cout);
		JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, -in_h*i + step, mod_q, enc_filter[(f_size-1)/2 - f_h*i], cout);

		for (int j=1; j<(f_h-1)/2+1; j++)
		{
			JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, j+ in_h*i + step, mod_q, enc_filter[(f_size-1)/2 + j+ f_h*i], cout);
			JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, -j + in_h*i + step, mod_q, enc_filter[(f_size-1)/2 - j + f_h*i], cout);
			JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, j - in_h*i + step, mod_q, enc_filter[(f_size-1)/2 + j - f_h*i], cout);
			JAGUAR_ROTATE_MULTIPLY_SCALAR_ADD_POLY_QUANT(cin, n, -j - in_h*i + step, mod_q, enc_filter[(f_size-1)/2 -j - f_h*i], cout);
		}
	}
}




template <int N>
struct QuantRandomShareCacheEntry {
    vector<Plaintext> ss_r;
    vector<vector<uint64_t>> ss_r_data;
};

static uint64_t jaguar_secure_random_u64()
{
    static thread_local sci::PRG128 prg;
    uint64_t value = 0;
    prg.random_data(&value, sizeof(value));
    return value;
}

template <int N>
static const QuantRandomShareCacheEntry<N> &get_random_ss_r_quant_cache(
    const int num_ct_out,
    const uint64_t log2_q,
    const uint64_t log2_p,
    const int discard_lsb)
{
    static std::mutex cache_mutex;
    static std::map<std::tuple<int, uint64_t, uint64_t, int>, QuantRandomShareCacheEntry<N>> cache;
    const auto key = std::make_tuple(num_ct_out, log2_q, log2_p, discard_lsb);
    // Jaguar_52: a fresh HE mask r for every layer and request (as in the
    // ISCA-Jaguar-CiM v51+ CPU baseline). The original reused r for repeated
    // shapes; JAGUAR_LEGACY_MASK_CACHE=1 restores it for comparison only.
    static const bool fresh_mask = [] {
        const char *v = std::getenv("JAGUAR_LEGACY_MASK_CACHE");
        return !(v != nullptr && std::string(v) == "1");
    }();
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (!fresh_mask && it != cache.end()) {
            return it->second;
        }
    }

    const int effective_bits_i = std::max<int>(1, static_cast<int>(log2_p) - discard_lsb);
    const uint64_t effective_bits = static_cast<uint64_t>(effective_bits_i);
    const uint64_t mask_p = (effective_bits >= 64) ? ~0ULL : ((1ULL << effective_bits) - 1ULL);
    const int shift = static_cast<int>(log2_q - log2_p);
    uint64_t qM1 = (1ULL << log2_q) - 1;

    QuantRandomShareCacheEntry<N> entry;
    entry.ss_r.assign(num_ct_out, Plaintext(N));
    entry.ss_r_data.assign(num_ct_out, vector<uint64_t>(N, 0));

    for (int i = 0; i < num_ct_out; ++i) {
        for (int j = 0; j < N; ++j) {
            uint64_t r = jaguar_secure_random_u64();
            if (log2_p < 64) {
                r &= mask_p;
            }
            entry.ss_r_data[i][j] = r;
            entry.ss_r[i].data(0)[j] = (r << shift) & qM1;
        }
    }

    if (fresh_mask) {
        static thread_local QuantRandomShareCacheEntry<N> fresh;
        fresh = std::move(entry);
        return fresh;
    }
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto result = cache.emplace(key, std::move(entry));
    return result.first->second;
}


template <int N>
void cloud_padded_new_CNN_quant_st(const int layer, const int flag_sw, const int flag_perm, const int seed_pre, int &seed, const int in_h, vector<vector<vector<uint64_t>>> &filter, const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p, const vector<Ciphertext> &cin, vector<Ciphertext> &cout, vector<vector<uint64_t> > &ss_r_data, const int discard_lsb=0)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_cloud_sum(0);
    chrono::microseconds time_quant_sum(0);
    chrono::microseconds time_off_cloud_sum(0);
    chrono::microseconds time_trunc_sum(0);

    int num_fout = filter.size();
    int num_in = filter[0].size();
    int f_size = filter[0][0].size();
    int f_h = sqrt(f_size);

    auto &context_data = *context.key_context_data();
	Modulus coeff_modulus = context_data.parms().coeff_modulus()[0];
    size_t coeff_modulus_size = context_data.parms().coeff_modulus().size();

    uint64_t mod_q = coeff_modulus.value();
    auto &plain_modulus = context_data.parms().plain_modulus();
    uint64_t mod_p = plain_modulus.value();

	const uint64_t delta = (1ULL<<(log2_q-log2_p));
    
    int p_l = (f_h - 1) / 2;
    int p_r = (f_h - 1) / 2 + ((f_h - 1) % 2 ? 1 : 0);
    int wPp = in_h + p_l + p_r;
    int in_size = wPp * wPp;
    int n = context_data.parms().poly_modulus_degree();
    int n_i;
    int num_out = f_size * num_in * num_fout;
	int first_col_num, col_num, col_num_num, remain_zero;

    if (in_size > n) {
		remain_zero = n%wPp;
		first_col_num = (n- remain_zero)/wPp;
		col_num = first_col_num - 2*p_l;
		col_num_num = ((wPp - first_col_num)%col_num)?((wPp - first_col_num)/col_num + 1):(wPp - first_col_num)/col_num;
        n_i = 1 + col_num_num;
    }
    else { 
        num_out = f_size * num_in * (num_fout / (n / in_size) + (num_fout % (n / in_size)? 1: 0));
        n_i = n / in_size;
    }
   
	int coeff_modulus_bit_count = get_significant_bit_count(coeff_modulus.value());

    int num_ct_in = cin.size(); // = # of input plaintext (packed or splited)
    int num_ct_out;
    if (in_size > n)
	{
        num_ct_out = num_fout * n_i;
		cout.resize(num_ct_out, cin[0]);
	}
    else
	{
		num_ct_out = num_fout;
		cout.resize(num_fout, cin[0]);
	}

	// Generate ss_r outside the timed conv body.
	const auto &share_cache = get_random_ss_r_quant_cache<N>(num_ct_out, log2_q, log2_p, discard_lsb);
	const auto &ss_r = share_cache.ss_r;
	auto ss_r_copy_start = chrono::high_resolution_clock::now();
	ss_r_data = share_cache.ss_r_data;
	auto ss_r_copy_end = chrono::high_resolution_clock::now();
	auto ss_r_copy_us = static_cast<uint64_t>(chrono::duration_cast<chrono::microseconds>(ss_r_copy_end - ss_r_copy_start).count());
	jaguar::LayerSummary ss_r_copy_summary;
	ss_r_copy_summary.kind = "excluded_overhead";
	ss_r_copy_summary.name = "ss_r_data_copy";
	ss_r_copy_summary.total_latency_us = ss_r_copy_us;
	ss_r_copy_summary.compute_latency_us = ss_r_copy_us;
	ss_r_copy_summary.comm_bytes = 0;
	ss_r_copy_summary.rounds = 0;
	ss_r_copy_summary.details = "num_ct_out=" + std::to_string(num_ct_out);
	jaguar::profiler().record_layer_summary(ss_r_copy_summary);

	int idx = 0;
	int loop = 1;

	vector<Ciphertext> result2_ct(n_i, cin[0]);
    Ciphertext result_ct;
    result_ct = cin[0];
    set_zero_poly(N, 1, result_ct.data(0));
    set_zero_poly(N, 1, result_ct.data(1));

    Ciphertext temp_ct;
    temp_ct = cin[0];
    set_zero_poly(N, 1, temp_ct.data(0));
    set_zero_poly(N, 1, temp_ct.data(1));

	time_start = chrono::high_resolution_clock::now();
	if (in_size <= n)
	{
		for (int i=0; i<num_fout; i++)
		{
			for (int j=0; j<num_in; j++)
			{
				if (j == 0)
				{
				    padded_new_conv_quant<N>(wPp, j, f_size, n_i, context, filter[i][j], cin[j/n_i], result_ct);
				}
				else if (j>0)
				{
				    padded_new_rotate_conv_quant<N>(wPp, j, f_size, n_i, context, filter[i][j], cin[j/n_i], result_ct);
				}
			}

            jaguar::rerandomize_conv_output(result_ct);
            auto trunc_start = chrono::high_resolution_clock::now();
            JAGUAR_DISCARD_LSB_CT(result_ct, discard_lsb, N);
            auto trunc_end = chrono::high_resolution_clock::now();
            time_trunc_sum += chrono::duration_cast<chrono::microseconds>(trunc_end - trunc_start);
            JAGUAR_ADD_PLAIN_PLAIN_QUANT(result_ct.data(0), ss_r[i].data(), log2_q, N);
            JAGUAR_REDUCE_QUANT_POLY(result_ct.data(0), cout[i].data(0), N, log2_q);
            JAGUAR_REDUCE_QUANT_POLY(result_ct.data(1), cout[i].data(1), N, log2_q);
		}
	}
	else
	{
		for (int i=0; i<num_fout; i++)
		{
			for (int j=0; j<num_in; j++)
			{
				for (int k=0; k<n_i; k++)
				{
					if (j==0)
					{
						padded_new_conv_quant<N>(wPp, j, f_size, n_i, context, filter[i][j], cin[k + j*n_i], result2_ct[k]);
					}

					else
					{
						padded_new_rotate_conv_quant<N>(wPp, j, f_size, n_i, context, filter[i][j], cin[k + j*n_i], result2_ct[k]);
					}
				}
			}
			for (int k=0; k<n_i; k++)
			{
                jaguar::rerandomize_conv_output(result2_ct[k]);
                auto trunc_start = chrono::high_resolution_clock::now();
                JAGUAR_DISCARD_LSB_CT(result2_ct[k], discard_lsb, N);
                auto trunc_end = chrono::high_resolution_clock::now();
                time_trunc_sum += chrono::duration_cast<chrono::microseconds>(trunc_end - trunc_start);
                JAGUAR_ADD_PLAIN_PLAIN_QUANT(result2_ct[k].data(0), ss_r[idx].data(), log2_q, N);
                JAGUAR_REDUCE_QUANT_POLY(result2_ct[k].data(0), cout[idx].data(0), N, log2_q);
                JAGUAR_REDUCE_QUANT_POLY(result2_ct[k].data(1), cout[idx].data(1), N, log2_q);
				idx++;
			}
		}
	}
	time_end = chrono::high_resolution_clock::now();
	time_cloud_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

	std::cout << "\nConv layer, N=" << n << ", width=" << wPp << ", filter width=" << f_h << ", input ch=" << num_in << ", output ch=" << num_fout << ", ct_# = " << cout.size() << "\n";

	std::cout << "Runtime for CNN layer (cloud): " << time_cloud_sum.count()/loop << " [usec]\n";
	std::cout << "Message size: " << 2 * (log2_q + 1 - discard_lsb) / 8 * n / 1024 * cout.size() << " [KB]\n" ;
    jaguar::LayerSummary layer_summary;
    layer_summary.kind = "conv_compute";
    layer_summary.name = "cloud_conv_quant";
    layer_summary.total_latency_us = static_cast<uint64_t>(time_cloud_sum.count()/loop);
    layer_summary.compute_latency_us = static_cast<uint64_t>(time_cloud_sum.count()/loop);
    layer_summary.comm_bytes = 0;
    layer_summary.rounds = 0;
    layer_summary.details = "N=" + std::to_string(n) +
                    ",width=" + std::to_string(wPp) +
                    ",filter=" + std::to_string(f_h) +
                    ",in_ch=" + std::to_string(num_in) +
                    ",out_ch=" + std::to_string(num_fout) +
                    ",ct_count=" + std::to_string(cout.size()) +
                    ",effective_bits=" + std::to_string(log2_q + 1 - discard_lsb);
    jaguar::profiler().record_layer_summary(layer_summary);

    if (discard_lsb > 0) {
        jaguar::LayerSummary trunc_summary;
        trunc_summary.kind = "truncation";
        trunc_summary.name = "discard_lsb_ct";
        trunc_summary.total_latency_us = static_cast<uint64_t>(time_trunc_sum.count());
        trunc_summary.compute_latency_us = static_cast<uint64_t>(time_trunc_sum.count());
        trunc_summary.comm_bytes = 0;
        trunc_summary.rounds = 0;
        trunc_summary.details = "discard_lsb=" + std::to_string(discard_lsb) +
                                ",ct_count=" + std::to_string(cout.size()) +
                                ",width=" + std::to_string(wPp);
        jaguar::profiler().record_layer_summary(trunc_summary);
    }
}



template <int N>
void cloud_padded_new_CNN_quant_mt(const int th_count, const int layer, const int flag_sw, const int flag_perm, const int seed_pre, int &seed, const int in_h, vector<vector<vector<uint64_t>>> &filter, const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p, const vector<Ciphertext> &cin, vector<Ciphertext> &cout, vector<vector<uint64_t> > &ss_r_data, const int discard_lsb=0)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_cloud_sum(0);
    chrono::microseconds time_trunc_sum(0);

    int num_fout = filter.size();
    int num_in = filter[0].size();
    int f_size = filter[0][0].size();
    int f_h = sqrt(f_size);

    auto &context_data = *context.key_context_data();
    int n = context_data.parms().poly_modulus_degree();

    int p_l = (f_h - 1) / 2;
    int p_r = (f_h - 1) / 2 + ((f_h - 1) % 2 ? 1 : 0);
    int wPp = in_h + p_l + p_r;
    int in_size = wPp * wPp;
    int n_i;
    int first_col_num, col_num, col_num_num, remain_zero;
    if (in_size > n) {
        remain_zero = n % wPp;
        first_col_num = (n - remain_zero) / wPp;
        col_num = first_col_num - 2 * p_l;
        col_num_num = ((wPp - first_col_num) % col_num) ? ((wPp - first_col_num) / col_num + 1) : (wPp - first_col_num) / col_num;
        n_i = 1 + col_num_num;
    } else {
        n_i = n / in_size;
    }

    int num_ct_out = (in_size > n) ? (num_fout * n_i) : num_fout;
    cout.resize(num_ct_out, cin[0]);

    const auto &share_cache = get_random_ss_r_quant_cache<N>(num_ct_out, log2_q, log2_p, discard_lsb);
    const auto &ss_r = share_cache.ss_r;
    auto ss_r_copy_start = chrono::high_resolution_clock::now();
	ss_r_data = share_cache.ss_r_data;
	auto ss_r_copy_end = chrono::high_resolution_clock::now();
	auto ss_r_copy_us = static_cast<uint64_t>(chrono::duration_cast<chrono::microseconds>(ss_r_copy_end - ss_r_copy_start).count());
	jaguar::LayerSummary ss_r_copy_summary;
	ss_r_copy_summary.kind = "excluded_overhead";
	ss_r_copy_summary.name = "ss_r_data_copy";
	ss_r_copy_summary.total_latency_us = ss_r_copy_us;
	ss_r_copy_summary.compute_latency_us = ss_r_copy_us;
	ss_r_copy_summary.comm_bytes = 0;
	ss_r_copy_summary.rounds = 0;
	ss_r_copy_summary.details = "num_ct_out=" + std::to_string(num_ct_out);
	jaguar::profiler().record_layer_summary(ss_r_copy_summary);

    const int actual_threads = std::max(1, std::min(th_count, num_fout));
    std::vector<std::thread> workers;
    std::vector<uint64_t> trunc_us(actual_threads, 0);

    auto worker = [&](int tid, int fout_begin, int fout_end) {
        if (in_size <= n) {
            for (int i = fout_begin; i < fout_end; ++i) {
                Ciphertext result_ct = cin[0];
                set_zero_poly(N, 1, result_ct.data(0));
                set_zero_poly(N, 1, result_ct.data(1));

                for (int j = 0; j < num_in; ++j) {
                    if (j == 0) {
                        padded_new_conv_quant<N>(wPp, j, f_size, n_i, context, filter[i][j], cin[j / n_i], result_ct);
                    } else {
                        padded_new_rotate_conv_quant<N>(wPp, j, f_size, n_i, context, filter[i][j], cin[j / n_i], result_ct);
                    }
                }

                jaguar::rerandomize_conv_output(result_ct);
                auto trunc_start = chrono::high_resolution_clock::now();
                JAGUAR_DISCARD_LSB_CT(result_ct, discard_lsb, N);
                auto trunc_end = chrono::high_resolution_clock::now();
                trunc_us[tid] += static_cast<uint64_t>(chrono::duration_cast<chrono::microseconds>(trunc_end - trunc_start).count());
                JAGUAR_ADD_PLAIN_PLAIN_QUANT(result_ct.data(0), ss_r[i].data(), log2_q, N);
                JAGUAR_REDUCE_QUANT_POLY(result_ct.data(0), cout[i].data(0), N, log2_q);
                JAGUAR_REDUCE_QUANT_POLY(result_ct.data(1), cout[i].data(1), N, log2_q);
            }
        } else {
            for (int i = fout_begin; i < fout_end; ++i) {
                std::vector<Ciphertext> result2_ct(n_i, cin[0]);
                for (int k = 0; k < n_i; ++k) {
                    set_zero_poly(N, 1, result2_ct[k].data(0));
                    set_zero_poly(N, 1, result2_ct[k].data(1));
                }

                for (int j = 0; j < num_in; ++j) {
                    for (int k = 0; k < n_i; ++k) {
                        if (j == 0) {
                            padded_new_conv_quant<N>(wPp, j, f_size, n_i, context, filter[i][j], cin[k + j * n_i], result2_ct[k]);
                        } else {
                            padded_new_rotate_conv_quant<N>(wPp, j, f_size, n_i, context, filter[i][j], cin[k + j * n_i], result2_ct[k]);
                        }
                    }
                }

                for (int k = 0; k < n_i; ++k) {
                    const int out_idx = i * n_i + k;
                    jaguar::rerandomize_conv_output(result2_ct[k]);
                    auto trunc_start = chrono::high_resolution_clock::now();
                    JAGUAR_DISCARD_LSB_CT(result2_ct[k], discard_lsb, N);
                    auto trunc_end = chrono::high_resolution_clock::now();
                    trunc_us[tid] += static_cast<uint64_t>(chrono::duration_cast<chrono::microseconds>(trunc_end - trunc_start).count());
                    JAGUAR_ADD_PLAIN_PLAIN_QUANT(result2_ct[k].data(0), ss_r[out_idx].data(), log2_q, N);
                    JAGUAR_REDUCE_QUANT_POLY(result2_ct[k].data(0), cout[out_idx].data(0), N, log2_q);
                    JAGUAR_REDUCE_QUANT_POLY(result2_ct[k].data(1), cout[out_idx].data(1), N, log2_q);
                }
            }
        }
    };

    time_start = chrono::high_resolution_clock::now();
    int chunk = (num_fout + actual_threads - 1) / actual_threads;
    for (int tid = 0; tid < actual_threads; ++tid) {
        int begin = tid * chunk;
        int end = std::min(num_fout, begin + chunk);
        if (begin >= end) {
            break;
        }
        workers.emplace_back(worker, tid, begin, end);
    }
    for (auto &t : workers) {
        t.join();
    }
    time_end = chrono::high_resolution_clock::now();
    time_cloud_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    for (uint64_t us : trunc_us) {
        time_trunc_sum += chrono::microseconds(us);
    }
    std::cout << "\nConv layer, N=" << n << ", width=" << wPp << ", filter width=" << f_h << ", input ch=" << num_in << ", output ch=" << num_fout << ", ct_# = " << cout.size() << "\n";
    std::cout << "Runtime for CNN layer (cloud): " << time_cloud_sum.count() << " [usec]\n";
    std::cout << "Message size: " << 2 * (log2_q + 1 - discard_lsb) / 8 * n / 1024 * cout.size() << " [KB]\n";


    jaguar::LayerSummary layer_summary;
    layer_summary.kind = "conv_compute";
    layer_summary.name = "cloud_conv_quant_mt";
    layer_summary.total_latency_us = static_cast<uint64_t>(time_cloud_sum.count());
    layer_summary.compute_latency_us = static_cast<uint64_t>(time_cloud_sum.count());
    layer_summary.comm_bytes = 0;
    layer_summary.rounds = 0;
    layer_summary.details = "N=" + std::to_string(n) +
                    ",width=" + std::to_string(wPp) +
                    ",filter=" + std::to_string(f_h) +
                    ",in_ch=" + std::to_string(num_in) +
                    ",out_ch=" + std::to_string(num_fout) +
                    ",ct_count=" + std::to_string(cout.size()) +
                    ",effective_bits=" + std::to_string(log2_q + 1 - discard_lsb) +
                    ",threads=" + std::to_string(actual_threads);
    jaguar::profiler().record_layer_summary(layer_summary);

    if (discard_lsb > 0) {
        jaguar::LayerSummary trunc_summary;
        trunc_summary.kind = "truncation";
        trunc_summary.name = "discard_lsb_ct";
        trunc_summary.total_latency_us = static_cast<uint64_t>(time_trunc_sum.count());
        trunc_summary.compute_latency_us = static_cast<uint64_t>(time_trunc_sum.count());
        trunc_summary.comm_bytes = 0;
        trunc_summary.rounds = 0;
        trunc_summary.details = "discard_lsb=" + std::to_string(discard_lsb) +
                                ",ct_count=" + std::to_string(cout.size()) +
                                ",width=" + std::to_string(wPp) +
                                ",threads=" + std::to_string(actual_threads);
        jaguar::profiler().record_layer_summary(trunc_summary);
    }
}



template <int N>
void cloud_depthwise_padded_new_CNN_quant_st(const int layer, const int flag_sw, const int flag_perm, const int seed_pre, int &seed, const int in_h, vector<vector<uint64_t> > &filter_depth, const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p, const vector<Ciphertext> &cin, vector<Ciphertext> &cout, vector<vector<uint64_t> > &ss_r_data, const int discard_lsb=0)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_cloud_sum(0);
    chrono::microseconds time_off_cloud_sum(0);

    int num_fout = filter_depth.size();
    int num_in = filter_depth.size();
    int f_size = filter_depth[0].size();
    int f_h = sqrt(f_size);

    auto &context_data = *context.key_context_data();
	Modulus coeff_modulus = context_data.parms().coeff_modulus()[0];
    size_t coeff_modulus_size = context_data.parms().coeff_modulus().size();

    uint64_t mod_q = coeff_modulus.value();
    auto &plain_modulus = context_data.parms().plain_modulus();
    uint64_t mod_p = plain_modulus.value();

	const uint64_t delta = (1ULL<<(log2_q-log2_p));
    
    int p_l = (f_h - 1) / 2;
    int p_r = (f_h - 1) / 2 + ((f_h - 1) % 2 ? 1 : 0);
    int wPp = in_h + p_l + p_r;
    int in_size = wPp * wPp;
    int n = context_data.parms().poly_modulus_degree();
    int n_i;
    int num_out = f_size * num_in * num_fout;
	int first_col_num, col_num, col_num_num, remain_zero;

    if (in_size > n) {
		remain_zero = n%wPp;
		first_col_num = (n- remain_zero)/wPp;
		col_num = first_col_num - 2*p_l;
		col_num_num = ((wPp - first_col_num)%col_num)?((wPp - first_col_num)/col_num + 1):(wPp - first_col_num)/col_num;
        n_i = 1 + col_num_num;
    }
    else { 
        num_out = f_size * num_in * (num_fout / (n / in_size) + (num_fout % (n / in_size)? 1: 0));
        n_i = n / in_size;
    }
   
    int coeff_modulus_bit_count = get_significant_bit_count(coeff_modulus.value());

    int num_ct_in = cin.size();
    int num_ct_out;
    if (in_size > n)
	{
        num_ct_out = num_fout * n_i;
		cout.resize(num_ct_out, cin[0]);
	}
    else
	{
		num_ct_out = num_fout;
		cout.resize(num_fout, cin[0]);
	}

	// Generate ss_r outside the timed conv body.
	const auto &share_cache = get_random_ss_r_quant_cache<N>(num_ct_out, log2_q, log2_p, discard_lsb);
	const auto &ss_r = share_cache.ss_r;
	auto ss_r_copy_start = chrono::high_resolution_clock::now();
	ss_r_data = share_cache.ss_r_data;
	auto ss_r_copy_end = chrono::high_resolution_clock::now();
	auto ss_r_copy_us = static_cast<uint64_t>(chrono::duration_cast<chrono::microseconds>(ss_r_copy_end - ss_r_copy_start).count());
	jaguar::LayerSummary ss_r_copy_summary;
	ss_r_copy_summary.kind = "excluded_overhead";
	ss_r_copy_summary.name = "ss_r_data_copy";
	ss_r_copy_summary.total_latency_us = ss_r_copy_us;
	ss_r_copy_summary.compute_latency_us = ss_r_copy_us;
	ss_r_copy_summary.comm_bytes = 0;
	ss_r_copy_summary.rounds = 0;
	ss_r_copy_summary.details = "num_ct_out=" + std::to_string(num_ct_out);
	jaguar::profiler().record_layer_summary(ss_r_copy_summary);

	int idx = 0;
	int loop = 1;
	vector<Ciphertext> result2_ct(n_i, cin[0]);
    Ciphertext result_ct;
    result_ct = cin[0];
    set_zero_poly(N, 1, result_ct.data(0));
    set_zero_poly(N, 1, result_ct.data(1));

    Ciphertext temp_ct;
    temp_ct = cin[0];
    set_zero_poly(N, 1, temp_ct.data(0));
    set_zero_poly(N, 1, temp_ct.data(1));

	time_start = chrono::high_resolution_clock::now();
	if (in_size <= n)
	{
		for (int j=0; j<num_in; j++)
		{
			padded_new_conv_quant<N>(wPp, j, f_size, n_i, context, filter_depth[j], cin[j/n_i], result_ct);
			pos_rotate_add_poly_quant(temp_ct, result_ct, mod_q, N, (j%n_i)*in_size);

            jaguar::rerandomize_conv_output(result_ct);
            JAGUAR_DISCARD_LSB_CT(result_ct, discard_lsb, N);
            JAGUAR_ADD_PLAIN_PLAIN_QUANT(result_ct.data(0), ss_r[j].data(), log2_q, N);
            JAGUAR_REDUCE_QUANT_POLY(result_ct.data(0), cout[j].data(0), N, log2_q);
            JAGUAR_REDUCE_QUANT_POLY(result_ct.data(1), cout[j].data(1), N, log2_q);
        }
	}
	else
	{
		for (int j=0; j<num_in; j++)
		{
    		for (int k=0; k<n_i; k++)
    		{
				padded_new_conv_quant<N>(wPp, j, f_size, n_i, context, filter_depth[j], cin[k + j*n_i], result_ct);
                jaguar::rerandomize_conv_output(result_ct);
                JAGUAR_DISCARD_LSB_CT(result_ct, discard_lsb, N);
                JAGUAR_ADD_PLAIN_PLAIN_QUANT(result_ct.data(0), ss_r[idx].data(), log2_q, N);
                JAGUAR_REDUCE_QUANT_POLY(result_ct.data(0), cout[idx].data(0), N, log2_q);
                JAGUAR_REDUCE_QUANT_POLY(result_ct.data(1), cout[idx].data(1), N, log2_q);
				idx++;
    		}
    	}
	}
	time_end = chrono::high_resolution_clock::now();
	time_cloud_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);
	std::cout << "\nConv layer, N=" << n << ", width=" << wPp << ", filter width=" << f_h << ", input ch=" << num_in << ", output ch=" << num_fout << ", ct_# = " << cout.size() << "\n";

	std::cout << "Runtime for CNN layer (cloud): " << time_cloud_sum.count()/loop << " [usec]\n";
	std::cout << "Message size: " << 2 * 60 / 8 * n / 1024 * cout.size() << " [KB]\n" ;
    jaguar::LayerSummary layer_summary;
    layer_summary.kind = "conv_compute";
    layer_summary.name = "cloud_conv";
    layer_summary.total_latency_us = static_cast<uint64_t>(time_cloud_sum.count()/loop);
    layer_summary.compute_latency_us = static_cast<uint64_t>(time_cloud_sum.count()/loop);
    layer_summary.comm_bytes = static_cast<uint64_t>((2 * 60 / 8) * n * cout.size());
    layer_summary.rounds = 0;
    layer_summary.details = "N=" + std::to_string(n) +
                    ",width=" + std::to_string(wPp) +
                    ",filter=" + std::to_string(f_h) +
                    ",in_ch=" + std::to_string(num_in) +
                    ",out_ch=" + std::to_string(num_fout) +
                    ",ct_count=" + std::to_string(cout.size());
    jaguar::profiler().record_layer_summary(layer_summary);
}


template <int N>
void cloud_depthwise_padded_new_CNN_quant_mobilenet_v2_st(
    const int layer, const int flag_sw, const int flag_perm, const int seed_pre,
    int &seed, const int in_h, std::vector<std::vector<std::vector<uint64_t>>> &filter_depthwise_full,
    const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p,
    const std::vector<Ciphertext> &cin, std::vector<Ciphertext> &cout,
    std::vector<std::vector<uint64_t>> &ss_r_data, const int discard_lsb=0)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_cloud_sum(0);
    chrono::microseconds time_trunc_sum(0);

    const int num_fout = static_cast<int>(filter_depthwise_full.size());
    const int num_in = num_fout;
    if (num_fout == 0) {
        cout.clear();
        ss_r_data.clear();
        return;
    }
    for (size_t oc = 0; oc < filter_depthwise_full.size(); ++oc) {
        if (filter_depthwise_full[oc].size() != 1) {
            throw std::runtime_error("mobilenet_v2 depthwise expects one kernel per output channel");
        }
    }

    const int f_size = static_cast<int>(filter_depthwise_full[0][0].size());
    const int f_h = sqrt(f_size);

    auto &context_data = *context.key_context_data();
    Modulus coeff_modulus = context_data.parms().coeff_modulus()[0];
    uint64_t mod_q = coeff_modulus.value();
    int n = context_data.parms().poly_modulus_degree();

    int p_l = (f_h - 1) / 2;
    int p_r = (f_h - 1) / 2 + ((f_h - 1) % 2 ? 1 : 0);
    int wPp = in_h + p_l + p_r;
    int in_size = wPp * wPp;

    int n_i;
    int first_col_num, col_num, col_num_num, remain_zero;
    if (in_size > n) {
        remain_zero = n % wPp;
        first_col_num = (n - remain_zero) / wPp;
        col_num = first_col_num - 2 * p_l;
        col_num_num = ((wPp - first_col_num) % col_num) ? ((wPp - first_col_num) / col_num + 1) : (wPp - first_col_num) / col_num;
        n_i = 1 + col_num_num;
    } else {
        n_i = n / in_size;
    }

    const int num_ct_out = (in_size > n) ? (num_fout * n_i) : num_fout;
    cout.resize(num_ct_out, cin[0]);

    const auto &share_cache = get_random_ss_r_quant_cache<N>(num_ct_out, log2_q, log2_p, discard_lsb);
    const auto &ss_r = share_cache.ss_r;
    auto ss_r_copy_start = chrono::high_resolution_clock::now();
	ss_r_data = share_cache.ss_r_data;
	auto ss_r_copy_end = chrono::high_resolution_clock::now();
	auto ss_r_copy_us = static_cast<uint64_t>(chrono::duration_cast<chrono::microseconds>(ss_r_copy_end - ss_r_copy_start).count());
	jaguar::LayerSummary ss_r_copy_summary;
	ss_r_copy_summary.kind = "excluded_overhead";
	ss_r_copy_summary.name = "ss_r_data_copy";
	ss_r_copy_summary.total_latency_us = ss_r_copy_us;
	ss_r_copy_summary.compute_latency_us = ss_r_copy_us;
	ss_r_copy_summary.comm_bytes = 0;
	ss_r_copy_summary.rounds = 0;
	ss_r_copy_summary.details = "num_ct_out=" + std::to_string(num_ct_out);
	jaguar::profiler().record_layer_summary(ss_r_copy_summary);

    time_start = chrono::high_resolution_clock::now();
    if (in_size <= n) {
        for (int oc = 0; oc < num_fout; ++oc) {
            Ciphertext result_ct = cin[oc / n_i];
            set_zero_poly(N, 1, result_ct.data(0));
            set_zero_poly(N, 1, result_ct.data(1));

            // MobileNetV2 depthwise: one packed input slice maps to one output channel.
            padded_new_rotate_conv_quant<N>(
                wPp, oc, f_size, n_i, context, filter_depthwise_full[oc][0],
                cin[oc / n_i], result_ct);

            jaguar::rerandomize_conv_output(result_ct);
            auto trunc_start = chrono::high_resolution_clock::now();
            JAGUAR_DISCARD_LSB_CT(result_ct, discard_lsb, N);
            auto trunc_end = chrono::high_resolution_clock::now();
            time_trunc_sum += chrono::duration_cast<chrono::microseconds>(trunc_end - trunc_start);
            JAGUAR_ADD_PLAIN_PLAIN_QUANT(result_ct.data(0), ss_r[oc].data(), log2_q, N);
            JAGUAR_REDUCE_QUANT_POLY(result_ct.data(0), cout[oc].data(0), N, log2_q);
            JAGUAR_REDUCE_QUANT_POLY(result_ct.data(1), cout[oc].data(1), N, log2_q);
        }
    } else {
        int idx = 0;
        for (int oc = 0; oc < num_fout; ++oc) {
            for (int k = 0; k < n_i; ++k) {
                Ciphertext result_ct = cin[k + oc * n_i];
                set_zero_poly(N, 1, result_ct.data(0));
                set_zero_poly(N, 1, result_ct.data(1));

                padded_new_conv_quant<N>(
                    wPp, oc, f_size, n_i, context, filter_depthwise_full[oc][0],
                    cin[k + oc * n_i], result_ct);

                jaguar::rerandomize_conv_output(result_ct);
                auto trunc_start = chrono::high_resolution_clock::now();
                JAGUAR_DISCARD_LSB_CT(result_ct, discard_lsb, N);
                auto trunc_end = chrono::high_resolution_clock::now();
                time_trunc_sum += chrono::duration_cast<chrono::microseconds>(trunc_end - trunc_start);
                JAGUAR_ADD_PLAIN_PLAIN_QUANT(result_ct.data(0), ss_r[idx].data(), log2_q, N);
                JAGUAR_REDUCE_QUANT_POLY(result_ct.data(0), cout[idx].data(0), N, log2_q);
                JAGUAR_REDUCE_QUANT_POLY(result_ct.data(1), cout[idx].data(1), N, log2_q);
                idx++;
            }
        }
    }
    time_end = chrono::high_resolution_clock::now();
    time_cloud_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    jaguar::LayerSummary layer_summary;
    layer_summary.kind = "conv_compute";
    layer_summary.name = "cloud_conv";
    layer_summary.total_latency_us = static_cast<uint64_t>(time_cloud_sum.count());
    layer_summary.compute_latency_us = static_cast<uint64_t>(time_cloud_sum.count());
    layer_summary.comm_bytes = static_cast<uint64_t>((2 * (log2_q + 1 - discard_lsb) / 8) * n * cout.size());
    layer_summary.rounds = 0;
    layer_summary.details = "N=" + std::to_string(n) +
                    ",width=" + std::to_string(wPp) +
                    ",filter=" + std::to_string(f_h) +
                    ",in_ch=" + std::to_string(num_in) +
                    ",out_ch=" + std::to_string(num_fout) +
                    ",ct_count=" + std::to_string(cout.size()) +
                    ",effective_bits=" + std::to_string(log2_q + 1 - discard_lsb) +
                    ",mobilenet_v2_depthwise=1";
    jaguar::profiler().record_layer_summary(layer_summary);

    if (discard_lsb > 0) {
        jaguar::LayerSummary trunc_summary;
        trunc_summary.kind = "truncation";
        trunc_summary.name = "discard_lsb_ct";
        trunc_summary.total_latency_us = static_cast<uint64_t>(time_trunc_sum.count());
        trunc_summary.compute_latency_us = static_cast<uint64_t>(time_trunc_sum.count());
        trunc_summary.comm_bytes = 0;
        trunc_summary.rounds = 0;
        trunc_summary.details = "discard_lsb=" + std::to_string(discard_lsb) +
                                ",ct_count=" + std::to_string(cout.size()) +
                                ",width=" + std::to_string(wPp) +
                                ",mobilenet_v2_depthwise=1";
        jaguar::profiler().record_layer_summary(trunc_summary);
    }
}



template <int N>
void cloud_depthwise_padded_new_CNN_quant_mobilenet_v2_mt(
    const int th_count, const int layer, const int flag_sw, const int flag_perm, const int seed_pre,
    int &seed, const int in_h, std::vector<std::vector<std::vector<uint64_t>>> &filter_depthwise_full,
    const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p,
    const std::vector<Ciphertext> &cin, std::vector<Ciphertext> &cout,
    std::vector<std::vector<uint64_t>> &ss_r_data, const int discard_lsb=0)
{
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_cloud_sum(0);

    const int num_fout = static_cast<int>(filter_depthwise_full.size());
    const int num_in = num_fout;
    if (num_fout == 0) {
        cout.clear();
        ss_r_data.clear();
        return;
    }
    for (size_t oc = 0; oc < filter_depthwise_full.size(); ++oc) {
        if (filter_depthwise_full[oc].size() != 1) {
            throw std::runtime_error("mobilenet_v2 depthwise expects one kernel per output channel");
        }
    }

    const int f_size = static_cast<int>(filter_depthwise_full[0][0].size());
    const int f_h = sqrt(f_size);

    auto &context_data = *context.key_context_data();
    Modulus coeff_modulus = context_data.parms().coeff_modulus()[0];
    uint64_t mod_q = coeff_modulus.value();
    int n = context_data.parms().poly_modulus_degree();

    int p_l = (f_h - 1) / 2;
    int p_r = (f_h - 1) / 2 + ((f_h - 1) % 2 ? 1 : 0);
    int wPp = in_h + p_l + p_r;
    int in_size = wPp * wPp;

    int n_i;
    int first_col_num, col_num, col_num_num, remain_zero;
    if (in_size > n) {
        remain_zero = n % wPp;
        first_col_num = (n - remain_zero) / wPp;
        col_num = first_col_num - 2 * p_l;
        col_num_num = ((wPp - first_col_num) % col_num) ? ((wPp - first_col_num) / col_num + 1) : (wPp - first_col_num) / col_num;
        n_i = 1 + col_num_num;
    } else {
        n_i = n / in_size;
    }

    const int num_ct_out = (in_size > n) ? (num_fout * n_i) : num_fout;
    cout.resize(num_ct_out, cin[0]);

    const auto &share_cache = get_random_ss_r_quant_cache<N>(num_ct_out, log2_q, log2_p, discard_lsb);
    const auto &ss_r = share_cache.ss_r;
    auto ss_r_copy_start = chrono::high_resolution_clock::now();
	ss_r_data = share_cache.ss_r_data;
	auto ss_r_copy_end = chrono::high_resolution_clock::now();
	auto ss_r_copy_us = static_cast<uint64_t>(chrono::duration_cast<chrono::microseconds>(ss_r_copy_end - ss_r_copy_start).count());
	jaguar::LayerSummary ss_r_copy_summary;
	ss_r_copy_summary.kind = "excluded_overhead";
	ss_r_copy_summary.name = "ss_r_data_copy";
	ss_r_copy_summary.total_latency_us = ss_r_copy_us;
	ss_r_copy_summary.compute_latency_us = ss_r_copy_us;
	ss_r_copy_summary.comm_bytes = 0;
	ss_r_copy_summary.rounds = 0;
	ss_r_copy_summary.details = "num_ct_out=" + std::to_string(num_ct_out);
	jaguar::profiler().record_layer_summary(ss_r_copy_summary);

    const int actual_threads = std::max(1, std::min(th_count, num_fout));
    vector<thread> workers(actual_threads);
    vector<uint64_t> trunc_us(actual_threads, 0);
    auto worker_fn = [&](int tid) {
        int chunk = (num_fout + actual_threads - 1) / actual_threads;
        int oc_begin = tid * chunk;
        int oc_end = std::min(num_fout, oc_begin + chunk);
        if (in_size <= n) {
            for (int oc = oc_begin; oc < oc_end; ++oc) {
                Ciphertext result_ct = cin[oc / n_i];
                set_zero_poly(N, 1, result_ct.data(0));
                set_zero_poly(N, 1, result_ct.data(1));
                padded_new_rotate_conv_quant<N>(
                    wPp, oc, f_size, n_i, context, filter_depthwise_full[oc][0],
                    cin[oc / n_i], result_ct);
                jaguar::rerandomize_conv_output(result_ct);
                auto trunc_start = chrono::high_resolution_clock::now();
                JAGUAR_DISCARD_LSB_CT(result_ct, discard_lsb, N);
                auto trunc_end = chrono::high_resolution_clock::now();
                trunc_us[tid] += static_cast<uint64_t>(chrono::duration_cast<chrono::microseconds>(trunc_end - trunc_start).count());
                JAGUAR_ADD_PLAIN_PLAIN_QUANT(result_ct.data(0), ss_r[oc].data(), log2_q, N);
                JAGUAR_REDUCE_QUANT_POLY(result_ct.data(0), cout[oc].data(0), N, log2_q);
                JAGUAR_REDUCE_QUANT_POLY(result_ct.data(1), cout[oc].data(1), N, log2_q);
            }
        } else {
            for (int oc = oc_begin; oc < oc_end; ++oc) {
                for (int k = 0; k < n_i; ++k) {
                    int out_idx = oc * n_i + k;
                    Ciphertext result_ct = cin[out_idx];
                    set_zero_poly(N, 1, result_ct.data(0));
                    set_zero_poly(N, 1, result_ct.data(1));
                    padded_new_conv_quant<N>(
                        wPp, oc, f_size, n_i, context, filter_depthwise_full[oc][0],
                        cin[out_idx], result_ct);
                    jaguar::rerandomize_conv_output(result_ct);
                    auto trunc_start = chrono::high_resolution_clock::now();
                    JAGUAR_DISCARD_LSB_CT(result_ct, discard_lsb, N);
                    auto trunc_end = chrono::high_resolution_clock::now();
                    trunc_us[tid] += static_cast<uint64_t>(chrono::duration_cast<chrono::microseconds>(trunc_end - trunc_start).count());
                    JAGUAR_ADD_PLAIN_PLAIN_QUANT(result_ct.data(0), ss_r[out_idx].data(), log2_q, N);
                    JAGUAR_REDUCE_QUANT_POLY(result_ct.data(0), cout[out_idx].data(0), N, log2_q);
                    JAGUAR_REDUCE_QUANT_POLY(result_ct.data(1), cout[out_idx].data(1), N, log2_q);
                }
            }
        }
    };

    time_start = chrono::high_resolution_clock::now();
    for (int tid = 0; tid < actual_threads; ++tid) workers[tid] = thread(worker_fn, tid);
    for (int tid = 0; tid < actual_threads; ++tid) workers[tid].join();
    time_end = chrono::high_resolution_clock::now();
    time_cloud_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    uint64_t trunc_total_us = 0;
    for (int tid = 0; tid < actual_threads; ++tid) trunc_total_us += trunc_us[tid];

    jaguar::LayerSummary layer_summary;
    layer_summary.kind = "conv_compute";
    layer_summary.name = "cloud_conv_quant_mt";
    layer_summary.total_latency_us = static_cast<uint64_t>(time_cloud_sum.count());
    layer_summary.compute_latency_us = static_cast<uint64_t>(time_cloud_sum.count());
    layer_summary.comm_bytes = static_cast<uint64_t>((2 * (log2_q + 1 - discard_lsb) / 8) * n * cout.size());
    layer_summary.rounds = 0;
    layer_summary.details = "N=" + std::to_string(n) +
                    ",width=" + std::to_string(wPp) +
                    ",filter=" + std::to_string(f_h) +
                    ",in_ch=" + std::to_string(num_in) +
                    ",out_ch=" + std::to_string(num_fout) +
                    ",ct_count=" + std::to_string(cout.size()) +
                    ",threads=" + std::to_string(actual_threads) +
                    ",effective_bits=" + std::to_string(log2_q + 1 - discard_lsb) +
                    ",mobilenet_v2_depthwise=1";
    jaguar::profiler().record_layer_summary(layer_summary);

    if (discard_lsb > 0) {
        jaguar::LayerSummary trunc_summary;
        trunc_summary.kind = "truncation";
        trunc_summary.name = "discard_lsb_ct";
        trunc_summary.total_latency_us = trunc_total_us;
        trunc_summary.compute_latency_us = trunc_total_us;
        trunc_summary.comm_bytes = 0;
        trunc_summary.rounds = 0;
        trunc_summary.details = "discard_lsb=" + std::to_string(discard_lsb) +
                                ",ct_count=" + std::to_string(cout.size()) +
                                ",width=" + std::to_string(wPp) +
                                ",threads=" + std::to_string(actual_threads) +
                                ",mobilenet_v2_depthwise=1";
        jaguar::profiler().record_layer_summary(trunc_summary);
    }
}




void data_stride(vector<vector<uint64_t> > &ss_r_data, const SEALContext &context, const int stride, const int f_w, const int w)
{
	auto &context_data = *context.key_context_data();
	const int N = context_data.parms().poly_modulus_degree();

	const int p_l = (f_w - 1)/ 2;
	const int p_r = (f_w - 1)/ 2 + ((f_w - 1)%2?1:0);
	const int wPp = w + p_l + p_r;
	const int new_w = 1 + (wPp - f_w) / stride;

	vector<vector<uint64_t> > ss_r_for_relu_data(1, vector<uint64_t> (new_w*new_w, 0));

	uint64_t coeff_temp;
	int cnt, idx;

	if (wPp*wPp <= N)
	{
		cnt = 0; idx = 0;
		for (int i=0; i<ss_r_data.size(); i++)
		{
			for (int j=0; j<new_w; j++)
			{
				for (int k=0; k<new_w; k++)
				{
					int start_j = j*stride + p_l;
					int start_k = k*stride + p_l;
					coeff_temp = ss_r_data[i][start_j*wPp + start_k];
					ss_r_for_relu_data[idx][cnt] = coeff_temp;
					cnt++;
				}
			}
			cnt = 0;
			idx++;
			if (i != ss_r_data.size()-1)
			{
				ss_r_for_relu_data.emplace_back(vector<uint64_t> (new_w * new_w, 0));
			}
		}
	}
	else
	{
		int first_col_num = (N - (N%wPp))/wPp;
		int col_num = first_col_num - 2*p_l;
		int col_num_num = ((wPp - first_col_num)%col_num)?((wPp - first_col_num)/col_num + 1) : (wPp - first_col_num)/col_num;
		int num_packed_inv = 1 + col_num_num;
		cnt = 0; idx = 0;
		int idx2 = 0;
		int start_j, start_k;
		vector<uint64_t> temp(N, 0);

		for (int i=0; i<ss_r_data.size(); i+=num_packed_inv)
		{
			idx2 = 0;
			start_j = p_l;
			temp = ss_r_data[i];
			for (int j=0; j<new_w; j++)
			{
				for (int k=0; k<new_w; k++)
				{
					if ((j*stride) / col_num > idx2)
					{
						temp = ss_r_data[i + idx2 + 1];
						idx2 ++;
						start_j = p_l + (j*stride)%col_num;
					}
					start_k = (k*stride + p_l);
					coeff_temp = temp[start_j*wPp + start_k];
					ss_r_for_relu_data[idx][cnt] = coeff_temp;
					cnt++;

					// Jaguar_52: same row-advance rule as client_ct_stride_dec_quant.
					if (k == new_w - 1)
						start_j += stride;
					}
				}
				if (cnt == new_w * new_w)
				{
					cnt = 0;
					idx++;
					ss_r_for_relu_data.emplace_back(vector<uint64_t> (new_w * new_w, 0));
				}
		}
		ss_r_for_relu_data.erase(ss_r_for_relu_data.end()-1);
	}
	ss_r_data.assign(ss_r_for_relu_data.size(), vector<uint64_t> (ss_r_for_relu_data[0].size()));
	copy(ss_r_for_relu_data.begin(), ss_r_for_relu_data.end(), ss_r_data.begin());
}


template <int N>
void cloud_complete_relu_share_quant(const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p, const int next_f_w, const vector<vector<uint64_t> > &cloud_relu_share, int &next_w, vector<Ciphertext> &ct_relu)
{
    if (cloud_relu_share.empty())
    {
        return;
    }
    next_w = static_cast<int>(std::sqrt(cloud_relu_share[0].size()));
    vector<vector<uint64_t> > cloud_relu_share_padded;
    zero_padding_to_N(context, next_f_w, cloud_relu_share, cloud_relu_share_padded);
    const uint64_t shift = log2_q - log2_p;
    const uint64_t q_mask = (log2_q == 64) ? std::numeric_limits<uint64_t>::max() : ((1ULL << log2_q) - 1);
    vector<Plaintext> plain_share(cloud_relu_share_padded.size(), Plaintext(N));
    for (size_t i = 0; i < cloud_relu_share_padded.size(); ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            plain_share[i].data()[j] = (cloud_relu_share_padded[i][j] << shift) & q_mask;
        }
        JAGUAR_ADD_PLAIN_PLAIN_QUANT(ct_relu[i].data(0), plain_share[i].data(), log2_q, N);
        JAGUAR_REDUCE_QUANT_POLY(ct_relu[i].data(0), ct_relu[i].data(0), N, log2_q);
        JAGUAR_REDUCE_QUANT_POLY(ct_relu[i].data(1), ct_relu[i].data(1), N, log2_q);
    }
}


void zero_padding_to_N(const SEALContext & context, const int f_w, const vector<vector<uint64_t> >  &din, vector<vector<uint64_t> > &din_new)
{
	const int p_l = (f_w - 1)/2; // left
	const int p_r = (f_w - 1)/2 + ((f_w - 1) % 2? 1: 0); // right

	const int din0_sqrt = sqrt(din[0].size()); // image width

	const int padded_size_sqrt = din0_sqrt + p_l + p_r; // image width including zero padding

	const int padded_size = padded_size_sqrt * padded_size_sqrt; // image size including zero padding

    auto &context_data = *context.key_context_data();
	const int n = context_data.parms().poly_modulus_degree(); // Polynomial degree n

	int num_packed = n / padded_size; // # of packed image into n, if padded_size < n
	int remain_zero = n%padded_size_sqrt;
	int num_packed_inv;
	int first_col_num;
	int col_num;
	int col_num_num;
	if (n<padded_size)
	{
		first_col_num = (n-remain_zero)/padded_size_sqrt;
		col_num = first_col_num - 2*p_l;
		col_num_num = ((padded_size_sqrt - first_col_num) % col_num)?((padded_size_sqrt - first_col_num)/col_num + 1) : (padded_size_sqrt-first_col_num)/col_num;
		num_packed_inv = 1 + col_num_num;
	}

	int row_new; // # of plaintext for total input.
	if (num_packed > 0)
		row_new = din.size() / num_packed + ((din.size() % num_packed) ? 1: 0);
	else
		row_new = din.size() * num_packed_inv;

	vector<uint64_t> zero_vector(padded_size_sqrt, 0); // zero padding for up and down side of image.
	din_new.resize(row_new, vector<uint64_t>(0));

	if (num_packed>0) // If n >= padded_size
	{
		remain_zero = n%padded_size;
		for (int i = 0; i < row_new; i++) {
            for (int j = 0; j < num_packed; j++) {
                // Zero padding
				for (int l=0; l<p_l; l++)
					din_new[i].insert(din_new[i].end(), zero_vector.begin(), zero_vector.end());
                if (i * num_packed + j < din.size())
                {
                    for (int k=0; k<din[0].size(); k+=din0_sqrt)
                    {
                        din_new[i].insert(din_new[i].end(), p_l, 0);
                        din_new[i].insert(din_new[i].end(), din[i * num_packed + j].begin() + k, din[i * num_packed + j].begin() + k+din0_sqrt);
                        din_new[i].insert(din_new[i].end(), p_r, 0);
                    }
					for (int l=0; l<p_l; l++)
						din_new[i].insert(din_new[i].end(), zero_vector.begin(), zero_vector.end());
                }
                else
                {
                    for (int k=0; k<din[0].size(); k+=din0_sqrt)
                    {
                        din_new[i].insert(din_new[i].end(), p_l, 0);
                        din_new[i].insert(din_new[i].end(), din[din.size() -1].begin() + k, din[din.size()-1].begin() + k + din0_sqrt);
                        din_new[i].insert(din_new[i].end(), p_r, 0);
                    }
					for (int l=0; l<p_l; l++)
						din_new[i].insert(din_new[i].end(), zero_vector.begin(), zero_vector.end());
                }
            }
			for (int j=0; j<remain_zero; j++)
				din_new[i].insert(din_new[i].end(), 1, 0);
        }

	}
	else // If padded_size > n
	{
		// Jaguar_52: chunk m of a channel holds padded rows
		// [m*col_num, m*col_num + first_col_num), each padded_size_sqrt wide, then
		// zeros up to n. This is the layout the cloud conv (row rotation by the
		// padded width) and client_ct_stride_dec_quant read back. The original
		// pointer walk stopped advancing at the negative top-padding offset (every
		// stem chunk after the first encrypted zeros) and filled middle chunks with
		// n instead of first_col_num*width entries (shifted 112-wide layers).
		for (size_t c = 0; c < din.size(); ++c) {
			for (int m = 0; m < num_packed_inv; ++m) {
				std::vector<uint64_t> &dst = din_new[c * num_packed_inv + m];
				dst.assign(n, 0);
				for (int r = 0; r < first_col_num; ++r) {
					const int data_row = m * col_num + r - p_l;
					if (data_row < 0 || data_row >= din0_sqrt) continue;
					std::copy(din[c].begin() + (size_t)data_row * din0_sqrt,
					          din[c].begin() + (size_t)(data_row + 1) * din0_sqrt,
					          dst.begin() + (size_t)r * padded_size_sqrt + p_l);
				}
			}
		}
	}
}



void avg_pooling(vector<vector<uint64_t> > &relu_data, int pooling_kernel_size, int pooling_stride, int pooling_padding)
{

	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_avg_pooling_sum(0);

	const int din0 = relu_data[0].size();
	const int din0_sqrt = sqrt(din0);// image width

	const int padded_size_sqrt = din0_sqrt + 2*pooling_padding; // image width including zero padding
	const int padded_size = padded_size_sqrt * padded_size_sqrt; // image size including zero padding

	const int new_w = 1 + (padded_size_sqrt - pooling_kernel_size)/pooling_stride;

	vector<uint64_t> zero_vector(padded_size_sqrt, 0); // zero padding for up and down side of image.

	vector<vector<uint64_t> > v_temp(relu_data.size());

	time_start = chrono::high_resolution_clock::now();
	if (pooling_padding != 0)
	{
		for (int i=0; i<relu_data.size(); i++)
		{
			for (int j=0; j<pooling_padding; j++)
			{
				for (int k=0; k<padded_size_sqrt; k++)
					v_temp[i].push_back(0);
			}
			for (int j=0; j<din0; j+=din0_sqrt)
			{
				for (int k=0; k<pooling_padding; k++)
					v_temp[i].push_back(0);
				for (int k=0; k<din0_sqrt; k++)
					v_temp[i].push_back(relu_data[i][j + k]);
				for (int k=0; k<pooling_padding; k++)
					v_temp[i].push_back(0);
			}
			for (int j=0; j<pooling_padding; j++)
			{
				for (int k=0; k<padded_size_sqrt; k++)
					v_temp[i].push_back(0);
			}
		}
		relu_data.assign(v_temp.size(), vector<uint64_t> (v_temp[0].size()));
		copy(v_temp.begin(), v_temp.end(), relu_data.begin());
	}
	else
	{
		v_temp.resize(relu_data.size(), vector<uint64_t> (relu_data[0].size()));
		copy(relu_data.begin(), relu_data.end(), v_temp.begin());
	}

	for (int i=0; i<relu_data.size(); i++)
	{
		for (int j=0; j<pooling_kernel_size; j++)
		{
			for (int k=0; k<pooling_kernel_size; k++)
			{
				if ((j == 0) && (k == 0))
						continue;
				for (int l=0; l<padded_size; l++)
				{
					if (l + padded_size_sqrt*j + k < relu_data[i].size())
						v_temp[i][l] += relu_data[i][l + padded_size_sqrt*j + k];
					else
						//v_temp[i][l] += relu_data[i][l + padded_size_sqrt*j + k - v_temp[i].size()];
						v_temp[i][l] += 0;
				}
			}
		}
	}

	time_end = chrono::high_resolution_clock::now();
	time_avg_pooling_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

	relu_data.assign(v_temp.size(), vector<uint64_t>(new_w*new_w, 0));

	time_start = chrono::high_resolution_clock::now();
	for (int i=0; i<relu_data.size(); i++)
	{
		int cnt = 0;
		for (int j=0; j<new_w; j++)
		{
			for (int k=0; k<new_w; k++)
			{
				int start_j = (j*pooling_stride);
				int start_k = (k*pooling_stride);
				relu_data[i][cnt] = v_temp[i][start_j * padded_size_sqrt + start_k];
				cnt++;
			}
		}
	}
	time_end = chrono::high_resolution_clock::now();
	time_avg_pooling_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

	std::cout << "\nAvg pooling w/ kernel size=" << pooling_kernel_size << ", stride=" << pooling_stride << ", padding=" << pooling_padding << ": " << din0_sqrt << " >> " << new_w << "\n";
	std::cout << "Runtime for avg pooling (us): " << time_avg_pooling_sum.count() << " [usec]\n";

    jaguar::LayerSummary layer_summary;
    layer_summary.kind = "avg_pool";
    layer_summary.name = "avg_pooling";
    layer_summary.total_latency_us = static_cast<uint64_t>(time_avg_pooling_sum.count());
    layer_summary.compute_latency_us = static_cast<uint64_t>(time_avg_pooling_sum.count());
    layer_summary.comm_bytes = 0;
    layer_summary.rounds = 0;
    layer_summary.details = "kernel=" + std::to_string(pooling_kernel_size) +
                    ",stride=" + std::to_string(pooling_stride) +
                    ",padding=" + std::to_string(pooling_padding) +
                    ",in_w=" + std::to_string(din0_sqrt) +
                    ",out_w=" + std::to_string(new_w) +
                    ",channels=" + std::to_string(relu_data.size());
    jaguar::profiler().record_layer_summary(layer_summary);
}



template <int N>
void cloud_add_residual_path_quant(const SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, int &seed, const int w, vector<vector<vector<uint64_t> > > &filter, const vector<Ciphertext> &cin0, const vector<Ciphertext> &cin1, vector<Ciphertext> &cout, vector<vector<uint64_t> > &ss_r_data, const int discard_lsb=0, bool use_lwe_output = false, std::vector<LweSharedGroup> *lwe_groups = nullptr)
{
    if (jaguar_quant::enabled() && jaguar_quant::is_client())
	{
		if (use_lwe_output) {
			if (lwe_groups == nullptr) {
				throw std::runtime_error("cloud_add_residual_path_quant: lwe_groups is null on client");
			}
			jaguar_quant::recv_lwe_groups(*lwe_groups);
			ss_r_data.clear();   
			cout.clear();        
		} else {
			jaguar_quant::recv_ciphertexts(context_quant, cout);
			jaguar_quant::recv_matrix_u64(ss_r_data);
		}
		return;
	}
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_cloud_sum(0);
    chrono::microseconds time_off_cloud_sum(0);

	if (cin0.size() != cin1.size())
	{
		std::cout << "cin0.size: "<< cin0.size() << "\n";
		std::cout << "cin1.size: "<< cin1.size() << "\n";
		std::cout << "Fail to add residual path";
		return;
	}

    auto &context_data = *context_quant.key_context_data();
	Modulus coeff_modulus = context_data.parms().coeff_modulus()[0];
    size_t coeff_modulus_size = context_data.parms().coeff_modulus().size();

    uint64_t mod_q = coeff_modulus.value();
    auto &plain_modulus = context_data.parms().plain_modulus();
    uint64_t mod_p = plain_modulus.value();

	//const uint64_t delta = mod_q/mod_p;
	const uint64_t delta = (1ULL) << (log2_q - log2_p);
    int n = context_data.parms().poly_modulus_degree();
	
	vector<Ciphertext> cin0_temp(cin0.size(), cin0[0]);
    for (int i=0; i<cin0.size(); i++)
        cin0_temp[i] = cin0[i];
	
	vector<Ciphertext> cout_temp(cin0.size(), cin0[0]);

	for (int i=0; i<cin0.size(); i++)
	{
		time_start = chrono::high_resolution_clock::now();
		JAGUAR_ADD_PLAIN_PLAIN_QUANT(cin0_temp[i].data(0), cin1[i].data(0), log2_q, n);
		JAGUAR_ADD_PLAIN_PLAIN_QUANT(cin0_temp[i].data(1), cin1[i].data(1), log2_q, n);
        JAGUAR_REDUCE_QUANT_POLY(cin0_temp[i].data(0), cout_temp[i].data(0), N, log2_q);
        JAGUAR_REDUCE_QUANT_POLY(cin0_temp[i].data(1), cout_temp[i].data(1), N, log2_q);
		time_end = chrono::high_resolution_clock::now();
		time_cloud_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);
	}

	std::cout << "\nAdd residual path , N=" << n << ", ct_# = " << cout.size() << "\n";
	std::cout << "Runtime for add residual path (cloud): " << time_cloud_sum.count() << " [usec]\n";

	int depth = filter.size();
	vector<vector<uint64_t> > filter_depth(depth, vector<uint64_t> (1, 1));

	cloud_depthwise_padded_new_CNN_quant_st<N>(1, 0, 0, 0, seed, w, filter_depth, context_quant, log2_q, log2_p, cout_temp, cout, ss_r_data, discard_lsb);

	std::cout << "Message size: " << 2 * 60 / 8 * n / 1024 * cout.size() << " [KB]\n" ;
    if (jaguar_quant::enabled() && jaguar_quant::is_cloud())
	{
		if (use_lwe_output) {
			if (lwe_groups == nullptr) {
				throw std::runtime_error("cloud_add_residual_path_quant: lwe_groups is null on cloud");
			}

			lwe_groups->clear();

			int next_w_dummy = 0;
			build_shared_lwe_groups_from_cout_quant<N>(
				cout,
				/*stride=*/1,
				/*f_w=*/1,
				/*w=*/w,
				/*num_output_channels=*/static_cast<int>(filter.size()),
				log2_q,
				*lwe_groups,
				next_w_dummy);

			jaguar_quant::send_lwe_groups(*lwe_groups);
		} else {
			jaguar_quant::send_ciphertexts(cout);
			jaguar_quant::send_matrix_u64(ss_r_data);
		}
	}
}

template <int N>
void cloud_complete_no_relu_quant(const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p, const int stride, const int f_w, vector<vector<uint64_t> > &ss_r_data, const bool pooling, const int pooling_kernel_size, const int pooling_stride, const int pooling_padding, const int next_f_w, const vector<Ciphertext> &ct_normal, int &next_w, vector<Ciphertext> &ct_relu)
{
	chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_cloud_complete_sum(0);

    fprintf(stderr, "[cloud_no_relu_quant][entry] stride=%d f_w=%d next_f_w=%d next_w=%d pooling=%d ct_normal.size=%zu ss_r_data.size=%zu\n",
            stride, f_w, next_f_w, next_w, pooling ? 1 : 0, ct_normal.size(), ss_r_data.size());
    if (ss_r_data.empty())
    {
        fprintf(stderr, "[cloud_no_relu_quant][fatal] ss_r_data is empty at entry\n");
        abort();
    }
    fprintf(stderr, "[cloud_no_relu_quant][entry] ss_r_data[0].size=%zu\n", ss_r_data[0].size());

	auto &context_data = *context.key_context_data();
	Modulus coeff_modulus_0 = context_data.parms().coeff_modulus()[0];
    uint64_t mod_q = coeff_modulus_0.value();
    auto &plain_modulus = context_data.parms().plain_modulus();
    uint64_t mod_p = plain_modulus.value();
    const uint64_t delta = (1ULL << (log2_q-log2_p));
    const uint64_t qM1 = mod_q-1;

	vector<vector<uint64_t> > ss_r_data_strided;
	stride_extract_packed_quant(ss_r_data, N, stride, f_w, next_w, ss_r_data_strided, next_w);
    fprintf(stderr, "[cloud_no_relu_quant][post_stride] next_w=%d ss_r_data_strided.size=%zu\n", next_w, ss_r_data_strided.size());
    if (ss_r_data_strided.empty())
    {
        fprintf(stderr, "[cloud_no_relu_quant][fatal] ss_r_data_strided is empty after stride extraction\n");
        abort();
    }
    fprintf(stderr, "[cloud_no_relu_quant][post_stride] ss_r_data_strided[0].size=%zu\n", ss_r_data_strided[0].size());

	vector<vector<uint64_t> > rr_r(ss_r_data_strided.size(), vector<uint64_t>(ss_r_data_strided[0].size(), 0));
	for (int i=0; i<ss_r_data_strided.size(); i++)
	{
		for (int j=0; j<ss_r_data_strided[i].size(); j++)
		{
			rr_r[i][j] = delta*(mod_p - ss_r_data_strided[i][j]);
            rr_r[i][j] = rr_r[i][j] & qM1;
		}
	}
	if (pooling)
	{
		avg_pooling(rr_r, pooling_kernel_size, pooling_stride, pooling_padding);
	}
	const int p_l = (next_f_w - 1)/ 2;
	const int p_r = (next_f_w - 1)/ 2 + ((next_f_w - 1)%2?1:0);

	int w = sqrt(rr_r[0].size());
	const int wPp = w + p_l + p_r;
	next_w = w;

	vector<vector<uint64_t> > rr_r_new;

	zero_padding_to_N(context, next_f_w, rr_r, rr_r_new);

    std::cout << "[cloud_no_relu_quant][debug] ct_normal.size=" << ct_normal.size()
              << " ss_r_data.size=" << ss_r_data.size()
              << " ss_r_data_strided.size=" << ss_r_data_strided.size()
              << " rr_r.size=" << rr_r.size()
              << " rr_r_new.size=" << rr_r_new.size() << std::endl;
    if (rr_r_new.size() != ct_normal.size())
    {
        std::cerr << "[cloud_no_relu_quant][fatal] rr_r_new.size=" << rr_r_new.size()
                  << " but ct_normal.size=" << ct_normal.size() << std::endl;
        std::abort();
    }

	ct_relu.resize(ct_normal.size(), ct_normal[0]);
    vector<Ciphertext> ct_temp(ct_normal.size(), ct_normal[0]);
	vector<Plaintext> pt_rr_r_new(rr_r_new.size(), Plaintext(N));
	for (int i=0; i<ct_relu.size(); i++)
	{
        const size_t coeff_count_local = rr_r_new[i].size();
        if (coeff_count_local > static_cast<size_t>(N))
        {
            std::cerr << "[cloud_no_relu_quant][fatal] padded row " << i << " has " << coeff_count_local
                      << " coefficients, larger than poly degree " << N << std::endl;
            std::abort();
        }
        for (int j = 0; j < N; ++j)
        {
            pt_rr_r_new[i].data()[j] = 0;
        }
        for (size_t j = 0; j < coeff_count_local; ++j)
        {
            pt_rr_r_new[i].data()[j] = rr_r_new[i][j];
        }
        ct_temp[i] = ct_normal[i];
	}

	time_start = chrono::high_resolution_clock::now();
	for (int i=0; i<ct_relu.size(); i++)
	{
        JAGUAR_ADD_PLAIN_PLAIN_QUANT(ct_temp[i].data(0), pt_rr_r_new[i].data(), log2_q, N);
        JAGUAR_REDUCE_QUANT_POLY(ct_temp[i].data(0), ct_relu[i].data(0), N, log2_q);
        JAGUAR_REDUCE_QUANT_POLY(ct_temp[i].data(1), ct_relu[i].data(1), N, log2_q);
	}
	time_end = chrono::high_resolution_clock::now();
	time_cloud_complete_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);
	std::cout << "\nNo ReLU (+pooling) w/ padding=" << p_l << ", width=" << w << " >> " << wPp << ", output ch=" << rr_r.size() << ", N=" << N << ", ct_# = " << ct_relu.size() << "\n";
	std::cout << "Runtime for no relu complete (us): " << time_cloud_complete_sum.count() << " [usec]\n";
}



template <int N>
void cloud_complete_avg_pooling_padding_quant(const SEALContext &context, const uint64_t log2_q, const uint64_t log2_p, const int stride, const int f_w, vector<vector<uint64_t> > &ss_r_data, const bool pooling, const int pooling_kernel_size, const int pooling_stride, const int pooling_padding, const int next_f_w, const vector<Ciphertext> &ct_normal, int &next_w, vector<Ciphertext> &ct_relu)
{
	chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds time_cloud_complete_sum(0);

	auto &context_data = *context.key_context_data();
	Modulus coeff_modulus_0 = context_data.parms().coeff_modulus()[0];
    uint64_t mod_q = coeff_modulus_0.value();
    auto &plain_modulus = context_data.parms().plain_modulus();
    uint64_t mod_p = plain_modulus.value();
	const uint64_t delta = mod_q/mod_p;

	data_stride(ss_r_data, context, stride, f_w, next_w);

	vector<vector<uint64_t> > rr_r_temp(ss_r_data.size(), vector<uint64_t>(ss_r_data[0].size(), 0));

	for (int i=0; i<ss_r_data.size(); i++)
	{
		for (int j=0; j<ss_r_data[i].size(); j++)
		{
			rr_r_temp[i][j] = delta*(mod_p - ss_r_data[i][j]);
		}
	}
	if (pooling)
	{
		avg_pooling(rr_r_temp, pooling_kernel_size, pooling_stride, pooling_padding);
	}
	vector<vector<uint64_t> > rr_r(2*rr_r_temp.size(), vector<uint64_t> (rr_r_temp[0].size(), 0));
	for (int i=0; i<rr_r_temp.size(); i++)
	{
		for (int j=0; j<rr_r_temp[i].size(); j++)
		{
			rr_r[i+(rr_r_temp.size()/2)][j] = rr_r_temp[i][j];
		}
	}

	const int p_l = (next_f_w - 1)/ 2;
	const int p_r = (next_f_w - 1)/ 2 + ((next_f_w - 1)%2?1:0);

	int w = sqrt(rr_r[0].size());
	const int wPp = w + p_l + p_r;
	next_w = w;

	vector<vector<uint64_t> > rr_r_new;

	zero_padding_to_N(context, next_f_w, rr_r, rr_r_new);

	ct_relu.resize(ct_normal.size(), ct_normal[0]);
    vector<Ciphertext> ct_normal_copy(ct_normal.size(), ct_normal[0]);
    for (int i=0; i<ct_normal.size(); i++)
    {
        ct_normal_copy[i] = ct_normal[i];
    }

	vector<Plaintext> pt_rr_r_new(rr_r_new.size(), Plaintext(N));
	for (int i=0; i<ct_relu.size(); i++)
	{
		for (int j=0; j<N; j++)
		{
			pt_rr_r_new[i].data(0)[j] = rr_r_new[i][j];
		}
	}

	time_start = chrono::high_resolution_clock::now();
	for (int i=0; i<ct_relu.size(); i++)
	{
        JAGUAR_ADD_PLAIN_PLAIN_QUANT(ct_normal_copy[i].data(0), pt_rr_r_new[i].data(), log2_q, N);
        JAGUAR_REDUCE_QUANT_POLY(ct_normal_copy[i].data(0), ct_relu[i].data(0), N, log2_q);
        JAGUAR_REDUCE_QUANT_POLY(ct_normal_copy[i].data(1), ct_relu[i].data(1), N, log2_q);
	}
	time_end = chrono::high_resolution_clock::now();
	time_cloud_complete_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);
	std::cout << "\nPadding for ResNet (+pooling) w/ padding=" << p_l << ", width=" << w << " >> " << wPp << ", output ch=" << rr_r.size() << ", N=" << N << ", ct_# = " << ct_relu.size() << "\n";
	std::cout << "Runtime for padding for ResNet complete (us): " << time_cloud_complete_sum.count() << " [usec]\n";

}

void print_title(string s)
{
	std::cout << "\n************************************\n" ;
	std::cout << s;
	std::cout << "\n************************************\n" ;
}