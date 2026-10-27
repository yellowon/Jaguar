// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/polyarithsmallmod.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintcore.h"

///////////////////
#include "seal/ciphertext.h"
#include "seal/plaintext.h"
#include <vector>
#include <immintrin.h>
#include <x86intrin.h>
#include <cstddef>
#include <cstdint>
///////////////////

#ifdef SEAL_USE_INTEL_HEXL
#include "hexl/hexl.hpp"
#endif

using namespace std;

namespace seal
{
    namespace util
    {
        void modulo_poly_coeffs(ConstCoeffIter poly, std::size_t coeff_count, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
#endif

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseReduceMod(result, poly, coeff_count, modulus.value(), modulus.value(), 1);
#else
            SEAL_ITERATE(
                iter(poly, result), coeff_count, [&](auto I) { get<1>(I) = barrett_reduce_64(get<0>(I), modulus); });
#endif
        }

        void add_poly_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, std::size_t coeff_count, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif
            const uint64_t modulus_value = modulus.value();

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseAddMod(&result[0], &operand1[0], &operand2[0], coeff_count, modulus_value);
#else

            SEAL_ITERATE(iter(operand1, operand2, result), coeff_count, [&](auto I) {
#ifdef SEAL_DEBUG
                if (get<0>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand1");
                }
                if (get<1>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand2");
                }
#endif
                std::uint64_t sum = get<0>(I) + get<1>(I);
                get<2>(I) = SEAL_COND_SELECT(sum >= modulus_value, sum - modulus_value, sum);
            });
#endif
        }

////////// Added by the authors //////////

        void extract_lwe_from_rlwe_quant(
            const seal::Ciphertext &ct,
            int coeff_index,
            std::vector<uint64_t> &a_out,
            uint64_t &b_out)
        {
            auto coeff_count = static_cast<int>(ct.poly_modulus_degree());
#ifdef SEAL_DEBUG
            if (ct.size() < 2)
            {
                throw std::invalid_argument("ct");
            }
            if (coeff_index < 0 || coeff_index >= coeff_count)
            {
                throw std::out_of_range("coeff_index");
            }
#endif
            a_out.resize(static_cast<size_t>(coeff_count));
            std::copy_n(ct.data(1), static_cast<size_t>(coeff_count), a_out.begin());
            b_out = ct.data(0)[coeff_index];
        }

        void decrypt_lwe_quant_reduce(
            const std::vector<uint64_t> &a,
            uint64_t b,
            ConstCoeffIter secret_key,
            const uint64_t log2_q,
            const uint64_t log2_p,
            const int N,
            const int discard_lsb,
            uint64_t &out)
        {
            uint64_t qM1 = (1ULL << log2_q) - 1;
            uint64_t pM1 = (1ULL << log2_p) - 1;
            uint64_t delta_div_2 = (1ULL << (log2_q - log2_p - 1 - discard_lsb));

            uint64_t acc = b & qM1;
            for (int i = 0; i < N; ++i)
            {
                acc = (acc + ((a[i] & qM1) * (secret_key[i] & qM1))) & qM1;
            }
            uint64_t operand_0 = ((acc + delta_div_2) & qM1) >> (log2_q - log2_p);
            out = operand_0 & pM1;
        }


        long unsigned int negate(const long unsigned int& val) {
            return -val;
        }
        long long unsigned int negate(long long unsigned int* val)
        {
            return *val;
        }
        template <typename... Types>
        auto negateTuple(const std::tuple<Types...>& t){
            return std::make_tuple(negate(std::get<Types>(t))...);
        }

        void multiply_poly_secret_quant(
            ConstCoeffIter poly0, //a
            ConstCoeffIter poly1, //secret_key
            const uint64_t q,
            CoeffIter dest, // destination
            const int N)
		{
            int NM1 = N-1;
            uint64_t qM1 = q-1;

            for (size_t i=0; i<N; i++)
            {
                dest[i] = 0;
                for (size_t j=0; j<i+1; j++)
                {
                    if (poly1[i-j] == 1)
                    {
                        dest[i] = (poly0[j] + dest[i]);
                    }
                    else if (poly1[i-j] == q-1)
                    {
                        uint64_t temp = q-poly0[j];
                        dest[i] = temp + dest[i];
                    }
                }
                for (size_t j=0; j<NM1-i; j++)
                {
                    if (poly1[NM1-j] == 1)
                    {
                        dest[i] = q-(poly0[i+j+1]) + dest[i];
                    }
                    else if (poly1[NM1-j] == q-1)
                    {
                        dest[i] = (poly0[i+j+1] + dest[i]);
                    }
                }
            }
		}

        void multiply_vec_poly_secret_quant(
            std::vector<CoeffIter> &poly0, //a
            ConstCoeffIter poly1, //secret_key
            const uint64_t q,
            std::vector<CoeffIter> &dest,
            const int N)
		{
            int NM1 = N-1;
            int length = poly0.size();

            SEAL_ITERATE(iter(poly0, dest), length, [&](auto I) {
                for (size_t i=0; i<N; i++)
                {
                    //get<1>(I)[i] = 0;
                    if (poly1[i] == 0)
                    {
                        //
                    }
                    else if (poly1[i] == 1)
                    {
                        for (size_t j=0; j<N-i; j++)
                        {
                            get<1>(I)[i+j] = (get<0>(I)[j] + get<1>(I)[i+j]);
                        }
                        for (size_t j=N-i; j<N; j++)
                        {
                            get<1>(I)[i+j-N] = (q-get<0>(I)[j] + get<1>(I)[i+j-N]);
                        }
                    }
                    else
                    {
                        for (size_t j=0; j<N-i; j++)
                        {
                            get<1>(I)[i+j] = (q-get<0>(I)[j] + get<1>(I)[i+j]);
                        }
                        for (size_t j=N-i; j<N; j++)
                        {
                            get<1>(I)[i+j-N] = (get<0>(I)[j] + get<1>(I)[i+j-N]);
                        }
                    }
                }
			});
		}


        void multiply_poly_poly(
            ConstCoeffIter poly0,
            ConstCoeffIter poly1,
			unsigned long long **operand_0,
            const uint64_t &mod_p,
            const int N)
		{
			unsigned long long *temp_0 = new unsigned long long [2];
            int NM1 = N-1;

            for (size_t i=0; i<N; i++)
            {
                operand_0[i][0] = 0;
                operand_0[i][1] = 0;
                for (size_t j=0; j<i+1; j++)
                {
                    multiply_uint64(poly0[j], poly1[i-j], temp_0);
				    add_uint128(operand_0[i], temp_0, operand_0[i]);
                }
                for (size_t j=0; j<NM1-i; j++)
                {
                    multiply_uint64(mod_p-poly0[i+j+1], poly1[NM1-j], temp_0);
				    add_uint128(operand_0[i], temp_0, operand_0[i]);
                }
            }
            
            delete [] temp_0;
		}

       
        void multiply_eval_poly_poly_mod(
            CoeffIter poly0,
            ConstCoeffIter poly1,
            const Modulus &modulus,
            const int N
			)
		{
			SEAL_ITERATE(iter(poly0, poly1), N, [&](auto I) {
                const uint64_t x = get<0>(I);
                const uint64_t y = get<1>(I);
                get<0>(I) = multiply_uint_mod(x, y, modulus);
			});
		}


		void set_plain_to_vector(
			Plaintext &plain,
			const std::vector<uint64_t> &vec, const int start, const int length
			)
		{
			SEAL_ITERATE(iter(plain.data(0) + start, &vec[0] + start), length, [&](auto I) {
				get<0>(I) = get<1>(I);
			});
		}

        void set_plain128_to_vector(
			unsigned long long **operand,
			const std::vector<uint64_t> &vec, const int start, const int length
			)
		{
			SEAL_ITERATE(iter(operand + start, &vec[0] + start), length, [&](auto I) {
				get<0>(I)[0] = get<1>(I);
			});
		}

		void set_vector_to_plain(
			std::vector<uint64_t> &vec, 
			CoeffIter plain, const int start, const int length
			)
		{
			SEAL_ITERATE(iter(plain + start, &vec[0] + start), length, [&](auto I) {
				get<1>(I) = get<0>(I);
			});
		}

		void add_ct_plain_ct128(
			const Ciphertext &ct, 
			const Plaintext &plain,
            unsigned long long **operand1_0, // destination
            unsigned long long **operand1_1, const int N)
        {
			unsigned long long *temp_0 = new unsigned long long [2];
			unsigned long long *temp_1 = new unsigned long long [2];
            SEAL_ITERATE(iter(ct.data(0), ct.data(1), plain.data(), operand1_0, operand1_1), N, [&](auto I) {
				temp_0[0] = get<0>(I);
				temp_0[1] = 0;
				temp_1[0] = get<2>(I);
				temp_1[1] = 0;
				add_uint128(temp_1, temp_0, get<3>(I));
				get<4>(I)[0] = get<1>(I);
				get<4>(I)[1] = 0;
            });
			delete [] temp_0;
			delete [] temp_1;
        }

        void add_plain_plain_quant(
            CoeffIter poly0, // destination
            //const Plaintext &plain, const int N)
            ConstCoeffIter poly1, const uint64_t log2_q, const int N)
        {
            SEAL_ITERATE(iter(poly0, poly1), N, [&](auto I) {
                get<0>(I) = get<0>(I) + get<1>(I);
            });
			//delete [] temp_0;
        }

        void add_plain_plain_quant_opt_fast(
            CoeffIter poly0,
            ConstCoeffIter poly1, const uint64_t log2_q, const int N)
        {
            (void)log2_q;
        #if defined(__AVX512F__)
            size_t i = 0;
            const size_t n = static_cast<size_t>(N);
            for (; i + 8 <= n; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const void *>(&poly0[i]));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const void *>(&poly1[i]));
                __m512i r = _mm512_add_epi64(a, b);
                _mm512_storeu_si512(reinterpret_cast<void *>(&poly0[i]), r);
            }
            for (; i < n; ++i)
            {
                poly0[i] = poly0[i] + poly1[i];
            }
        #else
            add_plain_plain_quant(poly0, poly1, log2_q, N);
        #endif
        }

        void add_ct_ct_quant(
            Ciphertext &poly0, // destination
            Ciphertext &poly1, const int N)
        {
            SEAL_ITERATE(iter(poly0.data(0), poly0.data(1), poly1.data(0), poly1.data(1)), N, [&](auto I) {
                get<0>(I) = get<0>(I) + get<2>(I);
                get<1>(I) = get<1>(I) + get<3>(I);
            });
        }

	static inline void add_u64_avx512_bulk(
		const uint64_t* __restrict in0,
		const uint64_t* __restrict in1,
		uint64_t* __restrict out,
		size_t n)
	{
		const size_t V = 8;
		const size_t U = 4;
		const size_t STEP = V * U;

		size_t i = 0;
		size_t n32 = n & ~(STEP - 1);
		for (; i < n32; i += STEP) {
			__m512i a0 = _mm512_load_si512((const __m512i*)(in0 + i + 0));
			__m512i b0 = _mm512_load_si512((const __m512i*)(in1 + i + 0));
			__m512i r0 = _mm512_add_epi64(a0, b0);
			_mm512_store_si512((__m512i*)(out + i + 0), r0);

                        __m512i a1 = _mm512_load_si512((const __m512i*)(in0 + i + 8));
                        __m512i b1 = _mm512_load_si512((const __m512i*)(in1 + i + 8));
                        __m512i r1 = _mm512_add_epi64(a1, b1);
                        _mm512_store_si512((__m512i*)(out + i + 8), r1);

                        __m512i a2 = _mm512_load_si512((const __m512i*)(in0 + i + 16));
                        __m512i b2 = _mm512_load_si512((const __m512i*)(in1 + i + 16));
                        __m512i r2 = _mm512_add_epi64(a2, b2);
                        _mm512_store_si512((__m512i*)(out + i + 16), r2);

                        __m512i a3 = _mm512_load_si512((const __m512i*)(in0 + i + 24));
                        __m512i b3 = _mm512_load_si512((const __m512i*)(in1 + i + 24));
                        __m512i r3 = _mm512_add_epi64(a3, b3);
                        _mm512_store_si512((__m512i*)(out + i + 24), r3);
		}
	}

	void add_ct_ct_quant_opt_fast(
		Ciphertext& dst,
		const Ciphertext& src,
		const int N)
	{
		uint64_t* d0 = dst.data(0);
		uint64_t* d1 = dst.data(1);
		const uint64_t* s0 = src.data(0);
		const uint64_t* s1 = src.data(1);

		add_u64_avx512_bulk(d0, s0, d0, (size_t)N);
		add_u64_avx512_bulk(d1, s1, d1, (size_t)N);
	}


        void add_plain_plain_quant_reduce(
            CoeffIter poly0, // destination
            //const Plaintext &plain, const int N)
            ConstCoeffIter poly1, const uint64_t log2_q, const int N)
        {
            uint64_t qM1 = (1ULL<<log2_q) - 1;
            SEAL_ITERATE(iter(poly0, poly1), N, [&](auto I) {
                get<0>(I) = (get<0>(I) + get<1>(I))&qM1;
            });
        }

        void add_poly128_plain_quant(
            unsigned long long **poly,
            ConstCoeffIter plain, const uint64_t log2_q, const int N, CoeffIter destination)
        {
            uint64_t qM1 = (1ULL<<log2_q) - 1;
			unsigned long long *temp_0 = new unsigned long long [2];
            SEAL_ITERATE(iter(poly, plain, destination), N, [&](auto I) {
                int no = add_uint64(get<0>(I)[0], get<1>(I), 0, temp_0);
                get<2>(I) = temp_0[0] & qM1;
            });
			delete [] temp_0;
        }


		void add_poly128_plain(
            unsigned long long **operand1_0, // destination
            //const Plaintext &plain, const int N)
            CoeffIter plain, const int N)
        {
			unsigned long long *temp_0 = new unsigned long long [2];
            SEAL_ITERATE(iter(operand1_0, plain), N, [&](auto I) {
				temp_0[0] = get<1>(I);
				temp_0[1] = 0;
				add_uint128(get<0>(I), temp_0, get<0>(I));
            });
			delete [] temp_0;
        }

		void add_poly128(
            unsigned long long **operand1_0, unsigned long long **operand1_1, // destination
            unsigned long long **operand2_0, unsigned long long **operand2_1, const int N)
        {
            SEAL_ITERATE(iter(operand1_0, operand1_1, operand2_0, operand2_1), N, [&](auto I) {
				add_uint128(get<0>(I), get<2>(I), get<0>(I));
				add_uint128(get<1>(I), get<3>(I), get<1>(I));
            });
        }
		
		void multiply_scalar_poly128(
				const Ciphertext &cin, 
				const int N, const uint64_t &scalar,
				unsigned long long **operand_0, unsigned long long **operand_1)
		{
			SEAL_ITERATE(iter(cin.data(0), cin.data(1), operand_0, operand_1), N, [&](auto I) {
				multiply_uint64(get<0>(I), scalar, get<2>(I));
				multiply_uint64(get<1>(I), scalar, get<3>(I));
			});
		}


        void multiply_scalar_poly_quant(
				const Ciphertext &cin, 
				const int N, const uint64_t &scalar,
				Ciphertext &cout)
		{
			SEAL_ITERATE(iter(cin.data(0), cin.data(1), cout.data(0), cout.data(1)), N, [&](auto I) {
                get<2>(I) = get<0>(I) * scalar;
                get<3>(I) = get<1>(I) * scalar;
			});
		}


	static inline void mul_scalar_u64_avx512_bulk(
			const uint64_t* __restrict in,
			uint64_t* __restrict out,
			size_t n,
			uint64_t scalar) 
	{
		const size_t V = 8;
		const size_t U = 4;
		const size_t STEP = V * U;

		__m512i vs = _mm512_set1_epi64((long long) scalar);

		size_t i = 0;
		size_t n32 = n & ~(STEP - 1);
		for (; i < n32; i += STEP) {
			__m512i a0 = _mm512_load_si512((const __m512i*)(in + i + 0));
			__m512i a1 = _mm512_load_si512((const __m512i*)(in + i + 8));
			__m512i a2 = _mm512_load_si512((const __m512i*)(in + i + 16));
			__m512i a3 = _mm512_load_si512((const __m512i*)(in + i + 24));

			__m512i r0 = _mm512_mullo_epi64(a0, vs);
			__m512i r1 = _mm512_mullo_epi64(a1, vs);
			__m512i r2 = _mm512_mullo_epi64(a2, vs);
			__m512i r3 = _mm512_mullo_epi64(a3, vs);

			_mm512_store_si512((__m512i*)(out + i+ 0), r0);
			_mm512_store_si512((__m512i*)(out + i+ 8), r1);
			_mm512_store_si512((__m512i*)(out + i+ 16), r2);
			_mm512_store_si512((__m512i*)(out + i+ 24), r3);
		}

		size_t n8 = n & ~((size_t)7);
		for (; i + 8 <= n8; i += 8) {
			__m512i a = _mm512_load_si512((const __m512i*)(in + i));
			__m512i r = _mm512_mullo_epi64(a, vs);
			_mm512_store_si512((__m512i*)(out + i), r);
		}
		for (; i < n; ++i) out[i] = in[i] * scalar;
	}

	void multiply_scalar_poly_quant_opt_fast(
		const Ciphertext& cin, const int N, const uint64_t& scalar, Ciphertext& cout) 
	{
		const uint64_t* in0 = cin.data(0);
		const uint64_t* in1 = cin.data(1);
		uint64_t* out0 = cout.data(0);
		uint64_t* out1 = cout.data(1);

		mul_scalar_u64_avx512_bulk(in0, out0, (size_t)N, scalar);
		mul_scalar_u64_avx512_bulk(in1, out1, (size_t)N, scalar);
	}

        void rotate_multiply_scalar_add_poly64(
				const Ciphertext &cin, 
				const int N, const int step, const uint64_t &mod_q, const Modulus &modulus, const uint64_t &scalar,
				Ciphertext &cout)
		{
			unsigned long long *temp_0 = new unsigned long long [2];
			unsigned long long *temp_1 = new unsigned long long [2];
            unsigned long long temp_2;
            unsigned long long temp_3;
            std::uint64_t sum0;
            std::uint64_t sum1;
            const uint64_t modulus_value = modulus.value();

			bool pos_step = (step>0)? 1: 0;
			if (step == 0)
			{
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), cout.data(0), cout.data(1)), N, [&](auto I) {
					multiply_uint64(get<0>(I), scalar, temp_0);
					multiply_uint64(get<1>(I), scalar, temp_1);
				    temp_2 = barrett_reduce_128(temp_0, modulus);
				    temp_3 = barrett_reduce_128(temp_1, modulus);
                    sum0 = temp_2 + get<2>(I);
					sum1 = temp_3 + get<3>(I);
                    get<2>(I) = SEAL_COND_SELECT(sum0 >= modulus_value, sum0 - modulus_value, sum0);
                    get<3>(I) = SEAL_COND_SELECT(sum1>= modulus_value, sum1 - modulus_value, sum1);
				});
			}
			else if (pos_step)
			{
				SEAL_ITERATE(iter(cin.data(0) + step, cin.data(1) + step, cout.data(0), cout.data(1)), N-step, [&](auto I) {
					multiply_uint64(get<0>(I), scalar, temp_0);
					multiply_uint64(get<1>(I), scalar, temp_1);
                    temp_2 = barrett_reduce_128(temp_0, modulus);
				    temp_3 = barrett_reduce_128(temp_1, modulus);
                    sum0 = temp_2 + get<2>(I);
					sum1 = temp_3 + get<3>(I);
                    get<2>(I) = SEAL_COND_SELECT(sum0 >= modulus_value, sum0 - modulus_value, sum0);
                    get<3>(I) = SEAL_COND_SELECT(sum1>= modulus_value, sum1 - modulus_value, sum1);

				});
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), cout.data(0) + N - step, cout.data(1) + N - step), step, [&](auto I) {
					multiply_uint64(mod_q - get<0>(I), scalar, temp_0);
					multiply_uint64(mod_q - get<1>(I), scalar, temp_1);
                    temp_2 = barrett_reduce_128(temp_0, modulus);
				    temp_3 = barrett_reduce_128(temp_1, modulus);
                    sum0 = temp_2 + get<2>(I);
					sum1 = temp_3 + get<3>(I);
                    get<2>(I) = SEAL_COND_SELECT(sum0 >= modulus_value, sum0 - modulus_value, sum0);
                    get<3>(I) = SEAL_COND_SELECT(sum1>= modulus_value, sum1 - modulus_value, sum1);
				});			
			}
			else
			{
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), cout.data(0) - step, cout.data(1) - step), N+step, [&](auto I) {
					multiply_uint64(get<0>(I), scalar, temp_0);
					multiply_uint64(get<1>(I), scalar, temp_1);
                    temp_2 = barrett_reduce_128(temp_0, modulus);
				    temp_3 = barrett_reduce_128(temp_1, modulus);
                    sum0 = temp_2 + get<2>(I);
					sum1 = temp_3 + get<3>(I);
                    get<2>(I) = SEAL_COND_SELECT(sum0 >= modulus_value, sum0 - modulus_value, sum0);
                    get<3>(I) = SEAL_COND_SELECT(sum1>= modulus_value, sum1 - modulus_value, sum1);
				});
				SEAL_ITERATE(iter(cin.data(0)+N+step, cin.data(1)+N+step, cout.data(0), cout.data(1)), -step, [&](auto I) {
					multiply_uint64(mod_q - get<0>(I), scalar, temp_0);
					multiply_uint64(mod_q - get<1>(I), scalar, temp_1);
                    temp_2 = barrett_reduce_128(temp_0, modulus);
				    temp_3 = barrett_reduce_128(temp_1, modulus);
                    sum0 = temp_2 + get<2>(I);
					sum1 = temp_3 + get<3>(I);
                    get<2>(I) = SEAL_COND_SELECT(sum0 >= modulus_value, sum0 - modulus_value, sum0);
                    get<3>(I) = SEAL_COND_SELECT(sum1>= modulus_value, sum1 - modulus_value, sum1);
				});			
			}
			delete [] temp_0;
			delete [] temp_1;
		}

        void rotate_multiply_scalar_add_poly128(
				const Ciphertext &cin, 
				const int N, const int step, const uint64_t &mod_q, const uint64_t &scalar,
				unsigned long long **destination_0, unsigned long long **destination_1)
		{
			unsigned long long *temp_0 = new unsigned long long [2];
			unsigned long long *temp_1 = new unsigned long long [2];

			bool pos_step = (step>0)? 1: 0;
			if (step == 0)
			{
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), destination_0, destination_1), N, [&](auto I) {
					multiply_uint64(get<0>(I), scalar, temp_0);
					multiply_uint64(get<1>(I), scalar, temp_1);
					add_uint128(temp_0, get<2>(I), get<2>(I));
					add_uint128(temp_1, get<3>(I), get<3>(I));
				});
			}
			else if (pos_step)
			{
				SEAL_ITERATE(iter(cin.data(0) + step, cin.data(1) + step, destination_0, destination_1), N-step, [&](auto I) {
					multiply_uint64(get<0>(I), scalar, temp_0);
					multiply_uint64(get<1>(I), scalar, temp_1);
					add_uint128(temp_0, get<2>(I), get<2>(I));
					add_uint128(temp_1, get<3>(I), get<3>(I));
				});
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), destination_0 + N - step, destination_1 + N - step), step, [&](auto I) {
					multiply_uint64(mod_q - get<0>(I), scalar, temp_0);
					multiply_uint64(mod_q - get<1>(I), scalar, temp_1);
					add_uint128(temp_0, get<2>(I), get<2>(I));
					add_uint128(temp_1, get<3>(I), get<3>(I));
				});			
			}
			else
			{
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), destination_0 - step, destination_1 - step), N+step, [&](auto I) {
					multiply_uint64(get<0>(I), scalar, temp_0);
					multiply_uint64(get<1>(I), scalar, temp_1);
					add_uint128(temp_0, get<2>(I), get<2>(I));
					add_uint128(temp_1, get<3>(I), get<3>(I));
				});
				SEAL_ITERATE(iter(cin.data(0)+N+step, cin.data(1)+N+step, destination_0, destination_1), -step, [&](auto I) {
					multiply_uint64(mod_q - get<0>(I), scalar, temp_0);
					multiply_uint64(mod_q - get<1>(I), scalar, temp_1);
					add_uint128(temp_0, get<2>(I), get<2>(I));
					add_uint128(temp_1, get<3>(I), get<3>(I));
				});			
			}
			delete [] temp_0;
			delete [] temp_1;
		}

		void rotate_multiply_scalar_add_poly_quant(
				const Ciphertext &cin, 
				const int N, const int step, const uint64_t &mod_q, const uint64_t &scalar,
				Ciphertext &cout)
		{
            unsigned long long temp_0;
            unsigned long long temp_1;

			bool pos_step = (step>0)? 1: 0;
			if (step == 0)
			{
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), cout.data(0), cout.data(1)), N, [&](auto I) {
                    temp_0 = get<0>(I) * scalar;
                    temp_1 = get<1>(I) * scalar;
                    get<2>(I) = temp_0 + get<2>(I);
                    get<3>(I) = temp_1 + get<3>(I);
				});
			}
			else if (pos_step)
			{
				SEAL_ITERATE(iter(cin.data(0) + step, cin.data(1) + step, cout.data(0), cout.data(1)), N-step, [&](auto I) {
                    temp_0 = get<0>(I) * scalar;
                    temp_1 = get<1>(I) * scalar;
                    get<2>(I) = temp_0 + get<2>(I);
                    get<3>(I) = temp_1 + get<3>(I);
				});
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), cout.data(0) + N - step, cout.data(1) + N - step), step, [&](auto I) {
                    temp_0 = (mod_q - get<0>(I)) * scalar;
                    temp_1 = (mod_q - get<1>(I)) * scalar;
                    get<2>(I) = temp_0 + get<2>(I);
                    get<3>(I) = temp_1 + get<3>(I);
				});			
			}
			else
			{
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), cout.data(0) - step, cout.data(1) - step), N+step, [&](auto I) {
                    temp_0 = get<0>(I) * scalar;
                    temp_1 = get<1>(I) * scalar;
                    get<2>(I) = temp_0 + get<2>(I);
                    get<3>(I) = temp_1 + get<3>(I);
				});
				SEAL_ITERATE(iter(cin.data(0)+N+step, cin.data(1)+N+step, cout.data(0), cout.data(1)), -step, [&](auto I) {
                    temp_0 = (mod_q - get<0>(I)) * scalar;
                    temp_1 = (mod_q - get<1>(I)) * scalar;
                    get<2>(I) = temp_0 + get<2>(I);
                    get<3>(I) = temp_1 + get<3>(I);
				});			
			}
		}


        static inline bool is_aligned_64(const void* p)
        {
            return (reinterpret_cast<std::uintptr_t>(p) & 63u) == 0;
        }

        static inline void muladd_u64_avx512_bulk(
            const uint64_t* in,
            uint64_t* out,
            size_t n,
            uint64_t scalar)
        {
        #if defined(__AVX512F__) && defined(__AVX512DQ__)
            constexpr size_t V = 8;
            constexpr size_t U = 4;
            constexpr size_t STEP = V * U;

            const __m512i vs = _mm512_set1_epi64((long long)scalar);
            const bool aligned = is_aligned_64(in) && is_aligned_64(out);

            size_t i = 0;
            const size_t n32 = n & ~(STEP - 1);

            if (aligned) {
                for (; i < n32; i += STEP) {
                    __m512i a0 = _mm512_load_si512((const void*)(in + i +  0));
                    __m512i a1 = _mm512_load_si512((const void*)(in + i +  8));
                    __m512i a2 = _mm512_load_si512((const void*)(in + i + 16));
                    __m512i a3 = _mm512_load_si512((const void*)(in + i + 24));

                    __m512i o0 = _mm512_load_si512((const void*)(out + i +  0));
                    __m512i o1 = _mm512_load_si512((const void*)(out + i +  8));
                    __m512i o2 = _mm512_load_si512((const void*)(out + i + 16));
                    __m512i o3 = _mm512_load_si512((const void*)(out + i + 24));

                    __m512i r0 = _mm512_add_epi64(o0, _mm512_mullo_epi64(a0, vs));
                    __m512i r1 = _mm512_add_epi64(o1, _mm512_mullo_epi64(a1, vs));
                    __m512i r2 = _mm512_add_epi64(o2, _mm512_mullo_epi64(a2, vs));
                    __m512i r3 = _mm512_add_epi64(o3, _mm512_mullo_epi64(a3, vs));

                    _mm512_store_si512((void*)(out + i +  0), r0);
                    _mm512_store_si512((void*)(out + i +  8), r1);
                    _mm512_store_si512((void*)(out + i + 16), r2);
                    _mm512_store_si512((void*)(out + i + 24), r3);
                }

                const size_t n8 = n & ~(V - 1);
                for (; i < n8; i += V) {
                    __m512i a = _mm512_load_si512((const void*)(in + i));
                    __m512i o = _mm512_load_si512((const void*)(out + i));
                    __m512i r = _mm512_add_epi64(o, _mm512_mullo_epi64(a, vs));
                    _mm512_store_si512((void*)(out + i), r);
                }
            } else {
                for (; i < n32; i += STEP) {
                    __m512i a0 = _mm512_loadu_si512((const void*)(in + i +  0));
                    __m512i a1 = _mm512_loadu_si512((const void*)(in + i +  8));
                    __m512i a2 = _mm512_loadu_si512((const void*)(in + i + 16));
                    __m512i a3 = _mm512_loadu_si512((const void*)(in + i + 24));

                    __m512i o0 = _mm512_loadu_si512((const void*)(out + i +  0));
                    __m512i o1 = _mm512_loadu_si512((const void*)(out + i +  8));
                    __m512i o2 = _mm512_loadu_si512((const void*)(out + i + 16));
                    __m512i o3 = _mm512_loadu_si512((const void*)(out + i + 24));

                    __m512i r0 = _mm512_add_epi64(o0, _mm512_mullo_epi64(a0, vs));
                    __m512i r1 = _mm512_add_epi64(o1, _mm512_mullo_epi64(a1, vs));
                    __m512i r2 = _mm512_add_epi64(o2, _mm512_mullo_epi64(a2, vs));
                    __m512i r3 = _mm512_add_epi64(o3, _mm512_mullo_epi64(a3, vs));

                    _mm512_storeu_si512((void*)(out + i +  0), r0);
                    _mm512_storeu_si512((void*)(out + i +  8), r1);
                    _mm512_storeu_si512((void*)(out + i + 16), r2);
                    _mm512_storeu_si512((void*)(out + i + 24), r3);
                }

                const size_t n8 = n & ~(V - 1);
                for (; i < n8; i += V) {
                    __m512i a = _mm512_loadu_si512((const void*)(in + i));
                    __m512i o = _mm512_loadu_si512((const void*)(out + i));
                    __m512i r = _mm512_add_epi64(o, _mm512_mullo_epi64(a, vs));
                    _mm512_storeu_si512((void*)(out + i), r);
                }
            }

            for (; i < n; ++i) {
                out[i] += in[i] * scalar;
            }
        #else
            for (size_t i = 0; i < n; ++i) {
                out[i] += in[i] * scalar;
            }
        #endif
        }

        static inline void muladd_neg_u64_avx512_bulk(
            const uint64_t* in,
            uint64_t* out,
            size_t n,
            uint64_t mod_q,
            uint64_t scalar)
        {
        #if defined(__AVX512F__) && defined(__AVX512DQ__)
            constexpr size_t V = 8;
            constexpr size_t U = 4;
            constexpr size_t STEP = V * U;

            const __m512i vq = _mm512_set1_epi64((long long)mod_q);
            const __m512i vs = _mm512_set1_epi64((long long)scalar);
            const bool aligned = is_aligned_64(in) && is_aligned_64(out);

            size_t i = 0;
            const size_t n32 = n & ~(STEP - 1);

            if (aligned) {
                for (; i < n32; i += STEP) {
                    __m512i a0 = _mm512_load_si512((const void*)(in + i +  0));
                    __m512i a1 = _mm512_load_si512((const void*)(in + i +  8));
                    __m512i a2 = _mm512_load_si512((const void*)(in + i + 16));
                    __m512i a3 = _mm512_load_si512((const void*)(in + i + 24));

                    __m512i o0 = _mm512_load_si512((const void*)(out + i +  0));
                    __m512i o1 = _mm512_load_si512((const void*)(out + i +  8));
                    __m512i o2 = _mm512_load_si512((const void*)(out + i + 16));
                    __m512i o3 = _mm512_load_si512((const void*)(out + i + 24));

                    __m512i b0 = _mm512_sub_epi64(vq, a0);
                    __m512i b1 = _mm512_sub_epi64(vq, a1);
                    __m512i b2 = _mm512_sub_epi64(vq, a2);
                    __m512i b3 = _mm512_sub_epi64(vq, a3);

                    __m512i r0 = _mm512_add_epi64(o0, _mm512_mullo_epi64(b0, vs));
                    __m512i r1 = _mm512_add_epi64(o1, _mm512_mullo_epi64(b1, vs));
                    __m512i r2 = _mm512_add_epi64(o2, _mm512_mullo_epi64(b2, vs));
                    __m512i r3 = _mm512_add_epi64(o3, _mm512_mullo_epi64(b3, vs));

                    _mm512_store_si512((void*)(out + i +  0), r0);
                    _mm512_store_si512((void*)(out + i +  8), r1);
                    _mm512_store_si512((void*)(out + i + 16), r2);
                    _mm512_store_si512((void*)(out + i + 24), r3);
                }

                const size_t n8 = n & ~(V - 1);
                for (; i < n8; i += V) {
                    __m512i a = _mm512_load_si512((const void*)(in + i));
                    __m512i o = _mm512_load_si512((const void*)(out + i));
                    __m512i b = _mm512_sub_epi64(vq, a);
                    __m512i r = _mm512_add_epi64(o, _mm512_mullo_epi64(b, vs));
                    _mm512_store_si512((void*)(out + i), r);
                }
            } else {
                for (; i < n32; i += STEP) {
                    __m512i a0 = _mm512_loadu_si512((const void*)(in + i +  0));
                    __m512i a1 = _mm512_loadu_si512((const void*)(in + i +  8));
                    __m512i a2 = _mm512_loadu_si512((const void*)(in + i + 16));
                    __m512i a3 = _mm512_loadu_si512((const void*)(in + i + 24));

                    __m512i o0 = _mm512_loadu_si512((const void*)(out + i +  0));
                    __m512i o1 = _mm512_loadu_si512((const void*)(out + i +  8));
                    __m512i o2 = _mm512_loadu_si512((const void*)(out + i + 16));
                    __m512i o3 = _mm512_loadu_si512((const void*)(out + i + 24));

                    __m512i b0 = _mm512_sub_epi64(vq, a0);
                    __m512i b1 = _mm512_sub_epi64(vq, a1);
                    __m512i b2 = _mm512_sub_epi64(vq, a2);
                    __m512i b3 = _mm512_sub_epi64(vq, a3);

                    __m512i r0 = _mm512_add_epi64(o0, _mm512_mullo_epi64(b0, vs));
                    __m512i r1 = _mm512_add_epi64(o1, _mm512_mullo_epi64(b1, vs));
                    __m512i r2 = _mm512_add_epi64(o2, _mm512_mullo_epi64(b2, vs));
                    __m512i r3 = _mm512_add_epi64(o3, _mm512_mullo_epi64(b3, vs));

                    _mm512_storeu_si512((void*)(out + i +  0), r0);
                    _mm512_storeu_si512((void*)(out + i +  8), r1);
                    _mm512_storeu_si512((void*)(out + i + 16), r2);
                    _mm512_storeu_si512((void*)(out + i + 24), r3);
                }

                const size_t n8 = n & ~(V - 1);
                for (; i < n8; i += V) {
                    __m512i a = _mm512_loadu_si512((const void*)(in + i));
                    __m512i o = _mm512_loadu_si512((const void*)(out + i));
                    __m512i b = _mm512_sub_epi64(vq, a);
                    __m512i r = _mm512_add_epi64(o, _mm512_mullo_epi64(b, vs));
                    _mm512_storeu_si512((void*)(out + i), r);
                }
            }

            for (; i < n; ++i) {
                out[i] += (mod_q - in[i]) * scalar;
            }
        #else
            for (size_t i = 0; i < n; ++i) {
                out[i] += (mod_q - in[i]) * scalar;
            }
        #endif
        }

    void rotate_multiply_scalar_add_poly_quant_opt_fast(
        const Ciphertext &cin,
        const int N,
        const int step,
        const uint64_t &mod_q,
        const uint64_t &scalar,
        Ciphertext &cout)
    {
        const uint64_t* in0 = cin.data(0);
        const uint64_t* in1 = cin.data(1);
        uint64_t* out0 = cout.data(0);
        uint64_t* out1 = cout.data(1);

        if (step == 0) {
            muladd_u64_avx512_bulk(in0, out0, (size_t)N, scalar);
            muladd_u64_avx512_bulk(in1, out1, (size_t)N, scalar);
            return;
        }

        if (step > 0) {
            const size_t s = (size_t)step;

            // cin[step ... N-1] -> cout[0 ... N-step-1]
            muladd_u64_avx512_bulk(in0 + s, out0, (size_t)N - s, scalar);
            muladd_u64_avx512_bulk(in1 + s, out1, (size_t)N - s, scalar);

            // cin[0 ... step-1] -> cout[N-step ... N-1], using (mod_q - x)
            muladd_neg_u64_avx512_bulk(in0, out0 + ((size_t)N - s), s, mod_q, scalar);
            muladd_neg_u64_avx512_bulk(in1, out1 + ((size_t)N - s), s, mod_q, scalar);
            return;
        }

        {
            const size_t s = (size_t)(-step);

            // cin[0 ... N-s-1] -> cout[s ... N-1]
            muladd_u64_avx512_bulk(in0, out0 + s, (size_t)N - s, scalar);
            muladd_u64_avx512_bulk(in1, out1 + s, (size_t)N - s, scalar);

            // cin[N-s ... N-1] -> cout[0 ... s-1], using (mod_q - x)
            muladd_neg_u64_avx512_bulk(in0 + ((size_t)N - s), out0, s, mod_q, scalar);
            muladd_neg_u64_avx512_bulk(in1 + ((size_t)N - s), out1, s, mod_q, scalar);
        }
    }

        void left_shift_plain(CoeffIter poly, const uint64_t &shift, const int N)
        {
            if (shift==0)
            {
                return ;
            }
            if (shift < 0)
            {
                throw invalid_argument("The left_shift cannot have a negative value for the shift amount.");
            }
            for (int i=0; i<N; i++)
            {
                poly[i] = poly[i] << shift;
            }
        }

        void discard_lsb_ct(Ciphertext &ct, const int discard_lsb, const int N)
        {
            if (discard_lsb == 0)
                return;
            
            SEAL_ITERATE(iter(ct.data(0), ct.data(1)), N, [&](auto I) {
				get<0>(I) = get<0>(I) >> discard_lsb;
				get<1>(I) = get<1>(I) >> discard_lsb;
			});
        }

        void discard_lsb_ct_opt_fast(Ciphertext &ct, const int discard_lsb, const int N)
        {
            if (discard_lsb == 0)
                return;
        #if defined(__AVX512F__)
            const __m512i vshift = _mm512_set1_epi64(static_cast<long long>(discard_lsb));
            uint64_t *c0 = ct.data(0);
            uint64_t *c1 = ct.data(1);
            size_t i = 0;
            const size_t n = static_cast<size_t>(N);
            for (; i + 8 <= n; i += 8)
            {
                __m512i a0 = _mm512_loadu_si512(reinterpret_cast<const void *>(c0 + i));
                __m512i a1 = _mm512_loadu_si512(reinterpret_cast<const void *>(c1 + i));
                a0 = _mm512_srlv_epi64(a0, vshift);
                a1 = _mm512_srlv_epi64(a1, vshift);
                _mm512_storeu_si512(reinterpret_cast<void *>(c0 + i), a0);
                _mm512_storeu_si512(reinterpret_cast<void *>(c1 + i), a1);
            }
            for (; i < n; ++i)
            {
                c0[i] >>= discard_lsb;
                c1[i] >>= discard_lsb;
            }
        #else
            discard_lsb_ct(ct, discard_lsb, N);
        #endif
        }

        void decrypt_quant_reduce(CoeffIter poly, 
                CoeffIter poly1, // destination
                const uint64_t log2_q, const uint64_t log2_p, const int N, const int discard_lsb)
        {
            uint64_t pM1 = (1ULL<<log2_p)-1;
            uint64_t delta_div_2 = (1ULL << (log2_q - log2_p - 1 - discard_lsb));

            uint64_t qM1 = (1ULL<<log2_q)-1;
            uint64_t operand_0;
            SEAL_ITERATE(iter(poly, poly1), N, [&](auto I) {
                    operand_0 = ((get<0>(I) + delta_div_2)&qM1) >> (log2_q - log2_p);
					get<1>(I) = operand_0 & pM1;
			});
        }

		void rotate_ct(
				const Ciphertext &cin, 
				const int N, const int step, const uint64_t &mod_q,
				unsigned long long **destination_0, unsigned long long **destination_1)
		{
			bool pos_step = (step>0)? 1: 0;
			if (step == 0)
			{
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), destination_0, destination_1), N, [&](auto I) {
					get<2>(I)[0] = get<0>(I);
					get<2>(I)[1] = 0;
					get<3>(I)[0] = get<1>(I);
					get<3>(I)[1] = 0;
				});
			}
			else if (pos_step)
			{
				SEAL_ITERATE(iter(cin.data(0) + step, cin.data(1) + step, destination_0, destination_1), N-step, [&](auto I) {
					get<2>(I)[0] = get<0>(I);
					get<2>(I)[1] = 0;
					get<3>(I)[0] = get<1>(I);
					get<3>(I)[1] = 0;
				});
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), destination_0 + N - step, destination_1 + N - step), step, [&](auto I) {
					get<2>(I)[0] = mod_q - get<0>(I);
					get<2>(I)[1] = 0;
					get<3>(I)[0] = mod_q - get<1>(I);
					get<3>(I)[1] = 0;
				});			
			}
			else
			{
				SEAL_ITERATE(iter(cin.data(0), cin.data(1), destination_0 - step, destination_1 - step), N+step, [&](auto I) {
					get<2>(I)[0] = get<0>(I);
					get<2>(I)[1] = 0;
					get<3>(I)[0] = get<1>(I);
					get<3>(I)[1] = 0;
				});
				SEAL_ITERATE(iter(cin.data(0)+N+step, cin.data(1)+N+step, destination_0, destination_1), -step, [&](auto I) {
					get<2>(I)[0] = mod_q - get<0>(I);
					get<2>(I)[1] = 0;
					get<3>(I)[0] = mod_q - get<1>(I);
					get<3>(I)[1] = 0;
				});			
			}
		}

        void reduce_quant_poly(
                CoeffIter operand,
                CoeffIter poly, // destination
				const int N, const uint64_t log2_q)
		{
            uint64_t qM1 = (1ULL<<log2_q) - 1;
			SEAL_ITERATE(iter(operand, poly), N, [&](auto I) {
                get<1>(I) = get<0>(I) & qM1;
			});
		}

        void reduce_quant_poly_opt_fast(
            CoeffIter operand,
            CoeffIter poly,
            const int N, const uint64_t log2_q)
        {
            uint64_t qM1 = (1ULL << log2_q) - 1;
        #if defined(__AVX512F__)
            const __m512i vmask = _mm512_set1_epi64(static_cast<long long>(qM1));
            size_t i = 0;
            const size_t n = static_cast<size_t>(N);
            for (; i + 8 <= n; i += 8)
            {
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const void *>(&operand[i]));
                __m512i r = _mm512_and_si512(a, vmask);
                _mm512_storeu_si512(reinterpret_cast<void *>(&poly[i]), r);
            }
            for (; i < n; ++i)
            {
                poly[i] = operand[i] & qM1;
            }
        #else
            reduce_quant_poly(operand, poly, N, log2_q);
        #endif
        }

        void reduce_quant_ntt_poly(
                CoeffIter operand,
                CoeffIter poly, // destination
				const int N, const uint64_t qPrime, const uint64_t log2_q)
		{
            uint64_t q = (1ULL<<log2_q);
            uint64_t qM1 = q - 1;
			SEAL_ITERATE(iter(operand, poly), N, [&](auto I) {
                if (get<0>(I) > (qPrime>>1))
                {
                    get<1>(I) = get<0>(I) - qPrime + q;
                    get<1>(I) = get<1>(I) & qM1;
                }
                else
                    get<1>(I) = get<0>(I) & qM1;
			});
		}

		void pos_rotate_add_poly64(
            const Ciphertext &ctemp,
            Ciphertext &cout,
			uint64_t mod_q,
			const int N, const int pos_step) 
        {
			uint64_t *temp_0 = new uint64_t [2];
			uint64_t *temp_1 = new uint64_t [2];
            std::uint64_t sum0;
            std::uint64_t sum1;

			if (pos_step < 0)
            {
                throw std::invalid_argument("pos_step");
            }
			else if (pos_step==0)
			{
				SEAL_ITERATE(iter(ctemp.data(0), ctemp.data(1), cout.data(0), cout.data(1)), N, [&](auto I) {
                    sum0 = get<0>(I) + get<2>(I);
					sum1 = get<1>(I) + get<3>(I);
                    get<2>(I) = SEAL_COND_SELECT(sum0 >= mod_q, sum0 - mod_q, sum0);
                    get<3>(I) = SEAL_COND_SELECT(sum1 >= mod_q, sum1 - mod_q, sum1);
				});
			}
			else
			{
				SEAL_ITERATE(iter(ctemp.data(0) + pos_step, ctemp.data(1) + pos_step, cout.data(0), cout.data(1)), N-pos_step, [&](auto I) {
                    sum0 = get<0>(I) + get<2>(I);
					sum1 = get<1>(I) + get<3>(I);
                    get<2>(I) = SEAL_COND_SELECT(sum0 >= mod_q, sum0 - mod_q, sum0);
                    get<3>(I) = SEAL_COND_SELECT(sum1 >= mod_q, sum1 - mod_q, sum1);
				});
				SEAL_ITERATE(iter(ctemp.data(0), ctemp.data(1), cout.data(0) + N - pos_step, cout.data(1) + N - pos_step), pos_step, [&](auto I) {
                    sum0 = mod_q - get<0>(I) + get<2>(I);
					sum1 = mod_q - get<1>(I) + get<3>(I);
                    get<2>(I) = SEAL_COND_SELECT(sum0 >= mod_q, sum0 - mod_q, sum0);
                    get<3>(I) = SEAL_COND_SELECT(sum1 >= mod_q, sum1 - mod_q, sum1);
				});			
			}
			delete [] temp_0;
			delete [] temp_1;
        }

        void pos_rotate_add_poly128(
            unsigned long long **operand_0, unsigned long long **operand_1,
            unsigned long long **destination_0, unsigned long long **destination_1,
			uint64_t *(q_128),
			const int N, const int pos_step) 
        {
			uint64_t *temp_0 = new uint64_t [2];
			uint64_t *temp_1 = new uint64_t [2];

			if (pos_step < 0)
            {
                throw std::invalid_argument("pos_step");
            }
			else if (pos_step==0)
			{
				SEAL_ITERATE(iter(operand_0, operand_1, destination_0, destination_1), N, [&](auto I) {
					add_uint128(get<2>(I), get<0>(I), get<2>(I));
					add_uint128(get<3>(I), get<1>(I), get<3>(I));
				});
			}
			else
			{
				SEAL_ITERATE(iter(operand_0 + pos_step, operand_1 + pos_step, destination_0, destination_1), N-pos_step, [&](auto I) {
					add_uint128(get<2>(I), get<0>(I), get<2>(I));
					add_uint128(get<3>(I), get<1>(I), get<3>(I));
				});
				SEAL_ITERATE(iter(operand_0, operand_1, destination_0 + N - pos_step, destination_1 + N - pos_step), pos_step, [&](auto I) {
					sub_uint(q_128, 2, reinterpret_cast<uint64_t *> (get<0>(I)), 2, 0, 2, temp_0);
					sub_uint(q_128, 2, reinterpret_cast<uint64_t *> (get<1>(I)), 2, 0, 2, temp_1);
					add_uint128(reinterpret_cast<unsigned long long *>(temp_0), get<2>(I), get<2>(I));
					add_uint128(reinterpret_cast<unsigned long long *>(temp_1), get<3>(I), get<3>(I));
				});			
			}
			delete [] temp_0;
			delete [] temp_1;
        }

		void pos_rotate_add_poly_quant(
            Ciphertext &operand,
            Ciphertext &destination,
			uint64_t q,
			const int N, const int pos_step) 
        {
			uint64_t temp_0;
			uint64_t temp_1;
            uint64_t qM1 = q - 1;

			if (pos_step < 0)
            {
                throw std::invalid_argument("pos_step");
            }
			else if (pos_step==0)
			{
				SEAL_ITERATE(iter(operand.data(0), operand.data(1), destination.data(0), destination.data(1)), N, [&](auto I) {
                    get<2>(I) = get<0>(I) + get<2>(I);
                    get<3>(I) = get<1>(I) + get<3>(I);
				});
			}
			else
			{
				SEAL_ITERATE(iter(operand.data(0) + pos_step, operand.data(1) + pos_step, destination.data(0), destination.data(1)), N-pos_step, [&](auto I) {
                    get<2>(I) = get<0>(I) + get<2>(I);
                    get<3>(I) = get<1>(I) + get<3>(I);
				});
				SEAL_ITERATE(iter(operand.data(0), operand.data(1), destination.data(0) + N - pos_step, destination.data(1) + N - pos_step), pos_step, [&](auto I) {
                    temp_0 = q - (get<0>(I) & qM1);
                    temp_1 = q - (get<1>(I) & qM1);

                    get<2>(I) = temp_0 + get<2>(I);
                    get<3>(I) = temp_1 + get<3>(I);
				});			
			}
        }

////////// Added by the authors end //////////

        void sub_poly_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, std::size_t coeff_count, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif

            const uint64_t modulus_value = modulus.value();
#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseSubMod(result, operand1, operand2, coeff_count, modulus_value);
#else
            SEAL_ITERATE(iter(operand1, operand2, result), coeff_count, [&](auto I) {
#ifdef SEAL_DEBUG
                if (get<0>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand1");
                }
                if (get<1>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand2");
                }
#endif
                unsigned long long temp_result;
                std::int64_t borrow = sub_uint64(get<0>(I), get<1>(I), &temp_result);
                get<2>(I) = temp_result + (modulus_value & static_cast<std::uint64_t>(-borrow));
            });
#endif
        }

        void add_poly_scalar_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, uint64_t scalar, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
            if (scalar >= modulus.value())
            {
                throw invalid_argument("scalar");
            }
#endif

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseAddMod(result, poly, scalar, coeff_count, modulus.value());
#else
            SEAL_ITERATE(iter(poly, result), coeff_count, [&](auto I) {
                const uint64_t x = get<0>(I);
                get<1>(I) = add_uint_mod(x, scalar, modulus);
            });
#endif
        }

        void sub_poly_scalar_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, uint64_t scalar, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
            if (scalar >= modulus.value())
            {
                throw invalid_argument("scalar");
            }
#endif

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseSubMod(result, poly, scalar, coeff_count, modulus.value());
#else
            SEAL_ITERATE(iter(poly, result), coeff_count, [&](auto I) {
                const uint64_t x = get<0>(I);
                get<1>(I) = sub_uint_mod(x, scalar, modulus);
            });
#endif
        }

        void multiply_poly_scalar_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, MultiplyUIntModOperand scalar, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
#endif

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseFMAMod(&result[0], &poly[0], scalar.operand, nullptr, coeff_count, modulus.value(), 8);
#else
            SEAL_ITERATE(iter(poly, result), coeff_count, [&](auto I) {
                const uint64_t x = get<0>(I);
                get<1>(I) = multiply_uint_mod(x, scalar, modulus);
            });
#endif
        }

        void dyadic_product_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, size_t coeff_count, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1)
            {
                throw invalid_argument("operand1");
            }
            if (!operand2)
            {
                throw invalid_argument("operand2");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (coeff_count == 0)
            {
                throw invalid_argument("coeff_count");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
#endif
#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseMultMod(&result[0], &operand1[0], &operand2[0], coeff_count, modulus.value(), 4);
#else
            const uint64_t modulus_value = modulus.value();
            const uint64_t const_ratio_0 = modulus.const_ratio()[0];
            const uint64_t const_ratio_1 = modulus.const_ratio()[1];

            SEAL_ITERATE(iter(operand1, operand2, result), coeff_count, [&](auto I) {
                // Reduces z using base 2^64 Barrett reduction
                unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
                multiply_uint64(get<0>(I), get<1>(I), z);

                // Multiply input and const_ratio
                // Round 1
                multiply_uint64_hw64(z[0], const_ratio_0, &carry);
                multiply_uint64(z[0], const_ratio_1, tmp2);
                tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, &tmp1);

                // Round 2
                multiply_uint64(z[1], const_ratio_0, tmp2);
                carry = tmp2[1] + add_uint64(tmp1, tmp2[0], &tmp1);

                // This is all we care about
                tmp1 = z[1] * const_ratio_1 + tmp3 + carry;

                // Barrett subtraction
                tmp3 = z[0] - tmp1 * modulus_value;

                // Claim: One more subtraction is enough
                get<2>(I) = SEAL_COND_SELECT(tmp3 >= modulus_value, tmp3 - modulus_value, tmp3);
            });
#endif
        }

        uint64_t poly_infty_norm_coeffmod(ConstCoeffIter operand, size_t coeff_count, const Modulus &modulus)
        {
#ifdef SEAL_DEBUG
            if (!operand && coeff_count > 0)
            {
                throw invalid_argument("operand");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
#endif
            // Construct negative threshold (first negative modulus value) to compute absolute values of coeffs.
            uint64_t modulus_neg_threshold = (modulus.value() + 1) >> 1;

            // Mod out the poly coefficients and choose a symmetric representative from
            // [-modulus,modulus). Keep track of the max.
            uint64_t result = 0;
            SEAL_ITERATE(operand, coeff_count, [&](auto I) {
                uint64_t poly_coeff = barrett_reduce_64(I, modulus);
                if (poly_coeff >= modulus_neg_threshold)
                {
                    poly_coeff = modulus.value() - poly_coeff;
                }
                if (poly_coeff > result)
                {
                    result = poly_coeff;
                }
            });

            return result;
        }

        void negacyclic_shift_poly_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, size_t shift, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly)
            {
                throw invalid_argument("poly");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (poly == result)
            {
                throw invalid_argument("result cannot point to the same value as poly");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
            if (util::get_power_of_two(static_cast<uint64_t>(coeff_count)) < 0)
            {
                throw invalid_argument("coeff_count");
            }
            if (shift >= coeff_count)
            {
                throw invalid_argument("shift");
            }
#endif
            // Nothing to do
            if (shift == 0)
            {
                set_uint(poly, coeff_count, result);
                return;
            }

            uint64_t index_raw = shift;
            uint64_t coeff_count_mod_mask = static_cast<uint64_t>(coeff_count) - 1;
            for (size_t i = 0; i < coeff_count; i++, poly++, index_raw++)
            {
                uint64_t index = index_raw & coeff_count_mod_mask;
                if (!(index_raw & static_cast<uint64_t>(coeff_count)) || !*poly)
                {
                    result[index] = *poly;
                }
                else
                {
                    result[index] = modulus.value() - *poly;
                }
            }
        }
    } // namespace util
} // namespace seal
