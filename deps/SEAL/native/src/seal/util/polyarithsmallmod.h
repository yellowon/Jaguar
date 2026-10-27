// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/modulus.h"
#include "seal/util/common.h"
#include "seal/util/defines.h"
#include "seal/util/iterator.h"
#include "seal/util/pointer.h"
#include "seal/util/polycore.h"
#include "seal/util/uintarithsmallmod.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "seal/plaintext.h"
#include <vector>
#include <immintrin.h>
#include <x86intrin.h>


namespace seal
{
    namespace util
    {
        void modulo_poly_coeffs(ConstCoeffIter poly, std::size_t coeff_count, const Modulus &modulus, CoeffIter result);

        inline void modulo_poly_coeffs(
            ConstRNSIter poly, std::size_t coeff_modulus_size, ConstModulusIter modulus, RNSIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(poly, modulus, result), coeff_modulus_size, [&](auto I) {
                modulo_poly_coeffs(get<0>(I), poly_modulus_degree, get<1>(I), get<2>(I));
            });
        }

        inline void modulo_poly_coeffs(
            ConstPolyIter poly_array, std::size_t size, ConstModulusIter modulus, PolyIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly_array && size > 0)
            {
                throw std::invalid_argument("poly_array");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly_array.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(poly_array, result), size, [&](auto I) {
                modulo_poly_coeffs(get<0>(I), coeff_modulus_size, modulus, get<1>(I));
            });
        }

        inline void negate_poly_coeffmod(
            ConstCoeffIter poly, std::size_t coeff_count, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw std::invalid_argument("poly");
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
            SEAL_ITERATE(iter(poly, result), coeff_count, [&](auto I) {
                auto coeff = get<0>(I);
#ifdef SEAL_DEBUG
                if (coeff >= modulus_value)
                {
                    throw std::out_of_range("poly");
                }
#endif
                std::int64_t non_zero = (coeff != 0);
                get<1>(I) = (modulus_value - coeff) & static_cast<std::uint64_t>(-non_zero);
            });
        }

        inline void negate_poly_coeffmod(
            ConstRNSIter poly, std::size_t coeff_modulus_size, ConstModulusIter modulus, RNSIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(poly, modulus, result), coeff_modulus_size, [&](auto I) {
                negate_poly_coeffmod(get<0>(I), poly_modulus_degree, get<1>(I), get<2>(I));
            });
        }

        inline void negate_poly_coeffmod(
            ConstPolyIter poly_array, std::size_t size, ConstModulusIter modulus, PolyIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly_array && size > 0)
            {
                throw std::invalid_argument("poly_array");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly_array.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(poly_array, result), size, [&](auto I) {
                negate_poly_coeffmod(get<0>(I), coeff_modulus_size, modulus, get<1>(I));
            });
        }

        void add_poly_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, std::size_t coeff_count, const Modulus &modulus,
            CoeffIter result);

////////// Added by the authors //////////

        void extract_lwe_from_rlwe_quant(
            const seal::Ciphertext &ct,
            int coeff_index,
            std::vector<uint64_t> &a_out,
            uint64_t &b_out);

        void decrypt_lwe_quant_reduce(
            const std::vector<uint64_t> &a,
            uint64_t b,
            seal::util::ConstCoeffIter secret_key,
            const uint64_t log2_q,
            const uint64_t log2_p,
            const int N,
            const int discard_lsb,
            uint64_t &out);


		void set_plain_to_vector(
			Plaintext &plain,
			const std::vector<uint64_t> &vec, const int start, const int length);

        void set_plain128_to_vector(
			unsigned long long **operand,
			const std::vector<uint64_t> &vec, const int start, const int length);

		void set_vector_to_plain(
			std::vector<uint64_t> &vec, 
			CoeffIter plain, const int start, const int length);

        void discard_lsb_ct(Ciphertext &ct, const int discard_lsb, const int N);

        void discard_lsb_ct_opt_fast(Ciphertext &ct, const int discard_lsb, const int N);

        void multiply_poly_secret_quant(
            ConstCoeffIter poly0, //a, destination
            ConstCoeffIter poly1, //secret_key
            const uint64_t q,
            CoeffIter dest,
            const int N);

        void multiply_vec_poly_secret_quant(
            std::vector<CoeffIter> &poly0, //a
            ConstCoeffIter poly1, //secret_key
            const uint64_t q,
            std::vector<CoeffIter> &dest,
            const int N);

        void multiply_poly_poly(
            ConstCoeffIter poly0,
            ConstCoeffIter poly1,
			unsigned long long **operand_0,
            const uint64_t &mod_p,
            const int N
			);

        void multiply_eval_poly_poly_mod(
            CoeffIter poly0,
            ConstCoeffIter poly1,
            const Modulus &modulus,
            const int N);

		void add_ct_plain_ct128(
			const Ciphertext &ct, 
			const Plaintext &plain,
            unsigned long long **operand1_0, // destination
            unsigned long long **operand1_1, const int N);

        void add_plain_plain_quant(
            CoeffIter poly0, // destination
            ConstCoeffIter poly1, const uint64_t log2_q, const int N);

        void add_plain_plain_quant_opt_fast(
            CoeffIter poly0,
            ConstCoeffIter poly1, const uint64_t log2_q, const int N);

        void add_plain_plain_quant_reduce(
            CoeffIter poly0, // destination
            ConstCoeffIter poly1, const uint64_t log2_q, const int N);

        void add_ct_ct_quant(
            Ciphertext &poly0, // destination
            Ciphertext &poly1, const int N);

	void add_ct_ct_quant_opt_fast(
		Ciphertext &dst,
		const Ciphertext &src, const int N);

        void add_poly128_plain_quant(
            unsigned long long **poly,
            ConstCoeffIter plain, const uint64_t log2_q, const int N, CoeffIter destination);

		void add_poly128_plain(
            unsigned long long **operand1_0, // destination
            CoeffIter plain, const int N);

		void add_poly128(
            unsigned long long **operand1_0, unsigned long long **operand1_1, 
            unsigned long long **operand2_0, unsigned long long **operand2_1,
			const int N);

        void left_shift_plain(CoeffIter poly, const uint64_t &shift, const int N);

        void decrypt_quant_reduce(CoeffIter poly, CoeffIter poly1, const uint64_t log2_q, const uint64_t log2_p, const int N, const int discard_lsb=0);

		void rotate_ct(
				const Ciphertext &cin, 
				const int N, const int step, const uint64_t &mod_q,
				unsigned long long **destination_0, unsigned long long **destination_1);


        void reduce_quant_poly(
                CoeffIter operand,
                CoeffIter poly,
				const int N, const uint64_t log2_q);

        void reduce_quant_poly_opt_fast(
            CoeffIter operand,
            CoeffIter poly,
            const int N, const uint64_t log2_q);

        void reduce_quant_ntt_poly(
                CoeffIter operand,
                CoeffIter poly, // destination
				const int N, const uint64_t qPrime, const uint64_t log2_q);


		void multiply_scalar_poly128(
				const Ciphertext &cin, 
				const int N, const uint64_t &scalar,
				unsigned long long **operand_0, unsigned long long **operand_1);


        void multiply_scalar_poly_quant(
				const Ciphertext &cin, 
				const int N, const uint64_t &scalar,
				Ciphertext &cout);

        void multiply_scalar_poly_quant_opt_fast(
                                const Ciphertext &cin,
                                const int N, const uint64_t &scalar,
                                Ciphertext &cout);	

        void pos_rotate_add_poly64(
            const Ciphertext &ctemp,
            Ciphertext &cout,
			uint64_t mod_q,
			const int N, const int pos_step);

		void pos_rotate_add_poly128(
            unsigned long long **operand_0, unsigned long long **operand_1,
            unsigned long long **destination_0, unsigned long long **destination_1,
			uint64_t *q_128,
			const int N, const int pos_step);

        void pos_rotate_add_poly_quant(
            Ciphertext &operand,
            Ciphertext &destination,
			uint64_t q,
			const int N, const int pos_step);

		void rotate_multiply_scalar_add_poly128(
				const Ciphertext &cin, 
				const int N, const int step, const uint64_t &mod_q, const uint64_t &scalar,
				unsigned long long **destination_0, unsigned long long **destination_1);

        void rotate_multiply_scalar_add_poly64(
				const Ciphertext &cin, 
				const int N, const int step, const uint64_t &mod_q, const Modulus &modulus, const uint64_t &scalar,
				Ciphertext &cout);

        void rotate_multiply_scalar_add_poly_quant(
				const Ciphertext &cin, 
				const int N, const int step, const uint64_t &mod_q, const uint64_t &scalar,
				Ciphertext &cout);

	void rotate_multiply_scalar_add_poly_quant_opt_fast(
    		const Ciphertext &cin,
    		const int N,
    		const int step,
    		const uint64_t &mod_q,
    		const uint64_t &scalar,
    		Ciphertext &cout);

////////// Added by the authors end //////////


        inline void add_poly_coeffmod(
            ConstRNSIter operand1, ConstRNSIter operand2, std::size_t coeff_modulus_size, ConstModulusIter modulus,
            RNSIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (operand1.poly_modulus_degree() != result.poly_modulus_degree() ||
                operand2.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(operand1, operand2, modulus, result), coeff_modulus_size, [&](auto I) {
                add_poly_coeffmod(get<0>(I), get<1>(I), poly_modulus_degree, get<2>(I), get<3>(I));
            });
        }

        inline void add_poly_coeffmod(
            ConstPolyIter operand1, ConstPolyIter operand2, std::size_t size, ConstModulusIter modulus, PolyIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && size > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && size > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (operand1.coeff_modulus_size() != result.coeff_modulus_size() ||
                operand2.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(operand1, operand2, result), size, [&](auto I) {
                add_poly_coeffmod(get<0>(I), get<1>(I), coeff_modulus_size, modulus, get<2>(I));
            });
        }

        void sub_poly_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, std::size_t coeff_count, const Modulus &modulus,
            CoeffIter result);

        inline void sub_poly_coeffmod(
            ConstRNSIter operand1, ConstRNSIter operand2, std::size_t coeff_modulus_size, ConstModulusIter modulus,
            RNSIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (operand1.poly_modulus_degree() != result.poly_modulus_degree() ||
                operand2.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(operand1, operand2, modulus, result), coeff_modulus_size, [&](auto I) {
                sub_poly_coeffmod(get<0>(I), get<1>(I), poly_modulus_degree, get<2>(I), get<3>(I));
            });
        }

        inline void sub_poly_coeffmod(
            ConstPolyIter operand1, ConstPolyIter operand2, std::size_t size, ConstModulusIter modulus, PolyIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && size > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && size > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (operand1.coeff_modulus_size() != result.coeff_modulus_size() ||
                operand2.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(operand1, operand2, result), size, [&](auto I) {
                sub_poly_coeffmod(get<0>(I), get<1>(I), coeff_modulus_size, modulus, get<2>(I));
            });
        }

        /**
        @param[in] scalar Must be less than modulus.value().
        */
        void add_poly_scalar_coeffmod(
            ConstCoeffIter poly, std::size_t coeff_count, std::uint64_t scalar, const Modulus &modulus,
            CoeffIter result);

        /**
        @param[in] scalar Must be less than modulus.value().
        */
        inline void add_poly_scalar_coeffmod(
            ConstRNSIter poly, std::size_t coeff_modulus_size, std::uint64_t scalar, ConstModulusIter modulus,
            RNSIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(poly, modulus, result), coeff_modulus_size, [&](auto I) {
                add_poly_scalar_coeffmod(get<0>(I), poly_modulus_degree, scalar, get<1>(I), get<2>(I));
            });
        }

        /**
        @param[in] scalar Must be less than modulus.value().
        */
        inline void add_poly_scalar_coeffmod(
            ConstPolyIter poly_array, std::size_t size, std::uint64_t scalar, ConstModulusIter modulus, PolyIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly_array && size > 0)
            {
                throw std::invalid_argument("poly_array");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly_array.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(poly_array, result), size, [&](auto I) {
                add_poly_scalar_coeffmod(get<0>(I), coeff_modulus_size, scalar, modulus, get<1>(I));
            });
        }

        /**
        @param[in] scalar Must be less than modulus.value().
        */
        void sub_poly_scalar_coeffmod(
            ConstCoeffIter poly, std::size_t coeff_count, std::uint64_t scalar, const Modulus &modulus,
            CoeffIter result);

        /**
        @param[in] scalar Must be less than modulus.value().
        */
        inline void sub_poly_scalar_coeffmod(
            ConstRNSIter poly, std::size_t coeff_modulus_size, std::uint64_t scalar, ConstModulusIter modulus,
            RNSIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(poly, modulus, result), coeff_modulus_size, [&](auto I) {
                sub_poly_scalar_coeffmod(get<0>(I), poly_modulus_degree, scalar, get<1>(I), get<2>(I));
            });
        }

        /**
        @param[in] scalar Must be less than modulus.value().
        */
        inline void sub_poly_scalar_coeffmod(
            ConstPolyIter poly_array, std::size_t size, std::uint64_t scalar, ConstModulusIter modulus, PolyIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly_array && size > 0)
            {
                throw std::invalid_argument("poly_array");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly_array.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(poly_array, result), size, [&](auto I) {
                sub_poly_scalar_coeffmod(get<0>(I), coeff_modulus_size, scalar, modulus, get<1>(I));
            });
        }

        void multiply_poly_scalar_coeffmod(
            ConstCoeffIter poly, std::size_t coeff_count, MultiplyUIntModOperand scalar, const Modulus &modulus,
            CoeffIter result);

        inline void multiply_poly_scalar_coeffmod(
            ConstCoeffIter poly, std::size_t coeff_count, std::uint64_t scalar, const Modulus &modulus,
            CoeffIter result)
        {
            // Scalar must be first reduced modulo modulus
            MultiplyUIntModOperand temp_scalar;
            temp_scalar.set(barrett_reduce_64(scalar, modulus), modulus);
            multiply_poly_scalar_coeffmod(poly, coeff_count, temp_scalar, modulus, result);
        }

        inline void multiply_poly_scalar_coeffmod(
            ConstRNSIter poly, std::size_t coeff_modulus_size, std::uint64_t scalar, ConstModulusIter modulus,
            RNSIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(poly, modulus, result), coeff_modulus_size, [&](auto I) {
                multiply_poly_scalar_coeffmod(get<0>(I), poly_modulus_degree, scalar, get<1>(I), get<2>(I));
            });
        }

        inline void multiply_poly_scalar_coeffmod(
            ConstPolyIter poly_array, std::size_t size, std::uint64_t scalar, ConstModulusIter modulus, PolyIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly_array && size > 0)
            {
                throw std::invalid_argument("poly_array");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly_array.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(poly_array, result), size, [&](auto I) {
                multiply_poly_scalar_coeffmod(get<0>(I), coeff_modulus_size, scalar, modulus, get<1>(I));
            });
        }

        void dyadic_product_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, std::size_t coeff_count, const Modulus &modulus,
            CoeffIter result);

        inline void dyadic_product_coeffmod(
            ConstRNSIter operand1, ConstRNSIter operand2, std::size_t coeff_modulus_size, ConstModulusIter modulus,
            RNSIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (operand1.poly_modulus_degree() != result.poly_modulus_degree() ||
                operand2.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(operand1, operand2, modulus, result), coeff_modulus_size, [&](auto I) {
                dyadic_product_coeffmod(get<0>(I), get<1>(I), poly_modulus_degree, get<2>(I), get<3>(I));
            });
        }

        inline void dyadic_product_coeffmod(
            ConstPolyIter operand1, ConstPolyIter operand2, std::size_t size, ConstModulusIter modulus, PolyIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && size > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && size > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (operand1.coeff_modulus_size() != result.coeff_modulus_size() ||
                operand2.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(operand1, operand2, result), size, [&](auto I) {
                dyadic_product_coeffmod(get<0>(I), get<1>(I), coeff_modulus_size, modulus, get<2>(I));
            });
        }

        std::uint64_t poly_infty_norm_coeffmod(ConstCoeffIter operand, std::size_t coeff_count, const Modulus &modulus);

        void negacyclic_shift_poly_coeffmod(
            ConstCoeffIter poly, std::size_t coeff_count, std::size_t shift, const Modulus &modulus, CoeffIter result);

        inline void negacyclic_shift_poly_coeffmod(
            ConstRNSIter poly, std::size_t coeff_modulus_size, std::size_t shift, ConstModulusIter modulus,
            RNSIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(poly, modulus, result), coeff_modulus_size, [&](auto I) {
                negacyclic_shift_poly_coeffmod(get<0>(I), poly_modulus_degree, shift, get<1>(I), get<2>(I));
            });
        }

        inline void negacyclic_shift_poly_coeffmod(
            ConstPolyIter poly_array, std::size_t size, std::size_t shift, ConstModulusIter modulus, PolyIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly_array && size > 0)
            {
                throw std::invalid_argument("poly_array");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly_array.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(poly_array, result), size, [&](auto I) {
                negacyclic_shift_poly_coeffmod(get<0>(I), coeff_modulus_size, shift, modulus, get<1>(I));
            });
        }

        inline void negacyclic_multiply_poly_mono_coeffmod(
            ConstCoeffIter poly, std::size_t coeff_count, std::uint64_t mono_coeff, std::size_t mono_exponent,
            const Modulus &modulus, CoeffIter result, MemoryPool &pool)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw std::invalid_argument("poly");
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
            SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count, pool);
            multiply_poly_scalar_coeffmod(poly, coeff_count, mono_coeff, modulus, temp);
            negacyclic_shift_poly_coeffmod(temp, coeff_count, mono_exponent, modulus, result);
        }

        inline void negacyclic_multiply_poly_mono_coeffmod(
            ConstRNSIter poly, std::size_t coeff_modulus_size, std::uint64_t mono_coeff, std::size_t mono_exponent,
            ConstModulusIter modulus, RNSIter result, MemoryPool &pool)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(poly, modulus, result), coeff_modulus_size, [&](auto I) {
                negacyclic_multiply_poly_mono_coeffmod(
                    get<0>(I), poly_modulus_degree, mono_coeff, mono_exponent, get<1>(I), get<2>(I), pool);
            });
        }

        inline void negacyclic_multiply_poly_mono_coeffmod(
            ConstPolyIter poly_array, std::size_t size, std::uint64_t mono_coeff, std::size_t mono_exponent,
            ConstModulusIter modulus, PolyIter result, MemoryPool &pool)
        {
#ifdef SEAL_DEBUG
            if (!poly_array && size > 0)
            {
                throw std::invalid_argument("poly_array");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly_array.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(poly_array, result), size, [&](auto I) {
                negacyclic_multiply_poly_mono_coeffmod(
                    get<0>(I), coeff_modulus_size, mono_coeff, mono_exponent, modulus, get<1>(I), pool);
            });
        }

        inline void negacyclic_multiply_poly_mono_coeffmod(
            ConstRNSIter poly, std::size_t coeff_modulus_size, ConstCoeffIter mono_coeff, std::size_t mono_exponent,
            ConstModulusIter modulus, RNSIter result, MemoryPool &pool)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!mono_coeff && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("mono_coeff");
            }
            if (!result && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && coeff_modulus_size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly.poly_modulus_degree() != result.poly_modulus_degree())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto poly_modulus_degree = result.poly_modulus_degree();
            SEAL_ITERATE(iter(poly, mono_coeff, modulus, result), coeff_modulus_size, [&](auto I) {
                negacyclic_multiply_poly_mono_coeffmod(
                    get<0>(I), poly_modulus_degree, get<1>(I), mono_exponent, get<2>(I), get<3>(I), pool);
            });
        }

        inline void negacyclic_multiply_poly_mono_coeffmod(
            ConstPolyIter poly_array, std::size_t size, ConstCoeffIter mono_coeff, std::size_t mono_exponent,
            ConstModulusIter modulus, PolyIter result, MemoryPool &pool)
        {
#ifdef SEAL_DEBUG
            if (!poly_array && size > 0)
            {
                throw std::invalid_argument("poly_array");
            }
            if (!mono_coeff && size > 0)
            {
                throw std::invalid_argument("mono_coeff");
            }
            if (!result && size > 0)
            {
                throw std::invalid_argument("result");
            }
            if (!modulus && size > 0)
            {
                throw std::invalid_argument("modulus");
            }
            if (poly_array.coeff_modulus_size() != result.coeff_modulus_size())
            {
                throw std::invalid_argument("incompatible iterators");
            }
#endif
            auto coeff_modulus_size = result.coeff_modulus_size();
            SEAL_ITERATE(iter(poly_array, result), size, [&](auto I) {
                negacyclic_multiply_poly_mono_coeffmod(
                    get<0>(I), coeff_modulus_size, mono_coeff, mono_exponent, modulus, get<1>(I), pool);
            });
        }
    } // namespace util
} // namespace seal
