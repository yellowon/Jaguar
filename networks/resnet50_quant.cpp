template <int N>
void resnet50_quant(string str, const SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, const Plaintext &secret_key_coeff, vector<vector<uint64_t> > &image, int &w, const SEALContext &context_ntt, const PublicKey &public_key, const SecretKey &secret_key, vector<Ciphertext> &ct, int &discard_lsb_model_output)
{
	vector<int> th_count(53, 0);
	vector<int> mt_count(53, 4);
	vector<int> discard_lsb(53, 9);
	discard_lsb_model_output = discard_lsb[52];

	if (str.compare("st") == 0)
	{
		print_title("QUANTIZATION ResNet-50 st");
	}
	else if (str.compare("mt") == 0)
	{
		print_title("QUANTIZATION ResNet-50 mt");
		copy(mt_count.begin(), mt_count.end(), th_count.begin());
	}
	else
	{
		throw std::runtime_error("Choose st / mt");
	}

    auto &context_ntt_data = *context_ntt.first_context_data();
    auto &parms_ntt = context_ntt_data.parms();
    auto &coeff_modulus_ntt = parms_ntt.coeff_modulus();
    uint64_t q_ntt = coeff_modulus_ntt[0].value();

	int seed;
	vector<Ciphertext> ct_in;
	vector<Ciphertext> ct_out;
	vector<Ciphertext> ct_out1;
    vector<vector<uint64_t> > ss_r_data;
    std::vector<LweSharedGroup> lwe_groups;

	string s = "../models/filters_resnet50_" + std::to_string(w) + "/";
    vector<vector<vector<vector<uint64_t>>>> filters;
	vector<uint64_t> v_f_w;
	vector<uint64_t> v_conv_stride;
	vector<bool> v_pooling;
	vector<int> v_pooling_k;
	vector<int> v_pooling_s;
	vector<int> v_pooling_p;

	if (w != 224)
	{
		throw std::runtime_error(
          "Only w == 224 is supported in this public Jaguar build.");
	}
	else // w == 224
	{
		for (int i=0; i<resnet50_in224_ch.size(); i++)
		{
			filters.push_back(allocate_filters(resnet50_in224_ch[i]));
			v_f_w.push_back(resnet50_in224_ch[i].filter_size);
			v_conv_stride.push_back(resnet50_in224_ch[i].stride);
			v_pooling.push_back(resnet50_in224_ch[i].pooling);
			v_pooling_k.push_back(resnet50_in224_ch[i].pooling_kernel_size);
			v_pooling_s.push_back(resnet50_in224_ch[i].pooling_stride);
			v_pooling_p.push_back(resnet50_in224_ch[i].pooling_padding);
		}
	}

	bool no_relu = 0;

	print_title("Client Input Encryption");

	jaguar_networks::configure_resnet50_runtime_params(filters);
	jaguar_networks::resnet50_log_model_summary();
	jaguar_networks::resnet50_set_quant_active(true);

	print_title("Client Input Encryption");
	client_enc_zero_padding_quant(context_quant, v_f_w[0], secret_key_coeff, image, ct_in, log2_q, log2_p, 0, context_ntt);

    // 0-th convolution
    // Match OpenCheetah's first-stage activation shape by downsampling before ReLU.
	call_conv_quant<N>(0, th_count[0], 1, 0, 0, 0, seed, w, filters[0], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[0], true, &lwe_groups, discard_lsb[0]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[0], v_f_w[0], w, 1, v_pooling[0], v_pooling_k[0], v_pooling_s[0], v_pooling_p[0], ss_r_data, v_f_w[1] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[0], true, &lwe_groups);

	print_title("Layer 1 (3)");
	print_title("Residual block 1");
	residual_path_B_quant<N>(4, th_count[4], 1, 0, 0, 0, seed, w, filters[0], v_conv_stride[0], v_f_w[0], v_pooling[0], v_pooling_k[0], v_pooling_s[0], v_pooling_p[0], filters[4], v_conv_stride[4], v_f_w[4], v_pooling[4], v_pooling_k[4], v_pooling_s[4], v_pooling_p[4], context_quant, log2_q, log2_p, secret_key_coeff, context_ntt, q_ntt, public_key, secret_key, ct_in, ct_out1, discard_lsb[0], discard_lsb[4]);

	// 1-th convolution
	call_conv_quant<N>(1, th_count[1], 1, 0, 0, 0, seed, w, filters[1], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[1], true, &lwe_groups, discard_lsb[1]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[1], v_f_w[1], w, 1, v_pooling[1], v_pooling_k[1], v_pooling_s[1], v_pooling_p[1], ss_r_data, v_f_w[2] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[1], true, &lwe_groups);

	// 2-th convolution
	call_conv_quant<N>(2, th_count[2], 1, 0, 0, 0, seed, w, filters[2], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[2], true, &lwe_groups, discard_lsb[2]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[2], v_f_w[2], w, 1, v_pooling[2], v_pooling_k[2], v_pooling_s[2], v_pooling_p[2], ss_r_data, v_f_w[3] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[2], true, &lwe_groups);

	// 3-th convolution
	call_conv_quant<N>(3, th_count[3], 1, 0, 0, 0, seed, w, filters[3], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[3], true, &lwe_groups, discard_lsb[3]);
	//no relu
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[3], v_f_w[3], w, no_relu, v_pooling[3], v_pooling_k[3], v_pooling_s[3], v_pooling_p[3], ss_r_data, v_f_w[4] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[3], true, &lwe_groups);


	print_title("Add residual path");
	//add residual_path
	cloud_add_residual_path_quant<N>(context_quant, log2_q, log2_p, seed, w, filters[4], ct_in, ct_out1, ct_out, ss_r_data, discard_lsb[4], true, &lwe_groups);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, 1, v_f_w[4], w, 1, v_pooling[4], v_pooling_k[4], v_pooling_s[4], v_pooling_p[4], ss_r_data, v_f_w[5] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[4], true, &lwe_groups);


	print_title("Residual block 2");
	// 5-th convolution
	call_conv_quant<N>(5, th_count[5], 1, 0, 0, 0, seed, w, filters[5], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[5], true, &lwe_groups, discard_lsb[5]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[5], v_f_w[5], w, 1, v_pooling[5], v_pooling_k[5], v_pooling_s[5], v_pooling_p[5], ss_r_data, v_f_w[6] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[5], true, &lwe_groups);

	// 6-th convolution
	call_conv_quant<N>(6, th_count[6], 1, 0, 0, 0, seed, w, filters[6], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[6], true, &lwe_groups, discard_lsb[6]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[6], v_f_w[6], w, 1, v_pooling[6], v_pooling_k[6], v_pooling_s[6], v_pooling_p[6], ss_r_data, v_f_w[7] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[6], true, &lwe_groups);

	// 7-th convolution
	call_conv_quant<N>(7, th_count[7], 1, 0, 0, 0, seed, w, filters[7], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[7], true, &lwe_groups, discard_lsb[7]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[7], v_f_w[7], w, 1, v_pooling[7], v_pooling_k[7], v_pooling_s[7], v_pooling_p[7], ss_r_data, v_f_w[8] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[7], true, &lwe_groups);

	print_title("Residual block 3");
	// 8-th convolution
	call_conv_quant<N>(8, th_count[8], 1, 0, 0, 0, seed, w, filters[8], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[8], true, &lwe_groups, discard_lsb[8]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[8], v_f_w[8], w, 1, v_pooling[8], v_pooling_k[8], v_pooling_s[8], v_pooling_p[8], ss_r_data, v_f_w[9] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[8], true, &lwe_groups);

	// 9-th convolution
	call_conv_quant<N>(9, th_count[9], 1, 0, 0, 0, seed, w, filters[9], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[9], true, &lwe_groups, discard_lsb[9]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[9], v_f_w[9], w, 1, v_pooling[9], v_pooling_k[9], v_pooling_s[9], v_pooling_p[9], ss_r_data, v_f_w[10] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[9], true, &lwe_groups);

	// 10-th convolution
	call_conv_quant<N>(10, th_count[10], 1, 0, 0, 0, seed, w, filters[10], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[10], true, &lwe_groups, discard_lsb[10]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[10], v_f_w[10], w, 1, v_pooling[10], v_pooling_k[10], v_pooling_s[10], v_pooling_p[10], ss_r_data, v_f_w[11] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[10], true, &lwe_groups);


	print_title("Layer 2 (4)");
	print_title("Residual block 1");
	residual_path_B_quant<N>(14, th_count[14], 1, 0, 0, 0, seed, w, filters[10], v_conv_stride[10], v_f_w[10], v_pooling[10], v_pooling_k[10], v_pooling_s[10], v_pooling_p[10], filters[14], v_conv_stride[14], v_f_w[14], v_pooling[14], v_pooling_k[14], v_pooling_s[14], v_pooling_p[14], context_quant, log2_q, log2_p, secret_key_coeff, context_ntt, q_ntt, public_key, secret_key, ct_in, ct_out1, discard_lsb[10], discard_lsb[14]);

	// 11-th convolution
	call_conv_quant<N>(11, th_count[11], 1, 0, 0, 0, seed, w, filters[11], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[11], true, &lwe_groups, discard_lsb[11]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[11], v_f_w[11], w, 1, v_pooling[11], v_pooling_k[11], v_pooling_s[11], v_pooling_p[11], ss_r_data, v_f_w[12] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[11], true, &lwe_groups);

	// 12-th convolution
	call_conv_quant<N>(12, th_count[12], 1, 0, 0, 0, seed, w, filters[12], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[12], true, &lwe_groups, discard_lsb[12]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[12], v_f_w[12], w, 1, v_pooling[12], v_pooling_k[12], v_pooling_s[12], v_pooling_p[12], ss_r_data, v_f_w[13] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[12], true, &lwe_groups);

	// 13-th convolution
	call_conv_quant<N>(13, th_count[13], 1, 0, 0, 0, seed, w, filters[13], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[13], true, &lwe_groups, discard_lsb[13]);
	//no relu
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[13], v_f_w[13], w, no_relu, v_pooling[13], v_pooling_k[13], v_pooling_s[13], v_pooling_p[13], ss_r_data, v_f_w[14] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[13], true, &lwe_groups);

	print_title("Add residual path");
	//add residual_path
	cloud_add_residual_path_quant<N>(context_quant, log2_q, log2_p, seed, w, filters[14], ct_in, ct_out1, ct_out, ss_r_data, discard_lsb[14], true, &lwe_groups);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, 1, v_f_w[14], w, 1, v_pooling[14], v_pooling_k[14], v_pooling_s[14], v_pooling_p[14], ss_r_data, v_f_w[15] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[14], true, &lwe_groups);

	print_title("Residual block 2");
	// 15-th convolution
	call_conv_quant<N>(15, th_count[15], 1, 0, 0, 0, seed, w, filters[15], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[15], true, &lwe_groups, discard_lsb[15]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[15], v_f_w[15], w, 1, v_pooling[15], v_pooling_k[15], v_pooling_s[15], v_pooling_p[15], ss_r_data, v_f_w[16] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[15], true, &lwe_groups);

	// 16-th convolution
	call_conv_quant<N>(16, th_count[16], 1, 0, 0, 0, seed, w, filters[16], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[16], true, &lwe_groups, discard_lsb[16]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[16], v_f_w[16], w, 1, v_pooling[16], v_pooling_k[16], v_pooling_s[16], v_pooling_p[16], ss_r_data, v_f_w[17] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[16], true, &lwe_groups);

	// 17-th convolution
	call_conv_quant<N>(17, th_count[17], 1, 0, 0, 0, seed, w, filters[17], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[17], true, &lwe_groups, discard_lsb[17]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[17], v_f_w[17], w, 1, v_pooling[17], v_pooling_k[17], v_pooling_s[17], v_pooling_p[17], ss_r_data, v_f_w[18] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[17], true, &lwe_groups);

	print_title("Residual block 3");
	// 18-th convolution
	call_conv_quant<N>(18, th_count[18], 1, 0, 0, 0, seed, w, filters[18], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[18], true, &lwe_groups, discard_lsb[18]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[18], v_f_w[18], w, 1, v_pooling[18], v_pooling_k[18], v_pooling_s[18], v_pooling_p[18], ss_r_data, v_f_w[19] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[18], true, &lwe_groups);

	// 19-th convolution
	call_conv_quant<N>(19, th_count[19], 1, 0, 0, 0, seed, w, filters[19], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[19], true, &lwe_groups, discard_lsb[19]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[19], v_f_w[19], w, 1, v_pooling[19], v_pooling_k[19], v_pooling_s[19], v_pooling_p[19], ss_r_data, v_f_w[20] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[19], true, &lwe_groups);

	// 20-th convolution
	call_conv_quant<N>(20, th_count[20], 1, 0, 0, 0, seed, w, filters[20], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[20], true, &lwe_groups, discard_lsb[20]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[20], v_f_w[20], w, 1, v_pooling[20], v_pooling_k[20], v_pooling_s[20], v_pooling_p[20], ss_r_data, v_f_w[21] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[20], true, &lwe_groups);

	print_title("Residual block 4");
	// 21-th convolution
	call_conv_quant<N>(21, th_count[21], 1, 0, 0, 0, seed, w, filters[21], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[21], true, &lwe_groups, discard_lsb[21]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[21], v_f_w[21], w, 1, v_pooling[21], v_pooling_k[21], v_pooling_s[21], v_pooling_p[21], ss_r_data, v_f_w[22] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[21], true, &lwe_groups);

	// 22-th convolution
	call_conv_quant<N>(22, th_count[22], 1, 0, 0, 0, seed, w, filters[22], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[22], true, &lwe_groups, discard_lsb[22]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[22], v_f_w[22], w, 1, v_pooling[22], v_pooling_k[22], v_pooling_s[22], v_pooling_p[22], ss_r_data, v_f_w[23] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[22], true, &lwe_groups);

	// 23-th convolution
	call_conv_quant<N>(23, th_count[23], 1, 0, 0, 0, seed, w, filters[23], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[23], true, &lwe_groups, discard_lsb[23]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[23], v_f_w[23], w, 1, v_pooling[23], v_pooling_k[23], v_pooling_s[23], v_pooling_p[23], ss_r_data, v_f_w[24] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[23], true, &lwe_groups);


	print_title("Layer 3 (6)");
	print_title("Residual block 1");
	residual_path_B_quant<N>(27, th_count[27], 1, 0, 0, 0, seed, w, filters[23], v_conv_stride[23], v_f_w[23], v_pooling[23], v_pooling_k[23], v_pooling_s[23], v_pooling_p[23], filters[27], v_conv_stride[27], v_f_w[27], v_pooling[27], v_pooling_k[27], v_pooling_s[27], v_pooling_p[27], context_quant, log2_q, log2_p, secret_key_coeff, context_ntt, q_ntt, public_key, secret_key, ct_in, ct_out1, discard_lsb[23], discard_lsb[27]);

	// 24-th convolution
	call_conv_quant<N>(24, th_count[24], 1, 0, 0, 0, seed, w, filters[24], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[24], true, &lwe_groups, discard_lsb[24]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[24], v_f_w[24], w, 1, v_pooling[24], v_pooling_k[24], v_pooling_s[24], v_pooling_p[24], ss_r_data, v_f_w[25] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[24], true, &lwe_groups);

	// 25-th convolution
	call_conv_quant<N>(25, th_count[25], 1, 0, 0, 0, seed, w, filters[25], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[25], true, &lwe_groups, discard_lsb[25]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[25], v_f_w[25], w, 1, v_pooling[25], v_pooling_k[25], v_pooling_s[25], v_pooling_p[25], ss_r_data, v_f_w[26] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[25], true, &lwe_groups);

	// 26-th convolution
	call_conv_quant<N>(26, th_count[26], 1, 0, 0, 0, seed, w, filters[26], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[26], true, &lwe_groups, discard_lsb[26]);
	//no relu
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[26], v_f_w[26], w, no_relu, v_pooling[26], v_pooling_k[26], v_pooling_s[26], v_pooling_p[26], ss_r_data, v_f_w[27] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[26], true, &lwe_groups);

	print_title("Add residual path");
	//add residual_path
	cloud_add_residual_path_quant<N>(context_quant, log2_q, log2_p, seed, w, filters[27], ct_in, ct_out1, ct_out, ss_r_data, discard_lsb[27], true, &lwe_groups);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, 1, v_f_w[27], w, 1, v_pooling[27], v_pooling_k[27], v_pooling_s[27], v_pooling_p[27], ss_r_data, v_f_w[28] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[27], true, &lwe_groups);

	print_title("Residual block 2");
	// 28-th convolution
	call_conv_quant<N>(28, th_count[28], 1, 0, 0, 0, seed, w, filters[28], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[28], true, &lwe_groups, discard_lsb[28]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[28], v_f_w[28], w, 1, v_pooling[28], v_pooling_k[28], v_pooling_s[28], v_pooling_p[28], ss_r_data, v_f_w[29] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[28], true, &lwe_groups);

	// 29-th convolution
	call_conv_quant<N>(29, th_count[29], 1, 0, 0, 0, seed, w, filters[29], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[29], true, &lwe_groups, discard_lsb[29]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[29], v_f_w[29], w, 1, v_pooling[29], v_pooling_k[29], v_pooling_s[29], v_pooling_p[29], ss_r_data, v_f_w[30] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[29], true, &lwe_groups);

	// 30-th convolution
	call_conv_quant<N>(30, th_count[30], 1, 0, 0, 0, seed, w, filters[30], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[30], true, &lwe_groups, discard_lsb[30]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[30], v_f_w[30], w, 1, v_pooling[30], v_pooling_k[30], v_pooling_s[30], v_pooling_p[30], ss_r_data, v_f_w[31] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[30], true, &lwe_groups);

	print_title("Residual block 3");
	// 31-th convolution
	call_conv_quant<N>(31, th_count[31], 1, 0, 0, 0, seed, w, filters[31], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[31], true, &lwe_groups, discard_lsb[31]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[31], v_f_w[31], w, 1, v_pooling[31], v_pooling_k[31], v_pooling_s[31], v_pooling_p[31], ss_r_data, v_f_w[32] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[31], true, &lwe_groups);

	// 32-th convolution
	call_conv_quant<N>(32, th_count[32], 1, 0, 0, 0, seed, w, filters[32], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[32], true, &lwe_groups, discard_lsb[32]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[32], v_f_w[32], w, 1, v_pooling[32], v_pooling_k[32], v_pooling_s[32], v_pooling_p[32], ss_r_data, v_f_w[33] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[32], true, &lwe_groups);

	// 33-th convolution
	call_conv_quant<N>(33, th_count[33], 1, 0, 0, 0, seed, w, filters[33], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[33], true, &lwe_groups, discard_lsb[33]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[33], v_f_w[33], w, 1, v_pooling[33], v_pooling_k[33], v_pooling_s[33], v_pooling_p[33], ss_r_data, v_f_w[34] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[33], true, &lwe_groups);

	print_title("Residual block 4");
	// 34-th convolution
	call_conv_quant<N>(34, th_count[34], 1, 0, 0, 0, seed, w, filters[34], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[34], true, &lwe_groups, discard_lsb[34]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[34], v_f_w[34], w, 1, v_pooling[34], v_pooling_k[34], v_pooling_s[34], v_pooling_p[34], ss_r_data, v_f_w[35] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[34], true, &lwe_groups);

	// 35-th convolution
	call_conv_quant<N>(35, th_count[35], 1, 0, 0, 0, seed, w, filters[35], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[35], true, &lwe_groups, discard_lsb[35]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[35], v_f_w[35], w, 1, v_pooling[35], v_pooling_k[35], v_pooling_s[35], v_pooling_p[35], ss_r_data, v_f_w[36] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[35], true, &lwe_groups);

	// 36-th convolution
	call_conv_quant<N>(36, th_count[36], 1, 0, 0, 0, seed, w, filters[36], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[36], true, &lwe_groups, discard_lsb[36]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[36], v_f_w[36], w, 1, v_pooling[36], v_pooling_k[36], v_pooling_s[36], v_pooling_p[36], ss_r_data, v_f_w[37] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[36], true, &lwe_groups);

	print_title("Residual block 5");
	// 37-th convolution
	call_conv_quant<N>(37, th_count[37], 1, 0, 0, 0, seed, w, filters[37], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[37], true, &lwe_groups, discard_lsb[37]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[37], v_f_w[37], w, 1, v_pooling[37], v_pooling_k[37], v_pooling_s[37], v_pooling_p[37], ss_r_data, v_f_w[38] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[37], true, &lwe_groups);

	// 38-th convolution
	call_conv_quant<N>(38, th_count[38], 1, 0, 0, 0, seed, w, filters[38], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[38], true, &lwe_groups, discard_lsb[38]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[38], v_f_w[38], w, 1, v_pooling[38], v_pooling_k[38], v_pooling_s[38], v_pooling_p[38], ss_r_data, v_f_w[39] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[38], true, &lwe_groups);

	// 39-th convolution
	call_conv_quant<N>(39, th_count[39], 1, 0, 0, 0, seed, w, filters[39], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[39], true, &lwe_groups, discard_lsb[39]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[39], v_f_w[39], w, 1, v_pooling[39], v_pooling_k[39], v_pooling_s[39], v_pooling_p[39], ss_r_data, v_f_w[40] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[39], true, &lwe_groups);

	print_title("Residual block 6");
	// 40-th convolution
	call_conv_quant<N>(40, th_count[40], 1, 0, 0, 0, seed, w, filters[40], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[40], true, &lwe_groups, discard_lsb[40]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[40], v_f_w[40], w, 1, v_pooling[40], v_pooling_k[40], v_pooling_s[40], v_pooling_p[40], ss_r_data, v_f_w[41] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[40], true, &lwe_groups);

	// 41-th convolution
	call_conv_quant<N>(41, th_count[41], 1, 0, 0, 0, seed, w, filters[41], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[41], true, &lwe_groups, discard_lsb[41]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[41], v_f_w[41], w, 1, v_pooling[41], v_pooling_k[41], v_pooling_s[41], v_pooling_p[41], ss_r_data, v_f_w[42] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[41], true, &lwe_groups);

	// 42-th convolution
	call_conv_quant<N>(42, th_count[42], 1, 0, 0, 0, seed, w, filters[42], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[42], true, &lwe_groups, discard_lsb[42]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[42], v_f_w[42], w, 1, v_pooling[42], v_pooling_k[42], v_pooling_s[42], v_pooling_p[42], ss_r_data, v_f_w[43] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[42], true, &lwe_groups);


	print_title("Layer 4 (3)");
	print_title("Residual block 1");
	residual_path_B_quant<N>(46, th_count[46], 1, 0, 0, 0, seed, w, filters[42], v_conv_stride[42], v_f_w[42], v_pooling[42], v_pooling_k[42], v_pooling_s[42], v_pooling_p[42], filters[46], v_conv_stride[46], v_f_w[46], v_pooling[46], v_pooling_k[46], v_pooling_s[46], v_pooling_p[46], context_quant, log2_q, log2_p, secret_key_coeff, context_ntt, q_ntt, public_key, secret_key, ct_in, ct_out1, discard_lsb[42], discard_lsb[46]);

	// 43-th convolution
	call_conv_quant<N>(43, th_count[43], 1, 0, 0, 0, seed, w, filters[43], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[43], true, &lwe_groups, discard_lsb[43]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[43], v_f_w[43], w, 1, v_pooling[43], v_pooling_k[43], v_pooling_s[43], v_pooling_p[43], ss_r_data, v_f_w[44] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[43], true, &lwe_groups);

	// 44-th convolution
	call_conv_quant<N>(44, th_count[44], 1, 0, 0, 0, seed, w, filters[44], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[44], true, &lwe_groups, discard_lsb[44]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[44], v_f_w[44], w, 1, v_pooling[44], v_pooling_k[44], v_pooling_s[44], v_pooling_p[44], ss_r_data, v_f_w[45] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[44], true, &lwe_groups);

	// 45-th convolution
	call_conv_quant<N>(45, th_count[45], 1, 0, 0, 0, seed, w, filters[45], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[45], true, &lwe_groups, discard_lsb[45]);
	//no relu
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[45], v_f_w[45], w, no_relu, v_pooling[45], v_pooling_k[45], v_pooling_s[45], v_pooling_p[45], ss_r_data, v_f_w[46] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[45], true, &lwe_groups);

	print_title("Add residual path");
	//add residual_path
	cloud_add_residual_path_quant<N>(context_quant, log2_q, log2_p, seed, w, filters[46], ct_in, ct_out1, ct_out, ss_r_data, discard_lsb[46], true, &lwe_groups);
	// conv stride = 1 (already strided)
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, 1, v_f_w[46], w, 1, v_pooling[46], v_pooling_k[46], v_pooling_s[46], v_pooling_p[46], ss_r_data, v_f_w[47] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[46], true, &lwe_groups);

	print_title("Residual block 2");
	// 47-th convolution
	call_conv_quant<N>(47, th_count[47], 1, 0, 0, 0, seed, w, filters[47], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[47], true, &lwe_groups, discard_lsb[47]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[47], v_f_w[47], w, 1, v_pooling[47], v_pooling_k[47], v_pooling_s[47], v_pooling_p[47], ss_r_data, v_f_w[48] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[47], true, &lwe_groups);

	// 48-th convolution
	call_conv_quant<N>(48, th_count[48], 1, 0, 0, 0, seed, w, filters[48], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[48], true, &lwe_groups, discard_lsb[48]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[48], v_f_w[48], w, 1, v_pooling[48], v_pooling_k[48], v_pooling_s[48], v_pooling_p[48], ss_r_data, v_f_w[49] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[48], true, &lwe_groups);

	// 49-th convolution
	call_conv_quant<N>(49, th_count[49], 1, 0, 0, 0, seed, w, filters[49], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[49], true, &lwe_groups, discard_lsb[49]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[49], v_f_w[49], w, 1, v_pooling[49], v_pooling_k[49], v_pooling_s[49], v_pooling_p[49], ss_r_data, v_f_w[50] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[49], true, &lwe_groups);

	print_title("Residual block 3");
	// 50-th convolution
	call_conv_quant<N>(50, th_count[50], 1, 0, 0, 0, seed, w, filters[50], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[50], true, &lwe_groups, discard_lsb[50]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[50], v_f_w[50], w, 1, v_pooling[50], v_pooling_k[50], v_pooling_s[50], v_pooling_p[50], ss_r_data, v_f_w[51] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[50], true, &lwe_groups);

	// 51-th convolution
	call_conv_quant<N>(51, th_count[51], 1, 0, 0, 0, seed, w, filters[51], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[51], true, &lwe_groups, discard_lsb[51]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[51], v_f_w[51], w, 1, v_pooling[51], v_pooling_k[51], v_pooling_s[51], v_pooling_p[51], ss_r_data, v_f_w[52] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[51], true, &lwe_groups);

	// 52-th convolution
	call_conv_quant<N>(52, th_count[52], 1, 0, 0, 0, seed, w, filters[52], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[52], true, &lwe_groups, discard_lsb[52]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[52], v_f_w[52], w, 1, v_pooling[52], v_pooling_k[52], v_pooling_s[52], v_pooling_p[52], ss_r_data, 1, ct, context_ntt, q_ntt, public_key, secret_key, discard_lsb[52], true, &lwe_groups);


	jaguar_networks::resnet50_set_quant_active(false);

	run_model_fc_quant(
		"resnet50",
		context_quant,
		log2_q,
		log2_p,
		secret_key_coeff,
		context_ntt,
		public_key,
		secret_key,
		ct,
		discard_lsb_model_output
	);
}
