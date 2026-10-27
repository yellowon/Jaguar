template <int N>
void resnet18_quant(string str, const SEALContext &context_quant, const uint64_t log2_q, const uint64_t log2_p, const Plaintext &secret_key_coeff, vector<vector<uint64_t> > &image, int &w, const SEALContext &context_ntt,  const PublicKey &public_key, const SecretKey &secret_key, vector<Ciphertext> &ct, int &discard_lsb_model_output)
{
	vector<int> th_count(20, 0);
	// Jaguar_52: 4 conv threads, matching ResNet-50 and MobileNetV2 (was 32).
	vector<int> mt_count(20, 4);
	vector<int> discard_lsb(20, 9);
    discard_lsb_model_output = discard_lsb[19];

	if (str.compare("st") == 0)
	{
		print_title("QUANTIZATION ResNet-18 st");
	}
	else if (str.compare("mt") == 0)
	{
		print_title("QUANTIZATION ResNet-18 mt");
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
	std::vector<LweSharedGroup> lwe_groups;
    vector<vector<uint64_t> > ss_r_data;

	string s = "../models/filters_resnet18_224_scale9/";
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
		std::cout << "[debug] w == 224\n" << std::flush;
		for (int i=0; i<resnet18_ch.size(); i++)
		{
			filters.push_back(read_filters(s + "layer_" + std::to_string(i) + ".txt", resnet18_ch[i].filter_size, resnet18_ch[i].filter_size));
			v_f_w.push_back(resnet18_ch[i].filter_size);
			v_conv_stride.push_back(resnet18_ch[i].stride);
			v_pooling.push_back(resnet18_ch[i].pooling);
			v_pooling_k.push_back(resnet18_ch[i].pooling_kernel_size);
			v_pooling_s.push_back(resnet18_ch[i].pooling_stride);
			v_pooling_p.push_back(resnet18_ch[i].pooling_padding);
		}
	}


	jaguar_networks::configure_resnet18_runtime_params(filters);
	jaguar_networks::resnet18_log_model_summary();
	jaguar_networks::resnet18_set_quant_active(true);

	print_title("Client Input Encryption");
	client_enc_zero_padding_quant(context_quant, v_f_w[0], secret_key_coeff, image, ct_in, log2_q, log2_p, 0, context_ntt);

	// 0-th convolution
	call_conv_quant<N>(0, th_count[0], 1, 0, 0, 0, seed, w, filters[0], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[0], true, &lwe_groups, discard_lsb[0]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[0], v_f_w[0], w, 1, v_pooling[0], v_pooling_k[0], v_pooling_s[0], v_pooling_p[0], ss_r_data, v_f_w[1] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[0], true, &lwe_groups);

	print_title("Residual group 0");
	// 1-th convolution
	call_conv_quant<N>(1, th_count[1], 1, 0, 0, 0, seed, w, filters[1], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[1], true, &lwe_groups, discard_lsb[1]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[1], v_f_w[1], w, 1, v_pooling[1], v_pooling_k[1], v_pooling_s[1], v_pooling_p[1], ss_r_data, v_f_w[2] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[1], true, &lwe_groups);

	// 2-th convolution
	call_conv_quant<N>(2, th_count[2], 1, 0, 0, 0, seed, w, filters[2], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[2], true, &lwe_groups, discard_lsb[2]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[2], v_f_w[2], w, 1, v_pooling[2], v_pooling_k[2], v_pooling_s[2], v_pooling_p[2], ss_r_data, v_f_w[3] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[2], true, &lwe_groups);

	// 3-th convolution
	call_conv_quant<N>(3, th_count[3], 1, 0, 0, 0, seed, w, filters[3], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[3], true, &lwe_groups, discard_lsb[3]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[3], v_f_w[3], w, 1, v_pooling[3], v_pooling_k[3], v_pooling_s[3], v_pooling_p[3], ss_r_data, v_f_w[4] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[3], true, &lwe_groups);

	// 4-th convolution
	//residual_path
	call_conv_quant<N>(4, th_count[4], 1, 0, 0, 0, seed, w, filters[4], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[4], true, &lwe_groups, discard_lsb[4]);
	residual_path_B_quant<N>(7, th_count[7], 1, 0, 0, 0, seed, w, filters[4], v_conv_stride[4], v_f_w[4], v_pooling[4], v_pooling_k[4], v_pooling_s[4], v_pooling_p[4], filters[7], v_conv_stride[7], v_f_w[7], v_pooling[7], v_pooling_k[7], v_pooling_s[7], v_pooling_p[7], context_quant, log2_q, log2_p, secret_key_coeff, context_ntt, q_ntt, public_key, secret_key, ct_in, ct_out1, discard_lsb[4], discard_lsb[6]);////
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[4], v_f_w[4], w, 1, v_pooling[4], v_pooling_k[4], v_pooling_s[4], v_pooling_p[4], ss_r_data, v_f_w[5] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[4], true, &lwe_groups);

	print_title("Residual group 1");
	// 5-th convolution
	call_conv_quant<N>(5, th_count[5], 1, 0, 0, 0, seed, w, filters[5], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[5], true, &lwe_groups, discard_lsb[5]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[5], v_f_w[5], w, 1, v_pooling[5], v_pooling_k[5], v_pooling_s[5], v_pooling_p[5], ss_r_data, v_f_w[6] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[5], true, &lwe_groups);

	// 6-th convolution
	//no relu
	call_conv_quant<N>(6, th_count[7], 1, 0, 0, 0, seed, w, filters[6], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[6], true, &lwe_groups, discard_lsb[6]);
	bool no_relu = 0;
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[6], v_f_w[6], w, no_relu, v_pooling[6], v_pooling_k[6], v_pooling_s[6], v_pooling_p[6], ss_r_data, v_f_w[7] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[6], true, &lwe_groups);

	print_title("Add residual path");
	//add residual_path
	cloud_add_residual_path_quant<N>(context_quant, log2_q, log2_p, seed, w, filters[7], ct_in, ct_out1, ct_out, ss_r_data, discard_lsb[7], true, &lwe_groups);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, 1, v_f_w[7], w, 1, v_pooling[7], v_pooling_k[7], v_pooling_s[7], v_pooling_p[7], ss_r_data, v_f_w[8] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[7], true, &lwe_groups);

	// 8-th convolution
	call_conv_quant<N>(8, th_count[8], 1, 0, 0, 0, seed, w, filters[8], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[8], true, &lwe_groups, discard_lsb[8]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[8], v_f_w[8], w, 1, v_pooling[8], v_pooling_k[8], v_pooling_s[8], v_pooling_p[8], ss_r_data, v_f_w[9] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[8], true, &lwe_groups);

	// 9-th convolution
	//residual_path
	call_conv_quant<N>(9, th_count[9], 1, 0, 0, 0, seed, w, filters[9], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[9], true, &lwe_groups, discard_lsb[9]);
	residual_path_B_quant<N>(12, th_count[12], 1, 0, 0, 0, seed, w, filters[9], v_conv_stride[9], v_f_w[9], v_pooling[9], v_pooling_k[9], v_pooling_s[9], v_pooling_p[9], filters[12], v_conv_stride[12], v_f_w[12], v_pooling[12], v_pooling_k[12], v_pooling_s[12], v_pooling_p[12], context_quant, log2_q, log2_p, secret_key_coeff, context_ntt, q_ntt, public_key, secret_key, ct_in, ct_out1, discard_lsb[9], discard_lsb[11]);
	
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[9], v_f_w[9], w, 1, v_pooling[9], v_pooling_k[9], v_pooling_s[9], v_pooling_p[9], ss_r_data, v_f_w[10] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[9], true, &lwe_groups);

	print_title("Residual group 2");
	// 10-th convolution
	call_conv_quant<N>(10, th_count[10], 1, 0, 0, 0, seed, w, filters[10], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[10], true, &lwe_groups, discard_lsb[10]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[10], v_f_w[10], w, 1, v_pooling[10], v_pooling_k[10], v_pooling_s[10], v_pooling_p[10], ss_r_data, v_f_w[11] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[10], true, &lwe_groups);

	// 11-th convolution
	//no relu
	call_conv_quant<N>(11, th_count[11], 1, 0, 0, 0, seed, w, filters[11], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[11], true, &lwe_groups, discard_lsb[11]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[11], v_f_w[11], w, no_relu, v_pooling[11], v_pooling_k[11], v_pooling_s[11], v_pooling_p[11], ss_r_data, v_f_w[12] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[11], true, &lwe_groups);

	print_title("Add residual path");
	//add residual_path
	cloud_add_residual_path_quant<N>(context_quant, log2_q, log2_p, seed, w, filters[12], ct_in, ct_out1, ct_out, ss_r_data, discard_lsb[12], true, &lwe_groups);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, 1, v_f_w[12], w, 1, v_pooling[12], v_pooling_k[12], v_pooling_s[12], v_pooling_p[12], ss_r_data, v_f_w[13] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[12], true, &lwe_groups);

	// 13-th convolution
	call_conv_quant<N>(13, th_count[13], 1, 0, 0, 0, seed, w, filters[13], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[13], true, &lwe_groups, discard_lsb[13]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[13], v_f_w[13], w, 1, v_pooling[13], v_pooling_k[13], v_pooling_s[13], v_pooling_p[13], ss_r_data, v_f_w[14] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[13], true, &lwe_groups);

	// 14-th convolution
	//residual_path
	call_conv_quant<N>(14, th_count[14], 1, 0, 0, 0, seed, w, filters[14], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[14], true, &lwe_groups, discard_lsb[14]);
	residual_path_B_quant<N>(17, th_count[17], 1, 0, 0, 0, seed, w, filters[14], v_conv_stride[14], v_f_w[14], v_pooling[14], v_pooling_k[14], v_pooling_s[14], v_pooling_p[14], filters[17], v_conv_stride[17], v_f_w[17], v_pooling[17], v_pooling_k[17], v_pooling_s[17], v_pooling_p[17], context_quant, log2_q, log2_p, secret_key_coeff, context_ntt, q_ntt, public_key, secret_key, ct_in, ct_out1, discard_lsb[14], discard_lsb[16]);

	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[14], v_f_w[14], w, 1, v_pooling[14], v_pooling_k[14], v_pooling_s[14], v_pooling_p[14], ss_r_data, v_f_w[15] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[14], true, &lwe_groups);


	print_title("Residual group 3");
	// 15-th convolution
	call_conv_quant<N>(15, th_count[15], 1, 0, 0, 0, seed, w, filters[15], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[15], true, &lwe_groups, discard_lsb[15]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[15], v_f_w[15], w, 1, v_pooling[15], v_pooling_k[15], v_pooling_s[15], v_pooling_p[15], ss_r_data, v_f_w[16] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[15], true, &lwe_groups);

	// 16-th convolution
	//no relu
	call_conv_quant<N>(16, th_count[16], 1, 0, 0, 0, seed, w, filters[16], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[16], true, &lwe_groups, discard_lsb[16]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[16], v_f_w[16], w, no_relu, v_pooling[16], v_pooling_k[16], v_pooling_s[16], v_pooling_p[16], ss_r_data, v_f_w[17] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[16], true, &lwe_groups);

	print_title("Add residual path");
	//add residual_path
	cloud_add_residual_path_quant<N>(context_quant, log2_q, log2_p, seed, w, filters[17], ct_in, ct_out1, ct_out, ss_r_data, discard_lsb[17], true, &lwe_groups);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, 1, v_f_w[17], w, 1, v_pooling[17], v_pooling_k[17], v_pooling_s[17], v_pooling_p[17], ss_r_data, v_f_w[18] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[17], true, &lwe_groups);

	// 18-th convolution
	call_conv_quant<N>(18, th_count[18], 1, 0, 0, 0, seed, w, filters[18], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[18], true, &lwe_groups, discard_lsb[18]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[18], v_f_w[18], w, 1, v_pooling[18], v_pooling_k[18], v_pooling_s[18], v_pooling_p[18], ss_r_data, v_f_w[19] , ct_in, context_ntt, q_ntt, public_key, secret_key, discard_lsb[18], true, &lwe_groups);

	// 19-th convolution
	call_conv_quant<N>(19, th_count[19], 1, 0, 0, 0, seed, w, filters[19], context_quant, log2_q, log2_p, ct_in, ct_out, ss_r_data, v_conv_stride[19], true, &lwe_groups, discard_lsb[19]);
	stride_relu_avg_pooling_quant<N>(context_quant, log2_q, log2_p, secret_key_coeff, ct_out, v_conv_stride[19], v_f_w[19], w, 1, v_pooling[19], v_pooling_k[19], v_pooling_s[19], v_pooling_p[19], ss_r_data, 1 , ct, context_ntt, q_ntt, public_key, secret_key, discard_lsb[19], true, &lwe_groups);

	jaguar_networks::resnet18_set_quant_active(false);

	run_model_fc_quant(
        "resnet18",
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

