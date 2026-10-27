#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <sstream>
#include <stdexcept>

struct LayerConfig {
	uint64_t input_channels;
	uint64_t output_channels;
	uint64_t filter_size;
	uint64_t stride;
	bool depthwise;
	bool pooling;
	int pooling_kernel_size;
	int pooling_stride;
	int pooling_padding;
};


std::vector<LayerConfig> resnet50_in64_ch = {
	{3, 64, 3, 1, 0, 0, 0, 0, 0}, // #0

    // Layer 1 (3)
    // Residual block 1 
	{64, 64, 1, 1, 0, 0, 0, 0, 0}, // 1 #1
	{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 2 #2
	{64, 256, 1, 1, 0, 0, 0, 0, 0}, // 3 #3
	{64, 256, 1, 1, 0, 0, 0, 0, 0}, // 4 #4 

    // Residual block 2 
	{256, 64, 1, 1, 0, 0, 0, 0, 0}, // 1 #5
    {64, 64, 3, 1, 0, 0, 0, 0, 0}, // 2 #6
	{64, 256, 1, 1, 0, 0, 0, 0, 0}, // 3 #7

    // Residual block 3
	{256, 64, 1, 1, 0, 0, 0, 0, 0}, // 1 #8
	{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 2 #9
	{64, 256, 1, 1, 0, 0, 0, 0, 0}, // 3 #10

    // Layer 2 (4)
    // Residual block 1
	{256, 128, 1, 1, 0, 0, 0, 0, 0}, // 1 #11
	{128, 128, 3, 2, 0, 0, 0, 0, 0}, // 2 #12
	{128, 512, 1, 1, 0, 0, 0, 0, 0}, // 3 #13
	{256, 512, 1, 2, 0, 0, 0, 0, 0}, // 4 #14 

    // Residual block 2
	{512, 128, 1, 1, 0, 0, 0, 0, 0}, // 1 #15
	{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 2 #16
	{128, 512, 1, 1, 0, 0, 0, 0, 0}, // 3 #17

    // Residual block 3
	{512, 128, 1, 1, 0, 0, 0, 0, 0}, // 1 #18
	{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 2 #19
	{128, 512, 1, 1, 0, 0, 0, 0, 0}, // 3 #20

    // Residual block 4
	{512, 128, 1, 1, 0, 0, 0, 0, 0}, // 1 #21
	{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 2 #22
	{128, 512, 1, 1, 0, 0, 0, 0, 0}, // 3 #23

    // Layer 3 (6)
    // Residual block 1
	{512, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #24
	{256, 256, 3, 2, 0, 0, 0, 0, 0}, // 2 #25
	{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #26
	{512, 1024, 1, 2, 0, 0, 0, 0, 0}, // 4 #27 /// downsample

    // Residual block 2
	{1024, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #28
	{257, 256, 3, 1, 0, 0, 0, 0, 0}, // 2 #29
	{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #30

    // Residual block 3
	{1024, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #31
	{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 2 #32
	{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #33

    // Residual block 4
	{1024, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #34
	{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 2 #35
	{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #36

    // Residual block 5
	{1024, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #37
	{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 2 #38
	{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #39

    // Residual block 6
	{1024, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #40
	{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 2 #41
	{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #42

    // Layer 4 (3)
    // Residual block 1
	{1024, 512, 1, 1, 0, 0, 0, 0, 0}, // 1 #43
	{512, 512, 3, 2, 0, 0, 0, 0, 0}, // 2 #44
	{512, 2048, 1, 1, 0, 0, 0, 0, 0}, // 3 #45
	{1024, 2048, 1, 2, 0, 0, 0, 0, 0}, // 4 #46 

    // Residual block 2
	{2048, 512, 1, 1, 0, 0, 0, 0, 0}, // 1 #47
	{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 2 #48
	{512, 2048, 1, 1, 0, 0, 0, 0, 0}, // 3 #49

    // Residual block 3
	{2048, 512, 1, 1, 0, 0, 0, 0, 0}, // 1 #50
	{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 2 #51
	{512, 2048, 1, 1, 0, 1, 7, 7, 0}, // 3 #52
};



std::vector<LayerConfig> resnet50_in224_ch = {
		{3, 64, 7, 2, 0, 1, 3, 2, 1}, 
        // Layer 1 (3)
        // Residual block 1 
		{64, 64, 1, 1, 0, 0, 0, 0, 0}, // 1 #1
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 2 #2
		{64, 256, 1, 1, 0, 0, 0, 0, 0}, // 3 #3
		{64, 256, 1, 1, 0, 0, 0, 0, 0}, // 4 #4 /// downsample

        // Residual block 2 
		{256, 64, 1, 1, 0, 0, 0, 0, 0}, // 1 #5
        {64, 64, 3, 1, 0, 0, 0, 0, 0}, // 2 #6
		{64, 256, 1, 1, 0, 0, 0, 0, 0}, // 3 #7

        // Residual block 3
		{256, 64, 1, 1, 0, 0, 0, 0, 0}, // 1 #8
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 2 #9
		{64, 256, 1, 1, 0, 0, 0, 0, 0}, // 3 #10

        // Layer 2 (4)
        // Residual block 1
		{256, 128, 1, 1, 0, 0, 0, 0, 0}, // 1 #11
		{128, 128, 3, 2, 0, 0, 0, 0, 0}, // 2 #12
		{128, 512, 1, 1, 0, 0, 0, 0, 0}, // 3 #13
		{256, 512, 1, 2, 0, 0, 0, 0, 0}, // 4 #14 /// downsample 

        // Residual block 2
		{512, 128, 1, 1, 0, 0, 0, 0, 0}, // 1 #15
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 2 #16
		{128, 512, 1, 1, 0, 0, 0, 0, 0}, // 3 #17

        // Residual block 3
		{512, 128, 1, 1, 0, 0, 0, 0, 0}, // 1 #18
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 2 #19
		{128, 512, 1, 1, 0, 0, 0, 0, 0}, // 3 #20

        // Residual block 4
		{512, 128, 1, 1, 0, 0, 0, 0, 0}, // 1 #21
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 2 #22
		{128, 512, 1, 1, 0, 0, 0, 0, 0}, // 3 #23

        // Layer 3 (6)
        // Residual block 1
		{512, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #24
		{256, 256, 3, 2, 0, 0, 0, 0, 0}, // 2 #25
		{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #26
		{512, 1024, 1, 2, 0, 0, 0, 0, 0}, // 4 #27 /// downsample

        // Residual block 2
		{1024, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #28
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 2 #29
		{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #30

        // Residual block 3
		{1024, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #31
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 2 #32
		{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #33

        // Residual block 4
		{1024, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #34
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 2 #35
		{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #36

        // Residual block 5
		{1024, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #37
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 2 #38
		{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #39

        // Residual block 6
		{1024, 256, 1, 1, 0, 0, 0, 0, 0}, // 1 #40
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 2 #41
		{256, 1024, 1, 1, 0, 0, 0, 0, 0}, // 3 #42

        // Layer 4 (3)
        // Residual block 1
		{1024, 512, 1, 1, 0, 0, 0, 0, 0}, // 1 #43
		{512, 512, 3, 2, 0, 0, 0, 0, 0}, // 2 #44
		{512, 2048, 1, 1, 0, 0, 0, 0, 0}, // 3 #45
		{1024, 2048, 1, 2, 0, 0, 0, 0, 0}, // 4 #46 /// downsample

        // Residual block 2
		{2048, 512, 1, 1, 0, 0, 0, 0, 0}, // 1 #47
		{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 2 #48
		{512, 2048, 1, 1, 0, 0, 0, 0, 0}, // 3 #49

        // Residual block 3
		{2048, 512, 1, 1, 0, 0, 0, 0, 0}, // 1 #50
		{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 2 #51
		{512, 2048, 1, 1, 0, 1, 7, 7, 0}, // 3 #52
	};



std::vector<LayerConfig> resnet18_ch = {
		{3, 64, 7, 2, 0, 1, 3, 2, 1}, // Suppose we have 64 output channels, each with 3 input channels, and each filter is 3x3 for layer0
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 1 // Residual block 0
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 2 
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 3
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 4 ///
		{64, 128, 3, 2, 0, 0, 0, 0, 0}, // 5 // Residual block 1
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 6
		{64, 128, 1, 2, 0, 0, 0, 0, 0}, // residual path 7
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 8
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 9
		{128, 256, 3, 2, 0, 0, 0, 0, 0}, // 10 // Residual block 2
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 11
		{128, 256, 1, 2, 0, 0, 0, 0, 0}, // 12
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 13
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 14 //
		{256, 512, 3, 2, 0, 0, 0, 0, 0}, // 15 // Residual block 3
		{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 16
		{256, 512, 1, 2, 0, 0, 0, 0, 0}, // residual path 17
		{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 18
		{512, 512, 3, 1, 0, 1, 7, 7, 0} // 19 // input size 224x224 //pooling padding 1?
	};



std::vector<LayerConfig> resnet18_in64_ch = {
		{3, 64, 3, 1, 0, 0, 0, 0, 0}, // Suppose we have 64 output channels, each with 3 input channels, and each filter is 3x3 for layer0
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 1 // Residual block 0
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 2 
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 3
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 4 ///
		{64, 128, 3, 2, 0, 0, 0, 0, 0}, // 5 // Residual block 1
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 6
		{64, 128, 1, 2, 0, 0, 0, 0, 0}, // residual path 7
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 8
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 9
		{128, 256, 3, 2, 0, 0, 0, 0, 0}, // 10 // Residual block 2
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 11
		{128, 256, 1, 2, 0, 0, 0, 0, 0}, // 12 // residual path 12
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 13
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 14 //
		{256, 512, 3, 2, 0, 0, 0, 0, 0}, // 15 // Residual block 3
		{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 16
		{256, 512, 1, 2, 0, 0, 0, 0, 0}, // residual path 17
		{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 18
		{512, 512, 3, 1, 0, 1, 8, 8, 0} // 19 // input size 64x64
	};


std::vector<LayerConfig> resnet18_in62_ch = {
		{3, 64, 7, 2, 0, 1, 3, 2, 1}, // Suppose we have 64 output channels, each with 3 input channels, and each filter is 3x3 for layer0
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 1 // Residual block 0
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 2 
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 3
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 4 ///
		{64, 128, 3, 2, 0, 0, 0, 0, 0}, // 5 // Residual block 1
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 6
		{64, 128, 1, 2, 0, 0, 0, 0, 0}, // residual path 7
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 8
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 9
		{128, 256, 3, 2, 0, 0, 0, 0, 0}, // 10 // Residual block 2
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 11
		{128, 256, 1, 2, 0, 0, 0, 0, 0}, // 12 // residual path 12
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 13
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 14 //
		{256, 512, 3, 2, 0, 0, 0, 0, 0}, // 15 // Residual block 3
		{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 16
		{256, 512, 1, 2, 0, 0, 0, 0, 0}, // residual path 17
		{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 18
		{512, 512, 3, 1, 0, 1, 2, 2, 0} // 19 // input size 64x64
	};




std::vector<LayerConfig> resnet18_in32_ch = {
		{3, 64, 7, 2, 0, 1, 3, 2, 1}, // Suppose we have 64 output channels, each with 3 input channels, and each filter is 3x3 for layer0
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 1 // Residual block 0
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 2 
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 3
		{64, 64, 3, 1, 0, 0, 0, 0, 0}, // 4 ///
		{64, 128, 3, 2, 0, 0, 0, 0, 0}, // 5 // Residual block 1
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 6
		{64, 128, 1, 2, 0, 0, 0, 0, 0}, // residual path 7
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 8
		{128, 128, 3, 1, 0, 0, 0, 0, 0}, // 9
		{128, 256, 3, 2, 0, 0, 0, 0, 0}, // 10 // Residual block 2
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 11
		{128, 256, 1, 2, 0, 0, 0, 0, 0}, // 12 // residual path 12
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 13
		{256, 256, 3, 1, 0, 0, 0, 0, 0}, // 14 //
		{256, 512, 3, 2, 0, 0, 0, 0, 0}, // 15 // Residual block 3
		{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 16
		{256, 512, 1, 2, 0, 0, 0, 0, 0}, // residual path 17
		{512, 512, 3, 1, 0, 0, 0, 0, 0}, // 18
		{512, 512, 3, 1, 0, 0, 0, 0, 0} // 19 // input size 64x64
	};



std::vector<LayerConfig> mobilenetv2_in224_ch = {
        {3, 32, 3, 2, 0, 0, 0, 0, 0},
        {32, 32, 3, 1, 1, 0, 0, 0, 0},
        {32, 16, 1, 1, 0, 0, 0, 0, 0},
        {16, 96, 1, 1, 0, 0, 0, 0, 0},
        {96, 96, 3, 2, 1, 0, 0, 0, 0},
        {96, 24, 1, 1, 0, 0, 0, 0, 0},
        {24, 144, 1, 1, 0, 0, 0, 0, 0},
        {144, 144, 3, 1, 1, 0, 0, 0, 0},
        {144, 24, 1, 1, 0, 0, 0, 0, 0},
        {24, 144, 1, 1, 0, 0, 0, 0, 0},
        {144, 144, 3, 2, 1, 0, 0, 0, 0},
        {144, 32, 1, 1, 0, 0, 0, 0, 0},
        {32, 192, 1, 1, 0, 0, 0, 0, 0},
        {192, 192, 3, 1, 1, 0, 0, 0, 0},
        {192, 32, 1, 1, 0, 0, 0, 0, 0},
        {32, 192, 1, 1, 0, 0, 0, 0, 0},
        {192, 192, 3, 1, 1, 0, 0, 0, 0},
        {192, 32, 1, 1, 0, 0, 0, 0, 0},
        {32, 192, 1, 1, 0, 0, 0, 0, 0},
        {192, 192, 3, 2, 1, 0, 0, 0, 0},
        {192, 64, 1, 1, 0, 0, 0, 0, 0},
        {64, 384, 1, 1, 0, 0, 0, 0, 0},
        {384, 384, 3, 1, 1, 0, 0, 0, 0},
        {384, 64, 1, 1, 0, 0, 0, 0, 0},
        {64, 384, 1, 1, 0, 0, 0, 0, 0},
        {384, 384, 3, 1, 1, 0, 0, 0, 0},
        {384, 64, 1, 1, 0, 0, 0, 0, 0},
        {64, 384, 1, 1, 0, 0, 0, 0, 0},
        {384, 384, 3, 1, 1, 0, 0, 0, 0},
        {384, 64, 1, 1, 0, 0, 0, 0, 0},
        {64, 384, 1, 1, 0, 0, 0, 0, 0},
        {384, 384, 3, 1, 1, 0, 0, 0, 0},
        {384, 96, 1, 1, 0, 0, 0, 0, 0},
        {96, 576, 1, 1, 0, 0, 0, 0, 0},
        {576, 576, 3, 1, 1, 0, 0, 0, 0},
        {576, 96, 1, 1, 0, 0, 0, 0, 0},
        {96, 576, 1, 1, 0, 0, 0, 0, 0},
        {576, 576, 3, 1, 1, 0, 0, 0, 0},
        {576, 96, 1, 1, 0, 0, 0, 0, 0},
        {96, 576, 1, 1, 0, 0, 0, 0, 0},
        {576, 576, 3, 2, 1, 0, 0, 0, 0},
        {576, 160, 1, 1, 0, 0, 0, 0, 0},
        {160, 960, 1, 1, 0, 0, 0, 0, 0},
        {960, 960, 3, 1, 1, 0, 0, 0, 0},
        {960, 160, 1, 1, 0, 0, 0, 0, 0},
        {160, 960, 1, 1, 0, 0, 0, 0, 0},
        {960, 960, 3, 1, 1, 0, 0, 0, 0},
        {960, 160, 1, 1, 0, 0, 0, 0, 0},
        {160, 960, 1, 1, 0, 0, 0, 0, 0},
        {960, 960, 3, 1, 1, 0, 0, 0, 0},
        {960, 320, 1, 1, 0, 0, 0, 0, 0},
        {320, 1280, 1, 1, 0, 1, 7, 1, 0}
    };



// Function to allocate filters for a given layer configuration
std::vector<std::vector<std::vector<uint64_t>>> allocate_filters(const LayerConfig& layer_config) {
	if (layer_config.depthwise == 0)
	{
		return std::vector<std::vector<std::vector<uint64_t>>>(
			layer_config.output_channels,
			std::vector<std::vector<uint64_t>>(
				layer_config.input_channels,
				std::vector<uint64_t>(
					layer_config.filter_size * layer_config.filter_size, 0
				)
			)
						    
		);
	}
	else
	{
		return std::vector<std::vector<std::vector<uint64_t>>>(
			1,
			std::vector<std::vector<uint64_t>>(
				layer_config.input_channels,
				std::vector<uint64_t>(
					layer_config.filter_size * layer_config.filter_size, 0
				)
			)
						    
		);
	}
}

// Function to allocate filters for all layers based on their configurations
std::vector<std::vector<std::vector<std::vector<uint64_t>>>> allocate_all_filters(const std::vector<LayerConfig>& all_layer_configs) {
	std::vector<std::vector<std::vector<std::vector<uint64_t>>>> all_filters;
	for (const auto& layer_config : all_layer_configs) {
		all_filters.push_back(allocate_filters(layer_config));
	}
	return all_filters;
}

// Write 2D filters to file
void write_2d_filter(std::ofstream& file, const std::vector<uint64_t>& filter, int height, int width) {
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				file << filter[i * width + j] << ' ';
			}
			file << '\n';
		}
}

void write_filters(const std::string& filename, const std::vector<std::vector<std::vector<uint64_t>>>& filters, int height, int width) {
	std::ofstream file(filename);
	if (!file) {
        std::cerr << "Could not open file: " << filename << std::endl;
        std::cerr << "Please make a directory" << std::endl;
		//return;

	}

    int output_channel_idx = 0;
	for (const auto& output_channel : filters) {
		file << "# output_channel_" << output_channel_idx++ << '\n';
		for (const auto& input_channel : output_channel) {
			write_2d_filter(file, input_channel, height, width);
			file << '\n';
		}
	}
}

// Jaguar_52: strict filter reader (from jwyoo's verified capture copy).
// The original read_2d_filter dropped each kernel's first row and set every
// 1x1 weight to 1; values are now read as signed integers in file order.
std::vector<std::vector<std::vector<uint64_t>>> read_filters(const std::string& filename, int height, int width) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open filter: "+filename);
    std::vector<std::vector<std::vector<uint64_t>>> filters;
    std::vector<uint64_t> kernel;
    std::string line;
    while (std::getline(file,line)) {
        if (line.empty()) continue;
        if (line[0]=='#') {
            if (!kernel.empty()) throw std::runtime_error("incomplete filter kernel");
            filters.push_back({}); continue;
        }
        if (filters.empty()) throw std::runtime_error("missing output-channel header");
        std::istringstream row(line);
        int64_t value;
        while (row>>value) {
            kernel.push_back(static_cast<uint64_t>(value));
            if (kernel.size()==size_t(height*width)) {
                filters.back().push_back(kernel); kernel.clear();
            }
        }
        if (!row.eof()) throw std::runtime_error("invalid filter integer");
    }
    if (!kernel.empty()) throw std::runtime_error("incomplete final filter kernel");
    return filters;
}

std::vector<std::vector<uint64_t>> resnet18_fc_weight;
std::vector<std::vector<uint64_t>> vgg16_fc_weight;

void init_fc_weight(std::vector<std::vector<uint64_t>> &W) {
    std::mt19937 gen(0);
    std::uniform_int_distribution<uint64_t> dist(0, 10);

    for (auto &row : W)
        for (auto &x : row)
            x = dist(gen);
}

void init_all_fc_weights() {
    resnet18_fc_weight.resize(1000, std::vector<uint64_t>(512));
    vgg16_fc_weight.resize(1000, std::vector<uint64_t>(512));

    init_fc_weight(resnet18_fc_weight);
    init_fc_weight(vgg16_fc_weight);
}