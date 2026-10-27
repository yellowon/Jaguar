#include "input_loader.h"

#include <fstream>
#include <iostream>
#include <unordered_map>

namespace jaguar_networks {
namespace {
const char *kShiftPretrainedDir = "../pretrained";

std::string default_input_path(const std::string &model_name) {
    static const std::unordered_map<std::string, std::string> kPaths = {
        {"resnet18", std::string(kShiftPretrainedDir) + "/resnet18_input_scale9_pred.inp"},
        {"resnet50", std::string(kShiftPretrainedDir) + "/resnet50_input_scale9_pred249.inp"},
        {"mobilenet_v2", std::string(kShiftPretrainedDir) + "/mobilenet_v2_input_scale9_248.inp"},
    };
    auto it = kPaths.find(model_name);
    return it == kPaths.end() ? std::string() : it->second;
}
} // namespace

bool load_input_from_inp(const std::string &path, int channels, int width, std::vector<std::vector<uint64_t>> &image) {
    if (path.empty()) return false;
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[Jaguar] failed to open input file: " << path << std::endl;
        return false;
    }
    const size_t elems_per_ch = static_cast<size_t>(width) * static_cast<size_t>(width);
    image.assign(channels, std::vector<uint64_t>(elems_per_ch, 0));
    for (int c = 0; c < channels; ++c) {
        for (size_t idx = 0; idx < elems_per_ch; ++idx) {
            long long value = 0;
            if (!(fin >> value)) {
                std::cerr << "[Jaguar] input file ended early: " << path << std::endl;
                return false;
            }
            image[c][idx] = static_cast<uint64_t>(value);
        }
    }
    std::cerr << "[Jaguar] loaded input .inp: " << path
              << " channels=" << channels << " width=" << width << std::endl;
    return true;
}

bool load_default_model_input(const std::string &model_name, int channels, int width, std::vector<std::vector<uint64_t>> &image) {
    const auto path = default_input_path(model_name);
    if (path.empty()) return false;
    return load_input_from_inp(path, channels, width, image);
}

void fill_debug_input(std::vector<std::vector<uint64_t>> &image) {
    for (size_t c = 0; c < image.size(); ++c) {
        for (size_t idx = 0; idx < image[c].size(); ++idx) {
            image[c][idx] = static_cast<uint64_t>(idx % 100);
        }
    }
}

} // namespace jaguar_networks
