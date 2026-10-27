#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace jaguar_networks {

bool load_input_from_inp(const std::string &path, int channels, int width, std::vector<std::vector<uint64_t>> &image);
bool load_default_model_input(const std::string &model_name, int channels, int width, std::vector<std::vector<uint64_t>> &image);
void fill_debug_input(std::vector<std::vector<uint64_t>> &image);

} // namespace jaguar_networks
