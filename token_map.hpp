// token_map.hpp

#ifndef KOKORO_TOKEN_MAP_HPP
#define KOKORO_TOKEN_MAP_HPP

#include <string>
#include <unordered_map>
#include <cstdint> // Required for int64_t

// FIX: Change token type to int64_t to match the model's tensor(int64) expectation.
using input_ids_type = int64_t;

// Function to load the phoneme-to-token ID map
std::unordered_map<std::string, input_ids_type> load_token_map();

#endif // KOKORO_TOKEN_MAP_HPP
