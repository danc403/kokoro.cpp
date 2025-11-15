// voice_loader.cpp

#include "voice_loader.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

// The file format is assumed to be:
// 4-byte int (N_rows) - The total number of style vectors (e.g., 510)
// 4-byte int (D_cols) - The dimension of each vector (e.g., 256)
// N_rows * D_cols * 4-byte float (data)

VoiceVectorStore load_raw_binary_vectors(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open voice vector file: " + filepath);
    }

    VoiceVectorStore store;
    int32_t N_rows_file = 0;
    int32_t D_cols_file = 0;

    // 1. Read N_rows (e.g., 510 - The number of available styles/lengths)
    if (!file.read(reinterpret_cast<char*>(&N_rows_file), sizeof(int32_t))) {
        throw std::runtime_error("Failed to read N_rows from file: " + filepath);
    }
    store.N_rows = (size_t)N_rows_file;

    // 2. Read D_cols (e.g., 256 - The style vector dimension)
    if (!file.read(reinterpret_cast<char*>(&D_cols_file), sizeof(int32_t))) {
        throw std::runtime_error("Failed to read D_cols from file: " + filepath);
    }
    store.D_cols = (size_t)D_cols_file;
    
    // --- VALIDATION AND MEMORY ALLOCATION ---
    if (store.N_rows == 0 || store.D_cols == 0) {
        throw std::runtime_error("Voice file contains zero elements (N or D is zero).");
    }
    
    size_t total_elements = store.N_rows * store.D_cols;
    
    // Allocate memory for ALL vectors in the file (510 * 256)
    store.data.resize(total_elements);

    // 3. Read Data
    size_t data_size_bytes = total_elements * sizeof(float);
    if (!file.read(reinterpret_cast<char*>(store.data.data()), data_size_bytes)) {
        throw std::runtime_error("Failed to read vector data from file: " + filepath);
    }
    file.close();

    std::cerr << "DEBUG: Voice file loaded. N_rows (Style Bank Size)=" << store.N_rows << ", D_cols (Style Dim)=" << store.D_cols << "." << std::endl;

    return store;
}
