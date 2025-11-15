// voice_loader.hpp

#ifndef VOICE_LOADER_HPP
#define VOICE_LOADER_HPP

#include <vector>
#include <string>

// Type alias for the loaded data container: a single flat vector of floats.
using VoiceData = std::vector<float>;

/**
 * @brief Structure to hold the raw vector data and its dimensions.
 */
struct VoiceVectorStore {
    VoiceData data;
    int N_rows = 0; // Number of vectors
    int D_cols = 0; // Dimension of each vector
};

/**
 * @brief Loads the raw binary vector data from a .dat file.
 * The file is expected to have the structure:
 * [4 bytes: int N (rows)] [4 bytes: int D (columns)] [raw float data (N * D * 4 bytes)]
 * * @param filepath Path to the .dat file.
 * @return A VoiceVectorStore structure containing the data and its dimensions.
 * @throws std::runtime_error on file I/O or header errors.
 */
VoiceVectorStore load_raw_binary_vectors(const std::string& filepath);

#endif // VOICE_LOADER_HPP
