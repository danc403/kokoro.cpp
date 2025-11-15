// voice_loader_cli.cpp

#include "voice_loader.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>

/**
 * @brief Prints the 2D vector data in a readable 1-vector-per-line format.
 */
void print_vector_store(const VoiceVectorStore& store) {
    if (store.data.empty()) {
        std::cout << "Data store is empty." << std::endl;
        return;
    }

    std::cout << "\n--- Loaded Vector Data ---" << std::endl;
    std::cout << "Dimensions: " << store.N_rows << " rows x " << store.D_cols << " columns." << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    for (int i = 0; i < store.N_rows; ++i) {
        std::cout << "Vector [" << i << "]: { ";
        
        for (int j = 0; j < store.D_cols; ++j) {
            float value = store.data[i * store.D_cols + j];
            std::cout << value;
            
            // Print only the first 5 elements for brevity
            if (j >= 4 && store.D_cols > 5) {
                std::cout << ", ... (" << store.D_cols - 5 << " more) ";
                break; 
            }
            if (j < store.D_cols - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " }" << std::endl;
        
        // Print only the first 10 vectors for brevity
        if (i >= 9 && store.N_rows > 10) {
            std::cout << "  [... " << store.N_rows - 10 << " vectors suppressed ...]" << std::endl;
            break; 
        }
    }
    std::cout << "--------------------------" << std::endl;
}

// The main function for the standalone testing executable
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dat_filepath>" << std::endl;
        std::cerr << "Example: " << argv[0] << " af_sky.dat" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];

    try {
        // Load the vectors using the library function
        VoiceVectorStore vectors = load_raw_binary_vectors(filepath);

        // Print the data using the local testing function
        print_vector_store(vectors);
        
    } catch (const std::runtime_error& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
