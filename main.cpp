/*
 * main.cpp: Application entry point. Loads configuration, initializes the TTS engine,
 * and executes synthesis based on configuration parameters.
 */
#include "tts_engine.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>

#include "json.hpp"

// Define the two search locations globally
const std::string SYSTEM_CONFIG_PATH = "/etc/kokoro/config.json";
const std::string LOCAL_CONFIG_PATH = "config.json";

// Define a structure for all configuration parameters, matching finalized config.json keys
struct TtsConfig {
    std::string onnxModelPath;
    std::string styleVectorDirectory; // Path to the directory of individual voice files (e.g., "voices")
    std::string defaultVoiceName;     // Name of the specific voice (e.g., "af_bella")
    std::string textToSynthesize;     // Sample text for initial testing
    
    // Model and playback configuration
    ma_uint32 audioSampleRate;
    size_t maxContextLength;
    float defaultPlaybackRate;
    float defaultPlaybackPitch;
};

/**
 * @brief Attempts to read configuration parameters from a JSON file.
 * @param filename The path to the configuration file.
 * @param config A reference to the TtsConfig structure to populate.
 * @return true if loading and parsing was successful, false otherwise.
 */
bool loadAndParseConfig(const std::string& filename, TtsConfig& config) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    try {
        nlohmann::json j;
        file >> j;

        config.onnxModelPath = j.at("onnxModelPath").get<std::string>();
        config.styleVectorDirectory = j.at("styleVectorDirectory").get<std::string>();
        config.defaultVoiceName = j.at("defaultVoiceName").get<std::string>();
        config.textToSynthesize = j.at("textToSynthesize").get<std::string>();
        config.audioSampleRate = j.at("audioSampleRate").get<ma_uint32>();
        config.maxContextLength = j.at("maxContextLength").get<size_t>();

        // Playback parameters (with robust defaults if missing)
        config.defaultPlaybackRate = j.value("defaultPlaybackRate", 1.0f);
        config.defaultPlaybackPitch = j.value("defaultPlaybackPitch", 0.0f);

    } catch (const nlohmann::json::exception& e) {
        std::cerr << "Error: JSON parsing failed in " << filename << ". Details: " << e.what() << std::endl;
        std::cerr << "Ensure all keys (onnxModelPath, styleVectorDirectory, defaultVoiceName, "
                  << "textToSynthesize, audioSampleRate, maxContextLength) are present." << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error: An unexpected error occurred while loading config from " << filename << ": " << e.what() << std::endl;
        return false;
    }

    std::cout << "Configuration loaded successfully from " << filename << "." << std::endl;
    return true;
}

int main() {
    TtsConfig config;
    bool config_loaded = false;
    std::string config_path;

    // 1. Attempt to load config from System Path or Local Path
    if (loadAndParseConfig(SYSTEM_CONFIG_PATH, config)) {
        config_loaded = true;
        config_path = SYSTEM_CONFIG_PATH;
    }

    if (!config_loaded && loadAndParseConfig(LOCAL_CONFIG_PATH, config)) {
        config_loaded = true;
        config_path = LOCAL_CONFIG_PATH;
    }

    if (!config_loaded) {
        std::cerr << "Fatal: Failed to load configuration from both " << SYSTEM_CONFIG_PATH
                  << " and " << LOCAL_CONFIG_PATH << ". Exiting." << std::endl;
        return 1;
    }

    KokoroTTS tts;
    
    // 1a. Initialize Engine with parameters from config.json
    std::cout << "Initializing TTS engine parameters..." << std::endl;
    tts.initializeEngine(config.audioSampleRate, config.maxContextLength);

    // 2. Load Model using parameter from config
    std::cout << "Loading model: " << config.onnxModelPath << "..." << std::endl;
    if (!tts.loadModel(config.onnxModelPath)) {
        std::cerr << "Fatal: Could not load model '" << config.onnxModelPath << "'. Exiting." << std::endl;
        return 1;
    }

    // 3. Load Voice using the new function signature: (directory path, voice name)
    std::cout << "Loading voice '" << config.defaultVoiceName << "' from directory: " << config.styleVectorDirectory << "..." << std::endl;
    if (!tts.loadVoiceData(config.styleVectorDirectory, config.defaultVoiceName)) {
        std::cerr << "Fatal: Could not load voice vector. Check voice file path (" 
                  << config.styleVectorDirectory << "/" << config.defaultVoiceName << ".dat) and dimension match." << std::endl;
        return 1;
    }

    std::cout << "\n--- Synthesis Parameters ---" << std::endl;
    std::cout << "Config loaded from: " << config_path << std::endl;
    std::cout << "Model Path: " << config.onnxModelPath << std::endl;
    std::cout << "Voice Directory: " << config.styleVectorDirectory << std::endl;
    std::cout << "Voice Name: " << config.defaultVoiceName << std::endl;
    std::cout << "Audio Sample Rate: " << config.audioSampleRate << "Hz" << std::endl;
    std::cout << "Max Context Length: " << config.maxContextLength << " tokens" << std::endl;
    std::cout << "Text: \"" << config.textToSynthesize << "\"" << std::endl;
    std::cout << "Speed (Rate): " << config.defaultPlaybackRate << std::endl;
    std::cout << "Pitch: " << config.defaultPlaybackPitch << " (Note: Pitch adjustment logic not yet implemented in tts_engine)" << std::endl;
    std::cout << "----------------------------" << std::endl;


    // 4. Synthesize and Stream
    std::cout << "Starting synthesis..." << std::endl;
    tts.synthesizeAndStream(config.textToSynthesize, config.defaultPlaybackRate);

    std::cout << "Synthesis complete." << std::endl;

    return 0;
}
