// tts_engine.hpp

#ifndef KOKORO_TTS_ENGINE_HPP
#define KOKORO_TTS_ENGINE_HPP

#include <string>
#include <vector>
#include <unordered_map> // CHANGED: Replaced <map> with <unordered_map>
#include <memory>
#include <cmath>
#include <iostream>

// ONNX Runtime includes
#include <onnxruntime_cxx_api.h>

// eSpeak-NG G2P
extern "C" {
    #include <espeak-ng/speak_lib.h>
}

// Miniaudio
#define MA_NO_DECODER
#define MA_NO_ENCODING
#define MA_NO_RESOURCE_MANAGER
#include "miniaudio.h"

// --- Type Definitions ---
// Define types based on ONNX model expectations (int64_t for tokens, float for styles)
using input_ids_type = int64_t;
using style_vector_type = float; 

// --- Forward Declarations ---
// CHANGED: Use std::unordered_map to match token_map.hpp and tts_engine.cpp
std::unordered_map<std::string, input_ids_type> load_token_map();

class KokoroTTS {
public:
    KokoroTTS();
    ~KokoroTTS();

    void initializeEngine(ma_uint32 sampleRate, size_t maxContextLength);
    bool loadModel(const std::string& modelPath);
    bool loadVoiceData(const std::string& voiceDirectory, const std::string& voiceName);
    
    // Main function to synthesize and play audio
    void synthesizeAndStream(const std::string& text, float speed);
    
    // Buffer to hold synthesized audio data (FP32)
    std::vector<float> synthesisBuffer;

private:
    // ONNX Runtime members
    Ort::Env env;
    std::unique_ptr<Ort::Session> session_;
    int outputDataType_; 
    std::string tokenInputName_;
    std::string outputName_;
    
    // Configuration members
    ma_uint32 sampleRate_;
    size_t maxContextLength_;
    size_t styleVectorDim_;
    
    // Voice Bank members
    std::vector<style_vector_type> currentStyleVector_;
    size_t voiceBankSize_;

    // G2P/Tokenization members
    // CHANGED: Use std::unordered_map to match implementation
    std::unordered_map<std::string, input_ids_type> tokenMap_;

    // Miniaudio playback members
    ma_device audioDevice_;
    size_t bufferReadIndex;
    int callbackDebugCounter; 

    // Helper functions
    std::vector<input_ids_type> phonemizeAndTokenize(const std::string& text, const std::string& languageCode);
    float convertHalfToFloat(uint16_t h);

    // Miniaudio callback (static)
    static void audioDataCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);
};

#endif // KOKORO_TTS_ENGINE_HPP
