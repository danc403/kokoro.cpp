// tts_engine.cpp

/*
 * Implementation of the KokoroTTS engine.
 * Includes miniaudio implementation, eSpeak-NG G2P, and ONNX Runtime inference logic.
 */
#include "tts_engine.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <filesystem>
#include "voice_loader.hpp" 
#include "token_map.hpp"

#include <espeak-ng/speak_lib.h>

namespace fs = std::filesystem; 

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

const input_ids_type PAD_TOKEN_ID = 0; 

// --- Helper for FP16 Conversion ---

float KokoroTTS::convertHalfToFloat(uint16_t h) {
    uint32_t f = 0;

    // Sign Bit (1-bit): Shift from bit 15 to bit 31
    f |= (h & 0x8000) << 16;

    // Extract Exponent (5-bit) and Mantissa (10-bit)
    int exponent = (h >> 10) & 0x1F; 
    uint32_t mantissa = h & 0x03FF; 

    if (exponent == 0x1F) {
        // Inf or NaN (Exponent is all 1s)
        exponent = 0xFF; 
        mantissa <<= 13;

    } else if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            return *(float*)&f; 
        } else {
            // Denormal/Subnormal
            exponent = 1;
            while ((mantissa & 0x400) == 0) { 
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FF;

            exponent += (127 - 15);
            mantissa <<= 13; 
        }
    } else {
        // Normal Number
        exponent += (127 - 15);
        mantissa <<= 13;
    }

    // Combine exponent and mantissa
    f |= (exponent << 23);
    f |= mantissa;

    return *(float*)&f;
}

// --- Constructor/Destructor ---

KokoroTTS::KokoroTTS()
    : env(ORT_LOGGING_LEVEL_WARNING, "KokoroTTS"), 
      // outputDataType_ is an int, initialized to 0 (UNKNOWN)
      outputDataType_(0), 
      sampleRate_(0), maxContextLength_(0), voiceBankSize_(0),
      // Initialize debug counter
      callbackDebugCounter(0),
      tokenMap_(load_token_map()) 
{
    std::memset(&audioDevice_, 0, sizeof(audioDevice_));

    int result = espeak_Initialize(AUDIO_OUTPUT_PLAYBACK, 0, nullptr, 0);
    if (result < 0) {
        std::cerr << "Error: Failed to initialize eSpeak-NG. Check 'libespeak-ng-dev' installation." << std::endl;
    }

    const char* voiceName = "en";
    int voice_result = espeak_SetVoiceByName(voiceName);
    if (voice_result != 0) {
        std::cerr << "Warning: Failed to set eSpeak voice to '" << voiceName << "'. Result: " << voice_result << std::endl;
    } else {
        std::cout << "DEBUG: eSpeak voice set to 'en'." << std::endl;
    }

    std::cout << "TTS Engine initialized. " << tokenMap_.size() << " tokens mapped." << std::endl;
}

KokoroTTS::~KokoroTTS() {
    espeak_Terminate();

    if (audioDevice_.type == ma_device_type_playback) {
        ma_device_uninit(&audioDevice_);
    }
}

// maxContextLength_ should store the actual fixed input tensor size (e.g., 512)
void KokoroTTS::initializeEngine(ma_uint32 sampleRate, size_t maxContextLength) {
    sampleRate_ = sampleRate;
    maxContextLength_ = maxContextLength; 
    std::cout << "Engine parameters set: Sample Rate=" << sampleRate_ 
              << "Hz, Max Context Length (Padded)=" << maxContextLength_ << " tokens." << std::endl;
}

// --- G2P (Phonemization) Logic ---

std::vector<input_ids_type> KokoroTTS::phonemizeAndTokenize(const std::string& text, const std::string& languageCode) {
    std::vector<input_ids_type> input_ids;

    std::cout << "DEBUG: Entering phonemizeAndTokenize. Text input size: " << text.length() << " characters." << std::endl;
    const void* text_ptr = text.c_str();
    if (text_ptr == nullptr) {
        std::cerr << "Error: Text pointer is null." << std::endl;
        return input_ids;
    }
    
    const char* ipa_output = espeak_TextToPhonemes(&text_ptr, 0, espeakPHONEMES_IPA);
    
    if (!ipa_output) {
        std::cerr << "Error: eSpeak-NG failed to generate phonemes or returned a null pointer." << std::endl;
        return input_ids;
    }
    
    std::string ipa_string(ipa_output);
    std::cout << "eSpeak IPA Output: " << ipa_string << std::endl;
    
    // Correct the colon (:) output from eSpeak to the expected Unicode Length Mark (\u02d0).
    const std::string length_mark_unicode = "\u02d0"; 
    
    size_t pos = ipa_string.find(':');
    while (pos != std::string::npos) {
        ipa_string.replace(pos, 1, length_mark_unicode);
        pos = ipa_string.find(':', pos + length_mark_unicode.length());
    }
    std::cout << "Phoneme string after colon cleanup: " << ipa_string << std::endl;

    // 2. Tokenization (Multi-Byte Lookup Logic - FIX FOR SUSPECT B)
    std::vector<input_ids_type> raw_tokens;
    const size_t max_unpadded_length = maxContextLength_ - 2; 

    size_t i = 0;
    while (i < ipa_string.length() && raw_tokens.size() < max_unpadded_length) {
        
        // Try looking up the longest valid UTF-8 token first (max 4 bytes).
        size_t best_match_len = 0;
        input_ids_type best_match_id = 0;
        
        // Iterate backwards from max possible token size (4) down to 1
        for (size_t len = std::min((size_t)4, ipa_string.length() - i); len >= 1; --len) {
            std::string token_slice = ipa_string.substr(i, len);
            
            if (tokenMap_.count(token_slice)) {
                best_match_len = len;
                best_match_id = tokenMap_.at(token_slice);
                break; // Found the longest possible token, stop searching shorter slices
            }
        }
        
        if (best_match_len > 0) {
            raw_tokens.push_back(best_match_id);
            i += best_match_len; // Advance by the size of the matched token
        } else {
            // Unmapped character: skip, but MUST advance by 1 byte to prevent infinite loop
            i += 1;
            // Optionally log skipped single byte here, if needed for debugging unmapped characters
        }
    }
    // --- END Tokenization FIX ---

    // B. Apply Padding to create the final input_ids (Size: unpadded_length + 2)
    // ONLY ONE PAD at start and one at end. NO padding to maxContextLength_
    input_ids.push_back(PAD_TOKEN_ID);
    input_ids.insert(input_ids.end(), raw_tokens.begin(), raw_tokens.end());
    input_ids.push_back(PAD_TOKEN_ID);

    // C. Remove the incorrect padding to MAX_CONTEXT_LENGTH. 
    /*
    while (input_ids.size() < maxContextLength_) {
        input_ids.push_back(PAD_TOKEN_ID);
    }
    */
    
    if (input_ids.size() > maxContextLength_) {
        // This should only happen if the raw_tokens were exactly maxContextLength_ - 2, 
        // and we added the two PAD_TOKEN_ID, pushing the size to maxContextLength_.
        input_ids.resize(maxContextLength_);
        std::cout << "DEBUG: Sequence truncated to " << maxContextLength_ << " tokens." << std::endl;
    }

    std::cout << "DEBUG: Final input_ids size: " << input_ids.size() << " (Padded with start/end tokens only)." << std::endl;
    std::cout << "DEBUG: Unpadded token count (for style lookup) is: " << raw_tokens.size() << " tokens." << std::endl;
    
    return input_ids;
}

// --- Model Loading ---

bool KokoroTTS::loadModel(const std::string& modelPath) {
    if (sampleRate_ == 0 || maxContextLength_ == 0) {
        std::cerr << "Fatal Error: Engine not initialized. Call initializeEngine() first." << std::endl;
        return false;
    }

    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);

        size_t numInputs = session_->GetInputCount();
        size_t numOutputs = session_->GetOutputCount();
        tokenInputName_ = "";
        styleVectorDim_ = 0;
        outputName_ = "";
        outputDataType_ = 0; 

        // 1. Probe Inputs
        for (size_t i = 0; i < numInputs; ++i) {
            auto name_ptr = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            std::string name = name_ptr.get();

            if (name == "input_ids" || name == "tokens") {
                tokenInputName_ = name;
                std::cout << "DEBUG: Detected token input name: '" << name << "'" << std::endl;
            } else if (name == "style") {
                Ort::TypeInfo typeInfo = session_->GetInputTypeInfo(i);
                Ort::ConstTensorTypeAndShapeInfo constTensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                std::vector<int64_t> shape = constTensorInfo.GetShape();

                if (shape.size() == 2 && shape[0] == 1) {
                    styleVectorDim_ = (size_t)shape[1];
                    std::cout << "DEBUG: Detected style vector dimension (D): " << styleVectorDim_ << std::endl;
                } else {
                    std::cerr << "Warning: Style input has unexpected rank or batch size: " << shape.size() << std::endl;
                }
            }
        }

        // 2. Probe Outputs
        if (numOutputs > 0) {
            auto name_ptr = session_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            outputName_ = name_ptr.get();

            Ort::TypeInfo typeInfo = session_->GetOutputTypeInfo(0); 
            Ort::ConstTensorTypeAndShapeInfo tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            // Assigning the C enum value (int) to the int member type.
            outputDataType_ = tensorInfo.GetElementType(); 

            std::cout << "DEBUG: Detected primary output name: '" << outputName_ << "'" << std::endl;
            std::cout << "DEBUG: Detected primary output data type (ORT Enum value): " << outputDataType_ << "\n";
        }

        // Since outputDataType_ is an int, compare against the known UNKNOWN value (0)
        if (tokenInputName_.empty() || styleVectorDim_ == 0 || outputName_.empty() || outputDataType_ == 0) {
            std::cerr << "Error: Could not find all necessary model inputs or primary output name/type." << std::endl;
            return false;
        }

        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.playback.format  = ma_format_f32;
        config.playback.channels = 1;
        config.sampleRate       = sampleRate_; 
        config.dataCallback     = KokoroTTS::audioDataCallback;
        config.pUserData        = this;

        ma_result result = ma_device_init(NULL, &config, &audioDevice_);
        if (result != MA_SUCCESS) {
            std::cerr << "Error: Failed to initialize miniaudio device. Result code: " << result << std::endl;
            return false;
        }

        std::cout << "Successfully loaded model and initialized miniaudio at " << sampleRate_ << "Hz." << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return false;
    }
}

// --- loadVoiceData Function ---

bool KokoroTTS::loadVoiceData(const std::string& voiceDirectory, const std::string& voiceName) {
    if (styleVectorDim_ == 0) {
        std::cerr << "Error: Model not loaded or style dimension not detected. Cannot load voice." << std::endl;
        return false;
    }
    
    fs::path fullPath = fs::path(voiceDirectory) / (voiceName + ".dat");
    std::string filepath = fullPath.string();

    std::cout << "Loading voice '" << voiceName << "' from directory: " << voiceDirectory << "..." << std::endl;

    try {
        VoiceVectorStore store = load_raw_binary_vectors(filepath);

        // 3. Validation
        if (store.N_rows == 0) {
            std::cerr << "Error: Voice file '" << filepath << "' expected N > 0 rows (styles)." << std::endl;
            return false;
        }
        if (store.D_cols != styleVectorDim_) {
            std::cerr << "Error: Voice file '" << filepath << "' dimension (D=" << store.D_cols 
                      << ") does not match model expectation (DIM=" << styleVectorDim_ << ")." << std::endl;
            return false;
        }

        // 4. Store the entire loaded bank (N_rows * D_cols elements)
        currentStyleVector_ = std::move(store.data); 
        voiceBankSize_ = store.N_rows; // Store the number of vectors in the bank

    } catch (const std::runtime_error& e) {
        std::cerr << "Error loading voice data from '" << filepath << "': " << e.what() << std::endl;
        return false;
    }

    std::cout << "Successfully loaded voice bank for '" << voiceName << "'." << std::endl;
    std::cout << "DEBUG: Voice bank size (N) is " << voiceBankSize_ << " vectors." << std::endl;
    return true;
}

// --- Audio Streaming Callback ---

void KokoroTTS::audioDataCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    KokoroTTS* tts = reinterpret_cast<KokoroTTS*>(pDevice->pUserData);
    
    // Check and print debug message only once
    if (tts->callbackDebugCounter == 0) {
        std::cout << "DEBUG: Miniaudio Callback running (expected)." << std::endl;
        // Increment the counter to prevent flooding the console
        tts->callbackDebugCounter++; 
    }
    
    float* pOut = reinterpret_cast<float*>(pOutput);

    size_t samples_to_copy = frameCount;
    size_t remaining_samples = tts->synthesisBuffer.size() - tts->bufferReadIndex;

    if (remaining_samples < samples_to_copy) {
        samples_to_copy = remaining_samples;
    }

    if (samples_to_copy > 0) {
        memcpy(pOut, tts->synthesisBuffer.data() + tts->bufferReadIndex, samples_to_copy * sizeof(float));
        tts->bufferReadIndex += samples_to_copy;
    }

    if (tts->bufferReadIndex >= tts->synthesisBuffer.size()) {
        ma_device_stop(pDevice);
        if (samples_to_copy < frameCount) {
              memset(pOut + samples_to_copy, 0, (frameCount - samples_to_copy) * sizeof(float));
        }
    } else if (samples_to_copy < frameCount) {
        memset(pOut + samples_to_copy, 0, (frameCount - samples_to_copy) * sizeof(float));
    }
}


// --- Main Synthesis and Stream Function ---

void KokoroTTS::synthesizeAndStream(const std::string& text, float speed) {
    std::cout << "DEBUG: synthesizeAndStream started. Speed: " << speed << std::endl;

    if (!session_ || currentStyleVector_.empty() || voiceBankSize_ == 0 || styleVectorDim_ == 0) {
        std::cerr << "Error: Model or Voice Bank not loaded correctly." << std::endl;
        return;
    }
    
    // 1. Phonemize and Tokenize (G2P)
    // input_ids now has variable size (e.g., 52)
    std::vector<input_ids_type> input_ids = phonemizeAndTokenize(text, "en");
    if (input_ids.empty() || input_ids.size() > maxContextLength_) { 
        std::cerr << "Error: Tokenization failed or resulted in invalid length (" << input_ids.size() << "). Expected <= " << maxContextLength_ << "." << std::endl;
        return;
    }

    const size_t num_elements = input_ids.size(); // Variable length (e.g., 52)
    
    // --- CRITICAL LOOKUP LOGIC ---
    size_t unpadded_length = 0;
    // Calculate the unpadded length (excluding the start/end padding tokens)
    unpadded_length = 0;
    for (size_t i = 1; i < input_ids.size() - 1; ++i) { // Check between the first and last element
        if (input_ids[i] != PAD_TOKEN_ID) {
            unpadded_length++;
        }
    }
    
    size_t style_index = unpadded_length;
    
    if (style_index >= voiceBankSize_) {
        std::cerr << "Warning: Input token length (" << unpadded_length << ") exceeds voice bank capacity (" << voiceBankSize_ << "). Using max style vector: index " << voiceBankSize_ - 1 << std::endl;
        style_index = voiceBankSize_ - 1;
    }
    
    size_t start_offset = style_index * styleVectorDim_;
    const style_vector_type* selected_style_vector_ptr = currentStyleVector_.data() + start_offset;
    
    std::cout << "DEBUG: Selected Style Vector for length " << unpadded_length << " is index " << style_index << "." << std::endl;
    std::cout << "DEBUG: Total voice bank size is " << voiceBankSize_ << "." << std::endl; 

    Ort::AllocatorWithDefaultOptions allocator;

    // 2. Prepare ONNX Input Tensors
    
    // a) input_ids Tensor (Shape: [1, num_elements])
    // The shape is now dynamic: [1, 52] for the test case.
    std::vector<int64_t> input_shape = {1, (int64_t)num_elements}; 
    Ort::Value inputIdsTensor = Ort::Value::CreateTensor<input_ids_type>(allocator, input_shape.data(), input_shape.size());
    input_ids_type* input_tensor_data_ptr = inputIdsTensor.GetTensorMutableData<input_ids_type>();
    std::copy(input_ids.begin(), input_ids.end(), input_tensor_data_ptr);
    
    // b) style Tensor (Shape: [1, styleVectorDim_])
    std::vector<int64_t> style_shape = {1, (int64_t)styleVectorDim_};
    Ort::Value styleTensor = Ort::Value::CreateTensor<style_vector_type>(allocator, style_shape.data(), style_shape.size());
    style_vector_type* style_tensor_data_ptr = styleTensor.GetTensorMutableData<style_vector_type>();
    std::copy(selected_style_vector_ptr, selected_style_vector_ptr + styleVectorDim_, style_tensor_data_ptr);

    // c) speed Tensor (Shape: [1])
    std::vector<float> speed_data = {speed};
    std::vector<int64_t> speed_shape = {1};
    Ort::Value speedTensor = Ort::Value::CreateTensor<float>(allocator, speed_shape.data(), speed_shape.size());
    float* speed_tensor_data_ptr = speedTensor.GetTensorMutableData<float>();
    std::copy(speed_data.begin(), speed_data.end(), speed_tensor_data_ptr);

    // d) input_lengths Tensor -- REMOVED

    // 3. Update Input Names and Tensors List (Back to 3 inputs)
    const char* inputNames[] = {tokenInputName_.c_str(), "style", "speed"};
    const char* outputNames[] = {outputName_.c_str()};

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputIdsTensor));
    inputTensors.push_back(std::move(styleTensor)); 
    inputTensors.push_back(std::move(speedTensor)); 

    std::cout << "DEBUG: All input tensors prepared. Starting session->Run()..." << std::endl;

    // 4. Run Inference
    std::cout << "Running ONNX inference..." << std::endl;

    try {
        std::vector<Ort::Value> outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            inputTensors.data(),
            inputTensors.size(),
            outputNames,
            1
        );

        std::cout << "DEBUG: Inference successful. Extracting output..." << std::endl;

        // 5. Extract Audio Output and Stream 
        Ort::Value& audioOut = outputTensors.front();
        size_t audioDataSize = audioOut.GetTensorTypeAndShapeInfo().GetElementCount();
        
        // --- Safety Truncation for Excessive Length ---
        // Max 300,000 samples = 12.5 seconds at 24000 Hz.
        const size_t MAX_SAFE_SAMPLES = 300000; 

        if (audioDataSize > MAX_SAFE_SAMPLES) {
            std::cerr << "WARNING: Excessive output length detected (" << audioDataSize << " samples). Truncating output to " << MAX_SAFE_SAMPLES << " samples to prevent memory overflow and playback hang." << std::endl;
            audioDataSize = MAX_SAFE_SAMPLES;
        }

        synthesisBuffer.resize(audioDataSize);

        float max_abs_amp = 0.0f;

        // --- Use Stored Output Data Type (int) for Correct Access and Conversion ---
        // 1 (FLOAT), 10 (FLOAT16), 5 (INT16).
        if (outputDataType_ == 1) { // 1 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            std::cout << "DEBUG: Processing output as FLOAT (FP32) with Dynamic Normalization." << std::endl;
            const float* audioData = audioOut.GetTensorData<float>();

            // 1. First Pass: Find Max Absolute Amplitude
            for (size_t i = 0; i < audioDataSize; ++i) { // Iterating only up to the (possibly truncated) size
                float abs_val = std::abs(audioData[i]);
                if (abs_val > max_abs_amp) {
                    max_abs_amp = abs_val;
                }
            }

            // 2. Calculate Normalization Factor
            float normalization_factor = 1.0f;
            if (max_abs_amp > 1e-6) { 
                normalization_factor = 1.0f / max_abs_amp;
            }

            std::cout << "DEBUG: Max raw amplitude: " << max_abs_amp << std::endl;
            std::cout << "DEBUG: Applying dynamic normalization factor: " << normalization_factor << std::endl;

            // 3. Second Pass: Normalize and Clamp to Synthesis Buffer
            for (size_t i = 0; i < audioDataSize; ++i) {
                float sample = audioData[i];
                float scaled_sample = sample * normalization_factor;

                // Clamp
                if (scaled_sample > 1.0f) { synthesisBuffer[i] = 1.0f; }
                else if (scaled_sample < -1.0f) { synthesisBuffer[i] = -1.0f; }
                else { synthesisBuffer[i] = scaled_sample; }
            }

        } else if (outputDataType_ == 10) { // 10 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            std::cout << "DEBUG: Processing output as FLOAT16. Converting to FP32." << std::endl;
            const uint16_t* halfAudioData = audioOut.GetTensorData<uint16_t>();

            // 1. First Pass: Find Max Absolute Amplitude (on the converted data)
            for (size_t i = 0; i < audioDataSize; ++i) { // Iterating only up to the (possibly truncated) size
                float sample = convertHalfToFloat(halfAudioData[i]);
                float abs_val = std::abs(sample);
                if (abs_val > max_abs_amp) {
                    max_abs_amp = abs_val;
                }
            }

            // 2. Calculate Normalization Factor
            float normalization_factor = 1.0f;
            if (max_abs_amp > 1e-6) {
                normalization_factor = 1.0f / max_abs_amp;
            }

            std::cout << "DEBUG: Max raw amplitude (FP16 converted): " << max_abs_amp << std::endl;
            std::cout << "DEBUG: Applying dynamic normalization factor: " << normalization_factor << std::endl;

            // 3. Second Pass: Convert, Normalize, and Clamp to Synthesis Buffer
            for (size_t i = 0; i < audioDataSize; ++i) {
                float sample = KokoroTTS::convertHalfToFloat(halfAudioData[i]);
                float scaled_sample = sample * normalization_factor;

                // Clamp
                if (scaled_sample > 1.0f) { synthesisBuffer[i] = 1.0f; }
                else if (scaled_sample < -1.0f) { synthesisBuffer[i] = -1.0f; }
                else { synthesisBuffer[i] = scaled_sample; }
            }

        } else if (outputDataType_ == 5) { // 5 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
             std::cout << "DEBUG: Processing output as INT16. Rescaling to FP32 [-1.0, 1.0]." << std::endl;
             const int16_t* intAudioData = audioOut.GetTensorData<int16_t>();

             const float FIXED_INT16_SCALE = 1.0f / 32768.0f;

             for (size_t i = 0; i < audioDataSize; ++i) { // Iterating only up to the (possibly truncated) size
                 float sample = static_cast<float>(intAudioData[i]) * FIXED_INT16_SCALE;

                 float abs_val = std::abs(sample);
                 if (abs_val > max_abs_amp) {
                     max_abs_amp = abs_val;
                 }

                 // Clamp
                 if (sample > 1.0f) { synthesisBuffer[i] = 1.0f; }
                 else if (sample < -1.0f) { synthesisBuffer[i] = -1.0f; }
                 else { synthesisBuffer[i] = sample; }
             }
             std::cout << "DEBUG: Max raw amplitude (INT16 scaled): " << max_abs_amp << std::endl;
        }
           else {
             std::cerr << "ERROR: Unsupported ONNX output data type: " << outputDataType_ << ". Cannot stream audio." << std::endl;
             return;
        }
        // --- End Dynamic Type Handling ---

        std::cout << "DEBUG: Output audio size: " << audioDataSize << " samples." << std::endl;

        bufferReadIndex = 0;

        // Reset the callback debug counter for the next run
        callbackDebugCounter = 0; 
        
        // --- RESTORED Standard Playback Logic ---
        if (ma_device_is_started(&audioDevice_)) {
            ma_device_stop(&audioDevice_);
        }

        if (ma_device_start(&audioDevice_) != MA_SUCCESS) {
            std::cerr << "Error: Failed to start miniaudio device." << std::endl;
            return;
        }

        std::cout << "Playing audio..." << std::flush;
        while (ma_device_is_started(&audioDevice_)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::cout << "\nSynthesis and playback complete." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error during inference: " << e.what() << std::endl;
        std::cerr << "This could be due to missing model files, incorrect paths, or corrupted tensors." << std::endl;
    }
}
