cmake_minimum_required(VERSION 3.10)
project(kokoro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- MANUAL PATH CHECK (UNCHANGED) ---
# Check that the user provided the necessary paths on the command line.
if(NOT ONNX_INCLUDE_DIR OR NOT ONNX_LIBRARY OR NOT ESPEAK_INCLUDE_DIR OR NOT ESPEAK_LIBRARY)
    message(FATAL_ERROR "Missing required manual paths. Please provide: ONNX_INCLUDE_DIR, ONNX_LIBRARY, ESPEAK_INCLUDE_DIR, and ESPEAK_LIBRARY.")
endif()
message(STATUS "All library paths manually configured.")

# --- Build Target (UPDATED TO INCLUDE NEW SOURCE FILES) ---

add_executable(${PROJECT_NAME}
    main.cpp
    tts_engine.cpp
    token_map.cpp
    voice_loader.cpp
)

# --- Link Libraries (Full Manual Force) ---

# ONNX Runtime Link
target_include_directories(${PROJECT_NAME} PUBLIC ${ONNX_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${ONNX_LIBRARY})

# eSpeak-NG Link
target_include_directories(${PROJECT_NAME} PUBLIC ${ESPEAK_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${ESPEAK_LIBRARY})

# Header Includes: This includes the main directory where tts_engine.hpp, token_map.hpp, 
# voice_loader.hpp, and now, json.hpp, are located.
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Miniaudio/System Dependencies (REQUIRED FOR AUDIO PLAYBACK)
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(ALSA QUIET alsa)
endif()

if(ALSA_FOUND)
    message(STATUS "Found ALSA for Miniaudio: ${ALSA_LIBRARIES}")
    target_link_libraries(${PROJECT_NAME} PUBLIC ${ALSA_LIBRARIES})
else()
    # Fallback/Common Linux dependency
    message(STATUS "ALSA not found, linking common sound library: -lasound")
    target_link_libraries(${PROJECT_NAME} PUBLIC "-lasound")
endif()
