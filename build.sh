#!/usr/bin/bash
#dos2unix *.hpp
#dos2unix *.cpp
rm kokoro

cd build
rm -rf ./*

cmake -DCMAKE_BUILD_TYPE=Release \
      -DONNX_INCLUDE_DIR=/usr/include/onnxruntime \
      -DONNX_LIBRARY=/usr/lib/x86_64-linux-gnu/libonnxruntime.so \
      -DESPEAK_INCLUDE_DIR=/usr/include/espeak-ng \
      -DESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so \
      ..

make
exit
