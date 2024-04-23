cmake_minimum_required(VERSION 3.14)
project("RnNoise" LANGUAGES C)

option(RTCD "Enable x86 run-time CPU detection" OFF)

file(STRINGS "rnnoise/model_version" VERSION LIMIT_COUNT 1)
set(MODEL "rnnoise_data-${VERSION}.tar.gz")
execute_process(
  COMMAND wget -c -N --progress=bar:force https://media.xiph.org/rnnoise/models/${MODEL}
  WORKING_DIRECTORY rnnoise
)
execute_process(COMMAND tar xvomf ${MODEL} WORKING_DIRECTORY rnnoise)

include_directories(rnnoise/include rnnoise/src)
set(RNNOISE_SOURCES
  rnnoise/src/celt_lpc.c
  rnnoise/src/denoise.c
  rnnoise/src/kiss_fft.c
  rnnoise/src/parse_lpcnet_weights.c
  rnnoise/src/pitch.c
  rnnoise/src/rnn.c
  rnnoise/src/rnnoise_data.c
  rnnoise/src/rnnoise_tables.c
  rnnoise/src/nnet.c
  rnnoise/src/nnet_default.c
)

if(RTCD)
  add_compile_definitions(RNN_ENABLE_X86_RTCD)
  list(APPEND RNNOISE_SOURCES
    rnnoise/src/x86/x86cpu.c
    rnnoise/src/x86/x86_dnn_map.c
    rnnoise/src/x86/nnet_avx2.c
    rnnoise/src/x86/nnet_sse4_1.c
  )
endif()

add_library(rnnoise SHARED ${RNNOISE_SOURCES})
install(TARGETS rnnoise LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/pyrnnoise)
