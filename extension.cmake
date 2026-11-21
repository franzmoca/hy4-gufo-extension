# Hy4 Flash external model extension.

# Host side: artifact probe, chat template, tool syntax and registration.
gufo_extension_target(
  NAME gufo_hy4flash
  SOURCES
    chat_template.cpp
    extension.cpp
    probe.cpp
    text_contract.cpp
    tool_calls.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
)

if(TARGET gufo_hy4flash)
  target_include_directories(gufo_hy4flash PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  # Registration leaves the provider to gufo_hy4flash_text in HIP builds.
  if(ENGINE_ENABLE_HIP)
    set_property(SOURCE "${GUFO_EXTENSION_BUILD_ROOT}/extension.cpp"
      APPEND PROPERTY COMPILE_DEFINITIONS ENGINE_ENABLE_HIP=1)
  endif()
endif()

if(ENGINE_ENABLE_HIP)
  set(HY4FLASH_ROCM_DIR "${GUFO_EXTENSION_BUILD_ROOT}/kernels/rocm")
  set(HY4FLASH_MMQ_DIR "${HY4FLASH_ROCM_DIR}/mmq")

  # Vendored llama.cpp quantized matvecs, from DeepSeek V4.1 Flash's
  # extension: their own C++17 dialect and GGML macros, so a separate target.
  # See kernels/rocm/mmq/VENDOR.md.
  set(HY4FLASH_MMQ_SOURCES
    ${HY4FLASH_MMQ_DIR}/hy4_ggml_stubs.hip.cpp
    ${HY4FLASH_MMQ_DIR}/hy4_mmq.hip.cpp
    ${HY4FLASH_MMQ_DIR}/hy4_mmq_d2r.hip.cpp
    ${HY4FLASH_MMQ_DIR}/mmid.hip.cpp
    ${HY4FLASH_MMQ_DIR}/mmvq.hip.cpp
    ${HY4FLASH_MMQ_DIR}/quantize.hip.cpp)
  add_library(gufo_hy4flash_mmq STATIC ${HY4FLASH_MMQ_SOURCES})
  foreach(hy4_mmq_type Q5_K Q6_K IQ1_S IQ1_M IQ2_XS IQ2_S)
    set(hy4_mmq_type_source
      "${CMAKE_CURRENT_BINARY_DIR}/hy4_mmq_v41_${hy4_mmq_type}.hip.cpp")
    file(GENERATE OUTPUT ${hy4_mmq_type_source} CONTENT
      "#define HY4_MMQ_TYPE GGML_TYPE_${hy4_mmq_type}\n#include \"hy4_mmq_v41_types.hip.cpp\"\n")
    target_sources(gufo_hy4flash_mmq PRIVATE ${hy4_mmq_type_source})
    list(APPEND HY4FLASH_MMQ_SOURCES ${hy4_mmq_type_source})
  endforeach()
  set_source_files_properties(${HY4FLASH_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_hy4flash_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_hy4flash_mmq PRIVATE
    GGML_USE_HIP
    GGML_HIP_NO_VMM
    HY4_ROCM_WMMA_MMQ_Y=64)
  target_compile_options(gufo_hy4flash_mmq PRIVATE
    -O3 -ffast-math -fno-finite-math-only -Wno-unused-value
    -Wno-unused-command-line-argument)
  target_include_directories(gufo_hy4flash_mmq SYSTEM PUBLIC ${HY4FLASH_MMQ_DIR})
  target_link_libraries(gufo_hy4flash_mmq PUBLIC hip::host roc::hipblas)
endif()

# Device side: model loading, evaluation and kernels.
gufo_extension_target(
  NAME gufo_hy4flash_hip
  SOURCES
    runtime/hy4_engram.cpp
    runtime/hy4_model.cpp
    runtime/hy4_dfly.cpp
    runtime/hy4_forward.cpp
    runtime/hy4_vision.cpp
    runtime/hy4_vision_gemm.cpp
    runtime/tokenizer.cpp
  LINK_LIBRARIES
    gufo_hy4flash_mmq
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
    roc::hipblas
    roc::hipblaslt
    Threads::Threads
)

if(TARGET gufo_hy4flash_hip)
  gufo_extension_hip_sources(gufo_hy4flash_hip
    SOURCES
      kernels/rocm/hy4_f16_mm.hip.cpp
      kernels/rocm/hy4_gemv.hip.cpp
      kernels/rocm/hy4_kernels.hip.cpp
      kernels/rocm/hy4_vision.hip.cpp
  )
  target_compile_options(gufo_hy4flash_hip PRIVATE
    -O3 $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
  target_include_directories(gufo_hy4flash_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>")
endif()

# Serving runner, prompt cache and the provider registration.
gufo_extension_target(
  NAME gufo_hy4flash_text
  SOURCES
    continuation_adapter.cpp
    image_resize.cpp
    provider.cpp
    runner.cpp
    vision_prompt.cpp
  LINK_LIBRARIES
    gufo_hy4flash_hip
    gufo_hy4flash
    gufo_ext_runtime
    gufo_ext_hip
)
if(TARGET gufo_hy4flash_text)
  target_compile_options(gufo_hy4flash_text PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

# Probes against a model: built by `hy4flash_probes`, not by default. Every
# declared target goes through gufo_extension_target, which disables it
# without testing.
if(BUILD_TESTING AND NOT TARGET hy4flash_probes)
  add_custom_target(hy4flash_probes)
endif()
foreach(name IN ITEMS eval_probe tokenize_probe vision_probe)
  gufo_extension_target(
    NAME hy4flash_${name}
    SOURCES tools/probe/${name}.cpp
    LINK_LIBRARIES gufo_hy4flash_hip gufo_ext_hip gufo_ext_runtime
  )
  if(TARGET hy4flash_${name} AND name STREQUAL "vision_probe")
    target_link_libraries(hy4flash_${name} PRIVATE gufo_hy4flash_text)
  endif()
  if(TARGET hy4flash_${name})
    set_target_properties(hy4flash_${name} PROPERTIES
      EXCLUDE_FROM_ALL TRUE OUTPUT_NAME hy4_${name})
    add_dependencies(hy4flash_probes hy4flash_${name})
  endif()
endforeach()

# Tests. Host contracts need no GPU; kernel checks need gfx1151 but no model.
# `hy4flash_tests` builds every test binary, so ctest never runs a stale one.
if(BUILD_TESTING AND NOT TARGET hy4flash_tests)
  add_custom_target(hy4flash_tests)
endif()

function(_hy4flash_test name labels)
  cmake_parse_arguments(ARG "" "" "LINK_LIBRARIES" ${ARGN})
  set(target hy4flash_${name}_test)
  gufo_extension_target(
    NAME ${target}
    SOURCES tests/${name}_test.cpp
    LINK_LIBRARIES ${ARG_LINK_LIBRARIES} gufo_ext_runtime
  )
  if(NOT TARGET ${target})
    return()
  endif()
  target_compile_definitions(${target} PRIVATE
    GUFO_HY4_CHAT_GOLDENS="${GUFO_EXTENSION_BUILD_ROOT}/tests/fixtures/chat_goldens.json")
  add_test(NAME hy4flash.${name} COMMAND ${target})
  set_tests_properties(hy4flash.${name} PROPERTIES
    LABELS "hy4flash;${labels}" SKIP_RETURN_CODE 77)
  add_dependencies(hy4flash_tests ${target})
endfunction()

_hy4flash_test(chat_template "cpu;tokenization;tools"
  LINK_LIBRARIES gufo_hy4flash gufo_ext_core)
_hy4flash_test(vision_prompt "cpu;vision"
  LINK_LIBRARIES gufo_hy4flash_text gufo_hy4flash_hip gufo_hy4flash gufo_ext_hip)
_hy4flash_test(kernels "hip;gpu;gfx1151"
  LINK_LIBRARIES gufo_hy4flash_hip gufo_hy4flash_mmq gufo_ext_hip)
_hy4flash_test(vision_kernels "hip;gpu;gfx1151"
  LINK_LIBRARIES gufo_hy4flash_hip gufo_ext_hip)

install(FILES
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSES/gufo.txt"
  DESTINATION share/licenses/gufo/extensions/hy4flash)
