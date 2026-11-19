# Hy4 Flash external model extension.

gufo_extension_target(
  NAME gufo_hy4flash
  SOURCES
    extension.cpp
    probe.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)

if(TARGET gufo_hy4flash)
  target_include_directories(gufo_hy4flash PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

install(FILES
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSES/gufo.txt"
  DESTINATION share/licenses/gufo/extensions/hy4flash)
