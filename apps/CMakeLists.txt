add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)

if(WIN32)
  # Ship the CUDA runtime next to the executables instead of relying on the system PATH.
  file(GLOB NINFER_CUDART_DLLS "${CUDAToolkit_BIN_DIR}/cudart64_*.dll"
                               "${CUDAToolkit_BIN_DIR}/x64/cudart64_*.dll")
  foreach(app ninfer ninfer-serve)
    add_custom_command(TARGET ${app} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NINFER_CUDART_DLLS} $<TARGET_FILE_DIR:${app}>)
  endforeach()
endif()
