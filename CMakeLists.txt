cmake_minimum_required(VERSION 3.20)
project(RPGBarEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
  add_compile_options(/O2 /openmp /utf-8)
else()
  add_compile_options(-O3 -fopenmp -Wall -Wextra)
  add_link_options(-fopenmp)
endif()

add_library(rpg_engine SHARED
  src/engine_stub.cc
  src/common.cc
  src/json.cc
  src/threads.cc
  src/mmap_file.cc
  src/quant.cc
  src/gguf.cc
  src/tokenizer.cc
)

target_compile_definitions(rpg_engine PRIVATE RPG_ENGINE_BUILD)

target_include_directories(rpg_engine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

if(WIN32)
  # 显式导出清单：不这么写，MinGW 会把内部 C++ 符号全导出（见 rpg_engine.def）。
  target_sources(rpg_engine PRIVATE src/rpg_engine.def)
  if(MINGW)
    target_link_options(rpg_engine PRIVATE -Wl,--exclude-all-symbols)
  else()
    set_target_properties(rpg_engine PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(rpg_engine PRIVATE Threads::Threads)

set_target_properties(rpg_engine PROPERTIES
  OUTPUT_NAME rpg_engine
  POSITION_INDEPENDENT_CODE ON
)

# 命令行工具（dump_gguf / tok_check）。链接同一批对象，避免重复编译。
if(WIN32)
  set(TOOL_SUFFIX ".exe")
else()
  set(TOOL_SUFFIX "")
endif()

foreach(tool dump_gguf tok_check selftest)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/${tool}.cc")
    add_executable(${tool} tools/${tool}.cc)
    target_include_directories(${tool} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # 复用 rpg_engine 的对象（排除 C ABI 桩，工具用不到）。
    target_sources(${tool} PRIVATE
      src/common.cc src/json.cc src/threads.cc src/mmap_file.cc
      src/quant.cc src/gguf.cc src/tokenizer.cc)
    target_link_libraries(${tool} PRIVATE Threads::Threads)
    set_target_properties(${tool} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endif()
endforeach()
