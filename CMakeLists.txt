cmake_minimum_required(VERSION 3.20)
project(RPGBarEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# 【不要加 OpenMP】并行走自带线程池（src/threads.h 里写了三个理由：零运行时依赖、
# decode 是很多次小并行区、可自己保证只有主线程发起并行）。
# libgomp / vcomp 对"直接丢进 Unity 游戏里"的库是实打实的额外依赖。
if(MSVC)
  add_compile_options(/O2 /utf-8)
else()
  add_compile_options(-O2 -Wall -Wextra)
endif()

# 引擎的全部 .cc（rpg_engine.def 与头文件不在此列）。
set(RPG_SOURCES
  src/engine_stub.cc
  src/common.cc
  src/json.cc
  src/threads.cc
  src/mmap_file.cc
  src/quant.cc
  src/gguf.cc
  src/tokenizer.cc
  src/model.cc
  src/qwen3.cc
)

add_library(rpg_engine SHARED ${RPG_SOURCES})

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
    # 全静态链接 libstdc++/libgcc/libwinpthread。少了它，dll 会依赖 MSYS2 的
    # libwinpthread-1.dll，宿主（Unity / Python）加载时报
    # "Could not find module ... or one of its dependencies"。
    # 本引擎用了 std::thread，libwinpthread 绕不开，只能静态吃进去。
    target_link_options(rpg_engine PRIVATE -static -static-libgcc -static-libstdc++)
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

# 命令行工具（dump_gguf / tok_check / selftest / forward_check）。
# 链接同一批源文件，避免重复编译；不含 engine_stub.cc（工具不走 C ABI）。
set(RPG_TOOLS dump_gguf tok_check selftest forward_check)
set(RPG_LIB_SOURCES ${RPG_SOURCES})
list(REMOVE_ITEM RPG_LIB_SOURCES src/engine_stub.cc)

foreach(tool IN LISTS RPG_TOOLS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/${tool}.cc")
    add_executable(${tool} tools/${tool}.cc)
    target_include_directories(${tool} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/tools)   # 工具的 npy.h 在 tools/ 下
    target_sources(${tool} PRIVATE ${RPG_LIB_SOURCES})
    target_link_libraries(${tool} PRIVATE Threads::Threads)
    set_target_properties(${tool} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endif()
endforeach()
