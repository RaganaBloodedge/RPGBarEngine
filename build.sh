#!/usr/bin/env bash
# build.sh —— MSYS2 / MinGW 快速构建（本机没有 cmake，走这条）。
# 有 cmake 的环境请用 CMakeLists.txt，两条路的产物一致。
#
# 关键点：必须 -static 全静态链接。否则 dll 会依赖 MSYS2 的 libwinpthread-1.dll，
# 宿主（Unity / Python）加载时报 "Could not find module ... or one of its dependencies"。
set -e
cd "$(dirname "$0")"
mkdir -p build

CXX="${CXX:-g++}"
"$CXX" -std=c++20 -O2 -fPIC -shared \
  -static -static-libgcc -static-libstdc++ \
  -Iinclude \
  -o build/rpg_engine.dll src/engine_stub.cc \
  -Wl,--out-implib,build/librpg_engine.a

echo "built: build/rpg_engine.dll"
