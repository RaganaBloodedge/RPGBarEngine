#!/usr/bin/env bash
# build.sh —— MSYS2 / MinGW-w64 快速构建（本机没装 cmake 时走这条）。
# 有 cmake 的环境请用 CMakeLists.txt，两条路的产物一致。
#
# 两个硬性要求，改这个脚本时别丢：
#
# 1) **必须 -static 全静态链接**。不静态链接时 dll 会依赖 MSYS2 的
#    libwinpthread-1.dll / libstdc++-6.dll，宿主（Unity / Python）加载时报
#    "Could not find module ... or one of its dependencies"。
#    本引擎用了 std::thread，所以 libwinpthread 是绕不开的 —— 只能静态吃进去。
#
# 2) **工具链的 g++ 必须是 MinGW 而非 Cygwin/MSYS 原生**。MSYS2 的
#    /mingw64/bin/g++ 产物只依赖系统 DLL；/usr/bin/g++（MSYS 原生）会拖
#    msys-2.0.dll。脚本默认取 PATH 上的 g++，请确保 PATH 里 mingw64 在前。
set -e
cd "$(dirname "$0")"

CXX="${CXX:-g++}"
mkdir -p build/obj

INCLUDES="-Iinclude -Isrc"
CXXFLAGS="-std=c++20 -O2 -fPIC -Wall -Wextra -DRPG_ENGINE_BUILD $INCLUDES"
LDFLAGS="-static -static-libgcc -static-libstdc++"

echo "[1/4] 编译引擎对象..."
OBJS=""
for f in src/*.cc; do
  o="build/obj/$(basename "${f%.cc}").o"
  "$CXX" $CXXFLAGS -c "$f" -o "$o"
  OBJS="$OBJS $o"
done

echo "[2/4] 链接动态库..."
# --exclude-all-symbols + .def：导出表里只留 C ABI 那 10 个函数。
# 缺了这两样，MinGW 会把内部 C++ 符号和 libgcc 的 _Unwind_* 一起导出。
"$CXX" -shared $LDFLAGS -o build/rpg_engine.dll $OBJS \
  src/rpg_engine.def -Wl,--exclude-all-symbols \
  -Wl,--out-implib,build/librpg_engine.a

echo "[3/4] 链接命令行工具..."
for t in dump_gguf tok_check selftest; do
  if [ -f "tools/$t.cc" ]; then
    "$CXX" $CXXFLAGS $LDFLAGS -o "build/$t.exe" "tools/$t.cc" $OBJS
  fi
done

echo "[4/4] 产物:"
ls -la build/rpg_engine.dll build/*.exe 2>/dev/null || true
echo "OK"
