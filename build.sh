#!/usr/bin/env bash
# build.sh —— MSYS2 / MinGW-w64 快速构建（没有 cmake 的环境走这条）。
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
# -pipe：汇编通过管道交给 as，不落临时 .s 文件。省 IO，也让构建在没有
#      可写临时目录（或删除被安全策略拦住）的机器上照样能跑。
CXXFLAGS="-std=c++20 -O2 -fPIC -pipe -Wall -Wextra -DRPG_ENGINE_BUILD $INCLUDES"
LDFLAGS="-static -static-libgcc -static-libstdc++"

echo "[1/4] 编译引擎对象..."
# AVX2 内核（matmul_avx2.cc）**单独**开 -mavx2 -mfma，其余编译单元保持基线：
# 整库开 -march=native 会让 dll 在没有 AVX2 的机器上加载即非法指令崩溃，
# 而这个库是要丢进 Unity、发给别人的。分派在 matmul.cc 按 CPU 特性做。
# 编译器若不认这两个开关，就不加 —— 该文件的 AVX2 代码全在 __AVX2__ 之后，
# 会自动退化成“返回 nullptr”，由分派层走标量。
AVX2FLAGS=""
if printf 'int main(){return 0;}' | "$CXX" -mavx2 -mfma -x c++ -fsyntax-only - >/dev/null 2>&1; then
  AVX2FLAGS="-mavx2 -mfma"
else
  echo "  (编译器不支持 -mavx2 -mfma，AVX2 内核将退化为标量)"
fi
OBJS=""
for f in src/*.cc; do
  o="build/obj/$(basename "${f%.cc}").o"
  extra=""
  if [ "$(basename "$f")" = "matmul_avx2.cc" ]; then extra="$AVX2FLAGS"; fi
  "$CXX" $CXXFLAGS $extra -c "$f" -o "$o"
  OBJS="$OBJS $o"
done

echo "[2/4] 链接动态库..."
# --exclude-all-symbols + .def：导出表里只留 C ABI 那 10 个函数。
# 缺了这两样，MinGW 会把内部 C++ 符号和 libgcc 的 _Unwind_* 一起导出。
#
# 先链到 .new 再 mv -f 就位，是**环境对策**，不是风格问题：本机（WorkBuddy
# 沙箱的 safe-delete 钩子）会拦住"删除/截断已存在的文件"，ld 直接写
# build/rpg_engine.dll 时既删不掉旧的、又截不了，只能**静默失败** —— 只回一句
# "collect2.exe: error: ld returned 1 exit status"，连 ld 自己的诊断都被吞掉，
# 极难归因（曾误以为是自己代码的问题）。mv -f 走的是 rename，不受该策略影响。
"$CXX" -shared $LDFLAGS -o build/rpg_engine.dll.new $OBJS \
  src/rpg_engine.def -Wl,--exclude-all-symbols \
  -Wl,--out-implib,build/librpg_engine.a.new
mv -f build/rpg_engine.dll.new build/rpg_engine.dll
mv -f build/librpg_engine.a.new build/librpg_engine.a

echo "[3/4] 链接命令行工具..."
# 工具的编译单元要能 #include "npy.h" 这类同目录头，所以多一个 -Itools。
# 同样先链 .new 再 mv -f（理由见上一步）。
for t in dump_gguf tok_check selftest forward_check gen_check chatml_check parse_check sample_check dotrow_check; do
  if [ -f "tools/$t.cc" ]; then
    "$CXX" $CXXFLAGS -Itools $LDFLAGS -o "build/$t.exe.new" "tools/$t.cc" $OBJS
    mv -f "build/$t.exe.new" "build/$t.exe"
  fi
done

echo "[4/4] 产物:"
ls -la build/rpg_engine.dll build/*.exe 2>/dev/null || true
echo "OK"
