#!/usr/bin/env python3
"""生成 src/unicode_tables.h —— 分词器所需的 Unicode 类别范围表。

为什么是"生成"而不是"手写近似"：
    Qwen2/Qwen3 的预分词正则里有 \\p{L} / \\p{N} / \\s。它们的定义是 Unicode
    通用类别（Letter = Lu+Ll+Lt+Lm+Lo；Number = Nd+Nl+No）与 Unicode 空白集。
    手写近似（"ASCII 字母 + CJK 区间"）在中文混排的常见文本上看起来没问题，
    但在韩文谚文、希腊字母、注音符号、全角数字上会悄悄切错 —— 而切错了不会
    报错，只会让 token id 序列和官方不一致。既然对拍要和官方逐 id 相等，
    就必须用准确的表。

表从 Python 的 unicodedata（UCD 数据）扫全码位生成，输出 (start,end) 闭区间
的 uint32 数组，体积只有几 KB。

用法：
    python tools/gen_unicode_tables.py > src/unicode_tables.h
"""
import sys
import unicodedata

MAX_CP = 0x110000

# 与 Rust regex 的 \s（Unicode 空白）保持一致。
# 注意 U+200B (ZWSP) 不是空白，U+180E 在较新 UCD 里已移出空白集。
UNICODE_SPACE = [
    0x0009, 0x000A, 0x000B, 0x000C, 0x000D,
    0x0020, 0x0085, 0x00A0, 0x1680,
    0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
    0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
]


def ranges(pred):
    out = []
    start = None
    for cp in range(MAX_CP):
        if 0xD800 <= cp <= 0xDFFF:  # 代理区不是码点
            ok = False
        else:
            ok = pred(cp)
        if ok:
            if start is None:
                start = cp
        else:
            if start is not None:
                out.append((start, cp - 1))
                start = None
    if start is not None:
        out.append((start, MAX_CP - 1))
    return out


def is_letter(cp):
    return unicodedata.category(chr(cp)).startswith("L")


def is_number(cp):
    return unicodedata.category(chr(cp)).startswith("N")


def emit(name, rs, out):
    out.append(f"// {name}: {len(rs)} 个区间")
    out.append(f"inline constexpr CodeRange {name}[] = {{")
    line = "   "
    for a, b in rs:
        item = f" {{0x{a:04X}, 0x{b:04X}}},"
        if len(line) + len(item) > 96:
            out.append(line)
            line = "   "
        line += item
    if line.strip():
        out.append(line)
    out.append("};")
    out.append("")


def main():
    print("// 本文件由 tools/gen_unicode_tables.py 生成，请勿手改。")
    print("// 重新生成：python tools/gen_unicode_tables.py > src/unicode_tables.h")
    print(f"// 数据来源：Python unicodedata（Unicode {unicodedata.unidata_version}）")
    print()
    print("#ifndef RPGBAR_UNICODE_TABLES_H_")
    print("#define RPGBAR_UNICODE_TABLES_H_")
    print()
    print("#include <cstdint>")
    print()
    print("namespace rpg {")
    print()
    print("struct CodeRange {")
    print("  uint32_t lo;")
    print("  uint32_t hi;  // 闭区间")
    print("};")
    print()
    out = []
    emit("kLetterRanges", ranges(is_letter), out)
    emit("kNumberRanges", ranges(is_number), out)
    # 空白集合很小，直接列单码点（用 lo==hi 表示）。
    emit("kSpaceRanges", [(c, c) for c in UNICODE_SPACE], out)
    sys.stdout.write("\n".join(out))
    print("}  // namespace rpg")
    print()
    print("#endif  // RPGBAR_UNICODE_TABLES_H_")


if __name__ == "__main__":
    main()
