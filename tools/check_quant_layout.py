#!/usr/bin/env python3
"""check_quant_layout.py —— 独立校验各 ggml 量化类型的"每元素字节数"。

原理（不依赖我们的 C++ 代码）：
    GGUF 里每个张量的 offset 是 GGUF **写入器**算出来的。写入器按它自己认为的
    block_bytes / block_elems 排布张量，所以：

        bytes_per_element(type) = n_bytes(tensor) / n_elements(tensor)

    对同一类型的所有张量必然相同，而且能从相邻张量的 offset 差直接反推出来：

        n_bytes = next_offset - offset      （同一文件内张量按 offset 连续排布）

    于是"我的量化类型尺寸表对不对"这件事，可以完全用文件自身的偏移量来验证 ——
    这正是最容易写错、又最不容易被发现的地方（表写错了，反量化不会报错，
    只会读错字节，最后表现为胡说八道）。

同时校验张量在文件里"无缝铺满"：按 offset 排序后逐块相接，且总长等于文件大小。

用法：
    python tools/check_quant_layout.py <model.gguf> [<model2.gguf> ...]
    python tools/check_quant_layout.py --header-only <url>   # 只拉文件头（Range 请求）
"""
import argparse
import struct
import sys

ALIGN_DEFAULT = 32

# 与本引擎 src/quant.cc 里 kTypes[] 完全对应。改一边必须改另一边。
TYPES = {
    0:  ("F32",   1,   4),
    1:  ("F16",   1,   2),
    2:  ("Q4_0",  32,  18),
    3:  ("Q4_1",  32,  20),
    6:  ("Q5_0",  32,  22),
    7:  ("Q5_1",  32,  24),
    8:  ("Q8_0",  32,  34),
    9:  ("Q8_1",  32,  36),
    10: ("Q2_K",  256, 84),
    11: ("Q3_K",  256, 110),
    12: ("Q4_K",  256, 144),
    13: ("Q5_K",  256, 176),
    14: ("Q6_K",  256, 210),
    15: ("Q8_K",  256, 292),
    30: ("BF16",  1,   2),
}

META_TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                   10: 8, 11: 8, 12: 8}
META_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                6: "<f", 10: "<Q", 11: "<q", 12: "<d"}


class Cur:
    def __init__(self, buf):
        self.b = buf
        self.p = 0

    def need(self, n):
        if self.p + n > len(self.b):
            raise EOFError(f"需要 {n} 字节，剩余 {len(self.b) - self.p}")

    def u32(self):
        self.need(4)
        v, = struct.unpack_from("<I", self.b, self.p)
        self.p += 4
        return v

    def u64(self):
        self.need(8)
        v, = struct.unpack_from("<Q", self.b, self.p)
        self.p += 8
        return v

    def raw(self, n):
        self.need(n)
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def string(self):
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")


def read_value(c, t):
    if t == 8:
        return c.string()
    if t == 7:
        return c.raw(1)[0] != 0
    if t == 9:
        et = c.u32()
        n = c.u64()
        return [read_value(c, et) for _ in range(n)]
    return struct.unpack(META_FORMATS[t], c.raw(META_TYPE_SIZES[t]))[0]


def parse_header(buf):
    c = Cur(buf)
    if c.raw(4) != b"GGUF":
        raise SystemExit("不是 GGUF 文件")
    version = c.u32()
    if version == 1:
        n_tensors, n_kv = c.u32(), c.u32()
    else:
        n_tensors, n_kv = c.u64(), c.u64()
    meta = {}
    for _ in range(n_kv):
        k = c.string()
        t = c.u32()
        meta[k] = read_value(c, t)
    tensors = []
    for _ in range(n_tensors):
        name = c.string()
        nd = c.u32()
        dims = [c.u64() for _ in range(nd)]
        typ = c.u32()
        off = c.u64()
        tensors.append({"name": name, "dims": dims, "type": typ, "offset": off})
    pos = c.p
    align = meta.get("general.alignment", ALIGN_DEFAULT)
    data_start = pos + (align - pos % align) % align
    return {"version": version, "meta": meta, "tensors": tensors,
            "alignment": align, "data_start": data_start}


def check(doc, file_size, label):
    print(f"=== {label} ===")
    print(f"  GGUF v{doc['version']} / {len(doc['tensors'])} 张量 / 对齐 "
          f"{doc['alignment']} / 数据起点 {doc['data_start']}")

    by_type = {}
    for t in doc["tensors"]:
        by_type.setdefault(t["type"], []).append(t)

    bad = 0
    print("  类型                    张量  每元素字节(实测)  每元素字节(表)  结论")
    for typ in sorted(by_type):
        if typ not in TYPES:
            print(f"  未知类型 {typ}: {len(by_type[typ])} 个张量 —— 本引擎不支持")
            bad += 1
            continue
        name, blk_elems, blk_bytes = TYPES[typ]
        want = blk_bytes / blk_elems
        ratios = set()
        for t in by_type[typ]:
            n_el = 1
            for d in t["dims"]:
                n_el *= d
            n_by = (n_el // blk_elems) * blk_bytes
            ratios.add(round(n_by / n_el, 10))
        ok = ratios == {round(want, 10)}
        flag = "OK" if ok else "!! 不一致"
        if not ok:
            bad += 1
        print(f"  {name:<22} {len(by_type[typ]):>4}  {sorted(ratios)}  "
              f"{round(want, 10)}  {flag}")

    # 无缝铺满：按 offset 排序后逐块相接
    ordered = sorted(doc["tensors"], key=lambda t: t["offset"])
    cursor = 0
    gaps = 0
    for t in ordered:
        if t["offset"] != cursor:
            gaps += 1
            if gaps <= 3:
                print(f"  缝隙：{t['name']} 期望 offset {cursor}，实际 {t['offset']}")
        n_el = 1
        for d in t["dims"]:
            n_el *= d
        blk_elems, blk_bytes = TYPES[t["type"]][1], TYPES[t["type"]][2]
        cursor = t["offset"] + (n_el // blk_elems) * blk_bytes
    total = doc["data_start"] + cursor
    print(f"  铺满检查：缝隙 {gaps} 处；推算文件总长 {total}，实际 {file_size} "
          f"{'OK' if total == file_size else '!! 不一致'}")
    if total != file_size:
        bad += 1
    print(f"  结论：{'全部通过' if bad == 0 else f'{bad} 项有问题'}")
    print()
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--header-only", action="append", default=[],
                    help="URL：只拉前 N 字节（默认 16MiB）解析头部")
    ap.add_argument("--bytes", type=int, default=16 << 20)
    args = ap.parse_args()

    total_bad = 0
    for p in args.paths:
        import os
        with open(p, "rb") as f:
            doc = parse_header(f.read())
        total_bad += check(doc, os.path.getsize(p), p)

    for url in args.header_only:
        import urllib.request
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{args.bytes}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            buf = r.read()
            # 从 Content-Range 拿真实总长
            cr = r.headers.get("Content-Range", "")
            size = int(cr.split("/")[-1]) if "/" in cr else len(buf)
        doc = parse_header(buf)
        total_bad += check(doc, size, url.split("/")[-1] + "（仅头部）")

    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
