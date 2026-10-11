#!/usr/bin/env python3
"""M4 验收：按**宿主的方式**调用 dll（ctypes），而不是进程内的 C++ 工具。

第 6/7/8 层验的是内核（前向 / 生成 / ChatML），这一层验的是"C 这一层"：
宿主拿到 dll 之后，能不能按 include/rpg_engine.h 的约定用起来 —— 加载、建会话、
生成、释放，以及句柄生命周期与错误通道。

三条判据：

  1. **结构**：给一份带工具定义的请求，能拿到 tool_calls[0].name == "read_scene"。
  2. **一致**：ABI 产出的 text 与工具调用，与工具链路径（chatml_check →
     gen_check → tok_check → parse_check）**完全相同**。两条路的工具不同、入口
     不同，若结果一致，说明 ABI 没有偷偷另走一套渲染或解码。
  3. **复用**：暖缓存（同一会话上继续请求）与冷缓存（新会话）的结果完全一致。
     最长公共前缀复用是整个 ABI 里最容易"静默出错"的地方 —— 复用错了不报错，
     只是让模型看到一段错的上下文。这条把它变成一次断言。

不需要 torch / transformers：第 2 条用的是纯进程内工具。

用法：
  python tools/check_abi.py
  python tools/check_abi.py --model models/Qwen3-1.7B-Q8_0.gguf --max-tokens 32
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START = time.time()


def log(msg: str = "") -> None:
    print(f"[{time.time() - START:6.1f}s] {msg}")


# ---------------------------------------------------------------- ctypes 绑定

def bind(dll_path: str):
    lib = ctypes.CDLL(dll_path)
    lib.rpg_engine_version.restype = ctypes.c_char_p
    lib.rpg_engine_last_error.restype = ctypes.c_char_p

    lib.rpg_engine_load_model.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.rpg_engine_load_model.restype = ctypes.c_int32
    lib.rpg_engine_unload_model.argtypes = [ctypes.c_int32]
    lib.rpg_engine_unload_model.restype = None

    lib.rpg_engine_session_create.argtypes = [ctypes.c_int32, ctypes.c_char_p]
    lib.rpg_engine_session_create.restype = ctypes.c_int32
    lib.rpg_engine_session_destroy.argtypes = [ctypes.c_int32]
    lib.rpg_engine_session_destroy.restype = None
    lib.rpg_engine_session_reset.argtypes = [ctypes.c_int32]
    lib.rpg_engine_session_reset.restype = ctypes.c_int32

    # 返回值是引擎分配的字符串，restype 用 c_void_p 拿裸指针（不能是 c_char_p，
    # 那样 ctypes 会自己拷一份，我们就没法把它交回 rpg_engine_free 了）。
    lib.rpg_engine_generate.argtypes = [ctypes.c_int32, ctypes.c_char_p]
    lib.rpg_engine_generate.restype = ctypes.c_void_p
    lib.rpg_engine_model_info.argtypes = [ctypes.c_int32]
    lib.rpg_engine_model_info.restype = ctypes.c_void_p
    lib.rpg_engine_free.argtypes = [ctypes.c_void_p]
    lib.rpg_engine_free.restype = None
    return lib


def last_error(lib) -> str:
    e = lib.rpg_engine_last_error()
    return e.decode("utf-8", errors="replace") if e else ""


def take_json(lib, ptr) -> dict | None:
    """接管引擎分配的字符串：拷贝出来 → 立刻 free（绝不泄漏）。"""
    if not ptr:
        return None
    try:
        return json.loads(ctypes.string_at(ptr).decode("utf-8"))
    finally:
        lib.rpg_engine_free(ctypes.c_void_p(ptr))


def gen(lib, session: int, request: dict) -> dict | None:
    payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
    ptr = lib.rpg_engine_generate(ctypes.c_int32(session), payload)
    return take_json(lib, ptr)


# ---------------------------------------------------------------- 工具链参考路

def read_ids(path: str) -> list[int]:
    with open(path, "r", encoding="ascii") as f:
        return [int(x) for x in f.read().strip().replace(",", " ").split()]


def header_abi_symbols() -> set[str]:
    """include/rpg_engine.h 里用 RPG_API 声明的函数名。"""
    path = os.path.join(REPO, "include/rpg_engine.h")
    src = open(path, encoding="utf-8").read()
    return set(re.findall(r"RPG_API\s+[^;]*?\b(rpg_engine_[a-z_]+)\s*\(", src))


def def_export_symbols() -> set[str]:
    """src/rpg_engine.def 的 EXPORTS 段。"""
    path = os.path.join(REPO, "src/rpg_engine.def")
    out: set[str] = set()
    in_exports = False
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s.startswith(";"):
            continue
        if s.upper().startswith("EXPORTS"):
            in_exports = True
            continue
        if in_exports and s:
            out.add(s.split()[0])
    return out


def parser_logs_invalid_calls() -> tuple[bool, str]:
    """静态守卫：解析层必须把"非法工具调用"记进日志，并带上模型原文。

    约定是**响应的 tool_calls 里只给 `arguments: null`，原文 arguments_raw 不进
    响应**（DESIGN §17.5）。那么原文唯一的出口就是日志 —— 这条分支要是被重构掉
    日志，宿主排错时就没有任何线索了，而且不会有别的测试因此变红（日志不影响任何
    返回值）。这种"删了也没人发现"的约定，值得钉一条守卫。
    """
    path = os.path.join(REPO, "src/tool_parser.cc")
    if not os.path.isfile(path):
        return False, f"找不到 {path}"
    src = open(path, encoding="utf-8").read()
    if "RPG_LOG_WARN" not in src:
        return False, "tool_parser.cc 里没有 RPG_LOG_WARN：非法工具调用的日志没了"
    if "arguments_raw.c_str()" not in src:
        return False, "日志里没带 arguments_raw：排错时看不到模型原始输出"
    return True, "解析层在非法工具调用分支打日志，且带上模型原文"


def reference_via_tools(args, case_path: str) -> tuple[str, list]:
    """渲染 → 生成 → 解码 → 解析，全程走命令行工具（无 torch）。"""
    runs = args.runs
    os.makedirs(runs, exist_ok=True)
    prompt_path = os.path.join(runs, "abi_ref.prompt.txt")
    prefix = os.path.join(runs, "abi_ref")

    subprocess.run([args.chatml_check, "--case", case_path, "--out", prompt_path],
                   check=True)
    subprocess.run(
        [args.gen_check, args.model, "--prompt-file", prompt_path,
         "--max-tokens", str(args.max_tokens), "--stop-eos", "--stop", "151645",
         "--no-kv-check", "--out", prefix],
        check=True, capture_output=True)

    ids_line = " ".join(str(i) for i in read_ids(prefix + ".gen.txt"))
    d = subprocess.run([args.tok_check, args.model, "--decode"],
                       input=ids_line.encode("utf-8"),
                       capture_output=True, check=True)
    # **不加 --skip-special**，理由见 DESIGN §16.4。
    text = d.stdout.split(b"\x00")[0].decode("utf-8", errors="replace")
    raw_path = prefix + ".raw.txt"
    with open(raw_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    p = subprocess.run([args.parse_check, "--text-file", raw_path, "--json"],
                       capture_output=True, text=True, encoding="utf-8", check=True)
    parsed = json.loads(p.stdout)
    return parsed["text"], parsed["tool_calls"]


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.path.join(REPO, "build/rpg_engine.dll"))
    ap.add_argument("--model", default=os.path.join(REPO, "models/Qwen3-1.7B-Q8_0.gguf"))
    ap.add_argument("--case", default=os.path.join(REPO, "tests/chatml/dm_toolcall.case.json"))
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--runs", default=os.path.join(REPO, "runs"))
    ap.add_argument("--gen-check", default=os.path.join(REPO, "build/gen_check.exe"))
    ap.add_argument("--chatml-check", default=os.path.join(REPO, "build/chatml_check.exe"))
    ap.add_argument("--parse-check", default=os.path.join(REPO, "build/parse_check.exe"))
    ap.add_argument("--tok-check", default=os.path.join(REPO, "build/tok_check.exe"))
    args = ap.parse_args()

    problems: list[str] = []
    passed = 0

    def check(cond: bool, what: str) -> None:
        nonlocal passed
        if cond:
            passed += 1
            print(f"  通过  {what}")
        else:
            print(f"  失败  {what}")
            problems.append(what)

    print("=" * 72)
    print("M4 验收：C ABI（宿主视角）")
    print("=" * 72)

    if not os.path.exists(args.lib):
        print(f"缺 {args.lib}（先跑 bash build.sh）")
        return 1
    if not os.path.exists(args.model):
        print(f"缺 {args.model}（python scripts/fetch_model.py --preset qwen3-1.7b-q8）")
        return 1

    lib = bind(args.lib)

    # ---------------------------------------------------------- 0. 空态与错误通道
    log("[0] 导出表 / 版本 / 错误通道")
    ver = lib.rpg_engine_version().decode()
    print(f"  版本 {ver}")

    # 三方一致：头文件声明、.def 导出清单、dll 实际导出。
    # 漏了 .def 的那条最容易发生（新增 ABI 函数时只改头文件），而它的症状是
    # 宿主侧"找不到入口点"，在 Unity / Python 里都很难看出原因。
    hdr, dfn = header_abi_symbols(), def_export_symbols()
    check(hdr == dfn and bool(hdr),
          f"头文件声明与 .def 导出清单一致（{len(hdr)} 个函数）")
    if hdr != dfn:
        print(f"  只在头文件：{sorted(hdr - dfn)}")
        print(f"  只在 .def  ：{sorted(dfn - hdr)}")
    missing = [n for n in sorted(hdr) if not hasattr(lib, n)]
    check(not missing, "全部 ABI 函数都能从 dll 解析到")
    if missing:
        print(f"  缺：{missing}")

    check(lib.rpg_engine_model_info(-1) is None, "无效模型句柄 → NULL")
    check(bool(last_error(lib)), "失败时 last_error 非空")

    # 接口形状守卫：两处"引擎只解析、不替宿主做决定"的约定，删了不会让别的检查
    # 变红，所以在这里钉住。
    ok, why = parser_logs_invalid_calls()
    check(ok, why)

    # ---------------------------------------------------------- 1. 加载
    log("[1] 加载模型")
    cfg = json.dumps({"threads": 0, "max_pos": 512, "max_seq": args.max_seq})
    t0 = time.time()
    model = lib.rpg_engine_load_model(args.model.encode("utf-8"), cfg.encode("utf-8"))
    check(model >= 0, f"load_model 返回句柄 {model}")
    if model < 0:
        print(f"  错误：{last_error(lib)}")
        return 1
    log(f"    加载耗时 {time.time() - t0:.1f}s")

    info = take_json(lib, lib.rpg_engine_model_info(ctypes.c_int32(model)))
    check(info is not None, "model_info 返回 JSON")
    if info:
        print(f"  {info['arch']} {info['n_layers']} 层 / hidden {info['hidden']} / "
              f"head_dim {info['head_dim']} / vocab {info['vocab']} / "
              f"tied {info['tied_embeddings']}")
        check(info["n_layers"] == 28 and info["head_dim"] == 128,
              "model_info 的层数与 head_dim 与设计一致")
        check(info["tied_embeddings"] is True, "Qwen3 是 tied embeddings")
        check(info["sessions"] == 0, "新模型下会话数为 0")

    # ---------------------------------------------------------- 2. 参考结果
    log("[2] 工具链参考结果（渲染 → 生成 → 解码 → 解析）")
    ref_text, ref_calls = reference_via_tools(args, args.case)
    print(f"  text={ref_text!r}")
    print(f"  tool_calls={[c['name'] for c in ref_calls]}")
    check(bool(ref_calls) and ref_calls[0]["name"] == "read_scene",
          "参考路径解析出 read_scene")

    # ---------------------------------------------------------- 3. ABI 生成
    case = json.load(open(args.case, encoding="utf-8"))
    request = {
        "messages": case["messages"],
        "tools": case.get("tools"),
        "max_tokens": args.max_tokens,
        "enable_thinking": case.get("enable_thinking", True),
    }

    log("[3] ABI：冷启动生成")
    s_cold = lib.rpg_engine_session_create(ctypes.c_int32(model), None)
    check(s_cold >= 0, f"session_create 返回句柄 {s_cold}")
    if s_cold < 0:
        print(f"  错误：{last_error(lib)}")
        return 1

    r_cold = gen(lib, s_cold, request)
    check(r_cold is not None, "generate 返回 JSON")
    if not r_cold:
        print(f"  错误：{last_error(lib)}")
        return 1
    print(f"  text={r_cold['text']!r}")
    print(f"  tool_calls={[c['name'] for c in r_cold['tool_calls']]} "
          f"finish={r_cold['finish_reason']} usage={r_cold['usage']}")

    check(len(r_cold["tool_calls"]) == 1 and
          r_cold["tool_calls"][0]["name"] == "read_scene",
          "ABI 解析出 read_scene")
    check(r_cold["tool_calls"][0]["arguments"].get("scene") == "神庙大厅",
          "工具参数是 {scene: 神庙大厅}")
    check(r_cold["finish_reason"] == "tool_calls",
          f"finish_reason 是 tool_calls（实为 {r_cold['finish_reason']!r}）")
    check(r_cold["text"] == ref_text, "ABI 的 text 与工具链路径完全相同")
    check(r_cold["tool_calls"] == [
        {"name": c["name"], "arguments": c["arguments"]} for c in ref_calls],
        "ABI 的工具调用与工具链路径完全相同")
    check(r_cold["usage"]["completion_tokens"] > 0, "usage 报告了生成 token 数")

    # 响应形状：**恰好**四键。这是一条防回归守卫 —— 将来若"顺手"把思考内容
    # （reasoning）或参数原文（arguments_raw）加进响应，这里会当场变红。
    # 定案理由：响应会流到玩家侧 UI，DM 的思考里写着剧本打算（防剧透）；参数原文
    # 走日志（见 parser_logs_invalid_calls）。
    check(set(r_cold.keys()) == {"text", "tool_calls", "finish_reason", "usage"},
          f"响应恰好 text/tool_calls/finish_reason/usage 四键"
          f"（实为 {sorted(r_cold.keys())}）")
    check(all(set(c.keys()) == {"name", "arguments"} for c in r_cold["tool_calls"]),
          "每个 tool_call 恰好 name/arguments 两键（思考内容与参数原文不下发）")

    # ---------------------------------------------------------- 4. 缓存复用
    log("[4] 缓存复用：暖缓存必须与冷缓存逐字节一致")
    r_warm_same = gen(lib, s_cold, request)
    check(r_warm_same == r_cold, "全前缀复用（同一请求重发）结果不变")

    # 续一轮：messages 追加一条，渲染后与上一轮的 prompt 只有尾部不同。
    ext = dict(request)
    ext["messages"] = list(case["messages"]) + [
        {"role": "user", "content": "我继续往大厅深处走。"}
    ]
    r_warm_ext = gen(lib, s_cold, ext)
    check(r_warm_ext is not None, "续写一轮：generate 成功")

    s_fresh = lib.rpg_engine_session_create(ctypes.c_int32(model), None)
    r_cold_ext = gen(lib, s_fresh, ext)
    check(r_cold_ext is not None, "冷会话上跑同一份续写请求：generate 成功")
    check(r_warm_ext == r_cold_ext,
          "部分前缀复用的结果与冷启动完全相同（前缀复用逐位等价）")
    if r_warm_ext and r_cold_ext:
        print(f"  暖: {r_warm_ext['text'][:60]!r}…")
        print(f"  冷: {r_cold_ext['text'][:60]!r}…")

    # ---------------------------------------------------------- 5. reset 与生命周期
    log("[5] 会话重置与句柄生命周期")
    check(lib.rpg_engine_session_reset(ctypes.c_int32(s_cold)) == 0, "session_reset 返回 0")
    r_after_reset = gen(lib, s_cold, request)
    check(r_after_reset == r_cold, "reset 后再生成 == 冷启动结果（缓存已清空）")

    check(gen(lib, 999999, request) is None, "无效会话句柄 → NULL")
    check(bool(last_error(lib)), "无效会话句柄的 last_error 非空")
    check(lib.rpg_engine_session_create(ctypes.c_int32(999999), None) < 0,
          "在无效模型上建会话 → 负数")

    check(take_json(lib, lib.rpg_engine_model_info(ctypes.c_int32(model)))["sessions"] == 2,
          "model_info 报告 2 个存活会话")

    lib.rpg_engine_session_destroy(ctypes.c_int32(s_fresh))
    check(gen(lib, s_fresh, request) is None, "destroy 之后旧句柄失效")

    lib.rpg_engine_unload_model(ctypes.c_int32(model))
    check(gen(lib, s_cold, request) is None, "unload_model 连带销毁其会话")
    check(lib.rpg_engine_model_info(ctypes.c_int32(model)) is None,
          "unload 之后旧模型句柄失效")
    check(gen(lib, 0, request) is None, "句柄 0 不是有效句柄")

    # ---------------------------------------------------------- 收尾
    print("=" * 72)
    if problems:
        for p in problems:
            print(f"未通过：{p}")
        print(f"通过 {passed} 项，失败 {len(problems)} 项。")
        return 1
    print(f"通过 {passed} 项，失败 0 项。")
    print("通过：ABI 结构与错误通道正确、结果与工具链一致、前缀复用逐位等价。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
