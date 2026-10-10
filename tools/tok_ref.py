#!/usr/bin/env python3
"""tok_ref.py —— 用 HF tokenizers 作为权威参考，对拍 RPGBarEngine 的分词器。

参考侧用 tokenizer.json（Qwen 官方发布的分词器定义），被测侧用 tok_check.exe
（本引擎自研 BBPE 实现）。协议见 tools/tok_check.cc：NUL 分隔的用例。

用法：
    python tools/tok_ref.py <model.gguf> <tokenizer.json> [--exe build/tok_check.exe]
                            [--dump-n 3] [--extra-file corpus.txt]
"""
import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path

SEP = "\x00"


def build_corpus():
    """测试语料。每一类都对应一个容易写错的分支。"""
    cases = []

    # --- 中文 ---
    cases += [
        "你好，世界。",
        "他在酒馆里遇见了神秘的老者。",
        "把门推开，一股霉味扑面而来。",
        "骰子落定：17。掷出了个奇迹。",
        "任务提示：前往北门寻找守夜人。",
        "「你确定要继续吗？」她低声问道。",
        "龙与地下城风格的角色扮演游戏。",
        "一二三四五六七八九十",
        "繁體字測試：龍、鳳、龜、麟",
    ]

    # --- 英文与缩写 ---
    cases += [
        "The quick brown fox jumps over the lazy dog.",
        "it's don't we'll I'm they've he'd",
        "It's a trap! Don't move, I'm serious.",
        "WE'LL SEE ABOUT THAT.",
        "he'd've been here",
        "cannot shouldn't you're",
    ]

    # --- 数字（\p{N} 是单字符分支，靠 BPE 合并回多位数字） ---
    cases += [
        "1234567890",
        "42",
        "3.14159",
        "2026-10-10",
        "HP: 100/100  MP: 25",
        "全角数字：１２３４５",
        "上标²³ 罗马ⅠⅡⅢ",
    ]

    # --- 空白（\s*[\r\n]+ / \s+(?!\S) / \s+ 三个分支的边界） ---
    cases += [
        "a b",
        "a  b",
        "a   b",
        "a\tb",
        "a\nb",
        "a\r\nb",
        "a \n b",
        "  leading",
        "trailing  ",
        "multiple   spaces   here",
        "\n\n\n",
        "line1\nline2\n\nline3",
        "a\u3000b",       # 表意空格
        "a\u00a0b",       # 不换行空格
        " \n \n ",
        "\t\t\n\t",
        "end with newline\n",
    ]

    # --- 标点与符号（ ?[^\s\p{L}\p{N}]+[\r\n]* ） ---
    cases += [
        "...",
        "!!!???",
        "###",
        "a...b",
        "<->",
        "[]{}()",
        "￥100 元",
        "→←↑↓",
        "★☆♦♣",
    ]

    # --- 特殊 token（必须整串命中，不参与 BPE） ---
    cases += [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
        "前缀<|im_start|>中间<|im_end|>后缀",
        "<tool_call>\n{\"name\": \"read_scene\", \"arguments\": {}}\n</tool_call>",
    ]

    # --- 真实的 chat 模板片段（引擎实际要处理的形态） ---
    cases += [
        "<|im_start|>system\n你是跑团主持人。<|im_end|>\n",
        "<|im_start|>user\n我要搜查房间。<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\n# Tools\n\nYou may call one or more functions "
        "to assist with the user query.\n<tools>\n{\"type\":\"function\"}\n"
        "</tools>\n<|im_end|>\n",
        "<|im_start|>assistant\n<think>\n\n</think>\n\n好的。",
    ]

    # --- 混合与长文本 ---
    cases += [
        "玩家 Lym1cPilebro 掷出了 1d20 = 18！DM 说：“你成功了。”",
        "温度 25°C，湿度 60%。",
        "已知 a=1, b=2，求 a+b。",
        "Hello 世界! 这是一个 mixed 句子 with English words.",
        "。".join(["测试"] * 50),
        " ".join(["word"] * 100),
        ("这是一段较长的中文文本，用来验证 BPE 合并在大段输入上的行为是否与官方一致。"
         "文本里混有 English words, numbers 12345 和标点符号！？以及换行。\n"
         "第二段：继续测试。\n") * 3,
    ]

    # --- 其他脚本 ---
    cases += [
        "こんにちは世界",
        "안녕하세요 세계",
        "Привет, мир",
        "Γειά σου Κόσμε",
        "مرحبا بالعالم",
        "नमस्ते दुनिया",
    ]

    # --- emoji / 边缘 ---
    cases += [
        "🎲🐉⚔️",
        "a🎲b",
        "\u200b零宽空格\u200b",
        "組合せ",
        "",
        " ",
        "\n",
    ]

    return cases


def read_corpus_file(path):
    text = Path(path).read_text(encoding="utf-8")
    return text.split("\n")


# 模糊测试用的字符池：刻意塞满"容易切错"的字符 —— 各种空白、缩写用的撇号、
# 数字与字母的边界、全角标点、表意空格、组合符号、代理对之外的补充平面字符。
FUZZ_ALPHABET = (
    "abcXYZ019"
    " \t\n\r"
    "'\".,!?;:-_"
    "，。！？、；：“”（）【】《》…—"
    "的一是不了在人有我他这为之大来以个中上们"
    "\u3000\u00a0\u2002"      # 各种空白
    "０１２９"                 # 全角数字
    "éüñß"                    # 带音调
    "\u0301"                  # 组合尖音符（非 NFC，会被归一化掉）
    "🎲🐉"                    # 补充平面
    "１２"                    # 全角
)


def gen_fuzz(n, seed=20261010):
    import random
    rng = random.Random(seed)
    pool = list(FUZZ_ALPHABET)
    out = []
    for _ in range(n):
        length = rng.randint(0, 60)
        out.append("".join(rng.choice(pool) for _ in range(length)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("tokenizer_json")
    ap.add_argument("--exe", default="build/tok_check.exe")
    ap.add_argument("--dump-n", type=int, default=5)
    ap.add_argument("--extra-file", default=None)
    ap.add_argument("--fuzz", type=int, default=0,
                    help="额外生成 N 条随机字符串一起对拍")
    args = ap.parse_args()

    from tokenizers import Tokenizer as HFTokenizer

    ref = HFTokenizer.from_file(args.tokenizer_json)

    cases = build_corpus()
    if args.extra_file:
        cases += read_corpus_file(args.extra_file)
    if args.fuzz:
        cases += gen_fuzz(args.fuzz)

    # 参考侧会做 NFC 归一化，我们的引擎刻意不做（见 tokenizer.h 的说明）。
    # 所以先把语料统一成 NFC，保证比较的是同一件事。
    cases = [unicodedata.normalize("NFC", c) for c in cases]

    # 参考结果
    expected = [ref.encode(c).ids for c in cases]

    # 被测结果：NUL 分隔喂给 tok_check
    payload = SEP.join(cases).encode("utf-8")
    proc = subprocess.run(
        [args.exe, args.gguf],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        print("tok_check 失败：", proc.stderr.decode("utf-8", "replace"),
              file=sys.stderr)
        return 2

    got = []
    for line in proc.stdout.decode("utf-8").split("\n"):
        if line.strip() == "":
            # 空用例对应空行；最后一个换行不算用例
            continue
        got.append([int(x) for x in line.split()])

    # 用例里可能有空串（tokens 为空），输出是空行，会被上面过滤掉 ——
    # 这里按"非空用例"对齐重算一遍，避免错位。
    idx_nonempty = [i for i, c in enumerate(cases) if c != ""]
    expected_nonempty = [expected[i] for i in idx_nonempty]
    ok = 0
    bad = 0
    shown = 0
    if len(got) != len(expected_nonempty):
        print(f"!! 输出行数不匹配：得到 {len(got)}，期望 "
              f"{len(expected_nonempty)}", file=sys.stderr)
        return 2
    for k, i in enumerate(idx_nonempty):
        if got[k] == expected_nonempty[k]:
            ok += 1
        else:
            bad += 1
            if shown < args.dump_n:
                shown += 1
                print(f"--- 不一致 #{i} ---")
                print(f"  输入: {cases[i]!r}")
                print(f"  参考: {expected_nonempty[k]}")
                print(f"  引擎: {got[k]}")
                # 逐 token 找第一个分歧点
                a, b = expected_nonempty[k], got[k]
                for t in range(min(len(a), len(b))):
                    if a[t] != b[t]:
                        print(f"  首个分歧 @{t}: 参考={a[t]} "
                              f"{ref.decode([a[t]])!r} vs 引擎={b[t]}")
                        break
    # 空用例也要对：参考给空 id 列表，引擎也应为空
    for i, c in enumerate(cases):
        if c == "" and expected[i] != []:
            print(f"注意：空用例参考侧非空 {expected[i]}")

    print(f"\n用例 {len(idx_nonempty)} 条（另有空串用例 "
          f"{len(cases) - len(idx_nonempty)} 条）：一致 {ok}，不一致 {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
