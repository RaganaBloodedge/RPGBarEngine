// dump_gguf —— 检视一个 GGUF 的结构：元数据、张量清单、量化类型分布，
// 并抽样反量化几个数确认解码链路通。
//
// 用法：
//   dump_gguf model.gguf                # 摘要
//   dump_gguf model.gguf --tensors      # + 张量清单
//   dump_gguf model.gguf --meta         # + 全部元数据（长数组截断）
//   dump_gguf model.gguf --sample <tensor> [n]   # 反量化某张量的前 n 个数

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "gguf.h"
#include "quant.h"

namespace {

void PrintMeta(const rpg::Json& j, const std::string& key, int depth) {
  const std::string pad(static_cast<size_t>(depth) * 2, ' ');
  switch (j.type()) {
    case rpg::Json::Type::kArray: {
      const size_t n = j.size();
      // 长数组（词表）只看头尾，避免刷屏。
      if (n > 8) {
        std::printf("%s%s = [" RPG_SZ " 项] %s ... %s\n", pad.c_str(),
                    key.c_str(), RPG_SZ_ARG(n),
                    j.At(0).IsString() ? j.At(0).AsString().c_str() : "?",
                    j.At(n - 1).IsString() ? j.At(n - 1).AsString().c_str() : "?");
      } else {
        std::printf("%s%s = [", pad.c_str(), key.c_str());
        for (size_t i = 0; i < n; ++i) {
          if (i) std::printf(", ");
          std::printf("%s", j.At(i).IsString() ? j.At(i).AsString().c_str()
                                               : j.At(i).Dump().c_str());
        }
        std::printf("]\n");
      }
      break;
    }
    case rpg::Json::Type::kString: {
      const std::string& s = j.AsString();
      if (s.size() > 200) {
        std::printf("%s%s = <字符串 " RPG_SZ " 字节> %.160s...\n", pad.c_str(),
                    key.c_str(), RPG_SZ_ARG(s.size()), s.c_str());
      } else {
        std::printf("%s%s = %s\n", pad.c_str(), key.c_str(), s.c_str());
      }
      break;
    }
    default:
      std::printf("%s%s = %s\n", pad.c_str(), key.c_str(), j.Dump().c_str());
      break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "用法: dump_gguf <model.gguf> [--tensors] [--meta] "
                 "[--sample <tensor> [n]]\n");
    return 2;
  }
  const std::string path = argv[1];
  bool show_tensors = false;
  bool show_meta = false;
  std::string sample_name;
  int sample_n = 8;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--tensors") == 0) {
      show_tensors = true;
    } else if (std::strcmp(argv[i], "--meta") == 0) {
      show_meta = true;
    } else if (std::strcmp(argv[i], "--sample") == 0 && i + 1 < argc) {
      sample_name = argv[++i];
      if (i + 1 < argc && argv[i + 1][0] != '-') sample_n = std::atoi(argv[++i]);
    }
  }

  try {
    const rpg::GgufFile g = rpg::GgufFile::Open(path);
    std::printf("%s", g.Describe().c_str());

    // 架构参数（按 GGUF 的 <arch>.* 键读，缺了就显示 "-"）
    const char* keys[] = {"block_count",      "embedding_length",
                          "feed_forward_length", "attention.head_count",
                          "attention.head_count_kv", "attention.key_length",
                          "attention.value_length", "context_length",
                          "rope.freq_base",   "attention.layer_norm_rms_epsilon"};
    std::printf("  参数:\n");
    for (const char* k : keys) {
      const std::string full = g.architecture() + "." + k;
      if (!g.HasMeta(full)) {
        std::printf("    %-32s = -\n", k);
      } else {
        const rpg::Json& v = g[full];
        std::printf("    %-32s = %s\n", k, v.Dump().c_str());
      }
    }
    if (g.HasMeta("tie_word_embeddings")) {
      std::printf("    %-32s = %s\n", "tie_word_embeddings",
                  g["tie_word_embeddings"].Dump().c_str());
    }
    std::printf("    %-32s = %s\n", "output.weight 是否存在",
                g.Find("output.weight") != nullptr ? "有" : "无（tied）");

    if (show_meta) {
      std::printf("元数据:\n");
      for (const auto& kv : g.meta().items()) PrintMeta(kv.second, kv.first, 1);
    }

    if (show_tensors) {
      std::printf("张量清单（" RPG_SZ " 个）:\n", RPG_SZ_ARG(g.tensors().size()));
      for (const auto& t : g.tensors()) {
        std::printf("  %-40s dims=[", t.name.c_str());
        for (size_t d = 0; d < t.dims.size(); ++d) {
          std::printf("%s%lld", d ? "," : "", static_cast<long long>(t.dims[d]));
        }
        std::printf("] %s %lld 元素 / %lld 字节\n", rpg::TypeName(t.type),
                    static_cast<long long>(t.n_elements),
                    static_cast<long long>(t.n_bytes));
      }
    }

    if (!sample_name.empty()) {
      const rpg::GgufTensor* t = g.Find(sample_name);
      if (t == nullptr) {
        std::fprintf(stderr, "找不到张量：%s\n", sample_name.c_str());
        return 1;
      }
      const int n = sample_n > 0 ? sample_n : 8;
      std::vector<float> buf(static_cast<size_t>(
          t->n_elements < n ? t->n_elements : n));
      const int64_t elems = static_cast<int64_t>(buf.size());
      rpg::DequantizeRow(t->type, g.TensorBytes(*t), elems, buf.data());
      std::printf("反量化 %s（前 %lld 个，%s）:\n", t->name.c_str(),
                  static_cast<long long>(elems), rpg::TypeName(t->type));
      for (int64_t i = 0; i < elems; ++i) {
        std::printf("  [%lld] %.8g\n", static_cast<long long>(i),
                    static_cast<double>(buf[static_cast<size_t>(i)]));
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误: %s\n", e.what());
    return 1;
  }
}
