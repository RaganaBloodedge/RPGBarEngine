// RPGBarEngine —— KV Cache
//
// 增量解码的核心：把每层、每个 KV 头、每个位置的 K/V 存下来，生成第 N 个
// token 时不必重算前 N-1 个。没有它，生成 64 个 token 要做 64 次全序列前向，
// 复杂度是 O(N²)。
//
// **这个文件里没有随机数，也不该有。** 见 docs/DESIGN.md §12 #6。

#ifndef RPGBAR_KV_CACHE_H_
#define RPGBAR_KV_CACHE_H_

#include <cstdint>
#include <vector>

#include "model.h"

namespace rpg {

// 布局 [n_layers, n_kv_heads, max_seq, head_dim]，K 与 V 分成两块连续内存。
//
// 为什么是这个维度顺序（层 → KV 头 → 位置 → 维度）：
//   * 层在最外 —— 同一层的 K/V 是连续的一大块，按层做预取 / 分块时不必跨段；
//   * 头在位置之外 —— 一个头的**连续位置**是连续的，增量解码遍历历史
//     （s = 0..pos）走的就是一段顺序内存，而不是每步跳 max_seq 那么远；
//   * head_dim 在最内 —— 一条 head 向量本身连续，`Dot` 直接读，不用跨步。
//
// 内存 = 2 * n_layers * n_kv_heads * max_seq * head_dim * 4 字节。
// 1.7B（28 层 / 8 KV 头 / head_dim 128）在 max_seq = 4096 时约 **940 MB**，
// 4B 在同样长度下约 1.2 GB。所以 max_seq 是构造参数、**刻意不给默认值** ——
// 默认值会让"这一下扣掉 1 GB 内存"变得看不见（1.7B 的权重本身才 1.7 GB）。
//
// 只支持**顺序追加**（写入的 pos 必须从当前长度开始连续）。理由：DM 与小助手
// 各自的会话都是一条直线，没有需要"插空"的场景；而支持任意写入会让 len 的
// 语义变模糊（"已写入数量"还是"最大位置"？），还要额外维护一张空洞表。
class KvCache {
 public:
  KvCache(const ModelConfig& c, int64_t max_seq);

  int64_t len() const { return len_; }
  int64_t max_seq() const { return max_seq_; }
  int32_t n_layers() const { return n_layers_; }
  int32_t n_kv_heads() const { return n_kv_heads_; }
  int32_t head_dim() const { return head_dim_; }

  // 第 layer 层、第 kv_head 个头、位置 pos 处的 K/V（head_dim 个 float）。
  // 带全量越界检查 —— 适合非热路径（prefill 写入、工具探针）。
  float* K(int32_t layer, int32_t kv_head, int64_t pos);
  float* V(int32_t layer, int32_t kv_head, int64_t pos);
  const float* K(int32_t layer, int32_t kv_head, int64_t pos) const;
  const float* V(int32_t layer, int32_t kv_head, int64_t pos) const;

  // 第 layer 层、第 kv_head 个头的**位置 0** 基址。
  // attention 每步要读 2*(pos+1) 次，逐次做越界检查太贵；它只需要检查一次
  // 层号与头号，位置方向由调用方自己保证（pos < len ≤ max_seq）。
  const float* KHead(int32_t layer, int32_t kv_head) const;
  const float* VHead(int32_t layer, int32_t kv_head) const;

  int64_t bytes() const {
    return (static_cast<int64_t>(k_.size()) + static_cast<int64_t>(v_.size())) *
           static_cast<int64_t>(sizeof(float));
  }

  // 长度归零但**保留内存** —— 复用会话（同一房间开新的一局）时不必重新
  // 分配那 940 MB。注意 k_/v_ 里还留着旧值，只是再也不会被读到（len 才是
  // 唯一的可见性判据，attention 只遍历 [0, len)）。
  void Reset() { len_ = 0; }

  // 已写入的位置数。prefill 一次加 seq 个；增量解码每次加 1。
  void SetLen(int64_t n);
  void AddLen(int64_t n) { SetLen(len_ + n); }

 private:
  int64_t Offset(int32_t layer, int32_t kv_head, int64_t pos) const;

  int32_t n_layers_ = 0;
  int32_t n_kv_heads_ = 0;
  int32_t head_dim_ = 0;
  int64_t max_seq_ = 0;
  int64_t len_ = 0;
  std::vector<float> k_;
  std::vector<float> v_;
};

}  // namespace rpg

#endif  // RPGBAR_KV_CACHE_H_
