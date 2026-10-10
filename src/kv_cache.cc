#include "kv_cache.h"

#include "common.h"

namespace rpg {

KvCache::KvCache(const ModelConfig& c, int64_t max_seq)
    : n_layers_(c.n_layers),
      n_kv_heads_(c.n_kv_heads),
      head_dim_(c.head_dim),
      max_seq_(max_seq) {
  RPG_CHECK(n_layers_ > 0 && n_kv_heads_ > 0 && head_dim_ > 0,
            "模型配置不完整，建不了 KV Cache（层 %d / KV 头 %d / head_dim %d）",
            n_layers_, n_kv_heads_, head_dim_);
  RPG_CHECK(max_seq_ > 0, "max_seq 必须为正，给了 %lld",
            static_cast<long long>(max_seq_));

  const size_t per = static_cast<size_t>(n_layers_) *
                     static_cast<size_t>(n_kv_heads_) *
                     static_cast<size_t>(max_seq_) *
                     static_cast<size_t>(head_dim_);
  // assign 而不是 resize：要的是**已清零**的内存。万一有地方读到了没写过的
  // 槽位，读到的是 0（一个显眼的错值）而不是复用来的垃圾（一个看起来像
  // "模型抽风"的错值 —— 那种最难查）。代价是首次分配要 memset 一遍。
  k_.assign(per, 0.0f);
  v_.assign(per, 0.0f);
}

int64_t KvCache::Offset(int32_t layer, int32_t kv_head, int64_t pos) const {
  return ((static_cast<int64_t>(layer) * n_kv_heads_ + kv_head) * max_seq_ +
          pos) *
         head_dim_;
}

float* KvCache::K(int32_t layer, int32_t kv_head, int64_t pos) {
  RPG_CHECK(layer >= 0 && layer < n_layers_, "层号 %d 越界（共 %d 层）", layer,
            n_layers_);
  RPG_CHECK(kv_head >= 0 && kv_head < n_kv_heads_, "KV 头号 %d 越界（共 %d 个）",
            kv_head, n_kv_heads_);
  RPG_CHECK(pos >= 0 && pos < max_seq_, "位置 %lld 越界（容量 %lld）",
            static_cast<long long>(pos), static_cast<long long>(max_seq_));
  return k_.data() + Offset(layer, kv_head, pos);
}

float* KvCache::V(int32_t layer, int32_t kv_head, int64_t pos) {
  RPG_CHECK(layer >= 0 && layer < n_layers_, "层号 %d 越界（共 %d 层）", layer,
            n_layers_);
  RPG_CHECK(kv_head >= 0 && kv_head < n_kv_heads_, "KV 头号 %d 越界（共 %d 个）",
            kv_head, n_kv_heads_);
  RPG_CHECK(pos >= 0 && pos < max_seq_, "位置 %lld 越界（容量 %lld）",
            static_cast<long long>(pos), static_cast<long long>(max_seq_));
  return v_.data() + Offset(layer, kv_head, pos);
}

const float* KvCache::K(int32_t layer, int32_t kv_head, int64_t pos) const {
  return const_cast<KvCache*>(this)->K(layer, kv_head, pos);
}

const float* KvCache::V(int32_t layer, int32_t kv_head, int64_t pos) const {
  return const_cast<KvCache*>(this)->V(layer, kv_head, pos);
}

const float* KvCache::KHead(int32_t layer, int32_t kv_head) const {
  RPG_CHECK(layer >= 0 && layer < n_layers_, "层号 %d 越界（共 %d 层）", layer,
            n_layers_);
  RPG_CHECK(kv_head >= 0 && kv_head < n_kv_heads_, "KV 头号 %d 越界（共 %d 个）",
            kv_head, n_kv_heads_);
  return k_.data() + Offset(layer, kv_head, 0);
}

const float* KvCache::VHead(int32_t layer, int32_t kv_head) const {
  RPG_CHECK(layer >= 0 && layer < n_layers_, "层号 %d 越界（共 %d 层）", layer,
            n_layers_);
  RPG_CHECK(kv_head >= 0 && kv_head < n_kv_heads_, "KV 头号 %d 越界（共 %d 个）",
            kv_head, n_kv_heads_);
  return v_.data() + Offset(layer, kv_head, 0);
}

void KvCache::SetLen(int64_t n) {
  RPG_CHECK(n >= 0 && n <= max_seq_, "长度 %lld 越界（容量 %lld）",
            static_cast<long long>(n), static_cast<long long>(max_seq_));
  len_ = n;
}

}  // namespace rpg
