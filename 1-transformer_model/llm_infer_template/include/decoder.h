#pragma once

#include <vector>

enum class DecoderStrategy {
    GREEDY,
    BEAM_SEARCH,
    TOP_P,
    TOP_K
};

class Decoder {
    public:
        // 定义纯虚函数, 提供接口规范和支持抽象类的实现。
        virtual std::vector<int> decode(
            std::vector<std::vector<float>> logits,
            DecoderStrategy decoder_strategy
        ) = 0;

        virtual int decode(
            std::vector<float> logits,
            DecoderStrategy decoder_strategy
        ) = 0;

        virtual ~Decoder() {};
};

