
#include "decoder.h"
#include <vector>
#include <iostream>
#include <algorithm>

class GreedyDecoder: public Decoder {
public:
    std::vector<int> decode(
        std::vector<std::vector<float>> last_logits,
        DecoderStrategy decoder_strategy) override
    {
        std::vector<int> output_ids;
        for (auto & logits: last_logits) {
            auto max_iter = std::max_element(logits.begin(), logits.end());
            int token_id = static_cast<int>(std::distance(logits.begin(), max_iter));
            output_ids.push_back(token_id);
        }
        return output_ids;
    }

    int decode(
        std::vector<float> logits,
        DecoderStrategy decoder_strategy) override
    {  
        auto max_iter = std::max_element(logits.begin(), logits.end());
        int token_id = static_cast<int>(std::distance(logits.begin(), max_iter));        
        return token_id;
    }

};

// 工厂函数：创建指定解码策略的解码器

std::shared_ptr<Decoder> createDecoder(DecoderStrategy decoder_strategy) {
    if (decoder_strategy == DecoderStrategy::GREEDY)
        return std::make_shared<GreedyDecoder>();
    else
        std::cerr << "Unsupported Decoder Strategy!" << std::endl;
        return nullptr;  // 返回空指针
}