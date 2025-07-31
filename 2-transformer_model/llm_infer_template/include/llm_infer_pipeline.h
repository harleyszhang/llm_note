#pragma once

#include <vector>
#include <string>

#include "tokenizer.h"
#include "llm_engine.h"
#include "decoder.h"

class LLMInferPipeline {
public:
    // 构造函数
    LLMInferPipeline(
        std::shared_ptr<Tokenizer> tokenizer,
        std::shared_ptr<LLMEngine> llm_engine,
        std::shared_ptr<Decoder> decoder
    );

    // 大模型推理非流式接口
    std::string generate(const std::string& ptompts);

private:
    std::shared_ptr<Tokenizer> m_tokenizer;
    std::shared_ptr<Decoder> m_decoder;
    std::shared_ptr<LLMEngine> m_llm_engine;
};

