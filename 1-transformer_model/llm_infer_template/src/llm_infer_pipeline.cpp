#include <vector>
#include <string>
#include <iostream>

#include "llm_infer_pipeline.h"

// 通过构造函数初始化列表初始化参数
LLMInferPipeline::LLMInferPipeline(
    std::shared_ptr<Tokenizer> tokenizer,
    std::shared_ptr<LLMEngine> llm_engine,
    std::shared_ptr<Decoder> decoder
) : m_tokenizer(tokenizer), m_llm_engine(llm_engine), m_decoder(decoder)
{
    if (!m_tokenizer || !m_llm_engine || !m_decoder) {
        throw std::invalid_argument("[LLMInferPipeline] One or more components are null.");
    }
}

// 单 sample 非流式推理生成接口
std::string LLMInferPipeline::generate(const std::string& ptompts)
{   
    std::vector<int> output_ids;
    // 1. 前处理：分词
    std::vector<int> input_ids = m_tokenizer->encode(ptompts);
    for (int i = 0; i<input_ids.size(); i++){
        // 2. 模型推理，没有使用 kv cache 优化
        std::vector<std::vector<float>> logits = m_llm_engine->run_forward(input_ids); // input_ids size: [seq_len, vocab_size]
        std::vector<float> last_logits = logits.back(); // back() 返回容器的最后一个元素, size: [vocab_size]
        // 3. 后处理：解码采样
        int next_token_id = m_decoder->decode(last_logits, DecoderStrategy::GREEDY);
        input_ids.push_back(next_token_id);
        output_ids.push_back(next_token_id);
    }

    // 4. tokens_id --> string
    std::string output_text = m_tokenizer->decode(output_ids);
    std::cout << "[LLMInferPipeline] Postprocessing done." << std::endl;

    return output_text;
}