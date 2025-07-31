#include <iostream>
#include <memory>
#include <string>

#include "decoder.h"
#include "tokenizer.h"
#include "llm_engine.h"
#include "llm_infer_pipeline.h"

extern std::shared_ptr<Tokenizer> createSimpleTokenizer(const std::string& vocabFile);
extern std::shared_ptr<Decoder> createDecoder(DecoderStrategy strategy);
extern std::shared_ptr<LLMEngine> createDummyLLMEngine(const std::string& engineFile);

int main() {
    // 1. 创建 tokenizer
    auto tokenizer = createSimpleTokenizer("vocab.txt");
    if (tokenizer == nullptr) {
        std::cerr << "Failed to create tokenizer!" << std::endl;
        return -1;
    }

    // 2. 创建 decoder
    auto decoder = createDecoder(DecoderStrategy::GREEDY);
    if (decoder == nullptr) {
        std::cerr << "Failed to create decoder!" << std::endl;
        return -1;
    }

    // 3. 创建 llm_engine
    auto llm_engine = createDummyLLMEngine("model.trt");
    if (llm_engine == nullptr) {
        std::cerr << "Failed to create llm_engine!" << std::endl;
        return -1;
    }

    // 4. 组装推理流程
    LLMInferPipeline pipeline(tokenizer, llm_engine, decoder);

    // 5. 推理示例
    std::string input_text = "How learn c++ programming?";
    std::string output_text = pipeline.generate(input_text);

    std::cout << "Final output: " << output_text << std::endl;
    return 0;

}