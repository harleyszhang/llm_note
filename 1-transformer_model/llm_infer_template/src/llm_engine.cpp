
#include <iostream>
#include <random>
#include "llm_engine.h"

class DummyLLMEngine : public LLMEngine {
public:
    bool init_engine(const std::string& engineFile) override {
        // 模拟加载 engine
        // 实际中可能需要创建 TRT engine 或加载 ONNX
        std::cout << "[DummyLLMEngine] Loading engine from: " << engineFile << std::endl;
        // 假装加载成功
        return true;
    }

    std::vector<std::vector<float>> run_forward(const std::vector<int>& inputIds) override {
        // 模拟输出，每个 token 返回 vocab_size=10 的随机 logits
        static constexpr int vocabSize = 10;
        std::vector<std::vector<float>> logitsPerStep;
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // 假设 input 长度 = seqLen，每个位置都生成一个 logits
        for (size_t i = 0; i < inputIds.size(); ++i) {
            std::vector<float> stepLogits(vocabSize);
            for (int j = 0; j < vocabSize; ++j) {
                stepLogits[j] = dist(gen);
            }
            logitsPerStep.push_back(stepLogits);
        }
        return logitsPerStep;
    }
};

std::shared_ptr<LLMEngine> createDummyLLMEngine(const std::string& engineFile) {
    auto engine = std::make_shared<DummyLLMEngine>();
    if (!engine->init_engine(engineFile)) {
        return nullptr;
    }
    return engine;
}