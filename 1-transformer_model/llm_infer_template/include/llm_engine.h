#pragma once

#include <vector>
#include <string>

class LLMEngine {
public:
    // 大模型推理引擎初始化
    virtual bool init_engine(const std::string& engine_file) = 0;

    // 执行推理，输入是 tokenIDs，输出是该序列各步的 logits
    virtual std::vector<std::vector<float>> run_forward(const std::vector<int>& input_ids) = 0;

    // 析构函数
    virtual ~LLMEngine() {};
};