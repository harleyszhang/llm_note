#pragma once

#include <vector>
#include <stdio.h>
#include <string>

class Tokenizer {
    public:
        // Tokenizer 词表加载初始化
        virtual bool loadVocabulary(const std::string& vocabFile) = 0;
        
        // 单 sample Tokenizer 编码: string -> vector<int> & 表示这是一个引用类型，vocabFile 并不会拷贝字符串，而是引用传入的实际对象。
        virtual std::vector<int> encode(const std::string& text) const = 0;
        
        // 单 sample Tokenizer 解码
        virtual std::string decode(const std::vector<int>& tokens_id) const = 0;
        
        // 析构函数
        virtual ~Tokenizer() {};
};
