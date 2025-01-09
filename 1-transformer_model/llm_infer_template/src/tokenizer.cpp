#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

class SimpleTokenizer : public Tokenizer {
public:
    bool loadVocabulary(const std::string& vocabFile) override {
        std::ifstream ifs(vocabFile);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open vocab file: " << vocabFile << std::endl;
            return false;
        }
        std::string token;
        int idx = 0;
        while (std::getline(ifs, token)) {
            m_token2id[token] = idx;
            m_id2token[idx] = token;
            idx++;
        }
        ifs.close();
        return true;
    }

分析下述代码，并找出 bug 和 修复，以及给出 c++ 读取模型词表文件 vocab.json 的方法
#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

class SimpleTokenizer : public Tokenizer {
public:
    bool loadVocabulary(const std::string& vocabFile) override {
        std::ifstream ifs(vocabFile);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open vocab file: " << vocabFile << std::endl;
            return false;
        }
        std::string token;
        int idx = 0;
        while (std::getline(ifs, token)) {
            m_token2id[token] = idx;
            m_id2token[idx] = token;
            idx++;
        }
        ifs.close();
        return true;
    }

    // const 修饰符：表明这个函数不会修改所属类的成员变量。
	// override 关键字：表示这个函数覆盖了基类中的虚函数（即父类中已有的 encode 函数）。
    std::vector<int> encode(const std::string& text) const override {
        // 简化：按空格切分
        std::istringstream iss(text);
        std::string word;
        std::vector<int> tokenIds;
        while (iss >> word) {
            auto it = m_token2id.find(word);
            if (it != m_token2id.end()) {
                tokenIds.push_back(it->second);
            } else {
                // OOV token
                tokenIds.push_back(m_token2id.at("[UNK]"));
            }
        }
        return tokenIds;
    }

    std::string decode(const std::vector<int>& tokenIds) const override {
        std::ostringstream oss;
        for (size_t i = 0; i < tokenIds.size(); ++i) {
            auto it = m_id2token.find(tokenIds[i]);
            if (it != m_id2token.end()) {
                oss << it->second << " ";
            } else {
                oss << "[UNK] ";
            }
        }
        return oss.str();
    }

private:
    std::unordered_map<std::string, int> m_token2id;
    std::unordered_map<int, std::string> m_id2token;
};

// 工厂函数：创建一个 SimpleTokenizer 实例
std::shared_ptr<Tokenizer> createSimpleTokenizer(const std::string& vocabFile) {
    auto tokenizer = std::make_shared<SimpleTokenizer>();
    if (!tokenizer->loadVocabulary(vocabFile)) {
        return nullptr;
    }
    return tokenizer;
}

    std::string decode(const std::vector<int>& tokenIds) const override {
        std::ostringstream oss;
        for (size_t i = 0; i < tokenIds.size(); ++i) {
            auto it = m_id2token.find(tokenIds[i]);
            if (it != m_id2token.end()) {
                oss << it->second << " ";
            } else {
                oss << "[UNK] ";
            }
        }
        return oss.str();
    }

private:
    std::unordered_map<std::string, int> m_token2id;
    std::unordered_map<int, std::string> m_id2token;
};

// 工厂函数：创建一个 SimpleTokenizer 实例
std::shared_ptr<Tokenizer> createSimpleTokenizer(const std::string& vocabFile) {
    auto tokenizer = std::make_shared<SimpleTokenizer>();
    if (!tokenizer->loadVocabulary(vocabFile)) {
        return nullptr;
    }
    return tokenizer;
}
