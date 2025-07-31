#include <iostream>
#include <vector>

// 简化的 Tensor 类示例
class Tensor {
public:
    Tensor() : data_(std::vector<float>(4, 0.0f)) {}

    // 常量版本
    template <typename T>
    const T* ptr() const {
        if (data_.empty()) {
            return nullptr;
        }
        // std::vector 的 .data() 成员函数返回一个指向 vector 中第一个元素的指针。
        return reinterpret_cast<const T*>(data_.data());
    }

    // 非常量版本
    template <typename T>
    T* ptr() {
        if (data_.empty()) {
            return nullptr;
        }
        return reinterpret_cast<T*>(data_.data());
    }

private:
    std::vector<float> data_;
};

void readTensor(const Tensor& tensor) {
    const float* data = tensor.ptr<float>();
    if (data) {
        std::cout << "Read Tensor data: ";
        for (int i = 0; i < 4; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
}

void writeTensor(Tensor& tensor) {
    float* data = tensor.ptr<float>();
    if (data) {
        for (int i = 0; i < 4; ++i) {
            data[i] = static_cast<float>(i + 1);
        }
        std::cout << "Modified Tensor data." << std::endl;
    }
}

int main() {
    Tensor tensor;

    // 非常量对象，调用非常量版本
    writeTensor(tensor);
    readTensor(tensor);

    const Tensor const_tensor;
    // 常量对象，调用常量版本
    readTensor(const_tensor);

    return 0;
}