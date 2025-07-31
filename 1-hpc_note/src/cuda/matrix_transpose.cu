// 矩阵转置 CUDA 程序
// By: ChatGPT

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>

// CUDA 错误检查宏
#define CHECK_CUDA_ERROR(call) {                                         \
    cudaError_t err = call;                                              \
    if(err != cudaSuccess) {                                             \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
}

// 转置核函数
__global__ void transpose(int *src, int *dst, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;

    // 计算全局线程的行和列索引
    int row = by * bh + ty;
    int col = bx * bw + tx;

    // 检查索引是否在有效范围内
    if (row < height && col < width) { 
        int srcIndex = row * width + col;       // 原矩阵的索引
        int dstIndex = col * height + row;      // 转置后的索引
        dst[dstIndex] = src[srcIndex];
    }
}

// 主函数
int main() {
    // 定义矩阵的行数和列数
    int m = 8; // 行数
    int n = 8; // 列数

    // 计算矩阵大小（字节）
    size_t bytes = m * n * sizeof(int);

    // 初始化主机矩阵
    std::vector<int> h_src(m * n);
    std::vector<int> h_dst(m * n, 0); // 转置后的矩阵
    std::vector<int> h_ref(n * m, 0); // 参考结果

    // 填充原矩阵
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            h_src[i * n + j] = i * n + j;
        }
    }

    // 分配设备内存
    int *d_src, *d_dst;
    CHECK_CUDA_ERROR(cudaMalloc(&d_src, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dst, bytes));

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dst, h_dst.data(), bytes, cudaMemcpyHostToDevice));

    // 定义线程块和网格大小
    dim3 DimBlock(16, 16, 1);
    dim3 DimGrid((n - 1) / DimBlock.x + 1, (m - 1) / DimBlock.y + 1, 1);

    // 启动核函数
    transpose<<<DimGrid, DimBlock>>>(d_src, d_dst, n, m);

    // 检查核函数执行是否有错误
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 复制结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_dst.data(), d_dst, bytes, cudaMemcpyDeviceToHost));

    // 在CPU上计算参考结果
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            h_ref[j * m + i] = h_src[i * n + j];
        }
    }

    // 验证结果
    bool correct = true;
    for(int i = 0; i < n * m; ++i) {
        if(h_dst[i] != h_ref[i]) {
            std::cerr << "Mismatch at index " << i << ": GPU result " << h_dst[i]
                      << " != CPU result " << h_ref[i] << std::endl;
            correct = false;
            break;
        }
    }

    if(correct) {
        std::cout << "Matrix transposition completed successfully!\n";
    } else {
        std::cout << "Matrix transposition failed.\n";
    }

    // 打印原矩阵和转置后的矩阵
    std::cout << "Original Matrix (" << m << "x" << n << "):\n";
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            std::cout << h_src[i * n + j] << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\nTransposed Matrix (" << n << "x" << m << "):\n";
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            std::cout << h_dst[i * m + j] << "\t";
        }
        std::cout << "\n";
    }

    // 释放设备内存
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}