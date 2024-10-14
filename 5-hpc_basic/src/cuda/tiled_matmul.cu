// This program computes matrix multiplication using shared memory tiling
// By: ChatGPT

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

using std::cout;
using std::generate;
using std::vector;

// 宏定义tile size
#define TILE_SIZE 4

// 矩阵大小
const int N = 8;

// CUDA 错误检查宏
#define CHECK_CUDA_ERROR(call) {                                         \
    cudaError_t err = call;                                              \
    if(err != cudaSuccess) {                                             \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
}

// 矩阵乘法核函数
__global__ void matrixMul(const int *a, const int *b, int *c) {
    // 计算每个线程的全局行和列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 声明共享内存
    __shared__ int s_a[TILE_SIZE][TILE_SIZE];
    __shared__ int s_b[TILE_SIZE][TILE_SIZE];

    // 临时变量用于累加结果
    int tmp = 0;

    // 按tile划分矩阵
    for (int i = 0; i < N; i += TILE_SIZE) {
        // 加载A矩阵的tile到共享内存
        s_a[threadIdx.y][threadIdx.x] = a[row * N + (i + threadIdx.x)];
        // 加载B矩阵的tile到共享内存
        s_b[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * N + col];

        // 同步线程，确保所有数据加载完毕
        __syncthreads();

        // 进行乘法累加
        for (int j = 0; j < TILE_SIZE; j++) {
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        // 同步线程，确保所有线程完成计算
        __syncthreads();
    }

    // 将结果写回全局内存
    c[row * N + col] = tmp;
}

// 在CPU上验证结果
void verify_result(const vector<int> &a, const vector<int> &b, const vector<int> &c) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }
            if (tmp != c[i * N + j]) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << "CPU = " << tmp << ", GPU = " << c[i * N + j] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}

int main() {
    // 矩阵大小（字节）
    size_t bytes = N * N * sizeof(int);

    // 主机端矩阵
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N, 0);

    // 初始化矩阵A和B
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    // 打印初始化的矩阵（可选）
    
    cout << "Matrix A:\n";
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            cout << h_a[i * N + j] << "\t";
        }
        cout << "\n";
    }

    cout << "\nMatrix B:\n";
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            cout << h_b[i * N + j] << "\t";
        }
        cout << "\n";
    }
    

    // 设备端指针
    int *d_a, *d_b, *d_c;

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, bytes));

    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // 定义每个线程块的线程数
    dim3 threads(TILE_SIZE, TILE_SIZE);
    // 定义网格的线程块数
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // 启动核函数
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

    // 检查核函数是否有错误
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 将结果从设备复制回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // 验证结果
    verify_result(h_a, h_b, h_c);

    cout << "COMPLETED SUCCESSFULLY\n";

    // 可选：打印结果矩阵
    
    cout << "\nMatrix C (Result):\n";
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            cout << h_c[i * N + j] << "\t";
        }
        cout << "\n";
    }
    

    // 释放设备内存
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    return 0;
}