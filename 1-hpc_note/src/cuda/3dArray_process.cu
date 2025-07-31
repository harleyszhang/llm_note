#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

__global__ void increment3DArray(int *array, 
    int width, 
    int height, 
    int depth) 
{    
    // 计算线程的 x y 和 z 方向的全局索引    
    int IDx = threadIdx.x + blockIdx.x * gridDim.x;
    int IDy = threadIdx.y + blockIdx.y * gridDim.y;
    int IDz = threadIdx.z + blockIdx.z * gridDim.z;
    // 确保索引在三维数组的有效范围内

    if (IDx < width && IDy < height && IDz < depth) {
        // 计算数组中对应的全局索引
        int globalIndex = IDz * width * height + IDy * width + IDx;
        // 对应位置的元素加一
        array[globalIndex] = 10;
    }
}

int main() 
{
    // 三维数组的尺寸
    int width = 4;
    int height = 4;
    int depth = 4;

    // 计算数组总大小
    int size = width * height * depth;

    // 在主机端分配内存
    int *h_array = new int[size];

    // 初始化主机端数组
    for (int i = 0; i < size; ++i) {
        h_array[i] = i;
    }

    // 在设备端分配内存
    int *d_array;
    cudaMalloc(&d_array, size * sizeof(int));

    // 从主机复制数据到设备
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    // 定义线程块和网格尺寸
    dim3 blockSize(2, 2, 2); // 每个线程块有8个线程
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
    (height + blockSize.y - 1) / blockSize.y, 
    (depth + blockSize.z - 1) / blockSize.z);

    // 执行核函数
    increment3DArray<<<gridSize, blockSize>>>(d_array, width, height, depth);

    // 将结果从设备复制回主机
    cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    // TODO: 验证结果或进行其他操作
    printf("Welcom comeback to cpu!\n");
     // 初始化主机端数组
     for (int i = 0; i < size; ++i) {
        printf("h_array[%d] is %d.\n", i, h_array[i]);
    }
    // 释放设备内存
    cudaFree(d_array);

    // 释放主机内存
    delete[] h_array;

    return 0;
}