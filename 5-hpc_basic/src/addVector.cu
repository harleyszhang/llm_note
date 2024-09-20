#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 数组初始化函数
void initialArrays(float *array1, float *array2, int numElements ) {
    for (int i = 0; i < numElements ; i++) {
        array1[i] = (float)rand() / RAND_MAX * 100.0; // 生成 0 到 100 之间的随机浮点数
        array2[i] = (float)rand() / RAND_MAX * 1000.0; // 生成 0 到 1000 之间的随机浮点数
        
    }
}
// 数组相加的函数
__global__ void addArrays(const float *array1, const float *array2, float *result, int numElements ) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x; // 线程索引id
    if(i < numElements ) result[i] = array1[i] + array2[i]; // 加了 if 判断来限制内核不能非法访问全局内存
}

// 打印数组的函数
void printArray(float *array, int numElements ) {
    for (int i = 0; i < numElements ; i++) {
        printf("%f ", *(array + i));
    }
    printf("\n");
}

int main() {
    int numElements  = 1000000;  // 数组的长度
    srand(time(NULL)); // 初始化随机数种子

    // 使用 malloc 动态分配 CPU 内存
    float *h_array1 = (float *)malloc(numElements  * sizeof(float));
    float *h_array2 = (float *)malloc(numElements  * sizeof(float));
    float *h_result = (float *)malloc(numElements  * sizeof(float));
    initialArrays(h_array1, h_array2, numElements ); // 初始化数组

    /* 1. 使用 cudaMalloc 动态分配 GPU 内存 */
    float *d_array1, *d_array2, *d_result; 
    cudaMalloc((void**)&d_array1, sizeof(float) * numElements ); // (void**) 强制类型转换
    cudaMalloc((void**)&d_array2, sizeof(float) * numElements );
    cudaMalloc((void**)&d_result, sizeof(float) * numElements )

    /* 1. 使用 cudaMemcpy 函数把数据从主机内存拷贝到 GPU 的全局内存中 */
    cudaMemcpy(d_array1, h_array1, numElements , cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, h_array2, numElements , cudaMemcpyHostToDevice);
    
    /* 3. 调用 CUDA 内核函数执行向量加法 */
    const int  threadsPerBlock = 256; // 线程块大小
    const int blocksPerGrid = (numElements  +  threadsPerBlock - 1) /  threadsPerBlock; // 网格大小：也是线程块数量
    addArrays<<<blocksPerGrid,  threadsPerBlock>>>(d_array1, d_array2, d_result, numElements );

    /* 4. 将数据从 GPU 复制回主机 */
    cudaMemcpy(h_result, d_result, numElements , cudaMemcpyDeviceToHost);

    // 打印结果数组
    printf("Result array: ");
    printArray(h_result, numElements );

    /* 5. 释放动态分配的 CPU 和 GPU 内存 */
    free(h_array1);
    free(h_array2);
    free(h_result);
    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_result);
    return 0;
}