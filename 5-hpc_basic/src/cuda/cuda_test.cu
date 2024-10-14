#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 数组初始化函数
void initialArrays(float *array1, float *array2, int length) {
    for (int i = 0; i < length; i++) {
        array1[i] = (float)rand() / RAND_MAX * 100.0; // 生成 0 到 100 之间的随机浮点数
        array2[i] = (float)rand() / RAND_MAX * 1000.0; // 生成 0 到 1000 之间的随机浮点数
        
    }
}
// 数组相加的函数
void addArrays(float *array1, float *array2, float *result, int length) {
    for (int i = 0; i < length; i++) {
        *(result + i) = *(array1 + i) + *(array2 + i);
    }
}

// 打印数组的函数
void printArray(float *array, int length) {
    for (int i = 0; i < length; i++) {
        printf("%f ", *(array + i));
    }
    printf("\n");
}

// 评估运行时间的函数
void evaluateTime(void (*func)(float *, float *, float *, int), float *array1, float *array2, float *result, int length) {
    clock_t start, end;
    double cpu_time_used;

    start = clock();  // 开始时间
    func(array1, array2, result, length);  // 调用要评估的函数
    end = clock();    // 结束时间

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;  // 计算运行时间
    printf("Time taken: %f seconds\n", cpu_time_used);
}

int main() {
    int length = 10000000;  // 数组的长度
    srand(time(NULL)); // 初始化随机数种子

    // 使用 malloc 动态分配内存
    float *array1 = (float *)malloc(length * sizeof(float));
    float *array2 = (float *)malloc(length * sizeof(float));
    float *result = (float *)malloc(length * sizeof(float));

    // 初始化数组
    initialArrays(array1, array2, length);
    
    // 评估数组相加的运行时间
    evaluateTime(addArrays, array1, array2, result, length);

    // 打印结果数组
    printf("Result array: ");
    // printArray(result, length);

    // 释放动态分配的内存
    free(array1);
    free(array2);
    free(result);

    return 0;
}