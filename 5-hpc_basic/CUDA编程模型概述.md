- [一，CUDA 编程模型概述](#一cuda-编程模型概述)
  - [1.1 CUDA 编程结构](#11-cuda-编程结构)
  - [1.2，内存管理](#12内存管理)
    - [1.2.1 内存层次结构](#121-内存层次结构)
  - [1.3，线程管理](#13线程管理)
    - [1.3.1 线程层次结构](#131-线程层次结构)
- [二，编写启动验证和计时核函数](#二编写启动验证和计时核函数)
  - [2.1，启动核函数的配置](#21启动核函数的配置)
  - [2.2，编写核函数的注意事项](#22编写核函数的注意事项)
  - [2.3，验证核函数](#23验证核函数)
    - [2.4，给核函数计时](#24给核函数计时)
    - [2.4.1，用 CPU 计时器计时](#241用-cpu-计时器计时)
    - [2.4.2 用 nvprof 工具计时](#242-用-nvprof-工具计时)
    - [2.4.3，GPU 的理论最大性能](#243gpu-的理论最大性能)

## 一，CUDA 编程模型概述

CUDA 是 NVIDIA 的一种通用的并行计算平台和编程模型，是在C语言基础上扩展的。借助于CUDA，你可以像编写 C/C++ 语言程序一样实现并行算法。

`CUDA` 编程模型提供了一个计算机架构抽象作为应用程序和其可用硬件之间的桥梁，并利用 GPU 架构的计算能力提供了以下几个特有功能：
1. 一种通过层次结构在 GPU 中组织线程的方法；
2. 一种通过层次结构在 GPU 中访问内存的方法。

### 1.1 CUDA 编程结构

在 CUDA 的架构下，一个程序分为两个部份：`host` 端和 `device` 端。Host 端是指在 CPU 上执行的部份，而 device 端则是 GPU 上执行的部份。Device 端的程序又称为 "kernel"。通常 host 端程序会将数据准备好后，复制到 GPU 内存中，再由 GPU 执行 device 端程序，完成后再由 host 端程序将结果从 GPU 内存中取回。

一个典型的 CUDA 程序实现流程遵循以下模式。
1. 把数据从 CPU 内存拷贝到 GPU 内存。
2. 调用核函数对存储在 GPU 内存中的数据进行操作。
3. 将数据从 GPU 内存传送回到 CPU 内存。

### 1.2，内存管理

CUDA 编程的一个关键是内存管理，，CUDA运行时除了负责分配与释放设备内存，也负责在主机内存和设备内存之间传输数据。表2-1 列出了标准的 C 函数以及相应地针对内存操作的 CUDA C 函数。

![cuda 内存操作函数](../images/cuda_model/cuda_memory_manage.png)

1，`cudaMalloc`: 负责分配 GPU 内存，函数原型为：
```cpp
__host__ ​__device__ ​cudaError_t cudaMalloc ( void** devPtr, size_t size )
```

**参数**：
- `devPtr`：指向已分配设备内存的指针
- `size`：请求的分配内存大小（以字节为单位）

`__device__` 是一个限定词，声明一个函数是：在设备上执行的，和仅可从设备调用。
`__host__` 也是限定词声明的函数是：在主机上执行的，和仅可从主机调用。

`__host__` 限定词也可以用于与 `__device__` 限定词的组合，这种的情况下，这个函数是为主机和设备双方编译。

2，`cudaMemcpy`: 函数负责主机和设备之间的数据传输，其函数原型为：

```cpp
__host__​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
```

此函数从 `src` 指向的源存储区复制一定数量的字节到 `dst` 指向的目标存储区。复制方向由 kind 指定，其中的 kind 有以下几种：

```bash
cudaMemcpyHostToHost = 0
Host -> Host
cudaMemcpyHostToDevice = 1
Host -> Device
cudaMemcpyDeviceToHost = 2
Device -> Host
cudaMemcpyDeviceToDevice = 3
Device -> Device
cudaMemcpyDefault = 4 # 在这种情况下，传输类型是从指针值推断出来的
```

`cudaMemcpy` 函数以同步方式执行，在函数返回以及传输操作完成之前主机应用程序是阻塞的。

#### 1.2.1 内存层次结构

CPU/GPU 内存往往存在一种组织结构（hierarchy）。在这种结构中，含有多种类型的内存，每种内存分别具有不同的容量和延迟（latency，可以理解为处理器等待内存数据的时间）。一般来说，延迟低（速度高）的内存容量小，延迟高（速度低）的内存容量大。

在 GPU 内存层次结构中，最主要的两种内存是全局内存和共享内存。全局类似于 CPU 的系统内存，而共享内存类似于 CPU 的缓存。但 GPU 的共享内存可以由 CUDA C 的内核直接控制。

![cuda 内存层次](../images/cuda_model/mamory_hierarchy.png)

通过两个数组相加的示例来学习如何在主机和设备之间进行数据传输，以及如何使用 `CUDA C` 编程。如下图所示，数组 a 的第一个元素与数组 b 的第一个元素相加，得到的结果作为数
组 c 的第一个元素，重复这个过程直到数组中的所有元素都进行了一次运算。

![两个数组相加](../images/cuda_model/array_add.png)

主机端的纯 C 语言代码如下:

```c
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

int main() {
    int length = 1000000;  // 数组的长度
    srand(time(NULL)); // 初始化随机数种子

    // 使用 malloc 动态分配内存
    float *array1 = (float *)malloc(length * sizeof(float));
    float *array2 = (float *)malloc(length * sizeof(float));
    float *result = (float *)malloc(length * sizeof(float));

    // 初始化数组
    initialArrays(array1, array2, length);
    // 调用函数进行数组相加
    addArrays(array1, array2, result, length);

    // 打印结果数组
    printf("Result array: ");
    printArray(result, length);

    // 释放动态分配的内存
    free(array1);
    free(array2);
    free(result);

    return 0;
}
```

**修改后的 CUDA C 代码如下所示**:

```c
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
__global__ void addArrays(float *array1, float *array2, float *result, int length) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x; // 线程索引id
    if(n < length) result[n] = array1[n] + array2[n]; // 加了 if 判断来限制内核不能非法访问全局内存
}

// 打印数组的函数
void printArray(float *array, int length) {
    for (int i = 0; i < length; i++) {
        printf("%f ", *(array + i));
    }
    printf("\n");
}

int main() {
    int length = 1000000;  // 数组的长度
    srand(time(NULL)); // 初始化随机数种子

    // 使用 malloc 动态分配 CPU 内存
    float *h_array1 = (float *)malloc(length * sizeof(float));
    float *h_array2 = (float *)malloc(length * sizeof(float));
    float *h_result = (float *)malloc(length * sizeof(float));
    initialArrays(h_array1, h_array2, length); // 初始化数组

    // 使用 cudaMalloc 动态分配 GPU 内存
    float *d_array1, *d_array2, *d_result; 
    cudaMalloc((void**)&d_array1, sizeof(float) * length); // (void**) 强制类型转换
    cudaMalloc((void**)&d_array2, sizeof(float) * length);
    cudaMalloc((void**)&d_result, sizeof(float) * length)

    // 使用 cudaMemcpy 函数把数据从主机内存拷贝到 GPU 的全局内存中
    cudaMemcpy(d_array1, h_array1, length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, h_array2, length, cudaMemcpyHostToDevice);
    
    // 调用核函数进行数组相加
    const int block_size = 256; // 线程块大小
    const int grid_size = (length + block_size - 1) / block_size; // 网格大小：也是线程块数量
    addArrays<<<grid_size, block_size>>>(d_array1, d_array2, d_result, length);

    cudaMemcpy(h_result, d_result, length, cudaMemcpyDeviceToHost);

    // 打印结果数组
    printf("Result array: ");
    printArray(h_result, length);

    // 释放动态分配的 CPU 和 GPU 内存
    free(h_array1);
    free(h_array2);
    free(h_result);
    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_result);
    return 0;
}
```

使用 `nvcc` 编译该程序，并指定与 GeForce RTX 2070 对应的计算能力（读者可以选用自己所用 GPU 的计算能力）：

```bash
$ nvcc -arch=sm_75 add1.cu
```

### 1.3，线程管理

当核函数在主机端启动时，它的执行会移动到设备上，此时设备中会产生大量的线程并且每个线程都执行由核函数指定的语句。

由一个内核启动所产生的所有线程统称为一个网格 `grid`，。同一网格中的所有线程共享相同的全局内存空间。一个网格由多个线程块 `block` 构成，一个线程块包含一组线程 `thread`，同一线程块内的线程协作可以通过以下方式来实现:
- 同步
- 共享内存
  
不同块内的线程不能协作。线程依靠以下两个坐标变量来区分彼此。
- `blockIdx`（线程块在线程网格内的索引）
- `threadIdx`（块内的线程索引）

上面这两个变量是核函数中**需要预初始化的内置变量**。当执行一个核函数时，**CUDA 运行时为每个线程分配坐标变量 blockIdx 和 threadIdx**，`dim3` 类型的变量，基于 `uint3` 定义的整数型向量，包含 `3` 个无符号整数的结构，可以通过 `x、y、z` 三个字段来指定。

```cpp
blockIdx.x
blockIdx.y
blockIdx.z
threadIdx.x
threadIdx.y
threadIdx.z
```

在核函数中，每个线程都可以输出自己的**线程索引、块索引**、块维度和网格维度。

**在启动内核之前就需要定义主机端的网格和块变量**，并从主机端通过由 `x、y、z` 三个字段决定的矢量结构来访问它们。注意，**网格大小是块大小的倍数**。

```cpp
int N = 1000;
dim3 block (1024);
dim3 grid ((N + block.x - 1) / block.x)
```

**注意**：使用 `(N + block.x - 1) / block.x` 而不是 `N / block.x` 的原因是为了**处理未对齐的元素和向上取整**。

对于一个给定的数据大小，确定网格和块尺寸的一般步骤为：
1. 确定块的大小
2. 在已知数据大小和块大小的基础上计算网格维度

而要确定块尺寸，通常需要考虑：
- 内核的性能特性
- GPU资源的限制

#### 1.3.1 线程层次结构

内核启动的网格和线程块的维度会影响性能（优化途径），另外网格和线程块的维度是存在限制的，线程块的主要限制因素就是可利用的计算资源，如寄存器，共享内存等。

网格和线程块从逻辑上代表了一个核函数的线程层次结构。

## 二，编写启动验证和计时核函数

### 2.1，启动核函数的配置

核函数的调用形式如下：
```cpp
kernel_name<<<gird, block>>>(argument list); 
```

`<<<>>>` 运算符内是核函数的执行配置，利用执行配置可以指定线程在 GPU 上调度运行的方式。执行配置：
- 第一个值是**网格 grid 维度**，也就是启动块的数目。
- 第二个值是**线程块 block 维度**，也就是每个块中线程的数目。

这也意味着通过指定网格和块的维度，我们可以配置：
- **内核函数中线程的数目**；
- **内核中使用的线程布局**。

同一个块中的线程之间可以相互协作，不同块内的线程不能协作。

假设有 `32` 个数据元素用于计算，每 `8` 个元素一个块，需要启动 `4`个块，配置代码和线程布局图如下所示:

```cpp
kernel_name<<<4, 8>>>(argument list); 
```

![线程布局可视化](../images/cuda_model/thread_layout.png)

核函数的调用与主机线程是异步的。核函数调用结束后，控制权立刻返回给主机端。我们可以**调用以下函数来强制主机端程序等待所有的核函数执行结束**：
```cpp
cudaError_t cudaDeviceSynchonize(void);
```

有些 CUDA runtime API 在主机和设备之间是隐式同步的，如 `cudaMemcpy` 函数在主机和设备之间拷贝数据时，主机端隐式同步，即主机端程序必须等待数据拷贝完成后才能继续执行程序。

### 2.2，编写核函数的注意事项

下表2-2总结了 CUDA C 程序中的**函数类型限定符**。函数类型限定符指定一个函数在主机上执行还是在设备上执行，以及可被主机调用还是被设备调用。

![函数类型限定符](../images/cuda_model/function_type_qualifiers.png)

注意，`__device__` 和 `__host__` 限定符可以一齐使用，这样函数可以同时在主机和设备端进行编译。

核函数是在设备端执行的代码，核函数具有以下限制:
- 只能访问设备内存
- 必须具有 `void` 返回类型
- 不支持可变数量的参数
- 不支持静态变量
- 显示异步行为

### 2.3，验证核函数

如何判断核函数是否正确运行呢？

首先可以通过在核函数中添加 `printf` 函数打印相关信息，其次是将执行参数设置为 `<<<1，1>>>`，强制用一个块和一个线程执行核函数，模拟串行执行程序，方便调试和验证结果是否正确。

#### 2.4，给核函数计时

#### 2.4.1，用 CPU 计时器计时

最简单的方法是在主机端使用一个 CPU 或 GPU 计时器来计算内核的执行时间。

1，借助 `time.h` 库实现计时功能:

```cpp
int iLen = 1024;
dim3 block (iLen);
dim3 grid ((N + block - 1) / block)

clock_t start, end;
double cpu_time_used;
start = clock();  // 开始时间

addArrays<<<grid, block>>>(d_array1, d_array2, d_result);
cudaDeviceSynchronize() // 等待所有的 GPU 线程运行结束

end = clock();    // 结束时间
cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;  // 计算运行时间

printf("sumArraysOnGPU <<<%d, %d>>> Time taken: %f seconds\n", grid.x, block.x, cpu_time_used);
```

#### 2.4.2 用 nvprof 工具计时

自CUDA 5.0以来，NVIDIA 提供了一个名为 `nvprof` 的命令行分析工具，可以帮助从应用程序的 CPU 和 GPU 活动情况中获取时间线信息，其包括内核执行、内存传输以及 CUDA API 的调用。其用法如下:

```bash
nvprof [nvprof_args] <application> [application_args]
```

1，基本使用：要分析一个 CUDA 程序，你可以直接用 nvprof 来运行它：
```bash
nvprof ./your_cuda_program
```

2，分析特定内核：如果你只对特定的内核感兴趣，可以使用 --kernels 选项来指定：

```bash
nvprof --kernels kernel_name ./your_cuda_program
```

3，查看内存传输情况：要查看内存传输（数据从主机到设备或设备到主机）的统计信息：
```bash
nvprof --print-gpu-trace ./your_cuda_program
```

4，结合其他工具：nvprof 可以与 nvvp（NVIDIA Visual Profiler）配合使用，生成 .nvprof 文件并用图形化工具进一步分析。
```bash
nvprof --output profile.nvprof ./your_cuda_program
nvvp profile.nvprof
```

注意：NVIDIA 已经建议使用 Nsight Compute 和 Nsight Systems 作为 nvprof 的替代工具，这些工具提供了更强大和全面的性能分析功能。

#### 2.4.3，GPU 的理论最大性能

应用程序的测量值与理论峰值进行比较，可以判定你的应用程序的性能是**受限于算法还是受限于内存带宽的**。以 Tesla K10 为例，可以得到理论上的**操作：字节比率**（`ops:byte ratio`）：

$$\frac{\text{算力}}{\text{内存带宽}} = \frac{4.58\ TFLOPS}{320\ GB/s} = 13.6$$

也就是 13.6个指令：1个字节。对于 Tesla K10 而言，如果你的应用程序每访问一个字节所产生的指令数多于 13.6，那么你的应用程序受算法性能限制，GPU 将被充分利用；反之则受访存限制，GPU 没有被充分应用。



