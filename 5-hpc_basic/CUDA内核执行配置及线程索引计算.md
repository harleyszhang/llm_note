- [一 内核执行配置](#一-内核执行配置)
  - [1.1 grids 和 blocks 的维度设置](#11-grids-和-blocks-的维度设置)
  - [1.2 grids 和 blocks 的尺寸设置](#12-grids-和-blocks-的尺寸设置)
- [二 通过全局线程索引访问数据](#二-通过全局线程索引访问数据)
  - [2.1 全局线程索引计算](#21-全局线程索引计算)
  - [2.2 如何通过全局线程索引访问数据](#22-如何通过全局线程索引访问数据)
  - [三 实践](#三-实践)
  - [3.1 一维块: 向量相加](#31-一维块-向量相加)
  - [3.2 二维网格和二维块: 矩阵转置](#32-二维网格和二维块-矩阵转置)
  - [3.3 二维网格和二维块: 矩阵相乘](#33-二维网格和二维块-矩阵相乘)
  - [3.4 二维网格和二维块: 分块矩阵相乘](#34-二维网格和二维块-分块矩阵相乘)
  - [3.5 三维网格和三维块: 三维张量的处理](#35-三维网格和三维块-三维张量的处理)
- [参考资料](#参考资料)


## 一 内核执行配置

### 1.1 grids 和 blocks 的维度设置

前面的内容我们知道可以为一个核函数配置多个线程（每个线程同一时刻只能处理一个数据），而这些线程的组织是通过内核调用函数的**执行配置** <<<grid_size, block_size>> >来决定的，执行配置决定了如何划分并行任务。

这里的 grid_size（网格大小）和 block_size（线程块大小）一般来说是一个 `dim3` 的结构体类型的变量，但也可以是一个普通的整型变量，即只有一个维度，其他维度为 `1`。其维度定义如下所示。
```bash
// 三维 block 和三维 grid
dim3 grid_size(Gx, Gy, Gz);
dim3 block_size(Bx, By, Bz);
// 二维 block 和二维 grid
dim3 grid_size(Gx, Gy);
dim3 block_size(Bx, By);
// 一维 block 和一维 grid
dim3 grid_size(Gx);
dim3 block_size(Bx);
```

三维 block 和三维 grid 的可视化如下图所示:

![dim3_block_grid](../images/cuda_exec_model/dim3_block_grid.png)

`dim3` 可以被看作是一个简单的三维向量，其中每个维度代表一个不同的轴：
1. `x`：第一个维度，通常用于表示线程或线程块的线性索引。
2. `y`：第二个维度，用于表示线程块的二维网格布局中的行数。
3. `z`：第三个维度，用于表示线程块的三维网格布局中的深度

对于不同的数据维度可以设置不同的 grids 和 blocks 维度。

### 1.2 grids 和 blocks 的尺寸设置

理解了执行配置有着不同的维度和维度代表的意义，我们还需要学会如何设置各个维度的尺寸大小。

**在 cuda 中希望尽可能多的线程并行，并保证所有数据被处理**，对于：

1. **一维数据(向量相加)**：假设有一个问题需要处理的数据量为 N，每个线程块可以处理 B 个数据项，如果问题规模是线性的，可以计算网格尺寸为 gridSize.x = (N + B - 1) / B，确保所有数据项都被处理，执行配置为：
```cpp
// Threads per CTA (1024)
int NUM_THREADS = 1 << 10;
// CTAs per Grid
int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
// Launch the kernel on the GPU
vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
```
2. **二维数据（图像处理）**：对于二维数据（如图像、矩阵），如果图像的宽度为 width，高度为 height，每个线程块处理 THREADS * THREADS 个像素，则执行配置为：
```cpp
// Threads per CTA dimension
int THREADS = 32;
BLOCKS_X = (width + THREADS - 1) / THREADS;
BLOCKS_Y = (height + THREADS - 1) / THREADS;
dim3 blockDim(THREADS, THREADS);
dim3 gridDim(BLOCKS_X, BLOCKS_Y);
// Launch kernel
matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, height, width);
```
3. **三维数据（cnn 特征数据）**：对于三维数据，其尺寸为 width = 512, height = 512, depth = 64，如果每个线程块的尺寸为 blockDim(4, 4, 4)：
```cpp
int width = 256;
int height = 128;
int depth = 64;
dim3 blockDim(4, 4, 4);
// 计算网格尺寸. 向上取整，以确保覆盖所有数据
int gridSizeX = (width + blockDim.x - 1) / blockDim.x;
int gridSizeY = (height + blockDim.y - 1) / blockDim.y;
int gridSizeZ = (depth + blockDim.z - 1) / blockDim.z;
// 设置网格尺寸dim3 
dim3 gridDim(gridSizeX, gridSizeY, gridSizeZ);
featuremapProcess<<<blockDim, gridDim>>>(d_a, d_b, d_c, height, width, depth);
```

## 二 通过全局线程索引访问数据

### 2.1 全局线程索引计算

CUDA 中每一个线程都有一个唯一的标识 ID，也叫全局线程索引，kernel 函数内部就是通过这个线程 ID 来访问 1D/2D/3D 张量元素的。要线程的唯一标识符，得先理解 kernel 函数内部一些结构结构体的定义：
1. kernel 执行配置参数的两个变量是赋值给：`gridDim` 和 `blockDim` 内建变量（built-in variable）中。它们都是类型为 `dim3` 结构体变量，具有 x、y、z 这 3 个成员。
	- `gridDim` 表示每个维度上的线程块数量，即每个网格的尺寸。
	- `blockDim` 表示每个维度上的线程数量，即每个线程块的尺寸。
2. 线程id 和 blockid 分别定义为 blockIdx 和 threadIdx，它们都是类型为 `uint3` 的结构体变量，具有 x、y、z 这 3 个成员。其中：
	- blockIdx.x 取值范围是 [0, gridDim.x - 1];
	- blockIdx.y 取值范围是 [0, gridDim.y - 1];
	- blockIdx.z 取值范围是 [0, gridDim.z - 1];
	- threadIdx.x 取值范围是 [0, blockDim.x - 1];
	- 等等

网上有很多资料直接诸如下述这种全局线程索引的计算公式。
```cpp
int blockId = blockIdx.x + blockIdx.y * gridDim.x
			+  blockIdx.z * gridDim.x * gridDim.y;  
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) 
			+ (threadIdx.z * (blockDim.x * blockDim.y))
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
```

但根据我个人理解，理解这种公式实践意义不大，真的是多维配置的情况下，我们应该关注的是如何**分别计算 x,y,z 这三个方向下的全局线程索引**，然后再来计算待访问内存数据地址。**多维的网格和线程块本质上还是一维的，就像多维数组本质上也是一维数组一样**，理解了这个再去计算线程索引就相对容易些了。一维 grid 和 一维的 block 的情况比比较简单，这里忽略。

对于二维 grids 和 blocks，x 和 y 方向上的全局线程索引计算如下:

<img src="../images/cuda_exec_config/2d_grids_blocks.png" width="60%" alt="2d_grids_blocks">

对于三维 grids 和 blocks，x、y 和 z 方向上的全局线程索引计算如下:

<img src="../images/cuda_exec_config/3d_grids_blocks.png" width="60%" alt="3d_grids_blocks">

### 2.2 如何通过全局线程索引访问数据

上一节，我们知道了如何计算不同维度在不同方向上的全局线程索引，那么如何根据这个索引来去访问数据呢，这也是我们编写正确 kernel 函数最难点！以下是在核函数中使用线程索引来访问数据的步骤：
1. 确定数据结构和计算不同方向上的全局线程索引；
2. 考虑数据存储方式：行优先（主流）和列优先；
3. 确保不同方向上的全局线程索引在有效范围内。

另外，我看了很多资料后，得出一个结论，理解这个知识点，一定得是通过实践案例来理解！

### 三 实践

### 3.1 一维块: 向量相加

输入输出都是向量（一维），核函数和核函数配置如下：
```cpp
// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < N) c[tid] = a[tid] + b[tid];
}

int main() {
    // Vectors for holding the host-side (CPU-side) data
    std::vector<int> a;
    a.reserve(N);
    std::vector<int> b;
    b.reserve(N);
    std::vector<int> c;
    c.reserve(N);
  
    int NUM_THREADS = 1 << 10; // 每个网格的 CTA（合作线程数组）
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS; // CTAs per Grid

    // Launch the kernel on the GPU
    // Kernel calls are asynchronous (the CPU program continues execution after
    // call, but no necessarily before the kernel finishes)
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
}
```
### 3.2 二维网格和二维块: 矩阵转置

二维矩阵转置对于二维矩阵（例如图像），如果我们要进行转置操作，核函数可能如下所示：
```cpp
__global__ void transpose(int *src, int *dst, int width, int height) {
    int tx = threadIdx.x
    int ty = threadIdx.y
    int bx = blockIdx.x
    int by = blockIdx.y
    int bw = blockDim.x
    int bh = blockDim.y
    // 计算 ID_y 和 ID_x
    int row = by * bh + ty;
    int col = bx * bw + tx;
    // 判断 x 和 y 方向的全局线程索引都在有效范围内
    if (row < height && col < width) { 
        int srcIndex = row * width + col; // 原来的矩阵元素内存索引
        int dstIndex = col * height + row;
        dst[dstIndex] = src[srcIndex];
    }}

int main() {
    // assume that the matrix is m × n,
    // m is number of rows, n is number of cols
    // input d_Pin has been allocated on and copied to device
    // output d_Pout has been allocated on device

    dim3 DimGrid((n-1)/16 + 1, (m-1)/16+1, 1);
    dim3 DimBlock(16, 16, 1);
    PictureKernel<<<DimGrid,DimBlock>>>(d_in, d_out, m, n);
}
```

### 3.3 二维网格和二维块: 矩阵相乘

正方形二维矩阵相乘，直接使用二维网格和二维块，核函数和核函数配置如下:

```cpp
__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    c[row * N + col] = 0;
    for (int k = 0; k < N; k++) {
        // Accumulate results for a single element
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}
int main() {
    // Threads per CTA dimension
    int THREADS = 32;
    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS; // N is matrix shape
    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);
    // Launch kernel
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N)
}
```

### 3.4 二维网格和二维块: 分块矩阵相乘

基于共享内存实现的了分块缓冲和分块矩阵乘法技术。

```cpp
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// Pull out matrix and shared memory tile size
const int M = 1 << 10;
const int N = 1 << 11;
const int K = 1 << 12;
const int SHMEM_SIZE = 1 << 10;

__global__ void matrixMul(const int *a, const int *b, int *c, int tile_size) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Statically allocated shared memory
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    // Accumulate in temporary variable
    int tmp = 0;

    // Sweep tile across matrix
    for (int i = 0; i < K; i += tile_size) {
    // Load in elements for this tile
    s_a[threadIdx.y * tile_size + threadIdx.x] = a[row * K + (i * tile_size+ threadIdx.x)];
    s_b[threadIdx.y * tile_size + threadIdx.x] = b[(i * tile_size * N + threadIdx.y * N) + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < tile_size; j++) {
        tmp +=
            s_a[threadIdx.y * tile_size + j] * s_b[j * tile_size + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
    }

    // Write back results
    c[row * N + col] = tmp;
}

int main() {
    // Size (in bytes) of matrix
    // MxN = MxK * KxN
    size_t bytes_a = M * K * sizeof(int);
    size_t bytes_b = K * N * sizeof(int);
    size_t bytes_c = M * N * sizeof(int);

    // Host vectors
    vector<int> h_a(M * K);
    vector<int> h_b(K * N);
    vector<int> h_c(M * N);

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides M and N evenly)
    int BLOCKS_X = N / THREADS;
    int BLOCKS_Y = M / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);

    // Launch kernel
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, THREADS);

    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

    // Check result
    verify_result(h_a, h_b, h_c);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```
### 3.5 三维网格和三维块: 三维张量的处理

在 CUDA 中处理三维数组时，核函数需要正确地计算每个线程负责的数据索引。假设我们有一个三维数组，我们想要实现一个简单的核函数，该函数将三维数组中的每个元素加一。首先，定义核函数，计算每个线程的全局索引，并据此访问三维数组。

```cpp
__global__ void increment3DArray(int *array, 
                                int width, 
                                int height, 
                                int depth) 
{    
    // 计算线程的全局索引    
    int idx = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;        
    // 计算三维数组中的行、列和层索引
    int z = idx / (width * height);
    int y = (idx % (width * height)) / width;
    int x = idx % width;
    // 确保索引在三维数组的有效范围内
    if (x < width && y < height && z < depth) {
        // 计算数组中对应的全局索引
        int globalIndex = z * width * height + y * width + x;
        // 对应位置的元素加一
        array[globalIndex] = array[globalIndex] + 1;
    }
}

int main() {    
    // 三维数组的尺寸
    int width = 16;
    int height = 16;
    int depth = 8;
    // 计算数组总大小 
    int size = width * height * depth;
    // 在主机端分配内存
    int *h_array = new int[size];
    /*省略部分代码*/
    dim3 blockSize(2, 2, 2); 
    // 每个线程块有8个线程
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                    (height + blockSize.y - 1) / blockSize.y,
                    (depth + blockSize.z - 1) / blockSize.z);
    // 执行核函数
    increment3DArray<<<gridSize, blockSize>>>(d_array, width, height, depth);

    return 0;}
```

## 参考资料

- [极智开发 | CUDA线程模型与全局索引计算方式](https://mp.weixin.qq.com/s/IyQaarSN6V_tukt6KigkGQ)
- [C++ CUDA 设置线程块尺寸和网格尺寸](https://mp.weixin.qq.com/s/FfMWa94nLFIejilfc3DCxg)
- [C++ CUDA 核函数中如何通过索引访问数据](https://mp.weixin.qq.com/s/VuGarPnZu56hYNlkRP5cyw)
- https://harmanani.github.io/classes/csc447/Notes/Lecture15.pdf