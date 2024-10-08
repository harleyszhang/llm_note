CUDA 程序的并行层次有两个：kernel 函数内部和 kernel 函数外部的并行。核函数外部的并行主要指：
1. 核函数计算与数据传输之间的并行。
2. 主机计算与数据传输之间的并行。
3. 不同的数据传输（`cudaMemcpy` 函数中的第 4 个参数）之间的并行。
4. 核函数计算与主机计算之间的并行。
5. 不同核函数之间的并行。

值的注意的是，cuda 程序开发中，需要尽量减少主机与设备之间的数据传输及主机中的计算，尽量在设备中完成所有计算。如果做到了这一点，上述前 4 种核函数外部的并行就显得不那么重要了。另外，如果单个核函数的并行规模已经足够大，在同一个设备中同时运行多个核函数也不会带来太多性能提升，上述第五种核函数外部的并行也将不重要。尽管如此，对有些应用，核
函数外部的并行还是比较重要的。为了实现这种并行，需要合理地使用 CUDA 流（CUDA
stream）。

## CUDA Stream

一个 CUDA 流指的是由主机发出的在一个设备中执行的 CUDA 操作（即和 CUDA 有
关的操作，如主机－设备数据传输和核函数执行）序列。一个 CUDA 流中各个操作的次序是由主机控制的，按照主机发布的次序执行。如果是来自于两个不同 CUDA 流中的操作，则不一定按照某个次序执行，而有可能并发或交错地执行。

任何 CUDA 操作都存在于某个 CUDA 流中，要么是默认流（default stream），也称为空流（null stream），要么是明确指定的非空流。在之前的内容中，都没有明确地指定 CUDA 流，是因为那里所有的 CUDA 操作都是**在默认的空流中执行的**。

2，一个 CUDA 流由类型为 cudaStream_t 的变量表示，由如下 CUDA Runtime API 创建：
```bash
__host__ ​cudaError_t cudaStreamCreate(cudaStream_t* pStream);
```
参数:
   - pStream:指向新流标识符的指针

返回值:
   - cudaSuccess, cudaErrorInvalidValue

功能：
   - 在当前调用主机线程的上下文中创建一个新的异步流。如果当前调用主机线程没有上下文，则会选择该设备的主上下文，并将其设置为当前上下文，初始化后在其上创建流。

3，CUDA Stream 由如下函数销毁

```cpp
__host__​__device__​cudaError_t cudaStreamDestroy ( cudaStream_t stream )
```

cudaStreamDestroy 输入参数是 cudaStream_t 类型的变量，即**流标识符**，返回是一个错误代号。

stream 的定义、创建、销毁示例代码如下：

```cpp
cudaStream_t stream1;
cudaStreamCreate(&stream1);
cudaStreamDestroy(stream1);
```