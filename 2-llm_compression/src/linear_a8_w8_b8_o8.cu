#include <torch/torch.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/arch/mma.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/reference/host/gemm.h>
#include <iostream>
#include <stdexcept>

// 假设 CUDA_ARCH 是通过编译器定义的宏
// 例如，-DCUDA_ARCH=800

// 自定义的线性变换函数，使用CUTLASS库进行INT8矩阵乘法
// 输入和权重均为INT8，输出也是INT8
// alpha 和 beta 为FP32，用于后续的缩放操作
torch::Tensor linear_a8_w8_b8_o8(torch::Tensor input,  // INT8输入张量，形状为 (M, K)
                                 torch::Tensor weight, // INT8权重张量，形状为 (N, K)
                                 torch::Tensor bias,   // INT8偏置张量，形状为 (1, N)
                                 float alpha,          // FP32缩放因子 alpha
                                 float beta            // FP32缩放因子 beta
) {
    // 获取输入矩阵的维度
    auto M = input.size(0); // 输入的行数
    auto N = weight.size(0); // 输出的行数（即权重的行数）
    auto K = input.size(1); // 输入的列数（即权重的列数）

    // 定义输出数据类型和计算类型
    using ElementOutput = int8_t;             // 输出元素类型为INT8
    using ElementAccumulator = int32_t;       // 累加器类型为INT32
    using ElementComputeEpilogue = float;     // 后处理计算类型为FP32
    using ElementInputA = int8_t;             // 输入矩阵A的元素类型为INT8
    using ElementInputB = int8_t;             // 输入矩阵B的元素类型为INT8

    // 定义矩阵布局
    // 输入矩阵A采用行主序，权重矩阵B采用列主序，输出矩阵C采用行主序
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // 根据CUDA架构选择不同的GEMM配置
#if CUDA_ARCH >= 800
    // 对于CUDA架构8.0及以上，使用Tensor Cores优化的GEMM配置
    using Gemm = cutlass::gemm::device::Gemm<
        int8_t, LayoutInputA,                // 元素类型和布局：A为INT8，行主序
        int8_t, LayoutInputB,                // 元素类型和布局：B为INT8，列主序
        ElementOutput, LayoutOutput,         // 元素类型和布局：C为INT8，行主序
        ElementAccumulator,                   // 累加器类型为INT32
        cutlass::arch::OpClassTensorOp,      // 操作类为TensorOp（Tensor Cores）
        cutlass::arch::Sm80,                  // NVIDIA GPU架构为SM80（如A100）
        cutlass::gemm::GemmShape<256, 128, 64>, // 矩阵块大小
        cutlass::gemm::GemmShape<64, 64, 64>,   // warp级别的矩阵块大小
        cutlass::gemm::GemmShape<16, 8, 32>,    // instruction级别的矩阵块大小
        cutlass::epilogue::thread::FastLinearCombinationClamp<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>, // 后处理操作：线性组合并夹紧
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,  // 线程块调度
        3 // Pipeline stages
    >;
#elif CUDA_ARCH >= 750
    // 对于CUDA架构7.5及以上，使用Tensor Cores优化的GEMM配置
    using Gemm = cutlass::gemm::device::Gemm<
        int8_t, LayoutInputA,
        int8_t, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75
    >;
#elif CUDA_ARCH >= 700
    // 对于CUDA架构7.0及以上，使用SIMT的GEMM配置
    using Gemm = cutlass::gemm::device::Gemm<
        int8_t, LayoutInputA,
        int8_t, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm70
    >;
#else
    // 对于不支持的CUDA架构，抛出编译错误
    #error "Unsupported CUDA architecture"
#endif

    // 定义矩阵的坐标（行数和列数）
    auto input_size = cutlass::MatrixCoord(M, K);
    auto weight_size = cutlass::MatrixCoord(K, N);
    auto output_size = cutlass::MatrixCoord(M, N);
    
    // 获取输入张量所在设备
    auto device = input.device();
    
    // 创建输出张量，初始化为偏置，并重复M次以匹配输出维度
    auto out = bias.to(device).view({1, -1}).repeat({M, 1}); // 形状为 (M, N)

    // 定义GEMM问题大小
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // 创建TensorRef，指向输入、权重和输出的数据
    // 这些引用将用于CUTLASS GEMM操作
    cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
        input.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
    cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
        weight.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
    cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
        out.data_ptr<int8_t>(), LayoutOutput::packed(output_size));

    // 创建GEMM操作的参数
    typename Gemm::Arguments arguments{
        problem_size, // GEMM问题大小
        input_ref,     // 矩阵A的引用
        weight_ref,    // 矩阵B的引用
        out_ref,       // 矩阵C的引用
        out_ref,       // 矩阵D的引用（用于结果）
        {alpha, beta}, // Epilogue参数：alpha和beta
        1              // 指标（未使用）
    };
    
    // 创建GEMM操作对象
    Gemm gemm_op;

    // 查询所需的额外工作空间大小
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // 分配工作空间内存
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // 检查GEMM操作是否可以实现给定的参数
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS cannot implement GEMM, status: " +
                                 std::to_string((int)status));
    }

    // 初始化GEMM操作，传入参数和工作空间指针
    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS cannot initialize GEMM, status: " +
                                 std::to_string((int)status));
    }

    // 执行GEMM操作
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS cannot run GEMM, status: " +
                                 std::to_string((int)status));
    }

    // 返回输出张量
    return out;
}
