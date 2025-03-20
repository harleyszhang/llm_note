## 一，模型推理流程

自定义的 llama 模型结构定义实现在 `lightllm` 框架中。`LightLLM` 类在 `models/light_llm.py` 文件中定义。

`LightLLM` 调用 `LlamaTpPartModel` 类（lightllm.models.llama.model ），继承 `TpPartBaseModel` 类（`lightllm/common/basemodel/basemodel.py`），主要成员函数有：forward()、\_prefill()、\_decode()、\_context_forward()、\_token_forward()。

`_context_forward()` 函数实现如下所示：

```python
# 变量或函数以单个前导下划线命名，表示它们是内部实现的一部分，不应该被外部直接访问。但这只是一种约定，Python 不会强制限制访问。
def _context_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.context_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.layers_num):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics
```

Lightllm 框架定义的模型结构，主要分为三种 layer，每种 layer 都会定义加载权重函数 `load_hf_weights`。

- LlamaPostLayerInfer，继承 PostLayerInferTpl。
- LlamaPreLayerInfer，继承 PreLayerInferTpl。
- LlamaTransformerLayerInfer，继承 TransformerLayerInferTpl，继承 TransformerLayerInfer。

模型推理的顺序：pre_infer.token_forward() -> self.layers_infer[i].token_forward() ->self.post_infer.token_forward()。

TransformerLayerInferTpl 类的主要函数定义如下：

```python
class TransformerLayerInferTpl(TransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        # need to set by subclass
        self.eps_ = 1e-5 
        self.tp_q_head_num_ = -1
        self.tp_k_head_num_ = -1
        self.tp_v_head_num_ = -1
        self.tp_o_head_num_ = -1
        self.head_dim_ = -1
        self.embed_dim_ = -1
        return
   def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        self._token_attention(input_embdings,
                                    infer_state,
                                    layer_weight=layer_weight)
        self._token_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings
      
    # this impl dont to use @mark_cost_time
   def _token_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        # 如果 world_size_ 大于 1（即在分布式环境中），使用 dist.all_reduce 对输出 o 进行求和操作。
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return
```

## 二，llama 模型

模型结构类定义在 `lightllm/models/llama/model.py` 文件中，类名：`LlamaTpPartModel`，继承自 `TpPartBaseModel`。

成员变量包括以下类对象:

```python
# weight class
pre_and_post_weight_class = LlamaPreAndPostLayerWeight
transformer_weight_class = LlamaTransformerLayerWeight

# infer class
pre_layer_infer_class = LlamaPreLayerInfer
post_layer_infer_class = LlamaPostLayerInfer
transformer_layer_infer_class = LlamaTransformerLayerInfer

# infer state class
infer_state_class = LlamaInferStateInfo
```

自身实现的成员函数有：_init_config(self)、_verify_params(self)、_init_mem_manager(self)、_init_custom(self)、_init_to_get_rotary(self, base=10000)。

### 2.1，llama 模型的 layer

- `pre_layer_infer`：模型输入预处理层，本质是 `Embedding` 层，作用是将输入的离散化表示（例如 token ids）转换为连续的低维向量表示。
- `transformer_layer_infer`：transformer 模型的所有 layer
- `post_layer_infer``: 模型输出后处理层，实际执行的是 Add & Norm -> Linear -> Softmax。

### 2.2，llama 模型的主要 kernel

```python
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd

from lightllm.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd, token_att_fwd_int8k
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2, token_att_fwd2_int8v
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
```

1，torch.embedding
    
```python
input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
```

2，_rms_norm_fwd_fused

```python
def rmsnorm_forward(x, weight, eps):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    # print("BLOCK_SIZE:", BLOCK_SIZE)
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # print(BLOCK_SIZE, num_warps, "block_size, numwarps")
    BLOCK_SIZE = 128 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    num_warps = 8
    # enqueue kernel
    _rms_norm_fwd_fused[(M,)](x_arg, y, weight,
                              x_arg.stride(0), N, eps,
                              BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y
```

3，context_attention_fwd

```python
@torch.no_grad()
def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    BLOCK = 128
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))

    tmp = torch.empty((batch, head, max_input_len + 256), device=q.device, dtype=torch.float32)
    num_warps = 4 if Lk <= 64 else 8
    # num_warps = 4
    _fwd_kernel[grid](
        q, k, v, sm_scale, b_start_loc, b_seq_len,
        tmp,
        o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        tmp.stride(0), tmp.stride(1), tmp.stride(2),
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return
```

4，`4` 种 token_attention 相关 kernel：

```python
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd, token_att_fwd_int8k
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2, token_att_fwd2_int8v
```

根据不同的 triton 版本选择不同的 `token` kernel：

```python
def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo):
    total_token_num = infer_state.total_token_num
    batch_size = infer_state.batch_size
    calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
    att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")

    token_att_fwd(q.view(calcu_shape1),
                    infer_state.mem_manager.key_buffer[self.layer_num_],
                    att_m_tensor,
                    infer_state.b_loc,
                    infer_state.b_start_loc,
                    infer_state.b_seq_len,
                    infer_state.max_len_in_batch)
    
    if triton.__version__ == "2.0.0":
        prob = torch.empty_like(att_m_tensor)
        token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
        att_m_tensor = None

        o_tensor = torch.empty_like(q)

        token_att_fwd2(prob,
                    infer_state.mem_manager.value_buffer[self.layer_num_],
                    o_tensor.view(calcu_shape1),
                    infer_state.b_loc,
                    infer_state.b_start_loc,
                    infer_state.b_seq_len,
                    infer_state.max_len_in_batch)
        prob = None
        return o_tensor
    elif triton.__version__ >= "2.1.0":
        o_tensor = torch.empty_like(q)
        from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import token_softmax_reducev_fwd
        token_softmax_reducev_fwd(att_m_tensor, 
                                    infer_state.mem_manager.value_buffer[self.layer_num_],
                                    o_tensor.view(calcu_shape1),
                                    infer_state.b_loc,
                                    infer_state.b_start_loc,
                                    infer_state.b_seq_len,
                                    infer_state.max_len_in_batch,
                                    infer_state.other_kv_index)
        return o_tensor
    else:
        raise Exception("not support triton version")
```

5，rotary_emb_fwd

```python

@torch.no_grad()
def rotary_emb_fwd(q, cos, sin):
    total_len = q.shape[0]
    head_num = q.shape[1]
    head_dim = q.shape[2]
    assert q.shape[0] == cos.shape[0] and q.shape[0] == sin.shape[0], f"q shape {q.shape} cos shape {cos.shape}"
    BLOCK_HEAD = 4
    BLOCK_SEQ = 32
    grid = (triton.cdiv(head_num, BLOCK_HEAD), triton.cdiv(total_len, BLOCK_SEQ))
    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    # triton kernel 函数
    _rotary_kernel[grid](
        q, cos, sin,
        q.stride(0), q.stride(1), q.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        total_len, head_num,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return
```

## 三，Triton 框架解析

1，核心 `api`

- `tl.load()` : 返回一个数据张量，其值是从内存中由指针定义的位置加载的。
- `tl.store()`: 将数据张量存储到由指针定义的内存位置。

2，数学相关的算子

| Math Ops 接口 |                作用                |
| :-----------: | :--------------------------------: |
|      abs      |      计算 x 的逐元素绝对值。       |
|      exp      |        计算 x 的逐元素指数         |
|      log      |     计算 x 的逐元素自然对数。      |
|     fdiv      |   返回 x 除以 y 的浮点结果张量。   |
|      cos      |       计算 x 的逐元素余弦。        |
|      sin      |       计算 x 的逐元素正弦。        |
|     sqrt      |      计算 x 的逐元素平方根。       |
|    sigmoid    |      计算 x 的按元素 sigmoid       |
|    softmax    |     计算 x 的逐元素 softmax。      |
|    umulhi     | 返回 x 和 y 乘积的最高有效 32 位。 |

For a row-major 2D tensor `X`, the memory location of `X[i, j]` is given b y `&X[i, j] = X + i*stride_xi + j*stride_xj`. Therefore, blocks of pointers for `A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and `B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:

```bash
&A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
&B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
```



3，对于 `softmax` 算子，其 triton 框架版本的 kernel 的实现如下所示:

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, 
                  BLOCK_SIZE: tl.constexpr):
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * input_row_stride
  col_offsets = tl.arange(0, BLOCK_SIZE)
  input_ptrs = row_start_ptr + col_offsets
  # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
  row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  enominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  # Write back output to DRAM
  output_row_start_ptr = output_ptr + row_idx * output_row_stride
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)
```

再创建一个辅助函数，将任何给定输入张量的内核及其（元）参数排入队列。

```python
def softmax(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
```

