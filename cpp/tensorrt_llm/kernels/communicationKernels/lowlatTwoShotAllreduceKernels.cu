/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "lowlatTwoShotAllreduceKernels.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <atomic>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <nvml.h>

namespace tensorrt_llm::kernels
{
namespace
{
__device__ bool isNegZero(float v)
{
    return v == 0.f && signbit(v);
}

__device__ bool isNegZero(__nv_bfloat16 val)
{
    return isNegZero(__bfloat162float(val));
}

template <typename T>
inline __device__ float toFloat(T val)
{
    return val;
}

template <>
inline __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

template <typename T>
inline __device__ T fromFloat(float val)
{
    return val;
}

template <>
inline __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val)
{
    return __float2bfloat16(val);
}

// CUDA barrier based on mcast memory, adapted from pytorch symmetric memory implementation
// Source:
// https://github.com/pytorch/pytorch/blob/c1f51cf2c4fc8259fa48bc506320118e0e907906/torch/csrc/distributed/c10d/CUDASymmetricMemory.cu#L453
template <std::memory_order Sem>
__device__ __forceinline__ uint32_t cas(uint32_t* addr, uint32_t compare, uint32_t val)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ref(*addr);
    ref.compare_exchange_strong(compare, val, cuda::std::memory_order(Sem));
    return compare;
}

__device__ __forceinline__ size_t global_timer_ns()
{
    size_t val;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(val) : : "memory");
    return val;
}

constexpr size_t ns_per_ms = 1e6;

template <std::memory_order Sem>
__device__ __forceinline__ bool try_put_signal(uint32_t* addr, size_t timeout_ms)
{
    size_t deadline = global_timer_ns() + (timeout_ms * ns_per_ms);
    while (cas<Sem>(addr, 0, 1) != 0)
    {
        if (timeout_ms != 0 && global_timer_ns() > deadline)
        {
            return false;
        }
    }
    return true;
}

template <std::memory_order Sem>
__device__ __forceinline__ bool try_wait_signal(uint32_t* addr, size_t timeout_ms)
{
    size_t deadline = global_timer_ns() + (timeout_ms * ns_per_ms);
    while (cas<Sem>(addr, 1, 0) != 1)
    {
        if (timeout_ms != 0 && global_timer_ns() > deadline)
        {
            return false;
        }
    }
    return true;
}

static __global__ void barrier_kernel(uint32_t** signal_pads, int channel, int rank, int world_size, size_t timeout_ms)
{
    if (threadIdx.x < world_size)
    {
        auto target_rank = threadIdx.x;
        if (target_rank == rank)
        {
            return;
        }
        auto put_success = try_put_signal<std::memory_order_release>(
            signal_pads[target_rank] + world_size * channel + rank, timeout_ms);
        if (!put_success)
        {
            printf(
                "[FATAL] CUDASymmetricMemory::barrier: rank %d failed to send signal "
                "to rank %d on channel %d after %lu microseconds\n",
                rank, target_rank, channel, timeout_ms);
            asm volatile("trap;");
        }
        auto wait_success = try_wait_signal<std::memory_order_acquire>(
            signal_pads[rank] + world_size * channel + target_rank, timeout_ms);
        if (!wait_success)
        {
            printf(
                "[FATAL] CUDASymmetricMemory::barrier: rank %d failed to receive signal "
                "from rank %d on channel %d after %lu microseconds\n",
                rank, target_rank, channel, timeout_ms);
            asm volatile("trap;");
        }
    }
}
} // namespace

void mcastGPUBarrier(uint32_t** signal_pads_dev_, int rank, int world_size, int8_t local_device_idx, size_t timeout_ms)
{
    c10::cuda::CUDAGuard guard(local_device_idx);
    // Supports sync up to 128 ranks; Should be enough for now since the largest NVL domain has 72 GPUs
    barrier_kernel<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint32_t**>(signal_pads_dev_), 0, rank, world_size, timeout_ms);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int WORLD_SIZE, typename T>
__global__ void two_shot_all_reduce_kernel(T* output_ptr, T* shard_ptr, T** input_ptrs, T* mcast_ptr,
    size_t input_offset, size_t clear_offset, int num_tokens, int buffer_M, int token_dim, uint32_t** signal_pads,
    int rank, bool wait_for_results)
{

    int elt = blockIdx.y * blockDim.x + threadIdx.x;

    if (elt >= token_dim)
        return;

    int token = blockIdx.x;

    cudaGridDependencySynchronize();

    if (elt < token_dim)
    {
        // Scatter token
        int dest_rank = token % WORLD_SIZE;
        int dest_token_offset = token / WORLD_SIZE;
        T val = shard_ptr[token * token_dim + elt];
        if (isNegZero(val))
            val = fromFloat<T>(0.f);
        input_ptrs[dest_rank][input_offset + dest_token_offset * token_dim * WORLD_SIZE + rank * token_dim + elt] = val;

        // Reduce and broadcast

        int global_token = token * WORLD_SIZE + rank;
        if (global_token < num_tokens)
        {

            float accum = 0.f;

            T values[WORLD_SIZE];

            for (int r = 0; r < WORLD_SIZE; r++)
            {
                input_ptrs[rank][clear_offset + token * token_dim * WORLD_SIZE + r * token_dim + elt]
                    = fromFloat<T>(-0.f);
            }

            while (1)
            {
                bool valid = true;
                for (int r = 0; r < WORLD_SIZE; r++)
                {
                    T volatile* lamport_ptr = (T volatile*) &input_ptrs[rank][input_offset
                        + token * token_dim * WORLD_SIZE + r * token_dim + elt];
                    values[r] = *lamport_ptr;
                    valid &= !isNegZero(values[r]);
                }
                if (valid)
                    break;
            }
            for (int r = 0; r < WORLD_SIZE; r++)
            {
                accum += toFloat<T>(values[r]);
            }
            mcast_ptr[input_offset + buffer_M * token_dim + global_token * token_dim + elt] = fromFloat<T>(accum);
        }
    }
    cudaTriggerProgrammaticLaunchCompletion();

    input_ptrs[rank][clear_offset + buffer_M * token_dim + token * token_dim + elt] = fromFloat<T>(-0.f);

    // Optionally wait for results if the next layer isn't doing the Lamport check
    if (wait_for_results)
    {
        T volatile* lamport_ptr
            = (T volatile*) &input_ptrs[rank][input_offset + buffer_M * token_dim + token * token_dim + elt];
        T val = *lamport_ptr;
        while (isNegZero(val))
            val = *lamport_ptr;

        // Copy if requested
        if (output_ptr)
            output_ptr[token * token_dim + elt] = val;
    }
}

#define LAUNCH_ALL_REDUCE_KERNEL(WORLD_SIZE, T)                                                                        \
    cudaLaunchKernelEx(&config, &two_shot_all_reduce_kernel<WORLD_SIZE, T>, reinterpret_cast<T*>(output.data_ptr()),   \
        reinterpret_cast<T*>(input.data_ptr()), reinterpret_cast<T**>(mcast_mem->getBufferPtrsDev()),                  \
        (T*) mcast_mem->getMulticastPtr(), comm_buffer.storage_offset() + buffer_offset,                               \
        comm_buffer.storage_offset() + clear_offset, num_tokens, buffer_M, token_dim,                                  \
        reinterpret_cast<uint32_t**>(mcast_mem->getSignalPadPtrsDev()), mcast_mem->getRank(), wait_for_results);

at::Tensor twoShotAllReduceDispatch(tensorrt_llm::runtime::McastDeviceMemory* mcast_mem, at::Tensor output,
    at::Tensor input, at::Tensor comm_buffer, int64_t buffer_offset, int64_t clear_offset, bool wait_for_results)
{
    TORCH_CHECK(input.is_contiguous(), "two_shot_all_reduce: input must be contiguous.");
    auto world_size = mcast_mem->getWorldSize();

    int buffer_M = comm_buffer.sizes()[2];
    int num_tokens = input.sizes()[0];
    int token_dim = input.sizes()[1];

    int num_threads = 128;
    int num_blocks = (token_dim + num_threads - 1) / num_threads;

    dim3 grid(num_tokens, num_blocks);
    TLLM_LOG_DEBUG(
        "[TwoShot AllReduce] twoshot allreduce on rank %d, world_size: %d, buffer_M: %d, num_tokens: %d, token_dim: "
        "%d, "
        "buffer_offset: %d, clear_offset: %d, wait_for_results: %d",
        mcast_mem->getRank(), world_size, buffer_M, num_tokens, token_dim, buffer_offset, clear_offset,
        wait_for_results);

    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    config.gridDim = grid;
    config.blockDim = num_threads;
    config.attrs = attrs;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs = 1;

    // TODO: Add more instantiations if desired
    if (input.scalar_type() == torch::kFloat)
    {
        switch (world_size)
        {
        case 2: LAUNCH_ALL_REDUCE_KERNEL(2, float); break;
        case 4: LAUNCH_ALL_REDUCE_KERNEL(4, float); break;
        case 8: LAUNCH_ALL_REDUCE_KERNEL(8, float); break;
        case 16: LAUNCH_ALL_REDUCE_KERNEL(16, float); break;
        case 32: LAUNCH_ALL_REDUCE_KERNEL(32, float); break;
        case 64: LAUNCH_ALL_REDUCE_KERNEL(64, float); break;
        default: assert(false);
        }
    }
    else if (input.scalar_type() == torch::kBFloat16)
    {
        switch (world_size)
        {
        case 2: LAUNCH_ALL_REDUCE_KERNEL(2, __nv_bfloat16); break;
        case 4: LAUNCH_ALL_REDUCE_KERNEL(4, __nv_bfloat16); break;
        case 8: LAUNCH_ALL_REDUCE_KERNEL(8, __nv_bfloat16); break;
        case 16: LAUNCH_ALL_REDUCE_KERNEL(16, __nv_bfloat16); break;
        case 32: LAUNCH_ALL_REDUCE_KERNEL(32, __nv_bfloat16); break;
        case 64: LAUNCH_ALL_REDUCE_KERNEL(64, __nv_bfloat16); break;
        default: assert(false);
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return input;
}

namespace
{

template <typename T_IN>
__device__ void copy_f4(T_IN* dst, T_IN* src)
{
    float4* dst4 = (float4*) dst;
    float4* src4 = (float4*) src;
    __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

template <typename T_IN>
__device__ void copy_f4_ldg(T_IN* dst, T_IN* src)
{
    float4* dst4 = (float4*) dst;
    float4* src4 = (float4*) src;
    *dst4 = *src4;
}

__device__ float4 loadfloat4(void const* ptr)
{

    float return_value[4];

    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(return_value[0]), "=f"(return_value[1]), "=f"(return_value[2]), "=f"(return_value[3])
                 : "l"(ptr));

    return *(float4*) return_value;
}

template <typename T>
inline __device__ T add(T a, T b)
{
    return a + b;
}

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = add<T>(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32)); //__shfl_sync bf16 return float when sm < 80
    return val;
}

inline __device__ float block_reduce_sum(float val)
{
    __shared__ float smem[32];
    int lane_id = threadIdx.x % 32, warp_id = threadIdx.x / 32, warp_num = blockDim.x / 32;
    val = warpReduceSum(val);
    if (lane_id == 0)
    {
        smem[warp_id] = val;
    }
    __syncthreads();
    val = lane_id < warp_num ? smem[lane_id] : 0.f;
    val = warpReduceSum(val);
    return val;
}

template <int DIM, int NUM_THREADS, int NUM_INPUTS, typename T_OUT, typename T_IN>
__global__ void __launch_bounds__(128, 1) RMSNorm(int rank, T_IN* input_plus_residual, T_OUT* output_norm, T_IN* input,
    T_IN* gamma, float epsilon, T_IN* residual, int batch_size)
{

    static bool const LAMPORT = true;

    extern __shared__ uint8_t smem[];

    auto start = clock64();
    int sample = blockIdx.y;

    static int const CGA_THREADS = NUM_THREADS * 1;

    static int const ITERS = DIM / CGA_THREADS;
    float r_input[ITERS];
    float r_gamma[ITERS];

    T_IN* sh_input = (T_IN*) &smem[0];
    T_IN* sh_residual = (T_IN*) &smem[NUM_INPUTS * NUM_THREADS * ITERS * sizeof(T_IN)];
    T_IN* sh_gamma = (T_IN*) &smem[(NUM_INPUTS + 1) * NUM_THREADS * ITERS * sizeof(T_IN)];

    static int const ELTS_PER_THREAD = sizeof(float4) / sizeof(T_IN);

    int offsets[NUM_INPUTS][DIM / (1 * ELTS_PER_THREAD * NUM_THREADS)];

    cudaTriggerProgrammaticLaunchCompletion();

    for (int i = 0; i < NUM_INPUTS; i++)
    {
        for (int j = 0; j < DIM / (1 * ELTS_PER_THREAD * NUM_THREADS); j++)
        {
            int k = j * NUM_THREADS + threadIdx.x;
            offsets[i][j] = i * batch_size * DIM + sample * DIM + blockIdx.x * DIM / 1 + k * ELTS_PER_THREAD;
        }
    }

#pragma unroll
    for (int j = 0; j < DIM / (1 * ELTS_PER_THREAD * NUM_THREADS); j++)
    {
        int i = j * NUM_THREADS + threadIdx.x;
        copy_f4(&sh_residual[i * ELTS_PER_THREAD], &residual[sample * DIM + blockIdx.x * DIM + i * ELTS_PER_THREAD]);
    }

    __pipeline_commit();

#pragma unroll
    for (int j = 0; j < DIM / (ELTS_PER_THREAD * NUM_THREADS); j++)
    {
        int i = j * NUM_THREADS + threadIdx.x;
        copy_f4(&sh_gamma[i * ELTS_PER_THREAD], &gamma[blockIdx.x * DIM + i * ELTS_PER_THREAD]);
    }

    __pipeline_commit();

    // This is where we synchronize
    auto setup_complete = clock64();

    // Load all inputs
    bool valid = false;

    if (!LAMPORT)
        cudaGridDependencySynchronize();

    while (!valid)
    {
        valid = true;
#pragma unroll
        for (int i = 0; i < NUM_INPUTS; i++)
        {
            for (int j = 0; j < DIM / (ELTS_PER_THREAD * NUM_THREADS); j++)
            {
                int k = j * NUM_THREADS + threadIdx.x;

                float4* dst4 = (float4*) &sh_input[i * NUM_THREADS * ITERS + k * ELTS_PER_THREAD];
                float4* src4 = (float4*) &input[offsets[i][j]];

                float4 value = loadfloat4(src4);
                if (LAMPORT)
                {
                    // Assume that the 16B were written atomically, so we only need to check one value
                    T_IN lowest_val = *(T_IN*) &value;
                    valid &= !isNegZero(lowest_val);
                }
                *dst4 = value;
            }
        }
    }

    auto loads_issued = clock64();
    __syncthreads();

    // Perform the initial input reduction
    if (NUM_INPUTS > 0)
    {

        T_IN accum[ELTS_PER_THREAD];
        float4* accum4 = (float4*) &accum;

        for (int j = 0; j < DIM / (ELTS_PER_THREAD * NUM_THREADS); j++)
        {
            int k = j * NUM_THREADS + threadIdx.x;

            *accum4 = *(float4*) &sh_input[k * ELTS_PER_THREAD];

            for (int i = 1; i < NUM_INPUTS; i++)
            {
                float4 data = *(float4*) &sh_input[i * NUM_THREADS * ITERS + k * ELTS_PER_THREAD];
                T_IN* p_d = (T_IN*) &data;
                for (int x = 0; x < ELTS_PER_THREAD; x++)
                {
                    accum[x] += p_d[x];
                }
            }

            // Write back to input 0's staging location.  No sync needed since all data localized to thread.
            *(float4*) &sh_input[k * ELTS_PER_THREAD] = *accum4;
        }
    }

    auto loads_complete = clock64();
    // Wait for residual
    __pipeline_wait_prior(1);
    __syncthreads();

    float thread_sum = 0.f;

#pragma unroll
    for (int io = 0; io < ITERS / ELTS_PER_THREAD; io++)
    {

        float4 inp4 = *(float4*) &sh_input[io * NUM_THREADS * ELTS_PER_THREAD + threadIdx.x * ELTS_PER_THREAD];
        float4 res4 = *(float4*) &sh_residual[io * NUM_THREADS * ELTS_PER_THREAD + threadIdx.x * ELTS_PER_THREAD];

        T_IN* r_inp = (T_IN*) &inp4;
        T_IN* r_res = (T_IN*) &res4;

        float4 out4;

        T_IN* r_out = (T_IN*) &out4;

        for (int ii = 0; ii < ELTS_PER_THREAD; ii++)
        {

            int i = io * ELTS_PER_THREAD + ii;

            T_IN inp_plus_resid = r_inp[ii] + r_res[ii];
            r_out[ii] = inp_plus_resid;
            r_input[i] = toFloat(inp_plus_resid);

            // Accumulate the squares for RMSNorm
            thread_sum += toFloat(inp_plus_resid * inp_plus_resid);
        }

        *(float4*) &input_plus_residual[sample * DIM + blockIdx.x * DIM + io * NUM_THREADS * ELTS_PER_THREAD
            + threadIdx.x * ELTS_PER_THREAD]
            = out4;
    }

    // Wait for Gamma.  There will be a global synchronization as part of the reduction
    __pipeline_wait_prior(0);
    auto reduce_start = clock64();

    float cluster_sum = block_reduce_sum(thread_sum);

    auto reduce_complete = clock64();

    float rcp_rms = rsqrtf(cluster_sum / DIM);

#pragma unroll
    for (int io = 0; io < ITERS / ELTS_PER_THREAD; io++)
    {

        float4 gamma4 = *(float4*) &sh_gamma[io * NUM_THREADS * ELTS_PER_THREAD + threadIdx.x * ELTS_PER_THREAD];
        T_IN* r_g4 = (T_IN*) &gamma4;

        float4 out4;
        // FIXME: this only works if T_OUT == T_IN
        T_OUT* r_out = (T_OUT*) &out4;

        for (int ii = 0; ii < ELTS_PER_THREAD; ii++)
        {
            int i = io * ELTS_PER_THREAD + ii;
            r_gamma[i] = toFloat(r_g4[ii]);
            r_out[ii] = fromFloat<T_OUT>(r_gamma[i] * r_input[i] * rcp_rms);
        }

        *(float4*) &output_norm[sample * DIM + blockIdx.x * DIM + io * NUM_THREADS * ELTS_PER_THREAD
            + threadIdx.x * ELTS_PER_THREAD]
            = out4;
    }

    __syncthreads();
    auto stop = clock64();
}

template <int H_DIM>
void _rmsnorm(int64_t rank, torch::Tensor prenorm_output, torch::Tensor normed_output, torch::Tensor input,
    torch::Tensor gamma, double epsilon, torch::Tensor residual)
{

    // input to rmsnorm is the buffer in the twoshot ar
    // We should use prenorm output to determine the actual used size
    int batch = normed_output.sizes()[0];
    int dim = normed_output.sizes()[1];
    int _rank{static_cast<int>(rank)};
    float _epsilon{static_cast<float>(epsilon)};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    assert(dim == H_DIM);

    static int const NUM_THREADS = 128;
    static int const CGA_THREADS = NUM_THREADS;

    int iters = dim / CGA_THREADS;

    dim3 grid(1, batch, 1);

    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.stream = at::cuda::getCurrentCUDAStream();
    config.gridDim = grid;
    config.blockDim = NUM_THREADS;
    config.attrs = attrs;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs = 1;

    if (normed_output.scalar_type() == torch::kFloat && input.scalar_type() == torch::kFloat
        && gamma.scalar_type() == torch::kFloat)
    {
        size_t shmem_size = 3 * NUM_THREADS * iters * sizeof(float);
        cudaFuncSetAttribute(
            &RMSNorm<H_DIM, NUM_THREADS, 1, float, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        config.dynamicSmemBytes = shmem_size;
        cudaLaunchKernelEx(&config, &RMSNorm<H_DIM, NUM_THREADS, 1, float, float>, _rank,
            prenorm_output.data_ptr<float>(), normed_output.data_ptr<float>(), input.data_ptr<float>(),
            gamma.data_ptr<float>(), _epsilon, residual.data_ptr<float>(), batch);
    }
    else if (normed_output.scalar_type() == torch::kBFloat16 && input.scalar_type() == torch::kBFloat16
        && gamma.scalar_type() == torch::kBFloat16)
    {
        size_t shmem_size = 3 * NUM_THREADS * iters * sizeof(__nv_bfloat16);
        cudaFuncSetAttribute(&RMSNorm<H_DIM, NUM_THREADS, 1, __nv_bfloat16, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        config.dynamicSmemBytes = shmem_size;
        cudaLaunchKernelEx(&config, &RMSNorm<H_DIM, NUM_THREADS, 1, __nv_bfloat16, __nv_bfloat16>, _rank,
            (__nv_bfloat16*) prenorm_output.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*) normed_output.data_ptr<at::BFloat16>(), (__nv_bfloat16*) input.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*) gamma.data_ptr<at::BFloat16>(), _epsilon,
            (__nv_bfloat16*) residual.data_ptr<at::BFloat16>(), batch);
    }
    else
    {
        assert(false);
    }
}
} // namespace

void twoShotRMSNorm(int64_t rank, torch::Tensor prenorm_output, torch::Tensor normed_output, torch::Tensor input,
    torch::Tensor gamma, double epsilon, torch::Tensor residual)
{
    int dim = normed_output.sizes()[1];
    switch (dim)
    {
    case 2048: _rmsnorm<2048>(rank, prenorm_output, normed_output, input, gamma, epsilon, residual); break;
    case 4096: _rmsnorm<4096>(rank, prenorm_output, normed_output, input, gamma, epsilon, residual); break;
    // Llama-4 Hidden Dimension
    case 5120: _rmsnorm<5120>(rank, prenorm_output, normed_output, input, gamma, epsilon, residual); break;
    // DeepSeek Hidden Dimension
    case 7168: _rmsnorm<7168>(rank, prenorm_output, normed_output, input, gamma, epsilon, residual); break;
    case 8192: _rmsnorm<8192>(rank, prenorm_output, normed_output, input, gamma, epsilon, residual); break;
    default: TORCH_CHECK(false, "Unsupported dimension for rmsnorm: ", dim);
    }
}

}; // namespace tensorrt_llm::kernels
