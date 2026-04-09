import torch
import triton
import triton.language as tl
import time

# Triton 矩阵乘法内核 (优化平铺版本)
@triton.jit
def matmul_kernel(
    # 指向输入输出矩阵数据块的指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 步长（相邻行/列之间的内存跨度）
    stride_am, stride_ak,  # A 的步长 (M, K)
    stride_bk, stride_bn,  # B 的步长 (K, N)
    stride_cm, stride_cn,  # C 的步长 (M, N)
    # 平铺大小 (Tile Size) - 编译器常量，用于优化
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    计算 C = A @ B 的 Triton 内核。
    使用二维网格布局和双重循环平铺以优化 CPU 缓存利用率。
    """
    # 当前程序负责计算输出矩阵 C 中的哪个分块 (pid_m, pid_n)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 计算当前分块在 A 和 B 中的起始内存偏移
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 初始化累加器（存放当前分块的部分和）
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 双重循环平铺: 沿 K 维度分块加载和计算，提高缓存命中率
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 从 A 加载当前块 (BLOCK_SIZE_M x BLOCK_SIZE_K)
        a = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        # 从 B 加载当前块 (BLOCK_SIZE_K x BLOCK_SIZE_N)
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        # 矩阵乘累加: [BLOCK_SIZE_M, BLOCK_SIZE_N] += [BLOCK_SIZE_M, BLOCK_SIZE_K] @ [BLOCK_SIZE_K, BLOCK_SIZE_N]
        accumulator += tl.dot(a, b)
        # 移动到 K 维度的下一个分块
        offs_k += BLOCK_SIZE_K

    # 将最终结果写回 C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a, b):
    """
    使用 Triton 内核执行矩阵乘法 a @ b。
    参数:
        a, b: 2D PyTorch 张量 (CPU)
    返回:
        c: 2D PyTorch 张量 (CPU), 结果为 a @ b
    """
    # 检查输入
    assert a.dim() == 2 and b.dim() == 2, "输入必须是二维张量"
    assert a.shape[1] == b.shape[0], f"矩阵维度不匹配: {a.shape} @ {b.shape}"
    
    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check, "内部维度 K 必须相等"
    
    # 预分配输出张量
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    
    # 定义网格和分块大小 (可根据硬件微调)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32  # 平铺参数
    
    # 计算二维网格大小 (每个输出分块一个线程)
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # 启动内核
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c

def test_correctness_and_performance():
    """
    主测试函数: 比较 Triton 与 PyTorch 实现的正确性和性能。
    """
    print("🔍 开始 Triton-CPU 矩阵乘法测试")
    print("-" * 50)
    
    # 测试用例 (可根据需要调整大小)
    test_cases = [
        (128, 256, 512),   # 小规模
        (512, 1024, 2048), # 中等规模
        (1024, 2048, 4096),# 大规模
    ]
    
    for idx, (M, N, K) in enumerate(test_cases):
        print(f"\n测试用例 {idx+1}: A[{M}x{K}] @ B[{K}x{N}] = C[{M}x{N}]")
        
        # 生成随机测试数据 (CPU)
        torch.manual_seed(42)
        a = torch.randn((M, K), device='cpu', dtype=torch.float32)
        b = torch.randn((K, N), device='cpu', dtype=torch.float32)
        
        # **1. 使用 Triton 计算**
        start_time = time.perf_counter()
        c_triton = triton_matmul(a, b)
        triton_time = time.perf_counter() - start_time
        
        # **2. 使用 PyTorch (参考实现) 计算**
        start_time = time.perf_counter()
        c_ref = torch.matmul(a, b)  # 或 a @ b
        torch_time = time.perf_counter() - start_time
        
        # **3. 正确性验证: 计算最大绝对误差**
        max_abs_error = torch.max(torch.abs(c_triton - c_ref)).item()
        # 相对误差容限 (考虑浮点计算误差)
        rtol, atol = 1e-3, 1e-3
        is_correct = torch.allclose(c_triton, c_ref, rtol=rtol, atol=atol)
        
        # **4. 性能比较**
        # 计算理论浮点运算次数 (FLOPs)
        flops = 2.0 * M * N * K  # 矩阵乘法浮点运算数
        triton_gflops = (flops / triton_time) / 1e9
        torch_gflops = (flops / torch_time) / 1e9
        speedup = torch_time / triton_time
        
        # **5. 打印结果**
        print(f"   正确性: {'✅ PASS' if is_correct else '❌ FAIL'}")
        if not is_correct:
            print(f"      最大绝对误差: {max_abs_error:.2e}")
            print(f"      容限 (rtol={rtol}, atol={atol})")
        print(f"   性能对比:")
        print(f"     - Triton: {triton_time*1000:.2f} ms, {triton_gflops:.2f} GFLOP/s")
        print(f"     - PyTorch: {torch_time*1000:.2f} ms, {torch_gflops:.2f} GFLOP/s")
        print(f"     加速比: {speedup:.2f}x {'(Triton更快)' if speedup > 1.0 else '(PyTorch更快)'}")
    
    print("\n" + "=" * 50)
    print("测试完成！")

if __name__ == "__main__":
    # 可选：预热运行（避免首次启动开销影响计时）
    print("预热运行...")
    warmup_a = torch.randn((64, 64), device='cpu')
    warmup_b = torch.randn((64, 64), device='cpu')
    _ = triton_matmul(warmup_a, warmup_b)
    _ = torch.matmul(warmup_a, warmup_b)
    
    # 运行主测试
    test_correctness_and_performance()

'''
python ./matmul.py
预热运行...
🔍 开始 Triton-CPU 矩阵乘法测试
--------------------------------------------------

测试用例 1: A[128x512] @ B[512x256] = C[128x256]
   正确性: ✅ PASS
   性能对比:
     - Triton: 5.23 ms, 6.41 GFLOP/s
     - PyTorch: 50.44 ms, 0.67 GFLOP/s
     加速比: 9.64x (Triton更快)

测试用例 2: A[512x2048] @ B[2048x1024] = C[512x1024]
   正确性: ✅ PASS
   性能对比:
     - Triton: 281.43 ms, 7.63 GFLOP/s
     - PyTorch: 2.98 ms, 720.99 GFLOP/s
     加速比: 0.01x (PyTorch更快)

测试用例 3: A[1024x4096] @ B[4096x2048] = C[1024x2048]
   正确性: ✅ PASS
   性能对比:
     - Triton: 2312.28 ms, 7.43 GFLOP/s
     - PyTorch: 16.26 ms, 1056.26 GFLOP/s
     加速比: 0.01x (PyTorch更快)

==================================================
测试完成！
'''
