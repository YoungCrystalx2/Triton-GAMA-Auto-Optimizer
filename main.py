"""
Main entry point for evolutionary Triton kernel optimization.
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
import triton
import triton.language as tl

from config import OptimizationConfig, load_config_from_env
from llm_client import create_llm_client
from evaluator import create_vector_add_evaluator, create_matmul_evaluator
from evolutionary_optimizer import EvolutionaryOptimizer, Individual
"""
进化式Triton内核优化主程序:整合评估器、LLM客户端、进化算法
核心逻辑:遵循EVOPROMPT框架,通过"初始种群→进化（变异/交叉）→评估→选择"迭代,生成最优Triton内核
"""

def extract_kernel_from_file(file_path: str, kernel_name: str = "matmul_kernel") -> str:
    """Extract kernel code from a Python file."""
    import ast
    import re
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Method 1: Try simpler regex pattern first (more reliable)
    pattern = rf'@triton\.jit\s+def\s+{kernel_name}.*?(?=\n\ndef\s|\nclass\s|\Z)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        code = match.group(0).strip()
        if '@triton.jit' in code and f'def {kernel_name}' in code:
            # Verify it's complete by checking for key elements
            if 'tl.store' in code or 'return' in code or len(code) > 1000:
                return code
    
    # Method 2: Try regex with multiline (less reliable, may stop early)
    pattern = rf'@triton\.jit\s*\n\s*def\s+{kernel_name}.*?(?=\n\n\ndef\s|\n\nclass\s|\Z)'
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        code = match.group(0).strip()
        if '@triton.jit' in code and f'def {kernel_name}' in code:
            # Only use if it seems complete
            if 'tl.store' in code or len(code) > 1000:
                return code
    
    # Method 3: Use AST to find function, then extract with decorator
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
                # Find the decorator line (check line before function definition)
                start_line = node.lineno - 1  # lineno is 1-indexed, convert to 0-indexed
                # Check if previous line has @triton.jit
                if start_line > 0 and '@triton.jit' in lines[start_line - 1]:
                    start_line = start_line - 1
                
                # Get end line - use end_lineno if available (it's 1-indexed)
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    end_line = node.end_lineno  # end_lineno is 1-indexed, but we need it for slicing
                else:
                    # Fallback: find next top-level definition
                    end_line = len(lines)
                    for i in range(node.lineno, len(lines)):  # Start from function definition line
                        stripped = lines[i].lstrip()
                        if stripped and not lines[i].startswith((' ', '\t', '#')) and (stripped.startswith('def ') or stripped.startswith('class ')):
                            end_line = i
                            break
                
                # Extract lines (start_line is 0-indexed, end_line is 1-indexed from AST)
                # Convert end_line to 0-indexed for slicing (end_lineno is inclusive)
                kernel_lines = lines[start_line:end_line]
                code = '\n'.join(kernel_lines).strip()
                if '@triton.jit' in code and f'def {kernel_name}' in code:
                    return code
    except Exception as e:
        pass  # Fall through to next method
    
    # Method 4: Line-by-line extraction
    kernel_lines = []
    found_decorator = False
    found_function = False
    
    for i, line in enumerate(lines):
        # Look for @triton.jit decorator
        if '@triton.jit' in line and not found_decorator:
            found_decorator = True
            kernel_lines.append(line)
            continue
        
        # After finding decorator, look for function definition
        if found_decorator and f'def {kernel_name}' in line:
            found_function = True
            kernel_lines.append(line)
            continue
        
        # After finding function, collect lines until next top-level definition
        if found_function:
            # Stop at empty line followed by def/class, or at next def/class at column 0
            stripped = line.lstrip()
            if stripped and not line.startswith((' ', '\t')) and (stripped.startswith('def ') or stripped.startswith('class ')):
                break
            kernel_lines.append(line)
    
    if kernel_lines and '@triton.jit' in '\n'.join(kernel_lines):
        return '\n'.join(kernel_lines).strip()
    
    raise ValueError(f"Could not extract {kernel_name} from {file_path}. Found {len(kernel_lines)} lines but missing @triton.jit decorator.")


def load_baseline_kernel(kernel_type: str, kernel_file: Optional[str] = None) -> str:
    """加载基准内核代码 Load baseline kernel code based on kernel type."""
    if kernel_type == "vector_add":
        return """
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
"""
    elif kernel_type == "matmul":
        # If kernel_file is provided, try to load from file
        if kernel_file and Path(kernel_file).exists():
            try:
                return extract_kernel_from_file(kernel_file, "matmul_kernel")
            except Exception as e:
                print(f"Warning: Could not load kernel from {kernel_file}: {e}")
                print("Falling back to default baseline kernel.")
        
        # Default baseline (from matmul.py structure)
        return """
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_SIZE_K

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
"""
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def save_optimized_kernels(results: List[Individual], output_dir: str, config: OptimizationConfig):
    """保存优化后的内核代码和结果摘要（便于后续分析）"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save individual kernels
    for i, individual in enumerate(results):
        kernel_file = Path(output_dir) / f"kernel_{i+1}.py"
        with open(kernel_file, 'w') as f:
            f.write("# Optimized Triton Kernel\n")
            f.write(f"# Fitness (Speedup): {individual.fitness:.4f}\n")
            f.write(f"# Functional: {individual.functional}\n")
            f.write("#\n")
            f.write("import torch\n")
            f.write("import triton\n")
            f.write("import triton.language as tl\n")
            f.write("\n")
            f.write(individual.code)
    
    # Save summary
    summary_file = Path(output_dir) / "summary.json"
    summary = {
        "kernel_type": config.kernel_type,
        "kernel_name": config.kernel_name,
        "num_versions": len(results),
        "functional_count": sum(1 for ind in results if ind.functional),
        "versions": [
            {
                "file": f"kernel_{i+1}.py",
                "functional": ind.functional,
                "fitness": ind.fitness,
                "performance_ms": ind.evaluation_result.performance if ind.evaluation_result else None,
                "speedup": ind.evaluation_result.speedup if ind.evaluation_result else 0.0
            }
            for i, ind in enumerate(results)
        ],
        "best_fitness": max((ind.fitness for ind in results if ind.functional), default=0.0),
        "total_tokens": None  # Will be filled by optimizer
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evolutionary Triton Kernel Optimizer")
    parser.add_argument("--kernel-type", type=str, default="vector_add",
                       choices=["vector_add", "matmul"],
                       help="Type of kernel to optimize")
    parser.add_argument("--llm-provider", type=str, default="deepseek",
                       choices=["deepseek", "qwen", "glm"],
                       help="LLM provider to use")
    parser.add_argument("--llm-model", type=str, default="deepseek-chat",
                       help="LLM model name")
    parser.add_argument("--llm-api-key", type=str, default=None,
                       help="LLM API key (or set via environment variable)")
    parser.add_argument("--population-size", type=int, default=5,
                       help="Population size")
    parser.add_argument("--max-iterations", type=int, default=10,
                       help="Maximum iterations")
    parser.add_argument("--max-time", type=int, default=1200,
                       help="Maximum time in seconds (default: 1200 = 20 minutes)")
    parser.add_argument("--max-tokens", type=int, default=200000,
                       help="Maximum token budget")
    parser.add_argument("--output-dir", type=str, default="optimized_kernels",
                       help="Output directory for optimized kernels")
    parser.add_argument("--config-file", type=str, default=None,
                       help="Path to configuration file (JSON)")
    parser.add_argument("--kernel-file", type=str, default=None,
                       help="Path to kernel file to load baseline from (e.g., matmul.py)")
    parser.add_argument("--matmul-M", type=int, default=None,
                       help="Matrix dimension M for matmul (default: from config)")
    parser.add_argument("--matmul-N", type=int, default=None,
                       help="Matrix dimension N for matmul (default: from config)")
    parser.add_argument("--matmul-K", type=int, default=None,
                       help="Matrix dimension K for matmul (default: from config)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = OptimizationConfig(**config_dict)
    else:
        config = load_config_from_env()
        # Override with command line arguments
        config.llm_provider = args.llm_provider
        config.llm_model = args.llm_model
        config.llm_api_key = args.llm_api_key or config.llm_api_key
        config.population_size = args.population_size
        config.max_iterations = args.max_iterations
        config.max_time = args.max_time
        config.max_tokens = args.max_tokens
        config.kernel_type = args.kernel_type
        config.output_dir = args.output_dir
    
    print("=" * 60)
    print("Evolutionary Triton Kernel Optimizer")
    print("=" * 60)
    print(f"Kernel type: {config.kernel_type}")
    print(f"LLM provider: {config.llm_provider}")
    print(f"LLM model: {config.llm_model}")
    print(f"Population size: {config.population_size}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Max time: {config.max_time}s")
    print(f"Max tokens: {config.max_tokens}")
    print("=" * 60)
    
    # Get device
    device = triton.runtime.driver.active.get_active_torch_device()
    print(f"Device: {device}")
    
    # Load baseline kernel
    kernel_file = args.kernel_file
    if kernel_file is None and config.kernel_type == "matmul":
        # Default to matmul.py in triton-cpu-scripts
        default_matmul_file = Path(__file__).parent / "triton-cpu-main" / "triton-cpu-scripts" / "matmul.py"
        if default_matmul_file.exists():
            kernel_file = str(default_matmul_file)
            print(f"Using default kernel file: {kernel_file}")
    
    baseline_code = load_baseline_kernel(config.kernel_type, kernel_file)
    
    # Create evaluator
    if config.kernel_type == "vector_add":
        evaluator = create_vector_add_evaluator(
            device, 
            config.vector_size, 
            config.block_size
        )
        config.kernel_name = "add_kernel"
    elif config.kernel_type == "matmul":
        evaluator = create_matmul_evaluator(
            device,
            config.matmul_M,
            config.matmul_N,
            config.matmul_K
        )
        config.kernel_name = "matmul_kernel"
    else:
        raise ValueError(f"Unknown kernel type: {config.kernel_type}")
    
    # Create LLM client
    # For Qwen, use DASHSCOPE_API_KEY; for others, use {PROVIDER}_API_KEY
    if config.llm_provider == "qwen":
        api_key = config.llm_api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    else:
        api_key = config.llm_api_key or os.getenv(f"{config.llm_provider.upper()}_API_KEY")
    llm_client = create_llm_client(config.llm_provider, config.llm_model, api_key)
    
    # Create optimizer
    optimizer = EvolutionaryOptimizer(
        baseline_code=baseline_code,
        evaluator=evaluator,
        llm_client=llm_client,
        population_size=config.population_size,
        max_iterations=config.max_iterations,
        max_time=config.max_time,
        max_tokens=config.max_tokens,
        mutation_rate=config.mutation_rate,
        crossover_rate=config.crossover_rate,
        selection_size=config.selection_size,
        kernel_name=config.kernel_name
    )
    
    # Run optimization
    print("\nStarting optimization...")
    results = optimizer.evolve()
    
    # Filter to only functional kernels and sort by fitness
    functional_results = [ind for ind in results if ind.functional]
    functional_results.sort(key=lambda x: x.fitness, reverse=True)
    
    # Take top N versions
    output_results = functional_results[:config.max_output_versions]
    
    # If no functional results, include baseline
    if not output_results:
        print("Warning: No functional optimized kernels found. Including baseline.")
        baseline_result = evaluator.evaluate(baseline_code, config.kernel_name)
        output_results = [optimizer.population[0] if optimizer.population else Individual(
            code=baseline_code,
            fitness=baseline_result.speedup,
            functional=baseline_result.functional,
            evaluation_result=baseline_result
        )]
    
    # Print results with performance comparison
    print("\n" + "=" * 60)
    print("Optimization Results - Performance Comparison")
    print("=" * 60)
    
    # Get baseline performance for comparison
    baseline_result = evaluator.evaluate(baseline_code, config.kernel_name)
    baseline_time = baseline_result.performance if baseline_result.functional else None
    
    print(f"\nBaseline (Original Triton Kernel):")
    if baseline_time:
        print(f"  Correctness: {'✅ PASS' if baseline_result.functional else '❌ FAIL'}")
        print(f"  Performance: {baseline_time:.4f} ms")
        if baseline_result.pytorch_time:
            print(f"  PyTorch Reference: {baseline_result.pytorch_time:.4f} ms")
            print(f"  Speedup vs PyTorch: {baseline_result.pytorch_time / baseline_time:.2f}x")
    else:
        print(f"  Correctness: {'✅ PASS' if baseline_result.functional else '❌ FAIL'}")
        print(f"  Status: {'Functional' if baseline_result.functional else 'Non-functional'}")
    
    print(f"\nOptimized Versions:")
    for i, individual in enumerate(output_results):
        print(f"\nVersion {i+1}:")
        print(f"  Functional: {individual.functional}")
        if individual.functional and individual.evaluation_result:
            opt_time = individual.evaluation_result.performance
            print(f"  Correctness: ✅ PASS (tested with tolerance 1e-3)")
            print(f"  Performance: {opt_time:.4f} ms")
            if baseline_time:
                speedup_vs_baseline = baseline_time / opt_time if opt_time > 0 else 0
                print(f"  Speedup vs Baseline: {speedup_vs_baseline:.2f}x")
            if individual.evaluation_result.pytorch_time:
                pytorch_time = individual.evaluation_result.pytorch_time
                print(f"  PyTorch Reference: {pytorch_time:.4f} ms")
                speedup_vs_pytorch = pytorch_time / opt_time if opt_time > 0 else 0
                print(f"  Speedup vs PyTorch: {speedup_vs_pytorch:.2f}x")
        else:
            print(f"  Correctness: ❌ FAIL")
            print(f"  Status: Non-functional")
            if individual.evaluation_result and individual.evaluation_result.error_message:
                print(f"  Error: {individual.evaluation_result.error_message[:150]}...")
    
    print(f"\nTotal tokens used: {llm_client.get_total_tokens()}")
    
    # Save results
    save_optimized_kernels(output_results, config.output_dir, config)
    print(f"\nResults saved to: {config.output_dir}")
    
    # Update summary with token count
    summary_file = Path(config.output_dir) / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        summary["total_tokens"] = llm_client.get_total_tokens()
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    print("\nOptimization complete!")
    return output_results


if __name__ == "__main__":
    main()
