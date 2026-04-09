"""
Evaluator for Triton kernels: functionality and performance testing.
"""
import torch
import triton
import triton.language as tl
import time
import traceback
from typing import Dict, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
import sys
import io
import tempfile
import os
import importlib.util
import types
import re


def _estimate_hardware_score(kernel_code: str, exec_time_ms: float, baseline_time_ms: float) -> float:
    """Lightweight hardware-oriented score without external profilers."""
    code_lower = kernel_code.lower()

    # Cache friendliness score
    cache_hits = 0
    for token in ["tl.load", "tl.store", "stride", "tl.arange", "mask"]:
        if token in code_lower:
            cache_hits += 1
    cache_friendly_score = min(cache_hits / 5.0, 1.0)

    # Memory alignment score (focus on block-size alignment hints)
    block_vals = [int(v) for v in re.findall(r"BLOCK_SIZE[_A-Z]*\s*=\s*(\d+)", kernel_code)]
    if block_vals:
        aligned = sum(1 for v in block_vals if v % 16 == 0)
        memory_alignment_score = aligned / len(block_vals)
    else:
        memory_alignment_score = 0.5

    # CPI estimation score (proxy with code complexity + observed timing)
    loop_count = code_lower.count("for ")
    mask_count = code_lower.count("mask")
    dot_count = code_lower.count("tl.dot")
    complexity_penalty = min((loop_count * 0.06) + (mask_count * 0.02) + (dot_count * 0.03), 0.45)
    speed_factor = 0.0
    if baseline_time_ms and baseline_time_ms > 0 and exec_time_ms > 0:
        speed_factor = min(baseline_time_ms / exec_time_ms, 2.0) / 2.0
    cpi_estimation_score = max(0.0, min(1.0, 0.65 + 0.35 * speed_factor - complexity_penalty))

    hardware_score = (
        0.4 * cache_friendly_score +
        0.3 * memory_alignment_score +
        0.3 * cpi_estimation_score
    )
    return max(0.0, min(1.0, hardware_score))


@dataclass
class EvaluationResult:
    """Result of kernel evaluation."""
    functional: bool  # Whether the kernel compiles and produces correct results
    performance: float  # Execution time in milliseconds (baseline time if functional=False)
    speedup: float  # Speedup ratio relative to baseline (0 if functional=False)
    hardware_score: float = 0.0
    fitness: float = 0.0
    error_message: Optional[str] = None
    kernel_code: Optional[str] = None
    pytorch_time: Optional[float] = None  # PyTorch reference implementation time
    baseline_time: Optional[float] = None  # Baseline Triton kernel time


# Global registry to keep temporary files alive
# These files are kept until process exit to allow Triton to inspect source code
_temp_files_registry = set()

def cleanup_temp_files():
    """Clean up all temporary kernel files. Call this at program exit."""
    import atexit
    for temp_file in list(_temp_files_registry):
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except:
            pass
    _temp_files_registry.clear()

# Register cleanup on exit
import atexit
atexit.register(cleanup_temp_files)

def _execute_triton_code_safely(code: str, namespace: dict, module_name: str = "triton_kernel_module"):
    """
    Safely execute Triton code that contains @triton.jit decorators.
    
    This function writes the code to a temporary file and imports it,
    which allows Triton to properly inspect the source code.
    
    IMPORTANT: Temporary files are NOT deleted immediately because Triton
    needs to access the source code via inspect.getsourcelines(). They will
    be cleaned up when the process exits.
    
    Args:
        code: The Triton kernel code as a string
        namespace: Dictionary containing required imports (torch, triton, tl, etc.)
        module_name: Name for the temporary module
        
    Returns:
        Dictionary containing the namespace after code execution
    """
    # Create a temporary file in a persistent location
    # Use a dedicated temp directory that won't be cleaned up immediately
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"triton_kernel_{module_name}_{os.getpid()}_{id(namespace)}.py")
    
    # Write the file
    with open(temp_file, 'w', encoding='utf-8') as f:
        # Write imports
        f.write("import torch\n")
        f.write("import triton\n")
        f.write("import triton.language as tl\n")
        f.write("\n")
        # Write the kernel code
        f.write(code)
    
    # Register the file so it's not garbage collected
    _temp_files_registry.add(temp_file)
    
    try:
        # Load the module from file
        spec = importlib.util.spec_from_file_location(module_name, temp_file)
        module = importlib.util.module_from_spec(spec)
        
        # Update module namespace with provided values
        module.__dict__.update(namespace)
        
        # Execute the module
        spec.loader.exec_module(module)
        
        # Return all names defined in the module
        result_namespace = {k: v for k, v in module.__dict__.items() 
                           if not k.startswith('__')}
        result_namespace.update(namespace)  # Keep original namespace values
        
        return result_namespace
    except Exception as e:
        # On error, try to clean up
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            _temp_files_registry.discard(temp_file)
        except:
            pass
        raise e
    # Note: We don't delete the file here because Triton needs it for inspect.getsourcelines()


class TritonKernelEvaluator:
    """Evaluator for Triton kernel code."""
    
    def __init__(self, baseline_kernel_code: str, test_function: Callable, baseline_time: float):
        """
        Args:
            baseline_kernel_code: Original kernel code for comparison
            test_function: Function that takes kernel code string and returns (func_result, time_ms)
            baseline_time: Baseline execution time for speedup calculation
        """
        self.baseline_kernel_code = baseline_kernel_code
        self.test_function = test_function
        self.baseline_time = baseline_time
    
    def evaluate(self, kernel_code: str, kernel_name: str = "optimized_kernel") -> EvaluationResult:
        """
        Evaluate a kernel by:
        1. Testing if it compiles and runs correctly
        2. Measuring its performance
        
        Returns:
            EvaluationResult with functional status, performance, and speedup
        """
        try:
            # Test functionality and measure performance
            result = self.test_function(kernel_code, kernel_name)
            
            # Handle different return formats
            if len(result) == 2:
                func_result, exec_time = result
                pytorch_time = None
                hardware_score = 0.0
            elif len(result) == 3:
                func_result, exec_time, pytorch_time = result
                hardware_score = 0.0
            elif len(result) == 4:
                func_result, exec_time, pytorch_time, hardware_score = result
            else:
                func_result, exec_time, pytorch_time, hardware_score = False, 0.0, None, 0.0
            
            if func_result:
                # Calculate speedup relative to baseline
                speedup = max((self.baseline_time / exec_time) - 1, 0.0) if self.baseline_time > 0 else 0.0
                return EvaluationResult(
                    functional=True,
                    performance=exec_time,
                    speedup=speedup,
                    hardware_score=max(0.0, min(1.0, float(hardware_score))),
                    kernel_code=kernel_code,
                    pytorch_time=pytorch_time,
                    baseline_time=self.baseline_time
                )
            else:
                return EvaluationResult(
                    functional=False,
                    performance=self.baseline_time,
                    speedup=0.0,
                    hardware_score=0.0,
                    error_message="Functional test failed",
                    kernel_code=kernel_code,
                    pytorch_time=None,
                    baseline_time=self.baseline_time
                )
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            return EvaluationResult(
                functional=False,
                performance=self.baseline_time,
                speedup=0.0,
                hardware_score=0.0,
                error_message=error_msg,
                kernel_code=kernel_code,
                pytorch_time=None,
                baseline_time=self.baseline_time
            )

    def _static_syntax_check(self, kernel_code: str) -> bool:
        """Level-1 static syntax check."""
        try:
            compile(kernel_code, "<kernel_code>", "exec")
            return True
        except Exception:
            return False

    def _quick_functional_check(self, kernel_code: str) -> bool:
        """Level-2 quick functional check with reduced kernel name path."""
        try:
            result = self.test_function(kernel_code, "quick_check_kernel")
            if len(result) >= 1:
                return bool(result[0])
            return False
        except Exception:
            return False

    def compute_fitness(self, result: EvaluationResult) -> float:
        """Compute multi-dimensional hardware-aware fitness."""
        correct_score = 1.0 if result.functional else 0.0
        speedup_score = min(result.speedup, 5.0) / 5.0
        hardware_score = max(0.0, min(1.0, result.hardware_score))
        fitness = correct_score * 0.4 + speedup_score * 0.4 + hardware_score * 0.2
        return fitness

    def cascade_evaluate(self, kernel_code: str, kernel_name: str) -> EvaluationResult:
        """Three-level cascading evaluation: static -> quick functional -> full run."""
        if not self._static_syntax_check(kernel_code):
            return EvaluationResult(
                functional=False,
                performance=self.baseline_time,
                speedup=0.0,
                hardware_score=0.0,
                fitness=0.0,
                error_message="Static syntax check failed",
                kernel_code=kernel_code,
                baseline_time=self.baseline_time,
            )

        if not self._quick_functional_check(kernel_code):
            return EvaluationResult(
                functional=False,
                performance=self.baseline_time,
                speedup=0.0,
                hardware_score=0.0,
                fitness=0.0,
                error_message="Quick functional check failed",
                kernel_code=kernel_code,
                baseline_time=self.baseline_time,
            )

        full_result = self.evaluate(kernel_code, kernel_name)
        full_result.fitness = self.compute_fitness(full_result)
        return full_result


def create_vector_add_evaluator(device, size: int = 98432, block_size: int = 1024):
    """Create an evaluator for vector addition kernel."""
    
    def test_kernel(kernel_code: str, kernel_name: str) -> Tuple[bool, float]:
        """Test kernel functionality and measure performance."""
        # Create a temporary module to execute the kernel code
        namespace = {
            'torch': torch,
            'triton': triton,
            'tl': tl,
            'DEVICE': device
        }
        
        try:
            # Execute kernel code safely using temporary file
            namespace = _execute_triton_code_safely(kernel_code, namespace, f"kernel_{kernel_name}")
            
            # Get the kernel function (try different names)
            kernel = None
            for name in ['add_kernel', 'kernel', kernel_name]:
                if name in namespace:
                    kernel = namespace[name]
                    break
            
            if kernel is None:
                return False, 0.0, None, 0.0
            
            # Test correctness
            torch.manual_seed(0)
            x = torch.rand(size, device=device)
            y = torch.rand(size, device=device)
            output_torch = x + y
            
            # Create wrapper function
            def add(x, y):
                output = torch.empty_like(x)
                n_elements = output.numel()
                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                try:
                    kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
                except Exception as e:
                    # Try with different BLOCK_SIZE
                    try:
                        kernel[grid](x, y, output, n_elements, BLOCK_SIZE=512)
                    except:
                        raise e
                return output
            
            output_triton = add(x, y)
            
            # Check correctness
            max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
            is_correct = max_diff < 1e-5
            
            if not is_correct:
                return False, 0.0, None, 0.0
            
            # Measure performance
            x = torch.rand(size, device=device)
            y = torch.rand(size, device=device)
            
            # Warmup
            for _ in range(10):
                try:
                    _ = add(x, y)
                except:
                    return False, 0.0, None, 0.0
            
            # Benchmark
            times = []
            for _ in range(50):
                try:
                    start = time.perf_counter()
                    _ = add(x, y)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms
                except:
                    return False, 0.0, None, 0.0
            
            avg_time = sum(times) / len(times) if times else float('inf')
            hardware_score = _estimate_hardware_score(kernel_code, avg_time, 0.0)
            return True, avg_time, None, hardware_score
            
        except Exception as e:
            print(f"Kernel test error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0, None, 0.0
    
    # Get baseline time
    baseline_code = """
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
    
    torch.manual_seed(0)
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    
    # Measure baseline
    namespace = {'torch': torch, 'triton': triton, 'tl': tl, 'DEVICE': device}
    namespace = _execute_triton_code_safely(baseline_code, namespace, "baseline_kernel")
    kernel = namespace['add_kernel']
    
    def add_baseline(x, y):
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
        return output
    
    # Warmup
    for _ in range(10):
        _ = add_baseline(x, y)
    
    # Baseline benchmark
    baseline_times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = add_baseline(x, y)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        baseline_times.append((end - start) * 1000)
    
    baseline_time = sum(baseline_times) / len(baseline_times)
    
    return TritonKernelEvaluator(baseline_code, test_kernel, baseline_time)


def create_matmul_evaluator(device, M: int = 512, N: int = 512, K: int = 512):
    """Create an evaluator for matrix multiplication kernel."""
    
    def test_kernel(kernel_code: str, kernel_name: str) -> Tuple[bool, float, Optional[float]]:
        """Test kernel functionality and measure performance.
        
        Returns:
            (is_correct, triton_time_ms, pytorch_time_ms)
        """
        namespace = {
            'torch': torch,
            'triton': triton,
            'tl': tl,
            'DEVICE': device
        }
        
        try:
            # Execute kernel code safely using temporary file
            namespace = _execute_triton_code_safely(kernel_code, namespace, f"matmul_kernel_{kernel_name}")
            
            if 'matmul_kernel' not in namespace:
                return False, 0.0, None, 0.0
            
            kernel = namespace['matmul_kernel']
            
            # Test correctness
            torch.manual_seed(0)
            a = torch.rand((M, K), device=device, dtype=torch.float32) - 0.5
            b = torch.rand((K, N), device=device, dtype=torch.float32) - 0.5
            output_torch = torch.matmul(a, b)
            
            # Create wrapper - try to match matmul.py signature
            def matmul(a, b):
                assert a.shape[1] == b.shape[0]
                M, K = a.shape
                K_check, N = b.shape
                assert K == K_check
                
                c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
                
                # First, try to detect what parameters the kernel needs
                # by inspecting the kernel signature
                kernel_needs_group = False
                kernel_needs_activation = False
                try:
                    import inspect
                    sig = inspect.signature(kernel.fn)
                    param_names = list(sig.parameters.keys())
                    kernel_needs_group = 'GROUP_SIZE_M' in param_names
                    kernel_needs_activation = 'ACTIVATION' in param_names
                except:
                    # If we can't inspect, we'll try different configs
                    pass
                
                # Try different block sizes (matching matmul.py default)
                # Build configs based on what the kernel needs
                base_configs = [
                    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
                    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
                    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64},
                    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16},
                ]
                
                configs = []
                for base_config in base_configs:
                    config = base_config.copy()
                    # Only add GROUP_SIZE_M and ACTIVATION if kernel needs them
                    if kernel_needs_group:
                        config['GROUP_SIZE_M'] = 8
                    if kernel_needs_activation:
                        config['ACTIVATION'] = ''
                    configs.append(config)
                
                # Also try configs with GROUP_SIZE_M/ACTIVATION if we're not sure
                if not kernel_needs_group and not kernel_needs_activation:
                    # Try a few with GROUP_SIZE_M to see if kernel actually needs it
                    for base_config in base_configs[:2]:
                        config = base_config.copy()
                        config['GROUP_SIZE_M'] = 8
                        config['ACTIVATION'] = ''
                        configs.append(config)
                
                last_error = None
                for config in configs:
                    try:
                        # Check if kernel needs GROUP_SIZE_M by inspecting signature or config
                        # For kernels with GROUP_SIZE_M, use 1D grid
                        # For kernels without, use 2D grid
                        has_group_size = 'GROUP_SIZE_M' in config and config.get('GROUP_SIZE_M') is not None
                        
                        if has_group_size:
                            # 1D grid for kernels with GROUP_SIZE_M
                            grid = (triton.cdiv(M, config['BLOCK_SIZE_M']) * triton.cdiv(N, config['BLOCK_SIZE_N']),)
                        else:
                            # 2D grid for standard kernels
                            grid = (
                                triton.cdiv(M, config['BLOCK_SIZE_M']),
                                triton.cdiv(N, config['BLOCK_SIZE_N']),
                            )
                        
                        # Build kwargs, only including parameters that kernel accepts
                        kwargs = {}
                        for key, value in config.items():
                            # Only include if kernel needs it, or if we're trying to discover
                            if key in ['BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K']:
                                kwargs[key] = value
                            elif key == 'GROUP_SIZE_M' and (kernel_needs_group or has_group_size):
                                kwargs[key] = value
                            elif key == 'ACTIVATION' and (kernel_needs_activation or kernel_needs_group):
                                kwargs[key] = value
                        
                        # Try calling with the config
                        kernel[grid](
                            a, b, c,
                            M, N, K,
                            a.stride(0), a.stride(1),
                            b.stride(0), b.stride(1),
                            c.stride(0), c.stride(1),
                            **kwargs
                        )
                        return c
                    except (TypeError, KeyError) as e:
                        # If it's a TypeError/KeyError about arguments, try next config
                        last_error = e
                        error_str = str(e)
                        # Check if kernel needs GROUP_SIZE_M or ACTIVATION
                        needs_group = 'GROUP_SIZE_M' in error_str and 'GROUP_SIZE_M' not in kwargs
                        needs_activation = 'ACTIVATION' in error_str and 'ACTIVATION' not in kwargs
                        has_group = 'GROUP_SIZE_M' in kwargs
                        has_activation = 'ACTIVATION' in kwargs
                        
                        if needs_group:
                            kernel_needs_group = True
                            continue
                        if needs_activation:
                            kernel_needs_activation = True
                            continue
                        if (has_group and 'GROUP_SIZE_M' in error_str and 'unrecognised' in error_str) or \
                           (has_activation and 'ACTIVATION' in error_str and 'unrecognised' in error_str):
                            # Kernel doesn't need these params but we provided them
                            # Update our knowledge and skip
                            if 'GROUP_SIZE_M' in error_str:
                                kernel_needs_group = False
                            if 'ACTIVATION' in error_str:
                                kernel_needs_activation = False
                            continue
                        else:
                            # Other TypeError/KeyError, try next config
                            if config == configs[0]:
                                print(f"  Kernel execution error with config {config}: {e}")
                            continue
                    except Exception as e:
                        last_error = e
                        # Print error for first config to help debugging
                        if config == configs[0]:
                            print(f"  Kernel execution error with config {config}: {e}")
                        continue
                
                # If all configs failed, raise with last error details
                error_msg = f"Failed to execute kernel with any configuration. Last error: {last_error}"
                if last_error:
                    error_msg += f"\n  Error type: {type(last_error).__name__}"
                raise RuntimeError(error_msg)
            
            output_triton = matmul(a, b)
            
            max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
            is_correct = max_diff < 1e-3  # More lenient for FP32
            
            if not is_correct:
                return False, 0.0, None, 0.0
            
            # Measure performance
            a = torch.rand((M, K), device=device, dtype=torch.float32)
            b = torch.rand((K, N), device=device, dtype=torch.float32)
            
            # Warmup (reduced for faster evaluation)
            for _ in range(3):
                _ = matmul(a, b)
                _ = torch.matmul(a, b)
            
            # Benchmark Triton (reduced iterations for faster evaluation)
            triton_times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = matmul(a, b)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                triton_times.append((end - start) * 1000)
            
            # Benchmark PyTorch (reduced iterations)
            pytorch_times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = torch.matmul(a, b)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                pytorch_times.append((end - start) * 1000)
            
            avg_triton_time = sum(triton_times) / len(triton_times)
            avg_pytorch_time = sum(pytorch_times) / len(pytorch_times)
            hardware_score = _estimate_hardware_score(kernel_code, avg_triton_time, 0.0)

            return True, avg_triton_time, avg_pytorch_time, hardware_score
            
        except Exception as e:
            print(f"Kernel test error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0, None, 0.0
    
    # Get baseline - matching matmul.py structure
    baseline_code = """
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
    
    # Load baseline kernel
    namespace = {'torch': torch, 'triton': triton, 'tl': tl, 'DEVICE': device}
    namespace = _execute_triton_code_safely(baseline_code, namespace, "baseline_matmul_kernel")
    
    if 'matmul_kernel' not in namespace:
        baseline_time = 1.0
    else:
        kernel = namespace['matmul_kernel']
        
        # Prepare test data (use float32 to match test_kernel)
        torch.manual_seed(0)
        a = torch.rand((M, K), device=device, dtype=torch.float32)
        b = torch.rand((K, N), device=device, dtype=torch.float32)
        
        def matmul_baseline(a, b):
            M, K = a.shape
            K_check, N = b.shape
            assert K == K_check
            c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
            grid = (
                triton.cdiv(M, 64),
                triton.cdiv(N, 64),
            )
            kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
            )
            return c
        
        # Warmup (reduced for faster evaluation)
        for _ in range(3):
            _ = matmul_baseline(a, b)
        
        # Benchmark baseline (reduced iterations)
        baseline_times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = matmul_baseline(a, b)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            baseline_times.append((end - start) * 1000)
        
        baseline_time = sum(baseline_times) / len(baseline_times)
    
    return TritonKernelEvaluator(baseline_code, test_kernel, baseline_time)



def create_rmsnorm_evaluator(device, batch_size: int = 32, feature_dim: int = 512):
    """Create an evaluator for RMSNorm kernel.
    
    Args:
        device: torch device (cpu/cuda)
        batch_size: Default batch size for testing
        feature_dim: Default feature dimension for testing
    """
    
    def pytorch_rms_norm(x, weight, eps=1e-5):
        """PyTorch reference implementation of RMSNorm (match your test code)"""
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        return (x / rms) * weight
    


    
    def test_kernel(kernel_code: str, kernel_name: str) -> Tuple[bool, float, Optional[float]]:
        """Test RMSNorm kernel functionality and measure performance.
        
        Returns:
            (is_correct, triton_time_ms, pytorch_time_ms)
        """
        namespace = {
            'torch': torch,
            'triton': triton,
            'tl': tl,
            'DEVICE': device
        }
        
        try:
            # Execute kernel code safely using temporary file
            namespace = _execute_triton_code_safely(kernel_code, namespace, f"rmsnorm_kernel_{kernel_name}")
            
            
            # Check if RMSNorm kernel exists
            if 'rms_norm_kernel' not in namespace:
                print(f"❌ Kernel 'rms_norm_kernel' not found in namespace")
                return False, 0.0, None, 0.0
            
            kernel = namespace['rms_norm_kernel']
            
            # Test correctness (match your test code)
            torch.manual_seed(42)  # Fixed seed for reproducibility
            # Generate correct test data for RMSNorm
            x = torch.randn(batch_size, feature_dim, device=device, dtype=torch.float32)
            weight = torch.randn(feature_dim, device=device, dtype=torch.float32)
            eps = 1e-5
            
            # PyTorch reference result
            output_torch = pytorch_rms_norm(x, weight, eps=eps)
            
            # Create correct wrapper for RMSNorm (match your test code)
            def triton_rms_norm(input_tensor, weight, eps=1e-5):
                """Wrapper for Triton RMSNorm kernel (match your test code)"""
                assert input_tensor.dim() == 2, "Input must be 2D [batch_size, feature_dim]"
                assert weight.dim() == 1, "Weight must be 1D [feature_dim]"
                assert input_tensor.shape[1] == weight.shape[0], "Feature dim mismatch"
                
                batch_size, feature_dim = input_tensor.shape
                output = torch.empty_like(input_tensor)
                
                # Try different block sizes (common values for CPU)
                block_sizes = [128, 256, 64, 512]
                last_error = None
                
                for BLOCK_SIZE in block_sizes:
                    try:
                        # RMSNorm uses 1D grid (one program per batch element)
                        grid = (batch_size,)
                        
                        # Call kernel with correct parameters (match your test code)
                        kernel[grid](
                            input_tensor, weight, output,
                            feature_dim,  # n_elements (feature dimension)
                            input_tensor.stride(0), input_tensor.stride(1),  # stride_batch, stride_feature
                            eps=eps,
                            BLOCK_SIZE=BLOCK_SIZE,
                        )
                        return output
                    except Exception as e:
                        last_error = e
                        print(f"  ❌ Failed with BLOCK_SIZE={BLOCK_SIZE}: {type(e).__name__}: {e}")
                        continue
                
                # If all block sizes failed
                raise RuntimeError(f"Failed to execute RMSNorm kernel with any block size. Last error: {last_error}")
            
            # Get Triton result
            output_triton = triton_rms_norm(x, weight, eps=eps)
            
            # Correctness check (match your test code's tolerance)
            rtol, atol = 1e-4, 1e-5
            is_correct = torch.allclose(output_triton, output_torch, rtol=rtol, atol=atol)
            
            if not is_correct:
                # Print error details for debugging
                abs_diff = torch.abs(output_triton - output_torch)
                max_abs_error = torch.max(abs_diff).item()
                rel_diff = abs_diff / (torch.abs(output_torch) + 1e-8)
                max_rel_error = torch.max(rel_diff).item()
                print(f"❌ Correctness check failed: max_abs_error={max_abs_error:.2e}, max_rel_error={max_rel_error:.2e}")
                return False, 0.0, None, 0.0
            
            # Measure performance (match your test code)
            # Warmup
            for _ in range(3):
                _ = triton_rms_norm(x, weight, eps=eps)
                _ = pytorch_rms_norm(x, weight, eps=eps)
            
            # Benchmark Triton RMSNorm
            triton_times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = triton_rms_norm(x, weight, eps=eps)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                triton_times.append((end - start) * 1000)  # Convert to ms
            
            # Benchmark PyTorch RMSNorm
            pytorch_times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = pytorch_rms_norm(x, weight, eps=eps)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                pytorch_times.append((end - start) * 1000)
            
            # Calculate average times
            avg_triton_time = sum(triton_times) / len(triton_times) if triton_times else float('inf')
            avg_pytorch_time = sum(pytorch_times) / len(pytorch_times) if pytorch_times else float('inf')
            hardware_score = _estimate_hardware_score(kernel_code, avg_triton_time, 0.0)
            
            print(f"✅ RMSNorm kernel test passed: Triton={avg_triton_time:.2f}ms, PyTorch={avg_pytorch_time:.2f}ms")
            return True, avg_triton_time, avg_pytorch_time, hardware_score
            
        except Exception as e:
            error_msg = f"Kernel test error: {e}\n{traceback.format_exc()}"
            print(error_msg)
            return False, 0.0, None, 0.0
    
    # Get baseline RMSNorm kernel (your original baseline code)
    baseline_code = """
@triton.jit
def rms_norm_kernel(
    # 输入/输出指针
    input_ptr, weight_ptr, output_ptr,
    # 张量维度信息
    n_elements,  # 每个样本的特征数 (标准化维度)
    stride_batch, stride_feature,  # 输入张量的内存步长
    # 参数
    eps: tl.constexpr,
    # 平铺参数
    BLOCK_SIZE: tl.constexpr,
):
    \"\"\"
    RMSNorm 内核: output = (input / sqrt(mean(input^2) + eps)) * weight
    
    计算过程:
    1. 计算输入张量的平方的均值 (RMS)
    2. 使用 RMS 对输入进行标准化
    3. 应用可学习的权重参数
    \"\"\"
    # 当前程序处理的批次索引
    batch_idx = tl.program_id(axis=0)
    
    # 计算当前批次在内存中的起始偏移
    input_batch_start = batch_idx * stride_batch
    output_batch_start = batch_idx * stride_batch
    
    # ====== 步骤1: 计算 RMS (均方根) ======
    mean_square = tl.zeros((1,), dtype=tl.float32)
    
    # 分块计算平方和 (减少内存压力)
    for offset in range(0, n_elements, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_elements
        
        # 加载当前块的数据
        input_vals = tl.load(
            input_ptr + input_batch_start + col_idx * stride_feature,
            mask=mask,
            other=0.0,
        )
        
        # 累加平方值
        mean_square += tl.sum(input_vals * input_vals, axis=0)
    
    # 计算均值并加上 epsilon
    rms = tl.sqrt(mean_square / n_elements + eps)
    
    # ====== 步骤2: 应用标准化和权重 ======
    for offset in range(0, n_elements, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_elements
        
        # 重新加载输入数据
        input_vals = tl.load(
            input_ptr + input_batch_start + col_idx * stride_feature,
            mask=mask,
            other=0.0,
        )
        
        # 加载权重 (广播到整个批次)
        weight_vals = tl.load(
            weight_ptr + col_idx * stride_feature,
            mask=mask,
            other=1.0,  # 默认权重为1
        )
        
        # RMSNorm 计算: (input / rms) * weight
        normalized = (input_vals / rms) * weight_vals
        
        # 存储结果
        tl.store(
            output_ptr + output_batch_start + col_idx * stride_feature,
            normalized,
            mask=mask,
        )
"""
    
    # Load baseline kernel and calculate baseline time
    namespace = {'torch': torch, 'triton': triton, 'tl': tl, 'DEVICE': device}
    namespace = _execute_triton_code_safely(baseline_code, namespace, "baseline_rmsnorm_kernel")
    
    baseline_time = 1.0  # Default if baseline load fails
    if 'rms_norm_kernel' in namespace:
        kernel = namespace['rms_norm_kernel']
        
        # Define baseline wrapper (match your test code)
        def rms_norm_baseline(input_tensor, weight, eps=1e-5):
            assert input_tensor.dim() == 2
            assert weight.dim() == 1
            batch_size, feature_dim = input_tensor.shape
            output = torch.empty_like(input_tensor)
            grid = (batch_size,)
            kernel[grid](
                input_tensor, weight, output,
                feature_dim,
                input_tensor.stride(0), input_tensor.stride(1),
                eps=eps,
                BLOCK_SIZE=128,
            )
            return output
        
        # Prepare test data for baseline benchmark
        torch.manual_seed(42)
        x = torch.randn(batch_size, feature_dim, device=device, dtype=torch.float32)
        weight = torch.randn(feature_dim, device=device, dtype=torch.float32)
        
        # Warmup
        for _ in range(3):
            _ = rms_norm_baseline(x, weight)
        
        # Benchmark baseline
        baseline_times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = rms_norm_baseline(x, weight)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            baseline_times.append((end - start) * 1000)
        
        baseline_time = sum(baseline_times) / len(baseline_times)
        print(f"📌 RMSNorm baseline time: {baseline_time:.2f} ms")
    
    return TritonKernelEvaluator(baseline_code, test_kernel, baseline_time)