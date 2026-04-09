"""
Configuration for evolutionary optimization.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizationConfig:
    """Configuration for evolutionary optimization."""
    # LLM settings
    llm_provider: str = "deepseek"  # deepseek, qwen, glm
    llm_model: str = "deepseek-chat"
    llm_api_key: Optional[str] = None
    
    # Evolutionary algorithm settings
    population_size: int = 5
    max_iterations: int = 10
    max_time: int = 1200  # 20 minutes in seconds
    max_tokens: int = 200000
    mutation_rate: float = 0.5
    crossover_rate: float = 0.5
    selection_size: int = 2
    
    # Kernel settings
    kernel_type: str = "vector_add"  # vector_add, matmul
    kernel_name: str = "add_kernel"
    
    # Vector add settings
    vector_size: int = 98432
    block_size: int = 1024
    
    # Matmul settings (use larger matrices for better performance comparison)
    matmul_M: int = 1024
    matmul_N: int = 1024
    matmul_K: int = 1024
    
    # Output settings
    output_dir: str = "optimized_kernels"
    max_output_versions: int = 5


def load_config_from_env() -> OptimizationConfig:
    """Load configuration from environment variables."""
    import os
    
    config = OptimizationConfig()
    
    # LLM settings
    config.llm_provider = os.getenv("LLM_PROVIDER", config.llm_provider)
    config.llm_model = os.getenv("LLM_MODEL", config.llm_model)
    config.llm_api_key = os.getenv("LLM_API_KEY")
    
    # Algorithm settings
    if os.getenv("POPULATION_SIZE"):
        config.population_size = int(os.getenv("POPULATION_SIZE"))
    if os.getenv("MAX_ITERATIONS"):
        config.max_iterations = int(os.getenv("MAX_ITERATIONS"))
    if os.getenv("MAX_TIME"):
        config.max_time = int(os.getenv("MAX_TIME"))
    if os.getenv("MAX_TOKENS"):
        config.max_tokens = int(os.getenv("MAX_TOKENS"))
    
    # Kernel settings
    config.kernel_type = os.getenv("KERNEL_TYPE", config.kernel_type)
    
    return config
