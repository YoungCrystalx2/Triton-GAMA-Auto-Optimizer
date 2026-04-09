"""
Evolutionary Algorithm-based Triton Kernel Optimizer.
Implements EvoPrompt-style evolutionary optimization for Triton kernels.
"""
import random
import time
import copy
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import re

from llm_client import LLMClient, LLMResponse
from evaluator import TritonKernelEvaluator, EvaluationResult
from agent.loop_agent import LoopAgent
from agent.memory_agent import MemoryAgent
from agent.vector_agent import VectorAgent
from agent.schedule_agent import ScheduleAgent


@dataclass
class Individual:
    """An individual in the evolutionary population."""
    code: str  # Triton kernel code
    fitness: float  # Fitness score (speedup ratio)
    functional: bool  # Whether the kernel is functionally correct
    evaluation_result: Optional[EvaluationResult] = None


class EvolutionaryOptimizer:
    """Evolutionary optimizer for Triton kernels."""
    
    def __init__(
        self,
        baseline_code: str,
        evaluator: TritonKernelEvaluator,
        llm_client: LLMClient,
        population_size: int = 5,
        max_iterations: int = 10,
        max_time: int = 1200,  # 20 minutes in seconds
        max_tokens: int = 200000,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.5,
        selection_size: int = 2,
        kernel_name: str = "kernel"
    ):
        """
        Args:
            baseline_code: Original kernel code to optimize
            evaluator: Evaluator for testing kernels
            llm_client: LLM client for code generation
            population_size: Size of the population
            max_iterations: Maximum number of iterations
            max_time: Maximum time in seconds
            max_tokens: Maximum token budget
            mutation_rate: Probability of mutation operation
            crossover_rate: Probability of crossover operation
            selection_size: Number of parents to select
            kernel_name: Name of the kernel function
        """
        self.baseline_code = baseline_code
        self.evaluator = evaluator
        self.llm_client = llm_client
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_size = selection_size
        self.kernel_name = kernel_name
        
        self.population: List[Individual] = []
        self.generation = 0
        self.start_time = None
        self.best_individual: Optional[Individual] = None
        self.best_fitness_stagnant = 0

        # Specialized mutation agents (used only inside mutate()).
        self.loop_agent = LoopAgent()
        self.memory_agent = MemoryAgent()
        self.vector_agent = VectorAgent()
        self.schedule_agent = ScheduleAgent()
        
    def initialize_population(self) -> List[Individual]:
        """初始化种群【初始种群环节】"""
        """Initialize population with baseline code and variations."""
        population = []
        
        # Add baseline
        baseline_result = self.evaluator.cascade_evaluate(self.baseline_code, self.kernel_name)
        population.append(Individual(
            code=self.baseline_code,
            fitness=baseline_result.fitness,
            functional=baseline_result.functional,
            evaluation_result=baseline_result
        ))
        
        # Generate initial variations using LLM
        initial_prompt = self._create_initial_variation_prompt(self.baseline_code)
        for i in range(self.population_size - 1):
            if self._should_stop():
                break
            
            try:
                response = self.llm_client.generate(
                    prompt=initial_prompt,
                    max_tokens=2000,
                    temperature=0.8
                )
                new_code = self._extract_kernel_code(response.content)
                
                if new_code:
                    result = self.evaluator.cascade_evaluate(new_code, f"{self.kernel_name}_init_{i}")
                    population.append(Individual(
                        code=new_code,
                        fitness=result.fitness,
                        functional=result.functional,
                        evaluation_result=result
                    ))
            except Exception as e:
                print(f"Error initializing individual {i}: {e}")
                # Use baseline as fallback
                population.append(Individual(
                    code=self.baseline_code,
                    fitness=0.0,
                    functional=True,
                    evaluation_result=baseline_result
                ))
        
        return population
    
    def select_parents(self, population: List[Individual], n: int) -> List[Individual]:
        """Select parents using tournament selection."""
        selected = []
        
        # Prefer functional individuals with higher fitness
        functional_individuals = [ind for ind in population if ind.functional]
        if len(functional_individuals) >= n:
            # Sort by fitness descending
            functional_individuals.sort(key=lambda x: x.fitness, reverse=True)
            return functional_individuals[:n]
        
        # If not enough functional individuals, add non-functional ones
        all_individuals = sorted(population, key=lambda x: (x.functional, x.fitness), reverse=True)
        return all_individuals[:n]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> str:
        """Crossover operation using LLM to combine two parent codes."""
        crossover_prompt = self._create_crossover_prompt(parent1.code, parent2.code)
        
        try:
            response = self.llm_client.generate(
                prompt=crossover_prompt,
                max_tokens=2000,
                temperature=0.7
            )
            new_code = self._extract_kernel_code(response.content)
            return new_code if new_code else parent1.code
        except Exception as e:
            print(f"Crossover error: {e}")
            return parent1.code
    
    def mutate(self, individual: Individual) -> str:
        """Mutation operation using LLM to modify the code."""
        agent = random.choice([
            self.loop_agent,
            self.memory_agent,
            self.vector_agent,
            self.schedule_agent
        ])
        mutation_prompt = agent.create_prompt(individual.code)
        
        try:
            response = self.llm_client.generate(
                prompt=mutation_prompt,
                max_tokens=2000,
                temperature=0.9  # Higher temperature for more diversity
            )
            new_code = self._extract_kernel_code(response.content)
            return new_code if new_code else individual.code
        except Exception as e:
            print(f"Mutation error: {e}")
            return individual.code
    
    def evolve(self) -> List[Individual]:
        """Main evolutionary loop."""
        self.start_time = time.time()
        self.population = self.initialize_population()
        self.generation = 0
        
        print(f"Initialized population with {len(self.population)} individuals")
        
        for generation in range(self.max_iterations):
            if self._should_stop():
                print(f"Stopping early due to resource limits")
                break
            
            self.generation = generation
            print(f"\n=== Generation {generation + 1}/{self.max_iterations} ===")
            
            # Evaluate all individuals (skip if already evaluated)
            for i, ind in enumerate(self.population):
                if ind.evaluation_result is None:
                    result = self.evaluator.cascade_evaluate(ind.code, f"{self.kernel_name}_gen{generation}_ind{i}")
                    ind.evaluation_result = result
                    ind.functional = result.functional
                    ind.fitness = result.fitness
            
            # Update best individual
            functional_individuals = [ind for ind in self.population if ind.functional]
            if functional_individuals:
                best = max(functional_individuals, key=lambda x: x.fitness)
                if self.best_individual is None or best.fitness > self.best_individual.fitness:
                    self.best_individual = best
                    self.best_fitness_stagnant = 0
                    print(f"New best fitness: {best.fitness:.4f}")
                else:
                    self.best_fitness_stagnant += 1
            else:
                self.best_fitness_stagnant += 1

            if self.best_fitness_stagnant >= 5:
                print("Early stopping: best fitness stagnated for 5 generations")
                break
            
            # Select parents
            parents = self.select_parents(self.population, self.selection_size)
            
            # Generate new offspring
            new_population = []
            
            # Keep best individuals (elitism)
            elite_size = min(2, len([ind for ind in self.population if ind.functional]))
            if elite_size > 0:
                elite = sorted([ind for ind in self.population if ind.functional], 
                             key=lambda x: x.fitness, reverse=True)[:elite_size]
                new_population.extend(elite)
            
            # Generate offspring
            while len(new_population) < self.population_size:
                if self._should_stop():
                    break
                
                # Choose operation
                if random.random() < self.mutation_rate and len(parents) > 0:
                    # Mutation
                    parent = random.choice(parents)
                    new_code = self.mutate(parent)
                elif random.random() < self.crossover_rate and len(parents) >= 2:
                    # Crossover
                    parent1, parent2 = random.sample(parents, 2)
                    new_code = self.crossover(parent1, parent2)
                else:
                    # Use baseline
                    new_code = self.baseline_code
                
                # Evaluate new individual
                result = self.evaluator.cascade_evaluate(
                    new_code, 
                    f"{self.kernel_name}_gen{generation}_new{len(new_population)}"
                )
                new_individual = Individual(
                    code=new_code,
                    fitness=result.fitness,
                    functional=result.functional,
                    evaluation_result=result
                )
                new_population.append(new_individual)
            
            self.population = new_population
            
            # Print statistics
            functional_count = sum(1 for ind in self.population if ind.functional)
            avg_fitness = sum(ind.fitness for ind in self.population if ind.functional) / max(functional_count, 1)
            print(f"Functional: {functional_count}/{len(self.population)}, "
                  f"Avg fitness: {avg_fitness:.4f}, "
                  f"Tokens used: {self.llm_client.get_total_tokens()}")
        
        # Final evaluation and selection
        final_results = []
        functional_individuals = [ind for ind in self.population if ind.functional]
        
        if functional_individuals:
            # Sort by fitness
            functional_individuals.sort(key=lambda x: x.fitness, reverse=True)
            # Return up to 5 best functional individuals
            final_results = functional_individuals[:5]
        else:
            # If no functional individuals, return baseline
            baseline_result = self.evaluator.cascade_evaluate(self.baseline_code, self.kernel_name)
            final_results = [Individual(
                code=self.baseline_code,
                fitness=baseline_result.fitness,
                functional=baseline_result.functional,
                evaluation_result=baseline_result
            )]
        
        return final_results
    
    def _should_stop(self) -> bool:
        """Check if optimization should stop due to resource limits."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        tokens_used = self.llm_client.get_total_tokens()
        
        if elapsed_time > self.max_time:
            print(f"Time limit reached: {elapsed_time:.1f}s > {self.max_time}s")
            return True
        if tokens_used > self.max_tokens:
            print(f"Token limit reached: {tokens_used} > {self.max_tokens}")
            return True
        return False
    
    def _extract_kernel_code(self, llm_output: str) -> Optional[str]:
        """Extract kernel code from LLM output."""
        # Try to find code blocks
        patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'@triton\.jit\s+def.*?(?=\n\n|\nclass|\Z)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, llm_output, re.DOTALL)
            if matches:
                code = matches[0].strip()
                # Ensure it's a complete kernel function
                if '@triton.jit' in code or 'def ' in code:
                    return code
        
        # If no code block found, try to extract function definition
        lines = llm_output.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if '@triton.jit' in line or (in_code and line.strip()):
                in_code = True
                code_lines.append(line)
            elif in_code and not line.strip():
                if code_lines:
                    code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return None
    
    def _create_initial_variation_prompt(self, baseline_code: str) -> str:
        """Create prompt for generating initial variations."""
        return f"""You are an expert in Triton-CPU kernel optimization. Given the following Triton matrix multiplication kernel code, generate an optimized version that maintains functional correctness while improving performance.

Original kernel code:
```python
{baseline_code}
```

Generate an optimized version of this kernel. The optimization should:
1. Maintain functional correctness (same input/output behavior, same numerical results)
2. Improve performance through:
   - Optimizing block sizes (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) for CPU cache hierarchy
   - Improving memory access patterns (coalesced access, prefetching)
   - Better loop tiling and unrolling strategies
   - Reducing redundant computations
   - Optimizing mask operations
   - Using appropriate data types (float32 for CPU)
   - Cache-friendly data access patterns

Important constraints:
- The kernel signature must match: matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr)
- Use 2D grid: grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
- Ensure all memory accesses are properly masked
- Use tl.float32 for accumulator (not float16)

Return only the complete optimized kernel code with the @triton.jit decorator. Do not include explanations, comments, or markdown formatting - just the Python code."""

    def _create_mutation_prompt(self, code: str) -> str:
        """Create prompt for mutation operation."""
        return f"""You are performing an evolutionary mutation on a Triton-CPU matrix multiplication kernel. This mutation is one step in an evolutionary optimization process (mutation, crossover, selection).
The explicit optimization objective of this mutation is to **reduce runtime on Triton-CPU** (wall-clock execution time), not GPU performance and not theoretical FLOPs.
The goal is to generate a slightly modified variant of the current code that may improve Triton-CPU execution time while strictly preserving correctness.

Current kernel code:
```python
{code}

Mutation rules:

Apply only one or two localized changes; do NOT perform a full rewrite
Keep the overall structure, algorithm, and function signature unchanged
The mutated code must be a plausible direct descendant of the current code

Choose exactly ONE primary mutation strategy (optionally one minor secondary tweak):
Mutate block sizes (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) using nearby values (e.g., 32, 64, 128) to better match CPU cache hierarchy
Slightly reorder memory load/store or pointer arithmetic to improve CPU cache locality
Apply a small modification to the K-dimension loop (tiling, offset update style, or light unrolling) targeting CPU instruction efficiency
Simplify or slightly adjust mask computations to reduce CPU-side control-flow overhead
Modify accumulator initialization or accumulation pattern while preserving fp32 correctness and reducing instruction count
Reduce redundant index or offset computations that increase CPU instruction pressure
CPU-specific mutation guidance (Triton-CPU focused):
Favor cache-friendly and sequential memory access on CPU
Avoid GPU-specific optimizations or assumptions
Avoid introducing new buffers or algorithmic changes
Prefer small constant-factor improvements that reduce instruction count or memory traffic

Hard constraints:
Do NOT introduce new features, APIs, or comments
Do NOT refactor the entire kernel
Do NOT include explanations, markdown, or extra text
Return only the complete mutated kernel code with @triton.jit decorator. Do not include explanations or markdown."""


    def _create_crossover_prompt(self, code1: str, code2: str) -> str:
        """Create prompt for crossover operation."""
        return f"""You are combining two Triton kernels to create an improved version. Take the best features from both and merge them into a single optimized kernel.

Kernel 1:
```python
{code1}
```

Kernel 2:
```python
{code2}
```

Combine the best optimization techniques from both kernels into a single optimized kernel. Maintain functional correctness.

Return only the combined kernel code. Do not include explanations."""
