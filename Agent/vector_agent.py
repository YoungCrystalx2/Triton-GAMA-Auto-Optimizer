from agent.base_agent import BaseAgent


class VectorAgent(BaseAgent):
    """Vectorization/SIMD-focused mutation agent."""

    def create_prompt(self, code: str) -> str:
        return f"""You are a Triton-CPU vectorization mutation agent.
Apply one or two small SIMD/vectorization-oriented mutations only.

Focus areas:
- vector-friendly BLOCK_SIZE tuning
- reducing scalar-like repetitive operations
- improving data-parallel arithmetic structure
- minimizing control-flow disruption to vectorization

Hard constraints:
- Keep function signature unchanged
- Keep algorithm and output behavior unchanged
- Do not rewrite whole kernel
- Return only complete Python kernel code with @triton.jit

Current kernel:
```python
{code}
```"""
