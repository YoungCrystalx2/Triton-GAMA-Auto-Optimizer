from agent.base_agent import BaseAgent


class MemoryAgent(BaseAgent):
    """Memory-access-focused mutation agent."""

    def create_prompt(self, code: str) -> str:
        return f"""You are a Triton-CPU memory optimization mutation agent.
Apply one or two small memory-access mutations only.

Focus areas:
- improve cache locality
- improve contiguous access patterns
- reduce redundant loads/stores
- improve pointer arithmetic efficiency

Hard constraints:
- Keep function signature unchanged
- Keep algorithm and output behavior unchanged
- Do not rewrite whole kernel
- Return only complete Python kernel code with @triton.jit

Current kernel:
```python
{code}
```"""
