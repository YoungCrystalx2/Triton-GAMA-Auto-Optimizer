from agent.base_agent import BaseAgent


class LoopAgent(BaseAgent):
    """Loop-focused mutation agent."""

    def create_prompt(self, code: str) -> str:
        return f"""You are a Triton-CPU loop optimization mutation agent.
Apply one or two small loop-level mutations only.

Focus areas:
- loop tiling adjustment
- light loop unrolling
- loop order swap if safe
- reducing loop-carried overhead

Hard constraints:
- Keep function signature unchanged
- Keep algorithm and output behavior unchanged
- Do not rewrite whole kernel
- Return only complete Python kernel code with @triton.jit

Current kernel:
```python
{code}
```"""
