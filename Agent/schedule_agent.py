from agent.base_agent import BaseAgent


class ScheduleAgent(BaseAgent):
    """Parallel scheduling-focused mutation agent."""

    def create_prompt(self, code: str) -> str:
        return f"""You are a Triton-CPU scheduling mutation agent.
Apply one or two small schedule-level mutations only.

Focus areas:
- grid/work partition adjustments
- balancing parallel tiles
- reducing scheduling overhead
- improving execution regularity

Hard constraints:
- Keep function signature unchanged
- Keep algorithm and output behavior unchanged
- Do not rewrite whole kernel
- Return only complete Python kernel code with @triton.jit

Current kernel:
```python
{code}
```"""
