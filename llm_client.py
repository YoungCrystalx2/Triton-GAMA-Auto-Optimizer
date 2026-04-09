"""
LLM Client for code generation and optimization.
Supports multiple LLM providers as specified in the requirements.
"""
import json
import os
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    tokens_used: int
    model: str
    finish_reason: str = "stop"


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.total_tokens = 0
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> LLMResponse:
        """
        Generate text from prompt.
        核心方法：根据输入提示词生成内容
        输入prompt:进化算法生成的"优化指令"
        输出:LLM生成的优化后内核代码
        """
        pass
    
    def get_total_tokens(self) -> int:
        """
        Get total tokens used.
        获取累计token使用量(用于限制最大预算)
        """
        return self.total_tokens


 
class QwenClient(LLMClient):
    """Qwen API client (Qwen3 series)."""
    
    def __init__(self, model_name: str = "qwen3-coder-plus", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        try:
            import dashscope
            self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            dashscope.api_key = self.api_key
        except ImportError:
            raise ImportError("Please install dashscope: pip install dashscope")
    
    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> LLMResponse:
        """Generate using Qwen API."""
        try:
            import dashscope
            from dashscope import Generation
            
            # For Qwen3-Coder-Plus and newer models, use messages format
            # For older models, try prompt format first
            if 'coder' in self.model_name.lower() or 'qwen3' in self.model_name.lower():
                # Use messages format for Qwen3-Coder models
                response = Generation.call(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    result_format='message',  # Ensures response is in structured message format
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if response.status_code == 200:
                    content = response.output.choices[0].message.content
                    tokens = response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else 100
                    self.total_tokens += tokens
                    return LLMResponse(
                        content=content,
                        tokens_used=tokens,
                        model=self.model_name,
                        finish_reason="stop"
                    )
                else:
                    raise Exception(f"API error: {response.code} - {response.message}")
            else:
                # Fallback to prompt format for older models
                response = Generation.call(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if response.status_code == 200:
                    content = response.output.text if hasattr(response.output, 'text') else str(response.output)
                    tokens = response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else 100
                    self.total_tokens += tokens
                    return LLMResponse(
                        content=content,
                        tokens_used=tokens,
                        model=self.model_name,
                        finish_reason="stop"
                    )
                else:
                    raise Exception(f"API error: {response.code} - {response.message}")
        except Exception as e:
            print(f"Warning: Qwen API error: {e}")
            import traceback
            traceback.print_exc()
            # Don't return mock response - raise the error so user knows
            raise



def create_llm_client(provider: str, model_name: str, api_key: Optional[str] = None) -> LLMClient:
    """Factory function to create LLM client."""
    provider = provider.lower()
    
    if provider == "qwen":
        return QwenClient(model_name, api_key)
    # elif provider == "deepseek":
    #     return DeepSeekClient(model_name, api_key)
    # elif provider == "glm":
    #     return GLMClient(model_name, api_key)
    else:
        # Default to a mock client for testing
        print(f"Warning: Unknown provider {provider}. Using mock client.")
        return DeepSeekClient(model_name, api_key)