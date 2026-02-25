"""
Unified LLM client for OpenAI and OpenRouter APIs.
Supports multiple models with consistent interface.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import httpx
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LLM_MODELS, LLMModel, JUDGE_MODEL,
    MAX_OUTPUT_TOKENS, TOP_P,
    OPENROUTER_BASE_URL
)

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    finish_reason: str


class LLMClient:
    """Unified client for OpenAI and OpenRouter APIs."""
    
    def __init__(self):
        """Initialize clients for both providers."""
        # OpenAI client
        self.openai_client = OpenAI()
        
        # OpenRouter client (using httpx for flexibility)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = OPENROUTER_BASE_URL
        
        if not self.openrouter_api_key:
            print("Warning: OPENROUTER_API_KEY not set. OpenRouter models will not work.")
    
    def _call_openai(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        top_p: float = TOP_P
    ) -> LLMResponse:
        """Call OpenAI API."""
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider="openai",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            finish_reason=response.choices[0].finish_reason
        )
    
    def _call_openrouter(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        top_p: float = TOP_P
    ) -> LLMResponse:
        """Call OpenRouter API."""
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/kazakh-rag-experiment",
            "X-Title": "Kazakh Legal RAG Experiment"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", model),
            provider="openrouter",
            usage=data.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }),
            finish_reason=data["choices"][0].get("finish_reason", "stop")
        )
    
    def generate(
        self,
        model: LLMModel,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate response from specified model.
        
        Args:
            model: LLMModel configuration
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Override default max tokens
            
        Returns:
            LLMResponse object
        """
        max_tokens = max_tokens or MAX_OUTPUT_TOKENS
        
        if model.provider == "openai":
            return self._call_openai(
                model=model.api_id,
                messages=messages,
                temperature=model.temperature,
                max_tokens=max_tokens
            )
        elif model.provider == "openrouter":
            return self._call_openrouter(
                model=model.api_id,
                messages=messages,
                temperature=model.temperature,
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Unknown provider: {model.provider}")
    
    def generate_with_system_prompt(
        self,
        model: LLMModel,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate response with system prompt.
        
        Args:
            model: LLMModel configuration
            system_prompt: System prompt text
            user_message: User message text
            max_tokens: Override default max tokens
            
        Returns:
            LLMResponse object
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.generate(model, messages, max_tokens)


def get_model_by_name(name: str) -> Optional[LLMModel]:
    """Get model config by name."""
    for model in LLM_MODELS:
        if model.name == name:
            return model
    return None


def get_all_models() -> List[LLMModel]:
    """Get all configured models."""
    return LLM_MODELS


def get_judge_model() -> LLMModel:
    """Get judge model config."""
    return JUDGE_MODEL


# Test function
def test_clients():
    """Test both API clients."""
    client = LLMClient()
    
    test_messages = [
        {"role": "user", "content": "Say 'test successful' in exactly 2 words."}
    ]
    
    print("Testing OpenAI...")
    try:
        openai_model = get_model_by_name("GPT-4o-mini")
        response = client.generate(openai_model, test_messages)
        print(f"  OpenAI response: {response.content}")
        print(f"  Tokens used: {response.usage}")
    except Exception as e:
        print(f"  OpenAI error: {e}")
    
    print("\nTesting OpenRouter...")
    try:
        openrouter_model = get_model_by_name("Qwen-2.5-72B")
        response = client.generate(openrouter_model, test_messages)
        print(f"  OpenRouter response: {response.content}")
        print(f"  Tokens used: {response.usage}")
    except Exception as e:
        print(f"  OpenRouter error: {e}")


if __name__ == "__main__":
    test_clients()
