import numpy as np
import os
from exo.helpers import DEBUG  # Make sure to import DEBUG

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from .shard import Shard
from exo.download.shard_download import ShardDownloader


class InferenceEngine(ABC):
  session = {}

  @abstractmethod
  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    pass

  @abstractmethod
  async def sample(self, x: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    pass

  @abstractmethod
  async def load_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_session(self, key, value):
    self.session[key] = value

  async def clear_session(self):
    self.session.empty()

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    tokens = await self.encode(shard, prompt)
    if shard.model_id != 'stable-diffusion-2-1-base':
      x = tokens.reshape(1, -1)
    else:
      x = tokens
    output_data, inference_state = await self.infer_tensor(request_id, shard, x, inference_state)

    return output_data, inference_state

  # New methods for multi-token generation support
  async def infer_tensor_multi(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict], Optional[list]]:
    """
    Extended infer_tensor that can return multiple tokens for speculative decoding.
    Returns (logits, inference_state, generated_tokens_list_or_None)
    
    For standard engines: generated_tokens_list_or_None = None (fall back to normal single-token processing)
    For speculative engines: generated_tokens_list_or_None = [token1, token2, ...] (multi-token result)
    """
    # Default implementation falls back to single-token behavior
    output_data, inference_state = await self.infer_tensor(request_id, shard, input_data, inference_state)
    return output_data, inference_state, None

  async def infer_prompt_multi(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict], Optional[list]]:
    """
    Extended infer_prompt that can return multiple tokens for speculative decoding.
    Returns (logits, inference_state, generated_tokens_list_or_None)
    """
    tokens = await self.encode(shard, prompt)
    if shard.model_id != 'stable-diffusion-2-1-base':
      x = tokens.reshape(1, -1)
    else:
      x = tokens
    output_data, inference_state, generated_tokens = await self.infer_tensor_multi(request_id, shard, x, inference_state)

    return output_data, inference_state, generated_tokens


inference_engine_classes = {
  "mlx": "MLXDynamicShardInferenceEngine",
  "tinygrad": "TinygradDynamicShardInferenceEngine",
  "torch": "TorchDynamicShardInferenceEngine",
  "dummy": "DummyInferenceEngine",
  "speculative": "SpeculativeInferenceEngine",
}


def get_inference_engine(inference_engine_name: str, shard_downloader: ShardDownloader, speculative_config: Optional[dict] = None):
  if DEBUG >= 2:
    print(f"get_inference_engine called with: {inference_engine_name}")
  
  if inference_engine_name == "mlx":
    from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
    return MLXDynamicShardInferenceEngine(shard_downloader)
  
  elif inference_engine_name == "tinygrad":
    from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
    import tinygrad.helpers
    tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))
    return TinygradDynamicShardInferenceEngine(shard_downloader)
  
  elif inference_engine_name == "torch":
    from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
    return TorchDynamicShardInferenceEngine(shard_downloader)
  
  elif inference_engine_name == "dummy":
    from exo.inference.dummy_inference_engine import DummyInferenceEngine
    return DummyInferenceEngine()
  
  elif inference_engine_name == "speculative":
    from exo.inference.speculative_inference_engine import SpeculativeInferenceEngine
    
    # Default speculative config
    config = {
      'target_engine_name': 'mlx',  # Default to MLX
      'draft_engine_name': None,    # No draft engine by default (use early exit)
      'target_model_id': None,      # Will be set at runtime
      'draft_model_id': None,       # Will be set at runtime
      'gamma': 5,
      'temperature': 1.0,
      'top_k_threshold': 0.9,
      'lenience': 1.0,
      'enable_speculative': True,
      'early_exit_layer': None
    }
    
    # Update with provided config
    if speculative_config:
      config.update(speculative_config)
    
    # Create target engine - ALWAYS create a new instance
    if config['target_engine_name'] == 'tinygrad':
      from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
      import tinygrad.helpers
      tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))
      target_engine = TinygradDynamicShardInferenceEngine(shard_downloader)
    elif config['target_engine_name'] == 'torch':
      from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
      target_engine = TorchDynamicShardInferenceEngine(shard_downloader)
    else:
      target_engine = get_inference_engine(config['target_engine_name'], shard_downloader)
    
    # Create draft engine if specified - ALWAYS create a separate instance
    draft_engine = None
    if config['draft_engine_name']:
      # 🔧 CRITICAL FIX: Create separate shard downloader to prevent model weight sharing
      # The cached downloader shares weights between engines with same class name
      from exo.download.new_shard_download import new_shard_downloader
      draft_shard_downloader = new_shard_downloader()
      
      if DEBUG >= 1:
        print(f"🔧 Created separate shard downloader for draft engine:")
        print(f"   Target downloader: {id(shard_downloader)}")
        print(f"   Draft downloader: {id(draft_shard_downloader)}")
      
      if config['draft_engine_name'] == 'tinygrad':
        from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine  
        import tinygrad.helpers
        tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))
        draft_engine = TinygradDynamicShardInferenceEngine(draft_shard_downloader)
      elif config['draft_engine_name'] == 'torch':
        from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine
        draft_engine = TorchDynamicShardInferenceEngine(draft_shard_downloader)
      else:
        draft_engine = get_inference_engine(config['draft_engine_name'], draft_shard_downloader)
    
    return SpeculativeInferenceEngine(
      target_engine=target_engine,
      draft_engine=draft_engine,
      gamma=config['gamma'],
      temperature=config['temperature'],
      top_k_threshold=config['top_k_threshold'],
      lenience=config['lenience'],
      target_model_id=config['target_model_id'],
      draft_model_id=config['draft_model_id']
    )
  
  raise ValueError(f"Unsupported inference engine: {inference_engine_name}")


def get_model_family_variants(model_id: str) -> dict:
  """
  Get draft model variants for a given target model from the same family.
  Returns suggested draft models that work well with the target.
  """
  family_variants = {
    # LLaMA family
    "llama-3.1-8b": ["llama-3.2-3b", "llama-3.2-1b"],
    "llama-3.1-70b": ["llama-3.2-3b", "llama-3.1-8b"],
    "llama-3.1-405b": ["llama-3.1-8b", "llama-3.2-3b"],
    "llama-3.2-3b": ["llama-3.2-1b"],
    "llama-3.3-70b": ["llama-3.2-3b", "llama-3.1-8b"],
    
    # Qwen family
    "qwen-2.5-7b": ["qwen-2.5-1.5b", "qwen-2.5-0.5b"],
    "qwen-2.5-14b": ["qwen-2.5-3b", "qwen-2.5-1.5b"],
    "qwen-2.5-32b": ["qwen-2.5-7b", "qwen-2.5-3b"],
    "qwen-2.5-72b": ["qwen-2.5-14b", "qwen-2.5-7b"],
    
    # DeepSeek family
    "deepseek-v3": ["deepseek-r1-distill-qwen-7b", "deepseek-r1-distill-qwen-1.5b"],
    "deepseek-r1": ["deepseek-r1-distill-llama-8b", "deepseek-r1-distill-qwen-7b"],
    
    # Gemma family
    "gemma2-27b": ["gemma2-9b"],
    
    # Phi family 
    "phi-4": ["phi-3.5-mini"],
  }
  
  return {
    "target": model_id,
    "suggested_drafts": family_variants.get(model_id, []),
    "family": _get_model_family(model_id)
  }


def _get_model_family(model_id: str) -> str:
  """Determine the model family from model ID."""
  if "llama" in model_id.lower():
    return "llama"
  elif "qwen" in model_id.lower():
    return "qwen"
  elif "deepseek" in model_id.lower():
    return "deepseek"
  elif "gemma" in model_id.lower():
    return "gemma"
  elif "phi" in model_id.lower():
    return "phi"
  elif "mistral" in model_id.lower():
    return "mistral"
  else:
    return "unknown"


def suggest_speculative_config(target_model_id: str, draft_model_id: Optional[str] = None) -> dict:
  """
  Suggest optimal speculative decoding configuration for given models.
  """
  # Import here to avoid circular imports
  from exo.helpers import get_system_info
  
  family_info = get_model_family_variants(target_model_id)
  
  # Auto-select draft model if not provided
  if not draft_model_id and family_info["suggested_drafts"]:
    draft_model_id = family_info["suggested_drafts"][0]
  
  # Auto-detect the best inference engine for the system
  system_info = get_system_info()
  default_engine = "mlx" if system_info == "Apple Silicon Mac" else "tinygrad"
  
  # Family-specific optimizations
  if family_info["family"] == "llama":
    # LLaMA models work well with higher gamma
    return {
      'target_model_id': target_model_id,
      'draft_model_id': draft_model_id,
      'gamma': 6,
      'temperature': 1.0,
      'top_k_threshold': 0.9,
      'lenience': 1.1,
      'target_engine_name': default_engine,
      'draft_engine_name': default_engine if draft_model_id else None,
      'enable_speculative': True,
      'early_exit_layer': None
    }
  elif family_info["family"] == "qwen":
    # Qwen models are more conservative
    return {
      'target_model_id': target_model_id,
      'draft_model_id': draft_model_id,
      'gamma': 4,
      'temperature': 0.8,
      'top_k_threshold': 0.85,
      'lenience': 1.0,
      'target_engine_name': default_engine,
      'draft_engine_name': default_engine if draft_model_id else None,
      'enable_speculative': True,
      'early_exit_layer': None
    }
  else:
    # Default configuration
    return {
      'target_model_id': target_model_id,
      'draft_model_id': draft_model_id,
      'gamma': 5,
      'temperature': 1.0,
      'top_k_threshold': 0.9,
      'lenience': 1.0,
      'target_engine_name': default_engine,
      'draft_engine_name': default_engine if draft_model_id else None,
      'enable_speculative': True,
      'early_exit_layer': None
    }
