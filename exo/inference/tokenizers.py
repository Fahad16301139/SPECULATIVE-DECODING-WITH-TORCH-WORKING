import traceback
from os import PathLike
from aiofiles import os as aios
from typing import Union
from transformers import AutoTokenizer, AutoProcessor
import numpy as np
from exo.helpers import DEBUG
from exo.download.new_shard_download import ensure_downloads_dir


class DummyTokenizer:
  def __init__(self):
    self.eos_token_id = 69
    self.vocab_size = 1000

  def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
    return "dummy_tokenized_prompt"

  def encode(self, text):
    return np.array([1])

  def decode(self, tokens):
    return "dummy" * len(tokens)


async def resolve_tokenizer(repo_id: Union[str, PathLike]):
  if repo_id == "dummy":
    return DummyTokenizer()
  local_path = await ensure_downloads_dir()/str(repo_id).replace("/", "--")
  if DEBUG >= 2: print(f"Checking if local path exists to load tokenizer from local {local_path=}")
  try:
    if local_path and await aios.path.exists(local_path):
      if DEBUG >= 2: print(f"Resolving tokenizer for {repo_id=} from {local_path=}")
      return await _resolve_tokenizer(local_path)
  except:
    if DEBUG >= 5: print(f"Local check for {local_path=} failed. Resolving tokenizer for {repo_id=} normally...")
    if DEBUG >= 5: traceback.print_exc()
  return await _resolve_tokenizer(repo_id)


async def _resolve_tokenizer(repo_id_or_local_path: Union[str, PathLike]):
  # Convert to string for easier handling
  repo_str = str(repo_id_or_local_path)
  
  # Special handling for unsloth LLaMA models
  if "unsloth" in repo_str and "Llama" in repo_str:
    if DEBUG >= 2: print(f"Detected unsloth LLaMA model: {repo_str}")
    
    # Try to map to base model for tokenizer
    base_model_mapping = {
      "unsloth/Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B",
      "unsloth/Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B", 
      "unsloth/Llama-3.2-8B-Instruct": "meta-llama/Llama-3.1-8B",
      "unsloth/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B",
      "unsloth/Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B",
    }
    
    # Check if we have a mapping
    if repo_str in base_model_mapping:
      base_model = base_model_mapping[repo_str]
      if DEBUG >= 2: print(f"Using base model tokenizer: {base_model}")
      try:
        return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
      except Exception as e:
        if DEBUG >= 2: print(f"Failed to load base model tokenizer {base_model}: {e}")
    
    # Fallback: try loading directly from local path if available
    try:
      if DEBUG >= 2: print(f"Trying to load tokenizer directly from local path: {repo_id_or_local_path}")
      return AutoTokenizer.from_pretrained(repo_id_or_local_path, trust_remote_code=True, local_files_only=True)
    except Exception as e:
      if DEBUG >= 2: print(f"Failed to load local tokenizer: {e}")

  try:
    if DEBUG >= 4: print(f"Trying AutoProcessor for {repo_id_or_local_path}")
    processor = AutoProcessor.from_pretrained(repo_id_or_local_path, use_fast=True if "Mistral-Large" in f"{repo_id_or_local_path}" else False, trust_remote_code=True)
    if not hasattr(processor, 'eos_token_id'):
      processor.eos_token_id = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).eos_token_id
    if not hasattr(processor, 'encode'):
      processor.encode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).encode
    if not hasattr(processor, 'decode'):
      processor.decode = getattr(processor, 'tokenizer', getattr(processor, '_tokenizer', processor)).decode
    return processor
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load processor for {repo_id_or_local_path}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  try:
    if DEBUG >= 4: print(f"Trying AutoTokenizer for {repo_id_or_local_path}")
    return AutoTokenizer.from_pretrained(repo_id_or_local_path, trust_remote_code=True)
  except Exception as e:
    if DEBUG >= 4: print(f"Failed to load tokenizer for {repo_id_or_local_path}. Error: {e}")
    if DEBUG >= 4: print(traceback.format_exc())

  # Final fallback: create a basic LLaMA tokenizer if this looks like a LLaMA model
  if "llama" in repo_str.lower() or "Llama" in repo_str:
    if DEBUG >= 2: print(f"Creating fallback LLaMA tokenizer for {repo_str}")
    try:
      # Use a known working LLaMA tokenizer as fallback
      return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
    except Exception as e:
      if DEBUG >= 2: print(f"Fallback LLaMA tokenizer failed: {e}")

  raise ValueError(f"[TODO] Unsupported model: {repo_id_or_local_path}")
