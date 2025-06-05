# Speculative Decoding in exo

This document describes the speculative decoding integration with exo, a distributed AI inference framework.

## Overview

Speculative decoding is an optimization technique that accelerates language model inference by generating multiple tokens in parallel and then verifying them with the main model. This can significantly reduce latency while maintaining the same output quality.

exo supports speculative decoding through a wrapper inference engine that can work with any underlying inference engine (MLX, TinyGrad, etc.) and supports **different models from the same family**, exactly as described in the original research papers.

## How It Works

### Draft-Target Approach (Recommended)
This follows the original speculative decoding research where different models from the same family are used:

1. **Draft Phase**: A smaller, faster model (e.g., LLaMA-3B) generates multiple candidate tokens
2. **Verification Phase**: The larger target model (e.g., LLaMA-8B) verifies these candidates in parallel
3. **Acceptance/Rejection**: Tokens are accepted or rejected based on probability distributions
4. **Speedup**: Multiple tokens can be generated in roughly the same time as one token

**Key Advantage**: Uses models specifically designed to work together from the same family.

### Early Exit Approach
1. **Early Layers**: Use early transformer layers to generate draft tokens
2. **Full Model**: Verify with the complete model
3. **Caching**: Reuse intermediate computations for efficiency

## Model Family Support

exo automatically suggests compatible draft models based on the target model:

### LLaMA Family
- **Target: llama-3.1-8b** → Draft: llama-3.2-3b, llama-3.2-1b
- **Target: llama-3.1-70b** → Draft: llama-3.2-3b, llama-3.1-8b
- **Target: llama-3.1-405b** → Draft: llama-3.1-8b, llama-3.2-3b
- **Target: llama-3.3-70b** → Draft: llama-3.2-3b, llama-3.1-8b

### Qwen Family
- **Target: qwen-2.5-7b** → Draft: qwen-2.5-1.5b, qwen-2.5-0.5b
- **Target: qwen-2.5-14b** → Draft: qwen-2.5-3b, qwen-2.5-1.5b
- **Target: qwen-2.5-72b** → Draft: qwen-2.5-14b, qwen-2.5-7b

### DeepSeek Family
- **Target: deepseek-v3** → Draft: deepseek-r1-distill-qwen-7b
- **Target: deepseek-r1** → Draft: deepseek-r1-distill-llama-8b

And more families supported!

## Usage

### Basic Usage with Model Families

```bash
# Automatic configuration (recommended)
exo --inference-engine speculative \
    --speculative-target-model llama-3.1-8b \
    --speculative-auto-config

# Manual model specification
exo --inference-engine speculative \
    --speculative-target-model llama-3.1-8b \
    --speculative-draft-model llama-3.2-3b

# Use with any model family
exo --inference-engine speculative \
    --speculative-target-model qwen-2.5-7b \
    --speculative-draft-model qwen-2.5-1.5b \
    --speculative-auto-config
```

### Advanced Configuration

```bash
# Full manual configuration
exo --inference-engine speculative \
    --speculative-target-model llama-3.1-8b \
    --speculative-draft-model llama-3.2-3b \
    --speculative-target-engine mlx \
    --speculative-draft-engine mlx \
    --speculative-gamma 6 \
    --speculative-temperature 1.0 \
    --speculative-top-k 0.9 \
    --speculative-lenience 1.1

# Auto-select draft model from same family
exo --inference-engine speculative \
    --speculative-target-model llama-3.1-70b \
    --speculative-auto-config

# Disable speculative decoding (for comparison)
exo --inference-engine speculative \
    --speculative-target-model llama-3.1-8b \
    --disable-speculative
```

### Running with Different Model Families

```bash
# LLaMA family (optimized settings)
exo run llama-3.1-8b \
    --inference-engine speculative \
    --speculative-target-model llama-3.1-8b \
    --speculative-draft-model llama-3.2-3b \
    --prompt "Explain quantum computing"

# Qwen family (conservative settings)
exo run qwen-2.5-7b \
    --inference-engine speculative \
    --speculative-target-model qwen-2.5-7b \
    --speculative-draft-model qwen-2.5-1.5b \
    --speculative-auto-config

# DeepSeek family
exo run deepseek-v3 \
    --inference-engine speculative \
    --speculative-target-model deepseek-v3 \
    --speculative-auto-config
```

## Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--speculative-target-model` | Target model ID (e.g., llama-3.1-8b) | None | Any supported model |
| `--speculative-draft-model` | Draft model ID (e.g., llama-3.2-3b) | None | Compatible model or None |
| `--speculative-auto-config` | Auto-configure based on model family | False | - |
| `--speculative-target-engine` | Target inference engine | mlx | mlx, tinygrad |
| `--speculative-draft-engine` | Draft inference engine | None | mlx, tinygrad, None |
| `--speculative-gamma` | Number of speculative tokens | 5 | 1-20 |
| `--speculative-temperature` | Sampling temperature | 1.0 | 0.1-2.0 |
| `--speculative-top-k` | Top-k filtering threshold | 0.9 | 0.1-1.0 |
| `--speculative-lenience` | Acceptance lenience factor | 1.0 | 0.5-2.0 |
| `--disable-speculative` | Disable speculation entirely | False | - |

## Family-Specific Optimizations

exo automatically optimizes settings based on model families:

### LLaMA Family
- **Gamma**: 6 (higher speculation length)
- **Temperature**: 1.0
- **Lenience**: 1.1 (more accepting)
- **Rationale**: LLaMA models have consistent architectures across sizes

### Qwen Family
- **Gamma**: 4 (conservative approach)
- **Temperature**: 0.8
- **Top-k**: 0.85
- **Rationale**: Qwen models benefit from more focused sampling

### Default Family
- **Gamma**: 5
- **Temperature**: 1.0
- **Top-k**: 0.9
- **Lenience**: 1.0

## Integration Examples

### Python API with Model Families

```python
from exo.inference.inference_engine import (
    get_inference_engine, 
    suggest_speculative_config,
    get_model_family_variants
)
from exo.download.shard_download import ShardDownloader

# Get suggestions for a target model
target_model = "llama-3.1-8b"
family_info = get_model_family_variants(target_model)
print(f"Suggested drafts: {family_info['suggested_drafts']}")

# Auto-configure
auto_config = suggest_speculative_config(target_model)
print(f"Auto config: {auto_config}")

# Create engine with family models
shard_downloader = ShardDownloader()
engine = get_inference_engine('speculative', shard_downloader, auto_config)

# Check compatibility stats
stats = engine.get_speculation_stats()
print(f"Model compatibility: {stats['model_family_compatibility']}")
```

### Monitoring Model Performance

```python
# Monitor family-specific performance
stats = engine.get_speculation_stats()
print(f"Model family compatibility: {stats['model_family_compatibility']}")
print(f"Acceptance rate: {stats['acceptance_rate']:.2f}")

# Adjust models if compatibility issues
if stats['model_family_compatibility'] == 'incompatible':
    # Switch to auto-config
    new_config = suggest_speculative_config(target_model)
    engine.set_models(new_config['target_model_id'], new_config['draft_model_id'])
```

## Distributed Usage with Model Families

```bash
# Device 1 (handles early layers)
exo --inference-engine speculative \
    --speculative-target-model llama-3.1-8b \
    --speculative-draft-model llama-3.2-3b

# Device 2 (handles final layers with speculation)
exo --inference-engine speculative \
    --speculative-target-model llama-3.1-8b \
    --speculative-draft-model llama-3.2-3b \
    --speculative-gamma 6
```

## Performance Guidelines

### Model Size Ratios
- **Optimal ratio**: 1:2 to 1:4 (draft:target parameters)
  - LLaMA-3B → LLaMA-8B (1:2.7 ratio) ✅
  - LLaMA-3B → LLaMA-70B (1:23 ratio) ⚠️ (may be less efficient)

### Family Compatibility
- ✅ **Same tokenizer**: LLaMA-3.2-3B + LLaMA-3.1-8B
- ✅ **Same architecture**: Qwen-2.5-1.5B + Qwen-2.5-7B
- ❌ **Different families**: LLaMA + Qwen (incompatible)

### Expected Speedup
- **LLaMA family**: 1.5-2.5x speedup
- **Qwen family**: 1.3-2.0x speedup
- **Mixed families**: No speedup (falls back to normal inference)

## Benchmarking with Families

```bash
# Test different family combinations
exo run llama-3.1-8b \
    --inference-engine speculative \
    --speculative-target-model llama-3.1-8b \
    --speculative-draft-model llama-3.2-3b \
    --prompt "Write a story" --time

# Compare with non-speculative
exo run llama-3.1-8b \
    --inference-engine mlx \
    --prompt "Write a story" --time
```

## Troubleshooting

### Model Incompatibility
```
Warning: Models may be incompatible for speculative decoding:
  Target: llama-3.1-8b, Draft: qwen-2.5-3b
  Tokens match: False, Vocab match: False
```
**Solution**: Use models from the same family or enable auto-config.

### Low Acceptance Rate with Same Family
**Possible causes**:
- Draft model too different from target
- Gamma too high for the model pair
- Temperature mismatch

**Solutions**:
1. Use auto-config: `--speculative-auto-config`
2. Try a closer draft model size
3. Reduce gamma: `--speculative-gamma 3`

### Performance Regression
**Check**:
1. Model family compatibility
2. Acceptance rate > 0.3
3. Draft model is actually smaller

## Future Improvements

- [ ] **Multi-draft models**: Use multiple draft models simultaneously
- [ ] **Adaptive model selection**: Switch draft models based on context
- [ ] **Cross-family compatibility**: Enable compatible models across families
- [ ] **Dynamic gamma**: Adjust speculation length based on acceptance rate
- [ ] **Model-specific fine-tuning**: Optimize draft models for specific targets

## References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) - Original paper with family models
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) - Improved sampling techniques
- [Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168) - Self-speculation approach 