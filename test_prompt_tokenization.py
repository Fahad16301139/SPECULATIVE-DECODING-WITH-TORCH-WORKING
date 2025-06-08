import asyncio
from exo.models import get_repo, build_full_shard
from exo.inference.tokenizers import resolve_tokenizer

async def test_tokenization():
    shard = build_full_shard('llama-3.2-3b', 'TinygradDynamicShardInferenceEngine')
    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, 'TinygradDynamicShardInferenceEngine'))
    
    prompt = "the future of AI is"
    tokens = tokenizer.encode(prompt)
    
    print(f'Prompt: "{prompt}"')
    print(f'Tokens: {tokens}')
    print(f'Shape: {len(tokens)} tokens')
    print(f'Decoded back: "{tokenizer.decode(tokens)}"')
    
    # Test if the issue is in the prompt processing
    import numpy as np
    token_array = np.array(tokens)
    print(f'NumPy array shape: {token_array.shape}')
    reshaped = token_array.reshape(1, -1)
    print(f'Reshaped for inference: {reshaped.shape}')

if __name__ == "__main__":
    asyncio.run(test_tokenization()) 