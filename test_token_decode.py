import asyncio
from exo.models import get_repo, build_full_shard
from exo.inference.tokenizers import resolve_tokenizer

async def test_decode():
    shard = build_full_shard('llama-3.2-3b', 'TinygradDynamicShardInferenceEngine')
    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, 'TinygradDynamicShardInferenceEngine'))
    
    tokens = [9906, 29021, 498, 220, 26576, 5932]
    print(f'Tokens: {tokens}')
    print(f'Decoded: "{tokenizer.decode(tokens)}"')
    
    # Also test individual tokens
    for i, token in enumerate(tokens):
        decoded = tokenizer.decode([token])
        print(f'Token {i}: {token} -> "{decoded}"')

if __name__ == "__main__":
    asyncio.run(test_decode()) 