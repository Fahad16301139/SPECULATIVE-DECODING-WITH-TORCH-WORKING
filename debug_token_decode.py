import asyncio
from exo.models import get_repo, build_full_shard
from exo.inference.tokenizers import resolve_tokenizer

async def test_decode():
    shard = build_full_shard('llama-3.2-3b', 'TinygradDynamicShardInferenceEngine')
    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, 'TinygradDynamicShardInferenceEngine'))
    
    # Test the specific tokens from debug output
    tokens = [791, 3938, 315, 32186, 11]
    print(f'Debug tokens: {tokens}')
    print(f'Decoded: "{tokenizer.decode(tokens)}"')
    
    # Also test what our prompt should be
    prompt_tokens = tokenizer.encode("the future of AI is")
    print(f'\\nExpected prompt tokens: {prompt_tokens}')
    print(f'Expected prompt decoded: "{tokenizer.decode(prompt_tokens)}"')
    
    # Check if they match
    if tokens == prompt_tokens:
        print("\\n✅ Tokens match - context is correct")
    else:
        print("\\n🚨 Token mismatch - context is wrong!")
        print(f"   Debug tokens: {tokens}")
        print(f"   Expected tokens: {prompt_tokens}")

if __name__ == "__main__":
    asyncio.run(test_decode()) 