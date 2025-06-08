import asyncio
from exo.models import get_repo, build_full_shard
from exo.inference.tokenizers import resolve_tokenizer

async def decode_debug_tokens():
    shard = build_full_shard('llama-3.2-3b', 'TinygradDynamicShardInferenceEngine')
    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, 'TinygradDynamicShardInferenceEngine'))
    
    print("=== CONTEXT CORRUPTION ANALYSIS ===")
    
    # Debug tokens from Round 2
    debug_tokens = [791, 3938, 315]
    print(f'Debug tokens from Round 2: {debug_tokens}')
    print(f'Decoded: "{tokenizer.decode(debug_tokens)}"')
    
    # What should be the prompt
    expected_prompt = "the future of AI is"
    expected_tokens = tokenizer.encode(expected_prompt)
    print(f'\nExpected prompt: "{expected_prompt}"')
    print(f'Expected tokens: {expected_tokens}')
    print(f'Expected decoded: "{tokenizer.decode(expected_tokens)}"')
    
    # Check what individual debug tokens mean
    print(f'\n=== INDIVIDUAL TOKEN ANALYSIS ===')
    for i, token in enumerate(debug_tokens):
        decoded = tokenizer.decode([token])
        print(f'Token {i}: {token} -> "{decoded}"')
    
    # Check the generated tokens
    round1_accepted = [791, 3938, 315]  # From debug: "Accepted tokens: [791, 3938, 315]"
    round2_accepted = [16088, 374, 539, 1120]  # From debug: "Accepted tokens: [16088, 374, 539, 1120]"
    
    print(f'\n=== GENERATION ANALYSIS ===')
    print(f'Round 1 accepted: {round1_accepted}')
    print(f'Round 1 decoded: "{tokenizer.decode(round1_accepted)}"')
    
    print(f'Round 2 accepted: {round2_accepted}')
    print(f'Round 2 decoded: "{tokenizer.decode(round2_accepted)}"')
    
    # Full sequence
    all_accepted = round1_accepted + round2_accepted
    print(f'\nFull generated sequence: {all_accepted}')
    print(f'Full decoded: "{tokenizer.decode(all_accepted)}"')

if __name__ == "__main__":
    asyncio.run(decode_debug_tokens()) 