import argparse
import asyncio
import atexit
import signal
import json
import platform
import os
import time
import traceback
import uuid
import numpy as np
from tqdm import tqdm
from exo.train.dataset import load_dataset, iterate_batches
from exo.networking.manual.manual_discovery import ManualDiscovery
from exo.orchestration.node import Node
from exo.networking.grpc.grpc_server import GRPCServer
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.tailscale.tailscale_discovery import TailscaleDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.api import ChatGPTAPI
from exo.download.shard_download import ShardDownloader, NoopShardDownloader
from exo.download.download_progress import RepoProgressEvent
from exo.download.new_shard_download import new_shard_downloader, has_exo_home_read_access, has_exo_home_write_access, ensure_exo_home, seed_models
from exo.helpers import print_yellow_exo, find_available_port, DEBUG, get_system_info, get_or_create_node_id, get_all_ip_addresses_and_interfaces, terminal_link, shutdown
from exo.inference.shard import Shard
from exo.inference.inference_engine import get_inference_engine
from exo.inference.tokenizers import resolve_tokenizer
from exo.models import build_base_shard, get_repo
from exo.viz.topology_viz import TopologyViz
import uvloop
import concurrent.futures
import resource
import psutil
import logging

# TODO: figure out why this is happening
os.environ["GRPC_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure uvloop for maximum performance
def configure_uvloop():
    uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Increase file descriptor limits on Unix systems
    if not psutil.WINDOWS:
      soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
      try: resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
      except ValueError:
        try: resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))
        except ValueError: pass

    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 4)))
    return loop

# parse args
parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")
parser.add_argument("command", nargs="?", choices=["run", "eval", "train"], help="Command to run")
parser.add_argument("model_name", nargs="?", help="Model name to run")
parser.add_argument("--default-model", type=str, default=None, help="Default model")
parser.add_argument("--iters", type=int, default=100, help="Training iterations")
parser.add_argument("--save-every", type=int, default=5, help="Save the model every N iterations.")
parser.add_argument("--data", type=str, default="exo/train/data/lora", help="Directory where training data lives")
parser.add_argument("--batch-size", type=int, default=1, help="Minibatch size.")
parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to a custom checkpoint to load")
parser.add_argument("--save-checkpoint-dir", type=str, default="checkpoints", help="Path to a folder where checkpoints are stored")
parser.add_argument("--node-id", type=str, default=None, help="Node ID")
parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
parser.add_argument("--node-port", type=int, default=None, help="Node port")
parser.add_argument("--models-seed-dir", type=str, default=None, help="Model seed directory")
parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
parser.add_argument("--download-quick-check", action="store_true", help="Quick check local path for model shards download")
parser.add_argument("--max-parallel-downloads", type=int, default=8, help="Max parallel downloads for model shards download")
parser.add_argument("--broadcast-port", type=int, default=5678, help="Broadcast port for discovery")
parser.add_argument("--discovery-module", type=str, choices=["udp", "tailscale", "manual"], default="udp", help="Discovery module to use")
parser.add_argument("--discovery-timeout", type=int, default=30, help="Discovery timeout in seconds")
parser.add_argument("--discovery-config-path", type=str, default=None, help="Path to discovery config json file")
parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait to connect to before starting")
parser.add_argument("--chatgpt-api-port", type=int, default=52415, help="ChatGPT API port")
parser.add_argument("--chatgpt-api-response-timeout", type=int, default=900, help="ChatGPT API response timeout in seconds")
parser.add_argument("--max-generate-tokens", type=int, default=10000, help="Max tokens to generate in each request")
parser.add_argument("--inference-engine", type=str, default=None, help="Inference engine to use (mlx, tinygrad, dummy, or speculative)")
parser.add_argument("--disable-tui", action=argparse.BooleanOptionalAction, help="Disable TUI")
parser.add_argument("--run-model", type=str, help="Specify a model to run directly")
parser.add_argument("--prompt", type=str, help="Prompt for the model when using --run-model", default="Who are you?")
parser.add_argument("--default-temp", type=float, help="Default token sampling temperature", default=0.0)
parser.add_argument("--tailscale-api-key", type=str, default=None, help="Tailscale API key")
parser.add_argument("--tailnet-name", type=str, default=None, help="Tailnet name")
parser.add_argument("--node-id-filter", type=str, default=None, help="Comma separated list of allowed node IDs (only for UDP and Tailscale discovery)")
parser.add_argument("--interface-type-filter", type=str, default=None, help="Comma separated list of allowed interface types (only for UDP discovery)")
parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for the ChatGPT API")
# Speculative decoding arguments
parser.add_argument("--speculative-target-engine", type=str, default="mlx", help="Target engine for speculative decoding (mlx, tinygrad)")
parser.add_argument("--speculative-draft-engine", type=str, default=None, help="Draft engine for speculative decoding (mlx, tinygrad, or None for early exit)")
parser.add_argument("--speculative-target-model", type=str, default=None, help="Target model for speculative decoding (e.g., llama-3.1-8b)")
parser.add_argument("--speculative-draft-model", type=str, default=None, help="Draft model for speculative decoding (e.g., llama-3.2-3b, or None for auto-selection)")
parser.add_argument("--speculative-gamma", type=int, default=5, help="Number of speculative tokens to generate")
parser.add_argument("--speculative-temperature", type=float, default=1.0, help="Temperature for speculative sampling")
parser.add_argument("--speculative-top-k", type=float, default=0.9, help="Top-k threshold for speculative decoding")
parser.add_argument("--speculative-lenience", type=float, default=1.0, help="Lenience factor for acceptance probability")
parser.add_argument("--disable-speculative", action="store_true", help="Disable speculative decoding optimization")
parser.add_argument("--speculative-auto-config", action="store_true", help="Automatically configure speculative decoding based on model family")
parser.add_argument("--verbose", "-v", type=int, default=0, help="Verbosity level (0-3)")
parser.add_argument("--model", type=str, default=None, help="Model to use for inference")
parser.add_argument("--discovery-port", type=int, default=5678, help="Port for discovery")
parser.add_argument("--default-chatgpt-model", type=str, default=None, help="Default ChatGPT model")
parser.add_argument("--web-chat-url", type=str, default=None, help="Web chat URL")
parser.add_argument("--disable-download-progress", action="store_true", help="Disable download progress")
args = parser.parse_args()

print_yellow_exo()

# Show debug information
if DEBUG >= 1:
    print(f"🐛 DEBUG Level: {DEBUG}")
    if DEBUG >= 2:
        print(f"   DEBUG >= 2: Detailed operation logs enabled")
    if DEBUG >= 3:
        print(f"   DEBUG >= 3: Token-level debugging enabled")
    if DEBUG >= 4:
        print(f"   DEBUG >= 4: Low-level computation debugging enabled")

print(f"Selected inference engine: {args.inference_engine}")

system_info = get_system_info()
print(f"Detected system: {system_info}")

shard_downloader: ShardDownloader = new_shard_downloader(args.max_parallel_downloads) if args.inference_engine != "dummy" else NoopShardDownloader()
inference_engine_name = args.inference_engine or ("mlx" if system_info == "Apple Silicon Mac" else "tinygrad")
print(f"Inference engine name after selection: {inference_engine_name}")

# Prepare speculative config if using speculative engine
speculative_config = None
if inference_engine_name == "speculative" or (args.speculative_target_model and args.speculative_draft_model):
    print(f"\n🎯 CONFIGURING SPECULATIVE DECODING")
    
    # Use target model from speculative args or default to the main model
    target_model = args.speculative_target_model or args.model_name or args.default_model
    
    if not target_model:
        print("⚠️  Warning: No target model specified for speculative decoding. Using dummy model.")
        target_model = "dummy"
    
    # Use the main inference engine for both target and draft if not specified
    target_engine_name = args.speculative_target_engine if args.speculative_target_engine != "mlx" else inference_engine_name
    draft_engine_name = args.speculative_draft_engine or inference_engine_name
    
    speculative_config = {
        'target_engine_name': target_engine_name,
        'draft_engine_name': draft_engine_name,
        'target_model_id': target_model,
        'draft_model_id': args.speculative_draft_model,
        'gamma': args.speculative_gamma,
        'temperature': args.speculative_temperature,
        'top_k_threshold': args.speculative_top_k,
        'lenience': args.speculative_lenience,
        'enable_speculative': not args.disable_speculative,
        'early_exit_layer': None
    }
    
    print(f"📋 Speculative Decoding Configuration:")
    print(f"   🎯 Target model: {speculative_config['target_model_id']}")
    print(f"   🎨 Draft model: {speculative_config['draft_model_id'] or 'early-exit'}")
    print(f"   🎲 Gamma (speculation length): {speculative_config['gamma']}")
    print(f"   🌡️  Temperature: {speculative_config['temperature']}")
    print(f"   ⚡ Enabled: {speculative_config['enable_speculative']}")
    
    # Force engine name to speculative if speculative models are provided
    if args.speculative_target_model and args.speculative_draft_model:
        inference_engine_name = "speculative"

inference_engine = get_inference_engine(inference_engine_name, shard_downloader, speculative_config)
print(f"Using inference engine: {inference_engine.__class__.__name__} with shard downloader: {shard_downloader.__class__.__name__}")

# Show model suggestions if using speculative decoding
if inference_engine_name == "speculative" and hasattr(inference_engine, 'target_model_id'):
    if inference_engine.target_model_id and inference_engine.target_model_id != "dummy":
        from exo.inference.inference_engine import get_model_family_variants
        family_info = get_model_family_variants(inference_engine.target_model_id)
        if family_info["suggested_drafts"]:
            print(f"💡 Suggested draft models for {inference_engine.target_model_id}: {', '.join(family_info['suggested_drafts'])}")
            if DEBUG >= 1:
                print(f"   Model family: {family_info['family']}")
                print(f"   All suggestions: {family_info['suggested_drafts']}")
        else:
            print(f"ℹ️  No specific draft model suggestions for {inference_engine.target_model_id} (family: {family_info['family']})")
    
    # Show debug tips based on DEBUG level
    if DEBUG == 0:
        print(f"\n💡 Debug Tips:")
        print(f"   Set DEBUG=1 for basic speculative decoding logs")
        print(f"   Set DEBUG=2 for detailed phase-by-phase logs") 
        print(f"   Set DEBUG=3 for token-level debugging")
        print(f"   Set DEBUG=4 for low-level computation logs")
        print(f"   Example: DEBUG=2 exo --inference-engine speculative ...")
    elif DEBUG >= 1:
        print(f"\n🐛 Debug Mode Active - you'll see detailed speculative decoding logs!")

if args.node_port is None:
  args.node_port = find_available_port(args.node_host)
  if DEBUG >= 1: print(f"Using available port: {args.node_port}")

args.node_id = args.node_id or get_or_create_node_id()
chatgpt_api_endpoints = [f"http://{ip}:{args.chatgpt_api_port}/v1/chat/completions" for ip, _ in get_all_ip_addresses_and_interfaces()]
web_chat_urls = [f"http://{ip}:{args.chatgpt_api_port}" for ip, _ in get_all_ip_addresses_and_interfaces()]
if DEBUG >= 0:
  print("Chat interface started:")
  for web_chat_url in web_chat_urls:
    print(f" - {terminal_link(web_chat_url)}")
  print("ChatGPT API endpoint served at:")
  for chatgpt_api_endpoint in chatgpt_api_endpoints:
    print(f" - {terminal_link(chatgpt_api_endpoint)}")

# Convert node-id-filter and interface-type-filter to lists if provided
allowed_node_ids = args.node_id_filter.split(',') if args.node_id_filter else None
allowed_interface_types = args.interface_type_filter.split(',') if args.interface_type_filter else None

if args.discovery_module == "udp":
  discovery = UDPDiscovery(
    args.node_id,
    args.node_port,
    args.listen_port,
    args.broadcast_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    allowed_node_ids=allowed_node_ids,
    allowed_interface_types=allowed_interface_types
  )
elif args.discovery_module == "tailscale":
  discovery = TailscaleDiscovery(
    args.node_id,
    args.node_port,
    lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
    discovery_timeout=args.discovery_timeout,
    tailscale_api_key=args.tailscale_api_key,
    tailnet=args.tailnet_name,
    allowed_node_ids=allowed_node_ids
  )
elif args.discovery_module == "manual":
  if not args.discovery_config_path:
    raise ValueError(f"--discovery-config-path is required when using manual discovery. Please provide a path to a config json file.")
  discovery = ManualDiscovery(args.discovery_config_path, args.node_id, create_peer_handle=lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities))
topology_viz = TopologyViz(chatgpt_api_endpoints=chatgpt_api_endpoints, web_chat_urls=web_chat_urls) if not args.disable_tui else None
node = Node(
  args.node_id,
  None,
  inference_engine,
  discovery,
  shard_downloader,
  partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
  max_generate_tokens=args.max_generate_tokens,
  topology_viz=topology_viz,
  default_sample_temperature=args.default_temp
)
server = GRPCServer(node, args.node_host, args.node_port)
node.server = server
api = ChatGPTAPI(
  node,
  node.inference_engine.__class__.__name__,
  response_timeout=args.chatgpt_api_response_timeout,
  on_chat_completion_request=lambda req_id, __, prompt: topology_viz.update_prompt(req_id, prompt) if topology_viz else None,
  default_model=args.default_model,
  system_prompt=args.system_prompt
)
buffered_token_output = {}
def update_topology_viz(req_id, tokens, __):
  if not topology_viz: return
  if not node.inference_engine.shard: return
  if node.inference_engine.shard.model_id == 'stable-diffusion-2-1-base': return
  if req_id in buffered_token_output: buffered_token_output[req_id].extend(tokens)
  else: buffered_token_output[req_id] = tokens
  topology_viz.update_prompt_output(req_id, node.inference_engine.tokenizer.decode(buffered_token_output[req_id]))
node.on_token.register("update_topology_viz").on_next(update_topology_viz)
def update_prompt_viz(request_id, opaque_status: str):
  if not topology_viz: return
  try:
    status = json.loads(opaque_status)
    if status.get("type") != "node_status" or status.get("status") != "start_process_prompt": return
    topology_viz.update_prompt(request_id, status.get("prompt", "corrupted prompt (this should never happen)"))
  except Exception as e:
    if DEBUG >= 2:
      print(f"Failed to update prompt viz: {e}")
      traceback.print_exc()
node.on_opaque_status.register("update_prompt_viz").on_next(update_prompt_viz)

def preemptively_load_shard(request_id: str, opaque_status: str):
  try:
    status = json.loads(opaque_status)
    if status.get("type") != "node_status" or status.get("status") != "start_process_prompt": return
    current_shard = node.get_current_shard(Shard.from_dict(status.get("shard")))
    if DEBUG >= 2: print(f"Preemptively starting download for {current_shard}")
    asyncio.create_task(node.inference_engine.ensure_shard(current_shard))
  except Exception as e:
    if DEBUG >= 2:
      print(f"Failed to preemptively start download: {e}")
      traceback.print_exc()
node.on_opaque_status.register("preemptively_load_shard").on_next(preemptively_load_shard)

last_events: dict[str, tuple[float, RepoProgressEvent]] = {}
def throttled_broadcast(shard: Shard, event: RepoProgressEvent):
  global last_events
  current_time = time.time()
  if event.status == "not_started": return
  last_event = last_events.get(shard.model_id)
  if last_event and last_event[1].status == "complete" and event.status == "complete": return
  if last_event and last_event[0] == event.status and current_time - last_event[0] < 0.2: return
  last_events[shard.model_id] = (current_time, event)
  asyncio.create_task(node.broadcast_opaque_status("", json.dumps({"type": "download_progress", "node_id": node.id, "progress": event.to_dict()})))
shard_downloader.on_progress.register("broadcast").on_next(throttled_broadcast)

async def run_model_cli(node: Node, model_name: str, prompt: str):
  # For speculative engines, use the target engine class name for shard building
  if hasattr(node.inference_engine, 'target_engine'):
    inference_class = node.inference_engine.target_engine.__class__.__name__
  else:
    inference_class = node.inference_engine.__class__.__name__
    
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  request_id = str(uuid.uuid4())
  callback_id = f"cli-wait-response-{request_id}"
  callback = node.on_token.register(callback_id)
  if topology_viz:
    topology_viz.update_prompt(request_id, prompt)
  prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

  try:
    print(f"Processing prompt: {prompt}")
    await node.process_prompt(shard, prompt, request_id=request_id)

    tokens = []
    def on_token(_request_id, _tokens, _is_finished):
      tokens.extend(_tokens)
      return _request_id == request_id and _is_finished
    await callback.wait(on_token, timeout=300)

    print("\nGenerated response:")
    print(tokenizer.decode(tokens))
  except Exception as e:
    print(f"Error processing prompt: {str(e)}")
    traceback.print_exc()
  finally:
    node.on_token.deregister(callback_id)

def clean_path(path):
    """Clean and resolve path"""
    if path.startswith("Optional("):
        path = path.strip('Optional("').rstrip('")')
    return os.path.expanduser(path)

async def hold_outstanding(node: Node):
  while node.outstanding_requests:
    await asyncio.sleep(.5)
  return

async def run_iter(node: Node, shard: Shard, train: bool, data, batch_size=1):
  losses = []
  tokens = []
  for batch in tqdm(iterate_batches(data, batch_size), total=len(data) // batch_size):
    _, _, lengths = batch
    losses.append(np.sum(lengths * await node.enqueue_example(shard, *batch, train=train)))
    tokens.append(np.sum(lengths))
  total_tokens = np.sum(tokens)
  total_loss = np.sum(losses) / total_tokens

  return total_loss, total_tokens

async def eval_model_cli(node: Node, model_name, dataloader, batch_size, num_batches=-1):
  # For speculative engines, use the target engine class name for shard building
  if hasattr(node.inference_engine, 'target_engine'):
    inference_class = node.inference_engine.target_engine.__class__.__name__
  else:
    inference_class = node.inference_engine.__class__.__name__
    
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  train, val, test = dataloader(tokenizer.encode)
  print(f"Evaluating {len(test)} examples with batch_size {batch_size}")
  loss, tokens = await run_iter(node, shard, False, test, batch_size)
  print(f"total | {loss=}, {tokens=}")
  print("Waiting for outstanding tasks")
  await hold_outstanding(node)

async def train_model_cli(node: Node, model_name, dataloader, batch_size, iters, save_interval=0, checkpoint_dir=None):
  # For speculative engines, use the target engine class name for shard building
  if hasattr(node.inference_engine, 'target_engine'):
    inference_class = node.inference_engine.target_engine.__class__.__name__
  else:
    inference_class = node.inference_engine.__class__.__name__
    
  shard = build_base_shard(model_name, inference_class)
  if not shard:
    print(f"Error: Unsupported model '{model_name}' for inference engine {inference_class}")
    return
  tokenizer = await resolve_tokenizer(get_repo(shard.model_id, inference_class))
  train, val, test = dataloader(tokenizer.encode)
  print(f"Training on {len(train)} examples with batch_size {batch_size} for {iters} epochs")
  for i in tqdm(range(3)):
    await asyncio.sleep(1)
  for epoch in range(iters):
    loss, tokens = await run_iter(node, shard, True, train, batch_size)
    print(f"epoch {epoch + 1}/{iters}\t| loss: {loss}, tokens: {tokens}")
    if save_interval > 0 and epoch > 0 and (epoch % save_interval) == 0 and checkpoint_dir is not None:
      await node.coordinate_save(shard, epoch, checkpoint_dir)
      await hold_outstanding(node)
  await hold_outstanding(node)

async def check_exo_home():
  home, has_read, has_write = await ensure_exo_home(), await has_exo_home_read_access(), await has_exo_home_write_access()
  if DEBUG >= 1: print(f"exo home directory: {home}")
  print(f"{has_read=}, {has_write=}")
  if not has_read or not has_write:
    print(f"""
          WARNING: Limited permissions for exo home directory: {home}.
          This may prevent model downloads from working correctly.
          {"❌ No read access" if not has_read else ""}
          {"❌ No write access" if not has_write else ""}
          """)

async def main():
  loop = asyncio.get_running_loop()

  try: await check_exo_home()
  except Exception as e: print(f"Error checking exo home directory: {e}")

  if not args.models_seed_dir is None:
    try:
      models_seed_dir = clean_path(args.models_seed_dir)
      await seed_models(models_seed_dir)
    except Exception as e:
      print(f"Error seeding models: {e}")

  def restore_cursor():
    if platform.system() != "Windows":
        os.system("tput cnorm")  # Show cursor

  # Restore the cursor when the program exits
  atexit.register(restore_cursor)

  # Use a more direct approach to handle signals
  def handle_exit():
    asyncio.ensure_future(shutdown(signal.SIGTERM, loop, node.server))

  if platform.system() != "Windows":
    for s in [signal.SIGINT, signal.SIGTERM]:
      loop.add_signal_handler(s, handle_exit)

  await node.start(wait_for_peers=args.wait_for_peers)

  if args.command == "run" or args.run_model:
    model_name = args.model_name or args.run_model
    if not model_name:
      print("Error: Model name is required when using 'run' command or --run-model")
      return
    await run_model_cli(node, model_name, args.prompt)
  elif args.prompt and (args.speculative_target_model or args.model_name):
    # Handle direct prompt when speculative models are provided
    model_name = args.speculative_target_model or args.model_name
    print(f"Processing prompt with speculative decoding: {args.prompt}")
    print(f"DEBUG: Using model_name: '{model_name}'")
    print(f"DEBUG: args.speculative_target_model: '{args.speculative_target_model}'")
    print(f"DEBUG: args.model_name: '{args.model_name}'")
    await run_model_cli(node, model_name, args.prompt)
    return
  elif args.command == "eval" or args.command == 'train':
    model_name = args.model_name
    dataloader = lambda tok: load_dataset(args.data, preprocess=lambda item: tok(item)
                                                   , loadline=lambda line: json.loads(line).get("text",""))
    if args.command == 'eval':
      if not model_name:
        print("Error: Much like a human, I can't evaluate anything without a model")
        return
      await eval_model_cli(node, model_name, dataloader, args.batch_size)
    else:
      if not model_name:
        print("Error: This train ain't leaving the station without a model")
        return
      await train_model_cli(node, model_name, dataloader, args.batch_size, args.iters, save_interval=args.save_every, checkpoint_dir=args.save_checkpoint_dir)

  else:
    asyncio.create_task(api.run(port=args.chatgpt_api_port))  # Start the API server as a non-blocking task
    await asyncio.Event().wait()

  if args.wait_for_peers > 0:
    print("Cooldown to allow peers to exit gracefully")
    for i in tqdm(range(50)):
      await asyncio.sleep(.1)

def run():
    loop = None
    try:
        loop = configure_uvloop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting")
    finally:
        if loop: loop.close()

if __name__ == "__main__":
  run()
