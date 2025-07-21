# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# Modified for single GPU inference

import importlib
import os
import torch
from torch.utils.data import DataLoader
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set random seed for reproducibility
torch.manual_seed(777)
if torch.cuda.is_available():
    torch.cuda.manual_seed(777)
    torch.cuda.manual_seed_all(777)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up tf32
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32

amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

# Load data
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

# Create dataloader without DistributedSampler
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True
)

# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(device)

# Load checkpoint directly without DDP wrapper
model.load_ckpt(config.training.checkpoint_dir)

print(f"Running inference; save results to: {config.inference_out_dir}")

# Initialize LPIPS if needed
import lpips
# Suppress the warning by setting weights_only=True
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Run inference
model.eval()

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda" if torch.cuda.is_available() else "cpu",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in dataloader:
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        result = model(batch)
        if config.inference.get("render_video", False):
            result = model.render_video(result, **config.inference.render_video_config)
        export_results(result, config.inference_out_dir, compute_metrics=config.inference.get("compute_metrics"))
    
    # Clear GPU memory cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Summarize evaluation and generate website
if config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference_out_dir}")

print("Inference completed successfully!")