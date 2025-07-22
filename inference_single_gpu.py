import torch
import importlib
from setup import init_config
from torch.utils.data import DataLoader
from PIL import Image

config = init_config()
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

config.training.batch_size_per_gpu = 1

dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(device)
model.load_ckpt(config.training.checkpoint_dir)

out_dir = "/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/LVSM/data/habitat_eval/out"

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    
    for batch in dataloader:
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        result = model(batch)
        print("Infering this batch")

        imgs = result["render"].squeeze(0)
        for i, img in enumerate(imgs):
            img = (img.permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
            img_pil = Image.fromarray(img)
            img_pil.save(f"{out_dir}/output_{i}.png")

print("Inference completed successfully.")