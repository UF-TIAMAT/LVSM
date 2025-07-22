from typing import Any, Dict
import time
import os
import random
import socket
import numpy as np
import requests
import cv2
import base64
from flask import Flask, jsonify, request
import torch
import importlib
from setup import init_config
from torch.utils.data import DataLoader
from PIL import Image
from omegaconf import OmegaConf


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError
    
def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        return jsonify(model.process_payload(payload))

    app.run(host="localhost", port=port)

def send_request(url: str, **kwargs: Any) -> dict:
    response = {}
    for attempt in range(10):
        try:
            response = _send_request(url, **kwargs)
            break
        except Exception as e:
            if attempt == 9:
                print(e)
                exit()
            else:
                print(f"Error: {e}. Retrying in 20-30 seconds...")
                time.sleep(20 + random.random() * 10)

    return response

def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str

def _send_request(url: str, **kwargs: Any) -> dict:
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    filename = url.replace("/", "_").replace(":", "_") + ".lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    try:
        while True:
            # Use a while loop to wait until this filename does not exist
            while os.path.exists(filename):
                # If the file exists, wait 50ms and try again
                time.sleep(0.05)

                try:
                    # If the file was last modified more than 120 seconds ago, delete it
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass

            rand_str = str(random.randint(0, 1000000))

            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.05)
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break
            except FileNotFoundError:
                pass

        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to strings
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            else:
                payload[k] = v

        # Set the headers
        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        while True:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=1)
                if resp.status_code == 200:
                    result = resp.json()
                    break
                else:
                    raise Exception("Request failed")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                print(e)
                if time.time() - start_time > 20:
                    raise Exception("Request timed out after 20 seconds")

        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass

    except Exception as e:
        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass
        raise e

    return result

class LVSMModel:

    def __init__(self):
        self.config = OmegaConf.load("configs/LVSM_scene_decoder_only.yaml")
        self.amp_dtype_mapping = {
            "fp16": torch.float16, 
            "bf16": torch.bfloat16, 
            "fp32": torch.float32, 
            'tf32': torch.float32
        }

        # Manual parameters.
        self.config.training.dataset_path="./data/habitat_eval/full_list.txt"
        self.config.training.batch_size_per_gpu=1
        self.config.training.target_has_input=False
        self.config.training.num_views=5
        self.config.training.square_crop=True
        self.config.training.num_input_views=2
        self.config.training.num_target_views=3
        self.config.inference.if_inference=True
        self.config.inference.compute_metrics=True
        self.config.inference.render_video=True
        self.config.inference_out_dir="./experiments/evaluation/test"

        dataset_name =  self.config.training.get("dataset_name", "data.dataset.Dataset")
        module, class_name = dataset_name.rsplit(".", 1)
        Dataset = importlib.import_module(module).__dict__[class_name]
        dataset = Dataset( self.config)

        self.config.training.batch_size_per_gpu = 1

        self.dataloader = DataLoader(
            dataset,
            batch_size= self.config.training.batch_size_per_gpu,
            shuffle=False,
            num_workers= self.config.training.num_workers,
            prefetch_factor= self.config.training.prefetch_factor,
            persistent_workers=True,
            pin_memory=False,
            drop_last=True,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        module, class_name =  self.config.model.class_name.rsplit(".", 1)
        LVSM = importlib.import_module(module).__dict__[class_name]
        self.model = LVSM( self.config).to(self.device)
        self.model.load_ckpt( self.config.training.checkpoint_dir)

        self.out_dir = "/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/LVSM/data/habitat_eval/out"

    def generate_novel_views(self):
        with torch.no_grad(), torch.autocast(
            enabled= self.config.training.use_amp,
            device_type="cuda",
            dtype=self.amp_dtype_mapping[ self.config.training.amp_dtype],
        ):
            
            for batch in self.dataloader:
                batch = {k: v.to(self.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
                result = self.model(batch)
                print("Infering this batch")

                imgs = result["render"].squeeze(0)
                for i, img in enumerate(imgs):
                    img = (img.permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
                    img_pil = Image.fromarray(img)
                    img_pil.save(f"{self.out_dir}/output_{i}.png") 

        return {"status": "Inference completed successfully."}

class LVSMModelClient:
    def __init__(self, port: int = 12200):
        self.url = f"http://localhost:{port}/lvsm"

    def generate_novel_views(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        print(f"LVSMModelClient.generate_novel_views: {payload}")
        response = send_request(self.url, payload=payload)
        return response["response"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12200)
    args = parser.parse_args()

    print("Loading model...")

    class LVSMModelServer(ServerMixin, LVSMModel):
        def process_payload(self, payload: dict) -> dict:
            print(f"LVSMModelServer.process_payload: {payload}")
            self.generate_novel_views()
            return {"response": "Novel views generated successfully."}


    lvsm = LVSMModelServer()
    print("Model loaded successfully.")
    print(f"Hosting model on port {args.port}...")
    host_model(lvsm, name="lvsm", port=args.port)