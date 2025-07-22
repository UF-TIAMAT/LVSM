from typing import Any, Dict
import time
import os
import random
import socket
import numpy as np
import requests
import cv2
import base64


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError

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
        pass

    def generate_novel_views():
        pass

class LVSMModelClient:
    def __init__(self, port: int = 12200):
        self.url = f"http://localhost:{port}/lvsm"

    def generate_novel_views(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        print(f"LVSMModelClient.generate_novel_views: {payload}")
        response = send_request(self.url, payload=payload)
        return response["response"]



# if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--port", type=int, default=12200)
    # args = parser.parse_args()

    # print("Loading model...")
    # model = LVSM(config).to(device)
    # model.load_ckpt(config.training.checkpoint_dir)

    # print("Starting inference...")
    # with torch.no_grad(), torch.autocast(
    #     enabled=config.training.use_amp,
    #     device_type="cuda",
    #     dtype=amp_dtype_mapping[config.training.amp_dtype],
    # ):
    #     for batch in dataloader:
    #         batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    #         result = model(batch)
    #         # Process result as needed