import argparse
import base64
import io
import json
import os
import traceback
from pathlib import Path

# Reduce extra TensorFlow/backend noise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import uvicorn


from typing import Optional


DEFAULT_BASE_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--openvla--openvla-7b/"
    "snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
)
OPENVLA_ROOT = Path(__file__).resolve().parent


class PredictRequest(BaseModel):
    instruction: str
    image_b64: str
    unnorm_key: Optional[str] = None
    do_sample: bool = False

class OpenVLAServingModel:
    @staticmethod
    def _resolve_adapter_run_dir(model_path: str, adapter_path: Optional[str]) -> Path:
        resolved = Path(model_path)
        if adapter_path is None:
            return resolved
        if (resolved / "dataset_statistics.json").exists():
            return resolved

        candidate = OPENVLA_ROOT / "openvla-runs" / Path(adapter_path).name
        if candidate.exists():
            print(f"[INFO] Inferred processor/stat run dir from adapter: {candidate}")
            return candidate
        return resolved

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        default_unnorm_key: str = "raccoon_pick_place",
        adapter_path: Optional[str] = None,
        base_model_path: str = DEFAULT_BASE_MODEL_PATH,
        merge_adapter: bool = False,
    ):
        resolved_model_path = self._resolve_adapter_run_dir(model_path, adapter_path)
        self.model_path = str(resolved_model_path)
        self.device = device
        self.default_unnorm_key = default_unnorm_key
        self.adapter_path = adapter_path
        self.base_model_path = base_model_path
        self.merge_adapter = merge_adapter

        processor_source = self.model_path
        if adapter_path is not None and not (resolved_model_path / "preprocessor_config.json").exists():
            processor_source = base_model_path
        self.processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=True,
        )

        load_path = base_model_path if adapter_path is not None else model_path
        self.vla = AutoModelForVision2Seq.from_pretrained(
            load_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if adapter_path is not None:
            self.vla = PeftModel.from_pretrained(self.vla, adapter_path)
            if merge_adapter:
                self.vla = self.vla.merge_and_unload()
        self.vla = self.vla.to(device)

        stats_path = resolved_model_path / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                norm_stats = json.load(f)
                self.vla.norm_stats = norm_stats
                if hasattr(self.vla, "base_model") and hasattr(self.vla.base_model, "model"):
                    self.vla.base_model.model.norm_stats = norm_stats
            print(f"[INFO] Loaded dataset statistics from: {stats_path}")
            print(f"[INFO] Available norm_stats keys: {list(self.vla.norm_stats.keys())}")
        else:
            print(f"[WARN] dataset_statistics.json not found at: {stats_path}")

    @torch.inference_mode()
    def predict(self, req: PredictRequest):
        image_bytes = base64.b64decode(req.image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = f"In: What action should the robot take to {req.instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        unnorm_key = req.unnorm_key or self.default_unnorm_key

        action = self.vla.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=req.do_sample,
        )

        if hasattr(action, "tolist"):
            action = action.tolist()

        action = [float(x) for x in action]
        if len(action) < 4:
            raise ValueError(f"Predicted action is too short: len={len(action)}, action={action}")

        print(f"[PREDICT] instruction={req.instruction}")
        print(f"[PREDICT] unnorm_key={unnorm_key}")
        print(f"[PREDICT] action={action}", flush=True)

        return {
            "action": action,
            "unnorm_key": unnorm_key,
            "prompt": prompt,
        }


def build_app(serving_model: OpenVLAServingModel):
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/predict")
    def predict(req: PredictRequest):
        try:
            return serving_model.predict(req)
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(exc))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--default-unnorm-key", type=str, default="raccoon_pick_place")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--merge_adapter", action="store_true")
    args = parser.parse_args()

    serving_model = OpenVLAServingModel(
        model_path=args.model_path,
        device=args.device,
        default_unnorm_key=args.default_unnorm_key,
        adapter_path=args.adapter_path,
        base_model_path=args.base_model_path,
        merge_adapter=args.merge_adapter,
    )
    app = build_app(serving_model)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
