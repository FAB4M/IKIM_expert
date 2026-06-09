"""OpenAI-kompatibler FastAPI-Server für Qwen (transformers), optional mit LoRA-Adapter.
Endpunkte: /v1/chat/completions, /v1/models, /health.

    python medrax/llm/serve_qwen.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter weights/llm/qwen_reports_lora
"""

import argparse
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OpenAI-compatible Qwen server (+ optional LoRA)")
    p.add_argument("--model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    p.add_argument("--adapter", default=os.environ.get("QWEN_ADAPTER", ""))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


def build_app(model_name: str, adapter_path: str = ""):
    import torch
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import List, Optional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Lade Modell:", model_name, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    if adapter_path and os.path.isdir(adapter_path):
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"LoRA-Adapter geladen: {adapter_path}", flush=True)
        except Exception as e:
            print(f"[WARN] Adapter konnte nicht geladen werden ({e}) -> nur Basismodell.", flush=True)
    elif adapter_path:
        print(f"[WARN] Adapter-Pfad nicht gefunden: {adapter_path} -> nur Basismodell.", flush=True)

    model.eval()
    print("Modell geladen.", flush=True)

    app = FastAPI()

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: Optional[str] = None
        messages: List[Message]
        temperature: Optional[float] = 0.2
        max_tokens: Optional[int] = 512
        top_p: Optional[float] = 0.9
        stream: Optional[bool] = False

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list",
                "data": [{"id": model_name, "object": "model",
                          "created": int(time.time()), "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        temp = req.temperature if req.temperature is not None else 0.2
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=req.max_tokens or 512,
                temperature=temp, top_p=req.top_p or 0.9, do_sample=temp > 0,
                pad_token_id=tokenizer.eos_token_id)
        new = out[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(new, skip_special_tokens=True).strip()
        return {
            "id": "chatcmpl-" + str(uuid.uuid4()), "object": "chat.completion",
            "created": int(time.time()), "model": req.model or model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": int(inputs["input_ids"].shape[-1]),
                      "completion_tokens": int(new.shape[-1]),
                      "total_tokens": int(inputs["input_ids"].shape[-1] + new.shape[-1])},
        }

    return app


def main(argv=None):
    args = parse_args(argv)
    import uvicorn

    app = build_app(args.model, args.adapter)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
