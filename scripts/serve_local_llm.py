"""Локальный OpenAI-совместимый LLM-сервер на transformers — «мини-vLLM» для обычной машины.

Зачем он нужен. В продакшене генерацию отдают специализированным серверам (vLLM, TGI —
см. профиль ``llm`` в docker-compose.yml): они быстрые, но vLLM не работает на Windows и
хочет мощную видеокарту. Этот скрипт поднимает тот же HTTP-контракт
(``POST /v1/chat/completions``) поверх обычной библиотеки transformers. Поэтому kbrag
подключается к нему без единой строчки нового кода — меняется только ``base_url`` в конфиге.
Это и есть смысл OpenAI-совместимого API: модель и сервер можно менять, не трогая RAG.

Запуск (в отдельном окружении с CUDA-сборкой torch, см. README):

    python scripts/serve_local_llm.py --model Qwen/Qwen2.5-1.5B-Instruct --quantize 4bit --port 8001

Проверка:

    curl http://127.0.0.1:8001/health

Подключение kbrag:

    KBRAG_GENERATION_BACKEND=openai KBRAG_LLM_BASE_URL=http://127.0.0.1:8001/v1 \\
    KBRAG_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct kbrag ask "Сколько дней отпуска положено?"

Ограничения (сознательные — это демо, а не прод): один запрос за раз, без батчинга и стриминга.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import uuid
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("serve_local_llm")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    # Клиенты присылают и другие поля OpenAI API (response_format, stop, …) — молча игнорируем их.
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(512, ge=1)


def load_model(model_name: str, quantize: str, device: str):
    """Загружает токенизатор и модель; 4/8-битная квантизация экономит видеопамять.

    Модель 1.5B в fp16 занимает ~3.1 ГБ, в 4 битах — ~1.3 ГБ: это разница между
    «не влезает» и «работает» на видеокарте с 4 ГБ, часть которых занята рабочим столом.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs: dict = {}
    if quantize != "none":
        if device != "cuda":
            raise SystemExit(
                "Квантизация bitsandbytes работает только на CUDA; используйте --quantize none на CPU."
            )
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantize == "4bit",
            load_in_8bit=quantize == "8bit",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = {"": 0}
    else:
        # На CPU fp16 медленный и неточный, поэтому там float32.
        kwargs["dtype"] = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if quantize == "none":
        model.to(device)
    model.eval()
    return tokenizer, model, device


def create_app(model_name: str, tokenizer, model, device: str, max_new_tokens_cap: int) -> FastAPI:
    app = FastAPI(title="local-llm", version="0.1.0")
    # generate() не потокобезопасен и всё равно занимает GPU целиком — обрабатываем по одному.
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model_name, "device": device}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict:
        messages = [m.model_dump() for m in req.messages]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        max_new = min(req.max_tokens, max_new_tokens_cap)
        do_sample = req.temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(temperature=req.temperature, top_p=req.top_p)
        started = time.perf_counter()
        try:
            with lock, torch.inference_mode():
                output = model.generate(**inputs, **gen_kwargs)
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            torch.cuda.empty_cache()
            raise HTTPException(status_code=503, detail="Не хватило видеопамяти для этого запроса") from exc
        new_tokens = output[0, prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = int(new_tokens.shape[0])
        logger.info(
            "prompt=%d completion=%d за %.1f с",
            prompt_tokens,
            completion_tokens,
            time.perf_counter() - started,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length" if completion_tokens >= max_new else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-new-tokens-cap", type=int, default=512)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tokenizer, model, device = load_model(args.model, args.quantize, args.device)
    if device == "cuda":
        logger.info("Видеопамять под модель: %.2f ГБ", torch.cuda.memory_allocated() / 1024**3)
    app = create_app(args.model, tokenizer, model, device, args.max_new_tokens_cap)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
