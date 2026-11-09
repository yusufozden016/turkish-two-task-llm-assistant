from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any


class ModelServiceError(RuntimeError):
    """Raised when the local router or LoRA-backed model cannot be used."""


class TwoLoraModelService:
    """Loads the router and both LoRA adapters once and serves one-shot prompts."""

    BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen3-0.6B")
    MAX_BERT_LENGTH = 512
    MAX_NEW_TOKENS = 192

    def __init__(self, project_root: Path | None = None) -> None:
        configured_project_root = os.getenv("PROJECT_ROOT")
        self.app_dir = Path(__file__).resolve().parent
        self.project_root = (
            project_root
            or (
                Path(configured_project_root).expanduser()
                if configured_project_root
                else Path(__file__).resolve().parents[1]
            )
        ).resolve()

        standalone_models_dir = self.app_dir / "models"
        legacy_outputs_dir = self.project_root / "outputs"

        self.summary_adapter_dir = self._path_from_env(
            "SUMMARY_ADAPTER_DIR",
            self._model_default_path(
                standalone_models_dir / "summary",
                legacy_outputs_dir
                / "qwen3-0.6b-tr-news-lora-6000-summary-length-balanced"
                / "final",
            ),
        )
        self.wiki_adapter_dir = self._path_from_env(
            "WIKI_ADAPTER_DIR",
            self._model_default_path(
                standalone_models_dir / "wiki",
                legacy_outputs_dir
                / "qwen3-0.6b-wikiner-lora-fp16"
                / "checkpoint-733",
            ),
        )
        self.router_dir = self._path_from_env(
            "ROUTER_DIR",
            self._model_default_path(
                standalone_models_dir / "router",
                legacy_outputs_dir / "turkish_small_bert_task_classifier",
            ),
        )

        self.router_device_name = os.getenv("ROUTER_DEVICE", "cpu")
        self.qwen_device_name = os.getenv(
            "QWEN_DEVICE",
            "cuda" if self._cuda_is_available() else "cpu",
        )

        self.router_tokenizer: Any | None = None
        self.router_model: Any | None = None
        self.qwen_tokenizer: Any | None = None
        self.qwen_model: Any | None = None
        self.torch: Any | None = None
        self._generation_lock = threading.Lock()
        self._loaded = False

    @staticmethod
    def _path_from_env(name: str, default: Path) -> Path:
        value = os.getenv(name)
        return Path(value).expanduser().resolve() if value else default.resolve()

    @staticmethod
    def _model_default_path(standalone_path: Path, legacy_path: Path) -> Path:
        if os.getenv("MODEL_ROOT"):
            return Path(os.environ["MODEL_ROOT"]).expanduser() / standalone_path.name
        if standalone_path.is_dir():
            return standalone_path
        return legacy_path

    @staticmethod
    def _cuda_is_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def load(self) -> None:
        """Load all local model artifacts once during application startup."""
        self._validate_artifacts()

        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ModelServiceError(
                "Model dependencies are missing. Install torch, transformers, "
                "peft, and accelerate before starting the server."
            ) from exc

        self.torch = torch
        router_device = torch.device(self.router_device_name)
        qwen_device = torch.device(self.qwen_device_name)

        if qwen_device.type == "cuda" and not torch.cuda.is_available():
            raise ModelServiceError(
                "QWEN_DEVICE=cuda was selected, but CUDA-enabled PyTorch is unavailable."
            )

        try:
            self.router_tokenizer = AutoTokenizer.from_pretrained(
                self.router_dir,
                local_files_only=True,
            )
            self.router_model = AutoModelForSequenceClassification.from_pretrained(
                self.router_dir,
                local_files_only=True,
            ).to(router_device)
            self.router_model.eval()

            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.summary_adapter_dir,
                local_files_only=True,
            )
            if self.qwen_tokenizer.pad_token is None:
                self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

            qwen_dtype = (
                torch.float16 if qwen_device.type == "cuda" else torch.float32
            )
            base_model = self._load_base_model(
                AutoModelForCausalLM,
                qwen_dtype,
            )
            qwen_model = PeftModel.from_pretrained(
                base_model,
                self.summary_adapter_dir,
                adapter_name="summary",
                is_trainable=False,
                local_files_only=True,
            )
            qwen_model.load_adapter(
                self.wiki_adapter_dir,
                adapter_name="wiki_analiz",
                is_trainable=False,
                local_files_only=True,
            )
            self.qwen_model = qwen_model.to(qwen_device)
            self.qwen_model.config.use_cache = True
            self.qwen_model.eval()
        except Exception as exc:
            raise ModelServiceError(
                "The local models could not be loaded. Check the base-model cache, "
                "model paths, and the Python/CUDA environment."
            ) from exc

        self._loaded = True

    def _load_base_model(self, model_class: Any, dtype: Any) -> Any:
        """Support both Transformers versions used by the training notebooks."""
        try:
            return model_class.from_pretrained(
                self.BASE_MODEL_NAME,
                dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        except TypeError:
            return model_class.from_pretrained(
                self.BASE_MODEL_NAME,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )

    def _validate_artifacts(self) -> None:
        for model_dir in (
            self.summary_adapter_dir,
            self.wiki_adapter_dir,
            self.router_dir,
        ):
            if not model_dir.is_dir():
                raise ModelServiceError(f"Model directory not found: {model_dir}")

    @staticmethod
    def normalize_turkish(text: str) -> str:
        return text.replace("I", "ı").lower()

    @classmethod
    def _explicit_task_hint(cls, text: str) -> str | None:
        """Recognize only strong instruction prefixes before trusting BERT."""
        first_line = cls.normalize_turkish(text.strip().splitlines()[0])
        instruction_part = first_line.split(":", 1)[0].strip()

        summary_hints = (
            "çok kısa özetle",
            "kısaca özetle",
            "kısa özetle",
            "özetle",
            "özet çıkar",
            "özet yaz",
            "özetini çıkar",
            "ana fikrini ve önemli noktalarını özetle",
            "önemli bilgileri çıkar",
        )
        entity_hints = (
            "metindeki varlıkları",
            "varlıkları belirtilen",
            "varlıkları türleriyle",
            "tanımlı etiketlerdeki varlıkları",
            "cümlede geçen hedef varlıkları",
            "kişi, yer, tarih, kurum",
            "anahtar kelimeleri çıkar",
            "önemli kelimeleri çıkar",
            "özellik çıkar",
        )

        if any(hint in instruction_part for hint in summary_hints):
            return "summarization"
        if any(hint in instruction_part for hint in entity_hints):
            return "entity_extraction"
        return None

    def _prepare_router_inputs(self, text: str) -> tuple[Any, int, bool]:
        assert self.router_tokenizer is not None
        assert self.torch is not None

        normalized_text = self.normalize_turkish(text)
        token_ids = self.router_tokenizer(
            normalized_text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]

        special_token_count = self.router_tokenizer.num_special_tokens_to_add(
            pair=False
        )
        total_token_count = len(token_ids) + special_token_count
        was_center_cropped = total_token_count > self.MAX_BERT_LENGTH

        if not was_center_cropped:
            encoded = self.router_tokenizer(
                normalized_text,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt",
            )
            return encoded, len(token_ids), False

        cls_token_id = self.router_tokenizer.cls_token_id
        sep_token_id = self.router_tokenizer.sep_token_id
        if cls_token_id is None or sep_token_id is None:
            raise ModelServiceError(
                "The BERT tokenizer does not define CLS/SEP tokens."
            )

        content_token_budget = self.MAX_BERT_LENGTH - 3
        head_count = content_token_budget // 2
        tail_count = content_token_budget - head_count
        cropped_token_ids = (
            [cls_token_id]
            + token_ids[:head_count]
            + [sep_token_id]
            + token_ids[-tail_count:]
            + [sep_token_id]
        )

        input_ids = self.torch.tensor([cropped_token_ids], dtype=self.torch.long)
        attention_mask = self.torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }, len(token_ids), True

    def classify_prompt(self, prompt: str) -> dict[str, Any]:
        if not self._loaded:
            raise ModelServiceError("The model service is not ready yet.")
        assert self.router_model is not None
        assert self.router_tokenizer is not None
        assert self.torch is not None

        start_time = time.perf_counter()
        encoded, original_token_count, was_center_cropped = (
            self._prepare_router_inputs(prompt)
        )
        router_device = next(self.router_model.parameters()).device
        encoded = {key: value.to(router_device) for key, value in encoded.items()}

        with self.torch.inference_mode():
            logits = self.router_model(**encoded).logits
            probabilities = self.torch.softmax(logits, dim=-1)[0]

        predicted_id = int(probabilities.argmax().item())
        configured_labels = self.router_model.config.id2label
        class_label = configured_labels.get(predicted_id)
        if class_label is None:
            class_label = configured_labels.get(str(predicted_id))
        if class_label is None:
            raise ModelServiceError(
                f"No router class label was found for class id {predicted_id}."
            )
        model_class_name = str(class_label).lower()
        confidence = float(probabilities[predicted_id].item())
        class_name = model_class_name
        routing_source = "bert"
        explicit_task = self._explicit_task_hint(prompt)
        if explicit_task is not None:
            class_name = explicit_task
            confidence = 1.0
            routing_source = "explicit_instruction"
        adapter_name = {
            "summarization": "summary",
            "entity_extraction": "wiki_analiz",
        }.get(class_name)
        if adapter_name is None:
            raise ModelServiceError(f"Unknown router class: {class_name}")

        return {
            "class_name": class_name,
            "model_class_name": model_class_name,
            "adapter_name": adapter_name,
            "confidence": confidence,
            "model_confidence": float(probabilities[predicted_id].item()),
            "routing_source": routing_source,
            "classification_seconds": time.perf_counter() - start_time,
            "original_token_count": original_token_count,
            "used_token_count": int(encoded["input_ids"].shape[1]),
            "was_center_cropped": was_center_cropped,
        }

    def _build_messages(self, prompt: str, class_name: str) -> list[dict[str, str]]:
        if class_name == "entity_extraction":
            return [
                {
                    "role": "system",
                    "content": (
                        "Sen bir Named Entity Recognition (NER) modelisin. "
                        "Verilen cümledeki varlıkları yalnızca PERSON, GPE, DATE, ORG, "
                        "NORP, TITLE ve EVENT etiketlerini kullanarak çıkar. "
                        'Yalnızca "Varlıklar: ETİKET: varlık; ..." biçiminde yanıt ver.'
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt: str) -> dict[str, Any]:
        if not prompt.strip():
            raise ModelServiceError("The prompt cannot be empty.")
        if not self._loaded:
            raise ModelServiceError("The model service is not ready yet.")
        assert self.qwen_model is not None
        assert self.qwen_tokenizer is not None
        assert self.torch is not None

        routing_result = self.classify_prompt(prompt)
        messages = self._build_messages(prompt, routing_result["class_name"])
        qwen_device = next(self.qwen_model.parameters()).device

        encoded = self.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(qwen_device) for key, value in encoded.items()}

        pad_token_id = (
            self.qwen_tokenizer.pad_token_id
            if self.qwen_tokenizer.pad_token_id is not None
            else self.qwen_tokenizer.eos_token_id
        )
        im_end_token_id = self.qwen_tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (
            im_end_token_id is None
            or im_end_token_id == self.qwen_tokenizer.unk_token_id
        ):
            im_end_token_id = self.qwen_tokenizer.eos_token_id

        generation_start_time = time.perf_counter()
        with self._generation_lock, self.torch.inference_mode():
            self.qwen_model.set_adapter(routing_result["adapter_name"])
            self.qwen_model.eval()
            generated_ids = self.qwen_model.generate(
                **encoded,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=im_end_token_id,
            )

        prompt_length = encoded["input_ids"].shape[1]
        new_token_ids = generated_ids[0, prompt_length:]
        answer = self.qwen_tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
        ).strip()
        generation_seconds = time.perf_counter() - generation_start_time

        return {
            "reply": answer,
            "details": {
                "class_name": routing_result["class_name"],
                "adapter_name": routing_result["adapter_name"],
                "confidence": routing_result["confidence"],
                "model_class_name": routing_result["model_class_name"],
                "model_confidence": routing_result["model_confidence"],
                "routing_source": routing_result["routing_source"],
                "classification_seconds": routing_result["classification_seconds"],
                "generation_seconds": generation_seconds,
                "generated_tokens": int(new_token_ids.shape[-1]),
                "original_router_tokens": routing_result["original_token_count"],
                "used_router_tokens": routing_result["used_token_count"],
                "router_center_cropped": routing_result["was_center_cropped"],
            },
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self._loaded else "loading",
            "router_device": self.router_device_name,
            "qwen_device": self.qwen_device_name,
            "adapters": ["summary", "wiki_analiz"],
        }
