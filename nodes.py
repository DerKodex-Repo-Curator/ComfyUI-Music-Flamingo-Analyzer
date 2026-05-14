import os
import re
import tempfile
from typing import Any, Dict, Optional, Tuple

import torch

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import torchaudio
except Exception:
    torchaudio = None


MODEL_ID = "nvidia/music-flamingo-2601-hf"
_MUSIC_FLAMINGO_CACHE: dict[tuple, tuple] = {}


class _MusicFlamingoBase:
    CATEGORY = "audio/MusicFlamingo"
    MODEL_MAX_LENGTH = 1200

    @staticmethod
    def _resolve_device(requested: str) -> str:
        req = str(requested or "auto").strip().lower()
        if req == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_dtype(dtype_name: str, device: str):
        name = str(dtype_name or "auto").strip().lower()
        if name == "float16":
            return torch.float16
        if name == "bfloat16":
            return torch.bfloat16
        if name == "float32":
            return torch.float32
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    @staticmethod
    def _resolve_audio_path(audio_file_path: str) -> str:
        raw = str(audio_file_path or "").strip().strip('"')
        if not raw:
            return ""
        if os.path.isabs(raw) and os.path.isfile(raw):
            return os.path.normpath(raw)

        candidates = [raw]
        if folder_paths is not None:
            candidates.extend(
                [
                    os.path.join(folder_paths.get_input_directory(), raw),
                    os.path.join(folder_paths.get_output_directory(), raw),
                ]
            )
            get_temp = getattr(folder_paths, "get_temp_directory", None)
            if callable(get_temp):
                candidates.append(os.path.join(get_temp(), raw))

        for candidate in candidates:
            full = os.path.normpath(candidate)
            if os.path.isfile(full):
                return full
        return ""

    @staticmethod
    def _audio_input_to_temp_path(
        audio: Dict[str, Any],
        max_seconds: Optional[float] = None,
        target_sample_rate: Optional[int] = None,
        force_mono: bool = False,
    ) -> str:
        if not isinstance(audio, dict):
            return ""
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if waveform is None or sample_rate is None:
            return ""
        if torchaudio is None:
            raise ImportError("torchaudio is required to use AUDIO input with Music Flamingo.")

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        if waveform.ndim == 3:
            waveform = waveform[0]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"Audio waveform must be [C,T] or [B,C,T], got {tuple(waveform.shape)}")

        sample_rate = int(sample_rate)
        if force_mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if max_seconds is not None and max_seconds > 0:
            waveform = waveform[:, : int(sample_rate * float(max_seconds))]
        if target_sample_rate and int(target_sample_rate) != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, int(target_sample_rate))
            sample_rate = int(target_sample_rate)

        if folder_paths is not None and callable(getattr(folder_paths, "get_temp_directory", None)):
            temp_root = folder_paths.get_temp_directory()
            os.makedirs(temp_root, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(
                prefix="music_flamingo_",
                suffix=".wav",
                dir=temp_root,
            )
            os.close(fd)
        else:
            fd, temp_path = tempfile.mkstemp(prefix="music_flamingo_", suffix=".wav")
            os.close(fd)

        torchaudio.save(temp_path, waveform.detach().cpu().float(), sample_rate, format="wav")
        return temp_path

    @classmethod
    def _get_model(
        cls,
        model_id: str,
        device: str,
        torch_dtype,
        quantization: str = "none",
        double_quant: bool = True,
        local_files_only: bool = False,
        cpu_offload: bool = False,
    ):
        try:
            from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration
        except Exception as exc:
            raise ImportError(
                "Music Flamingo 2601 requires a Transformers build that includes "
                "MusicFlamingoForConditionalGeneration. Update ComfyUI's Python environment with: "
                "python -m pip install --upgrade transformers accelerate"
            ) from exc

        quantization = str(quantization or "none").strip().lower()
        if device != "cuda":
            quantization = "none"

        key = (
            str(model_id),
            str(device),
            str(torch_dtype),
            quantization,
            bool(double_quant),
            bool(local_files_only),
            bool(cpu_offload),
        )
        cached = _MUSIC_FLAMINGO_CACHE.get(key)
        if cached is not None:
            return cached

        processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=bool(local_files_only),
            trust_remote_code=True,
        )
        kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if quantization in {"4bit", "8bit"}:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:
                raise ImportError(
                    "4-bit/8-bit Music Flamingo loading requires bitsandbytes. Install it with: "
                    "python -m pip install bitsandbytes"
                ) from exc

            if quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=bool(double_quant),
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=bool(cpu_offload),
                )
            kwargs["device_map"] = "auto"
            kwargs.pop("torch_dtype", None)
        elif device == "cuda":
            kwargs["device_map"] = "auto"

        try:
            model = MusicFlamingoForConditionalGeneration.from_pretrained(
                model_id,
                local_files_only=bool(local_files_only),
                trust_remote_code=True,
                **kwargs,
            )
        except ValueError as exc:
            message = str(exc)
            if "Some modules are dispatched on the CPU or the disk" in message:
                raise ValueError(
                    "Music Flamingo did not fit fully on the GPU with the current quantization settings. "
                    "Use quantization=4bit for the recommended RTX workstation path, or enable "
                    "cpu_offload when using quantization=8bit. Also close other ComfyUI models/workflows "
                    "that are occupying VRAM."
                ) from exc
            raise
        if device == "cpu":
            model = model.to(device)
        model.eval()
        _MUSIC_FLAMINGO_CACHE[key] = (processor, model)
        return processor, model

    @staticmethod
    def _prompt_allows_lyrics(prompt: str) -> bool:
        text = str(prompt or "").lower()
        if any(term in text for term in ("do not summarize lyrics", "do not analyze lyrics")):
            return False
        lyric_opt_in_terms = ("lyric", "lyrics", "words", "transcribe")
        if not any(term in text for term in lyric_opt_in_terms):
            return False
        return True

    @classmethod
    def _strip_lyric_sections(cls, text: str) -> str:
        patterns = [
            r"(?ims)(?:^|\n)\s*(?:lyrical\s+themes?|lyrics?|lyric\s+meaning)\s*:\s*.*?(?=\n\s*(?:[A-Z][A-Za-z &/.-]{2,40}:)|\Z)",
            r"(?ims)(?:^|\n)\s*(?:lyrical\s+themes?|lyrics?|lyric\s+meaning)\s*\n[-=]*\n.*?(?=\n\s*(?:[A-Z][A-Za-z &/.-]{2,40}:)|\Z)",
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned).strip()
        return cleaned

    @staticmethod
    def _model_float_dtype(model, fallback):
        for values in (model.parameters(), model.buffers()):
            for value in values:
                if getattr(value, "is_floating_point", lambda: False)():
                    return value.dtype
        return fallback

    @classmethod
    def _move_batch_to_model(cls, inputs, model, torch_dtype):
        moved = inputs.to(model.device)
        model_dtype = cls._model_float_dtype(model, torch_dtype)
        for key, value in list(moved.items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                moved[key] = value.to(dtype=model_dtype)
        return moved

class MusicFlamingoAnalyzer(_MusicFlamingoBase):
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "analyze"
    CATEGORY = "audio/MusicFlamingo"

    DNA_PROMPT = (
        "Extract the music DNA for prompt reuse. Provide BPM, musical key, primary instruments, "
        "genre tag, energy, mood, arrangement notes, and production style. "
        "Do not summarize lyrics or provide lyric themes. Keep it concise and structured."
    )
    DEFAULT_MAX_AUDIO_SECONDS = 30.0
    DEFAULT_TARGET_SAMPLE_RATE = 16000
    DEFAULT_MAX_NEW_TOKENS = 992

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_id": ([MODEL_ID], {"default": MODEL_ID}),
                "prompt": ("STRING", {"default": cls.DNA_PROMPT, "multiline": True}),
                "quantization": (["4bit", "8bit", "none"], {"default": "4bit"}),
                "max_audio_seconds": (
                    "FLOAT",
                    {"default": cls.DEFAULT_MAX_AUDIO_SECONDS, "min": 1.0, "max": 300.0, "step": 1.0},
                ),
                "target_sample_rate": (
                    "INT",
                    {"default": cls.DEFAULT_TARGET_SAMPLE_RATE, "min": 8000, "max": 192000, "step": 1000},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": cls.DEFAULT_MAX_NEW_TOKENS, "min": 32, "max": 1024, "step": 32},
                ),
                "unload_after_run": ("BOOLEAN", {"default": False}),
                "compute_dtype": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                "double_quant": ("BOOLEAN", {"default": True}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio_file_path": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Optional path. Leave empty to use AUDIO input.",
                    },
                ),
            },
        }

    @staticmethod
    def _safe_float(value, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = default
        if value != value:
            value = default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _safe_int(value, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    def analyze(
        self,
        audio: Dict[str, Any],
        model_id: str,
        prompt: str,
        quantization: str,
        max_audio_seconds: float,
        target_sample_rate: int,
        max_new_tokens: int,
        unload_after_run: bool,
        compute_dtype: str = "auto",
        double_quant: bool = True,
        local_files_only: bool = False,
        cpu_offload: bool = False,
        audio_file_path: Optional[str] = "",
    ) -> Tuple[str]:
        max_audio_seconds = self._safe_float(
            max_audio_seconds,
            self.DEFAULT_MAX_AUDIO_SECONDS,
            1.0,
            300.0,
        )
        target_sample_rate = self._safe_int(
            target_sample_rate,
            self.DEFAULT_TARGET_SAMPLE_RATE,
            8000,
            192000,
        )
        max_new_tokens = self._safe_int(
            max_new_tokens,
            self.DEFAULT_MAX_NEW_TOKENS,
            32,
            1024,
        )

        resolved_path = self._resolve_audio_path(audio_file_path or "")
        if not resolved_path:
            resolved_path = self._audio_input_to_temp_path(
                audio,
                max_seconds=max_audio_seconds,
                target_sample_rate=target_sample_rate,
                force_mono=True,
            )
        if not resolved_path:
            raise ValueError("Provide a valid audio_file_path or connect an AUDIO input.")

        device_name = self._resolve_device("auto")
        torch_dtype = self._resolve_dtype(compute_dtype, device_name)
        model_id = str(model_id or MODEL_ID).strip() or MODEL_ID
        user_prompt = str(prompt or "").strip() or self.DNA_PROMPT

        processor, model = self._get_model(
            model_id,
            device_name,
            torch_dtype,
            quantization=quantization,
            double_quant=bool(double_quant),
            local_files_only=bool(local_files_only),
            cpu_offload=bool(cpu_offload),
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "audio", "path": resolved_path},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = self._move_batch_to_model(inputs, model, torch_dtype)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        prompt_tokens = int(inputs.input_ids.shape[1])
        decoded = processor.batch_decode(outputs[:, prompt_tokens:], skip_special_tokens=True)
        raw_text = "\n".join(str(item).strip() for item in decoded if str(item).strip()).strip()
        if not self._prompt_allows_lyrics(user_prompt):
            raw_text = self._strip_lyric_sections(raw_text)

        if unload_after_run:
            _MUSIC_FLAMINGO_CACHE.clear()
            try:
                del model
                del processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return (raw_text,)


NODE_CLASS_MAPPINGS = {
    "MusicFlamingoAnalyzer": MusicFlamingoAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusicFlamingoAnalyzer": "Music Flamingo Analyzer",
}
