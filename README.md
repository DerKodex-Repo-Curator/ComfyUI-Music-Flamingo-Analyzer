# ComfyUI Custom Node for Music Flamingo

**Custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows you to analyze songs using NVIDIA's [Music Flamingo 2601](https://huggingface.co/nvidia/music-flamingo-2601-hf).**

![Screenshot](/example_workflow/screenshot.png?raw=true)


## Installation

1. Clone this repo into `custom_nodes`:
    ```
    cd ComfyUI/custom_nodes
    git clone https://github.com/tavy315/ComfyUI-Music-Flamingo-Analyzer
    ```
2. Install the node dependencies in the same Python environment ComfyUI uses:
    ```
    python -m pip install -r requirements.txt
    ```

The node defaults to [nvidia/music-flamingo-2601-hf](https://huggingface.co/nvidia/music-flamingo-2601-hf). The first run downloads the model through Hugging Face, so it can take a while.

## Nodes

- **Music Flamingo Analyzer**: optimized MVP node for prompt reuse. It defaults to 4-bit NF4 loading with `bitsandbytes`, nested/double quantization, automatic bf16/fp16 compute dtype selection, trims AUDIO input to a 30-second mono window at the model's native 16 kHz rate, and returns `description`.

After the first successful model download, enable `local_files_only` to stop Hugging Face from making remote metadata checks on each fresh ComfyUI process. Keep `unload_after_run=false` to reuse the in-memory model across repeated runs in the same ComfyUI session.

Prefer `quantization=4bit` for VRAM-constrained workflows. If you intentionally use `quantization=8bit` and Transformers reports that modules were dispatched to CPU or disk, enable `cpu_offload`; it is slower but avoids the 8-bit loader failure.

Use `description` with a text display or prompt-combiner node, then append your generation suffix for Suno or another music model.

Default analyzer prompts avoid lyric summaries and lyric-theme interpretation.

## Example Workflows

- `workflow/example.json`
