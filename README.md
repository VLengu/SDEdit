
# SDEdit — Guided Image Synthesis and Editing with SDEs

A from-scratch implementation of **SDEdit** (Stochastic Differential Editing), 
the image editing algorithm introduced in:

> **SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations**  
> Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon  
> *ICLR 2022* · [arXiv:2108.01073](https://arxiv.org/abs/2108.01073)

## How It Works

SDEdit edits real images using a pre-trained diffusion model without any fine-tuning:

```
Input Image → Encode → Add Noise to t₀ → Reverse Denoise with Prompt → Result
```

The key insight: by partially noising a real image (to an intermediate timestep `t₀`)
and then running the reverse diffusion process guided by a text prompt, the model 
can **reimagine** the image while preserving its overall structure. The parameter 
`t₀` (controlled by `strength`) balances:

| Strength | Effect |
|----------|--------|
| ~0.0–0.3 | Minimal changes, preserves structure |
| ~0.3–0.5 | Moderate edits, good structure preservation |
| ~0.5–0.7 | Significant changes, some structure retained |
| ~0.7–1.0 | Heavy edits approaching full generation |

## Project Structure

```
SDEdit/
├── cli.py                  # Command-line interface
├── requirements.txt         # Python dependencies
├── src/
│   └── sdedit/             # Core package
│       ├── __init__.py
│       ├── scheduler.py    # DDIM scheduler (from scratch)
│       ├── pipeline.py     # SDEdit editing pipeline
│       └── utils.py        # Image loading/saving utilities
├── README.md
└── LICENSE
```

## What's Implemented From Scratch

| Component | Status | Description |
|-----------|--------|-------------|
| **DDIM Scheduler** | ✅ | Noise schedule, forward diffusion, reverse denoising steps — all the math |
| **SDEdit Pipeline** | ✅ | Image encoding, noise injection, CFG-guided sampling, decoding |
| **CLI** | ✅ | Full command-line interface with progress tracking |

The project uses **HuggingFace Diffusers** only for loading pre-trained weights 
(UNet, VAE, CLIP text encoder). The core SDEdit algorithm and DDIM sampling 
logic are implemented entirely from scratch.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Edit an Image

```bash
python cli.py input.jpg --prompt "a watercolor painting of a house" --strength 0.4
```

### 3. Examples

```bash
# Moderate edit with specific seed for reproducibility
python cli.py photo.jpg --prompt "a cute cat" --strength 0.4 --seed 42

# Heavy edit with higher guidance
python cli.py photo.jpg --prompt "cyberpunk city, neon lights" --strength 0.7 --guidance 8.5

# With progress visualization
python cli.py photo.jpg --prompt "a lion" --strength 0.5 --show-progress
```

### Command-Line Options

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `input` | — | required | Path to input image |
| `--prompt` | `-p` | required | Text prompt guiding the edit |
| `--strength` | `-s` | 0.4 | Editing strength [0.0–1.0] |
| `--steps` | `-n` | 50 | Number of denoising steps |
| `--guidance` | `-g` | 7.5 | Classifier-free guidance scale |
| `--seed` | — | random | Random seed for reproducibility |
| `--output` | `-o` | auto | Output image path |
| `--model` | — | runwayml/stable-diffusion-v1-5 | HuggingFace model ID |
| `--eta` | — | 0.0 | DDIM stochasticity |
| `--device` | — | auto | Device (cuda/cpu) |
| `--show-progress` | — | off | Save intermediate images |

## License

[MIT License](LICENSE)
