
"""
SDEdit CLI — Edit images using SDEdit from the command line.

Usage:
    python cli.py input.jpg --prompt "a cat on a chair" --strength 0.4
    python cli.py input.jpg --prompt "sunset" --strength 0.6 --steps 50
"""

import os
import sys

# =========================================================================
# SSL fix: must run BEFORE any HuggingFace imports
# =========================================================================
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
except ImportError:
    pass

# Force huggingface_hub to use an unsafe httpx client
try:
    import huggingface_hub
    from huggingface_hub import configure_http_backend

    def _make_unsafe_client():
        import httpx
        return httpx.Client(verify=False)

    configure_http_backend(_make_unsafe_client)
except ImportError:
    pass

# =========================================================================
# Normal imports
# =========================================================================
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sdedit import SDEditPipeline, utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="SDEdit: Guided Image Synthesis and Editing with SDEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py photo.jpg --prompt "a watercolor painting" --strength 0.5 -o out.png
  python cli.py photo.jpg --prompt "cyberpunk style" --strength 0.7 --guidance 8.5
        """,
    )

    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument(
        "--prompt", "-p", type=str, required=True,
        help="Text prompt guiding the edit"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output image path (default: input_edited.png)"
    )
    parser.add_argument(
        "--strength", "-s", type=float, default=0.4,
        help="Editing strength [0.0-1.0] (default: 0.4)"
    )
    parser.add_argument(
        "--steps", "-n", type=int, default=50,
        help="Number of denoising steps (default: 50)"
    )
    parser.add_argument(
        "--guidance", "-g", type=float, default=7.5,
        help="CFG scale, higher = stronger prompt adherence (default: 7.5)"
    )
    parser.add_argument(
        "--eta", type=float, default=0.0,
        help="DDIM stochasticity [0-1] (default: 0.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model", type=str, default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID (default: runwayml/stable-diffusion-v1-5)"
    )
    parser.add_argument(
        "--size", type=int, default=512,
        help="Processing size (default: 512)"
    )
    parser.add_argument(
        "--show-progress", action="store_true",
        help="Save intermediate progress images"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda', 'cpu' (auto-detect if not set)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 0 <= args.strength <= 1:
        print("Error: strength must be between 0.0 and 1.0")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found")
        sys.exit(1)

    print(f"Loading input image: {args.input}")
    input_image = utils.load_image(args.input, size=args.size)

    output_path = args.output or (
        os.path.splitext(args.input)[0] + "_edited.png"
    )

    device = args.device
    if device is not None:
        import torch
        device = torch.device(device)

    print(f"Initializing SDEdit pipeline (model: {args.model})...")
    pipeline = SDEditPipeline(
        model_id=args.model,
        device=device,
    )

    if args.show_progress:
        print("\nRunning SDEdit with progress tracking...")
        result, intermediates = pipeline.edit_with_progress(
            image=input_image, prompt=args.prompt,
            strength=args.strength, num_inference_steps=args.steps,
            guidance_scale=args.guidance, eta=args.eta,
            seed=args.seed, size=args.size,
        )
        if intermediates:
            labels = [
                "Noised" if i == 0 else f"Step {i * (args.steps // 5)}"
                for i in range(len(intermediates))
            ]
            grid = utils.make_grid(intermediates, labels=labels)
            grid_path = os.path.splitext(output_path)[0] + "_progress.png"
            utils.save_image(grid, grid_path)
    else:
        result = pipeline.edit(
            image=input_image, prompt=args.prompt,
            strength=args.strength, num_inference_steps=args.steps,
            guidance_scale=args.guidance, eta=args.eta,
            seed=args.seed, size=args.size,
        )

    utils.save_image(result, output_path)

    comparison = utils.make_grid(
        [input_image, result], labels=["Input", "Edited"], cols=2,
    )
    comp_path = os.path.splitext(output_path)[0] + "_comparison.png"
    utils.save_image(comparison, comp_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
