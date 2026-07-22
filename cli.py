
"""
SDEdit CLI — Edit images using SDEdit from the command line.

Usage:
    # Basic usage
    python cli.py input.jpg --prompt "a cat sitting on a chair" --strength 0.4
    
    # With custom parameters
    python cli.py input.jpg --prompt "sunset landscape" --strength 0.6 \\
        --steps 50 --guidance 7.5 --seed 42 --output result.png
    
    # Get intermediate progress images
    python cli.py input.jpg --prompt "a lion" --strength 0.5 --show-progress
"""

import argparse
import sys
import os

# Ensure the src directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sdedit import SDEditPipeline, utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="SDEdit: Guided Image Synthesis and Editing with SDEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py photo.jpg --prompt "a watercolor painting" --strength 0.5 -o watercolor.png
  python cli.py photo.jpg --prompt "a lion in the savanna" --strength 0.7 --guidance 8.5
  python cli.py photo.jpg --prompt "cyberpunk style" --strength 0.4 --show-progress
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
        help="Editing strength [0.0-1.0]. 0=none, 0.3-0.5=moderate, "
             "0.5-0.7=significant, 1.0=full generation (default: 0.4)"
    )
    parser.add_argument(
        "--steps", "-n", type=int, default=50,
        help="Number of denoising steps (default: 50)"
    )
    parser.add_argument(
        "--guidance", "-g", type=float, default=7.5,
        help="Classifier-free guidance scale. Higher = stronger prompt "
             "adherence (default: 7.5)"
    )
    parser.add_argument(
        "--eta", type=float, default=0.0,
        help="DDIM stochasticity [0-1]. 0=deterministic, 1=DDPM-like "
             "(default: 0.0)"
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
        help="Processing size in pixels (default: 512)"
    )
    parser.add_argument(
        "--show-progress", action="store_true",
        help="Save intermediate progress images"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cuda', 'cpu' (auto-detect if not set)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate strength
    if not 0 <= args.strength <= 1:
        print("Error: strength must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Load input image
    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found")
        sys.exit(1)
    
    print(f"Loading input image: {args.input}")
    input_image = utils.load_image(args.input, size=args.size)
    
    # Set output path
    output_path = args.output or (
        os.path.splitext(args.input)[0] + "_edited.png"
    )
    
    # Determine device
    device = args.device
    if device is not None:
        import torch
        device = torch.device(device)
    
    # Initialize pipeline
    print(f"Initializing SDEdit pipeline (model: {args.model})...")
    pipeline = SDEditPipeline(
        model_id=args.model,
        device=device,
    )
    
    # Run SDEdit
    if args.show_progress:
        print("\nRunning SDEdit with progress tracking...")
        result, intermediates = pipeline.edit_with_progress(
            image=input_image,
            prompt=args.prompt,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            eta=args.eta,
            seed=args.seed,
            size=args.size,
        )
        
        # Save intermediates as a grid
        if intermediates:
            labels = [
                "Noised Input" if i == 0 else f"Step {i * (args.steps // 5)}"
                for i in range(len(intermediates))
            ]
            grid = utils.make_grid(intermediates, labels=labels)
            grid_path = os.path.splitext(output_path)[0] + "_progress.png"
            utils.save_image(grid, grid_path)
            print(f"Progress grid saved: {grid_path}")
    else:
        result = pipeline.edit(
            image=input_image,
            prompt=args.prompt,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            eta=args.eta,
            seed=args.seed,
            size=args.size,
        )
    
    # Save result
    utils.save_image(result, output_path)
    
    # Also save a side-by-side comparison
    comparison = utils.make_grid(
        [input_image, result],
        labels=["Input", "Edited"],
        cols=2,
    )
    comp_path = os.path.splitext(output_path)[0] + "_comparison.png"
    utils.save_image(comparison, comp_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
