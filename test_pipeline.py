
"""
Diagnostic script to test each stage of the SDEdit pipeline.
Run this first to find where the issue is.
"""

import os
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
except ImportError:
    pass
try:
    import huggingface_hub
    from huggingface_hub import configure_http_backend
    def _make_unsafe_client():
        import httpx
        return httpx.Client(verify=False)
    configure_http_backend(_make_unsafe_client)
except ImportError:
    pass

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import torch
from sdedit import SDEditPipeline, utils


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "企鹅.png"
    
    print("=" * 60)
    print("DIAGNOSTIC 1: VAE encode + decode (reconstruction test)")
    print("=" * 60)
    
    pipeline = SDEditPipeline()
    pipeline._load_models()
    
    image = utils.load_image(image_path, size=512)
    image.save("_debug_input.png")
    print(f"Input image: {image.size}")
    
    # Encode
    tensor = utils.preprocess_image(image, size=512)
    print(f"Preprocessed tensor: {tensor.shape}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")
    
    with torch.no_grad():
        latents = pipeline._vae.encode(tensor.to(pipeline.device, pipeline.dtype)).latent_dist.sample()
        print(f"VAE encoder output (raw): shape={latents.shape}, "
              f"range=[{latents.min():.3f}, {latents.max():.3f}], "
              f"std={latents.std():.3f}")
        
        latents_scaled = latents * pipeline._vae.config.scaling_factor
        print(f"After scaling factor ({pipeline._vae.config.scaling_factor}): "
              f"range=[{latents_scaled.min():.3f}, {latents_scaled.max():.3f}]")
        
        # Decode immediately (no diffusion)
        latents_unscaled = latents_scaled / pipeline._vae.config.scaling_factor
        reconstructed = pipeline._vae.decode(latents_unscaled).sample
        print(f"VAE decoder output: range=[{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
        
        result = utils.postprocess_image(reconstructed)
        result.save("_debug_reconstruction.png")
        print("Saved: _debug_reconstruction.png (should look like the original)")
    
    print()
    print("=" * 60)
    print("DIAGNOSTIC 2: Forward noise + reverse denoise (reproduce editing)")
    print("=" * 60)
    
    # Run SDEdit with a simple prompt
    result = pipeline.edit(
        image=image,
        prompt="a photo of a penguin",
        strength=0.3,      # low strength for diagnostic
        num_inference_steps=30,
        guidance_scale=5.0,  # low guidance for diagnostic
        seed=42,
    )
    result.save("_debug_edit.png")
    print("Saved: _debug_edit.png")
    
    print()
    print("=" * 60)
    print("Check the debug images above.")
    print("- If _debug_reconstruction.png looks GOOD, the VAE works fine.")
    print("- If _debug_reconstruction.png has BLOCKS/NOISE, the VAE is broken.")
    print("  (try: pip install --upgrade diffusers transformers)")
    print("- If _debug_edit.png has BLOCKS/NOISE, the diffusion scheduler has a bug.")
    print("=" * 60)


if __name__ == "__main__":
    main()
