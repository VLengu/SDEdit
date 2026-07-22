
"""
Utility functions for image loading, preprocessing, and visualization.
"""

import torch
from PIL import Image
from typing import Optional, Union, List
import numpy as np


def load_image(
    path: str,
    size: Optional[int] = None,
    mode: str = "RGB",
) -> Image.Image:
    """Load an image from disk and optionally resize.
    
    Args:
        path: Path to image file.
        size: If set, resize the shorter edge to this size.
        mode: Color mode (RGB or L).
        
    Returns:
        PIL Image.
    """
    image = Image.open(path).convert(mode)
    
    if size is not None:
        # Resize keeping aspect ratio
        w, h = image.size
        if w < h:
            new_w = size
            new_h = int(h * size / w)
        else:
            new_h = size
            new_w = int(w * size / h)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image


def preprocess_image(
    image: Image.Image,
    size: int = 512,
) -> torch.Tensor:
    """Preprocess a PIL image for the VAE encoder.
    
    Steps:
        1. Resize so shorter edge = size
        2. Center crop to size × size
        3. Normalize from [0, 255] to [-1, 1]
        4. Add batch dimension
    
    Args:
        image: Input PIL image.
        size: Target size for the crop.
        
    Returns:
        Tensor [1, 3, size, size] normalized to [-1, 1].
    """
    # Resize keeping aspect ratio, then center crop
    w, h = image.size
    if w < h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    
    image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    image = image.crop((left, top, left + size, top + size))
    
    # Convert to tensor and normalize to [-1, 1]
    tensor = torch.from_numpy(np.array(image)).float()
    tensor = tensor.permute(2, 0, 1)  # HWC → CHW
    tensor = tensor / 127.5 - 1.0      # [0, 255] → [-1, 1]
    tensor = tensor.unsqueeze(0)       # Add batch dim
    
    return tensor


def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a VAE decoder output tensor back to a PIL image.
    
    Args:
        tensor: Output tensor from VAE decoder [1, 3, H, W] in [-1, 1].
        
    Returns:
        PIL Image in RGB.
    """
    tensor = tensor.detach().cpu()
    tensor = tensor.squeeze(0)           # Remove batch dim: [3, H, W]
    tensor = (tensor + 1.0) / 2.0        # [-1, 1] → [0, 1]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = tensor.permute(1, 2, 0)     # CHW → HWC
    tensor = (tensor * 255).byte()        # [0, 1] → [0, 255]
    
    return Image.fromarray(tensor.numpy())


def make_grid(
    images: List[Image.Image],
    labels: Optional[List[str]] = None,
    cols: int = 4,
) -> Image.Image:
    """Arrange multiple images in a grid for comparison.
    
    Args:
        images: List of PIL Images.
        labels: Optional labels for each image.
        cols: Number of columns in the grid.
        
    Returns:
        Combined grid image.
    """
    if not images:
        raise ValueError("No images provided")
    
    rows = (len(images) + cols - 1) // cols
    first = images[0]
    w, h = first.size
    
    grid = Image.new("RGB", (cols * w, rows * h))
    
    for i, img in enumerate(images):
        row, col = divmod(i, cols)
        grid.paste(img.resize((w, h), Image.LANCZOS), (col * w, row * h))
        
        if labels and i < len(labels):
            # Use PIL to draw text labels
            from PIL import ImageDraw
            draw = ImageDraw.Draw(grid)
            draw.text((col * w + 5, row * h + 5), labels[i], fill="white")
    
    return grid


def save_image(
    image: Image.Image,
    path: str,
    quality: int = 95,
):
    """Save an image to disk.
    
    Args:
        image: PIL Image to save.
        path: Output path.
        quality: JPEG quality (only used for JPEG format).
    """
    image.save(path, quality=quality)
    print(f"Saved: {path}")


def setup_seed(seed: int):
    """Set random seeds for reproducibility across all backends.
    
    Args:
        seed: Random seed.
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
