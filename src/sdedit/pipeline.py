
"""
SDEdit Pipeline: Guided Image Synthesis and Editing with SDEs.

Core algorithm:
    1. Encode input image to latent space (via VAE encoder)
    2. Add noise to the latent up to a chosen timestep t₀
    3. Run the reverse denoising process from t₀ → 0
       with classifier-free guidance from a text prompt
    4. Decode the result back to pixel space (via VAE decoder)

Reference:
    Meng et al. "SDEdit: Guided Image Synthesis and Editing with 
    Stochastic Differential Equations" (ICLR 2022)
    https://arxiv.org/abs/2108.01073
"""

import torch
from typing import Optional, Tuple, List
from PIL import Image

from .scheduler import DDIMScheduler
from diffusers import DDIMScheduler as DiffusersDDIMScheduler
from . import utils

# Patch SSL verification for HuggingFace downloads
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""


class SDEditPipeline:
    """Pipeline that runs the SDEdit image editing algorithm.
    
    Uses a pre-trained latent diffusion model (Stable Diffusion) as the
    backbone, but implements the SDEdit-specific editing logic:
    - Forward noising to an intermediate state
    - Reverse denoising with text-guided sampling
    - Configurable editing strength via t₀ selection
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            model_id: HuggingFace model ID for the pretrained diffusion model.
            device: Computation device (auto-detected if None).
            dtype: Precision for model weights.
        """
        self.model_id = model_id
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        
        # These will be loaded on-demand
        self._vae = None          # AutoencoderKL (latent space encoder/decoder)
        self._unet = None         # UNet2DConditionModel (noise predictor)
        self._text_encoder = None # CLIPTextModel (text → embedding)
        self._tokenizer = None    # CLIPTokenizer
        self._scheduler = None    # DDIMScheduler
    
    def _load_models(self):
        """Lazy-load all model components from HuggingFace."""
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        
        print(f"[SDEdit] Loading models from '{self.model_id}'...")
        print(f"[SDEdit] Using device: {self.device}")
        
        # --- VAE (encoder → latent space, decoder ← latent space) ---
        self._vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)
        
        # --- UNet (noise predictor conditioned on text embeddings) ---
        self._unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", torch_dtype=self.dtype
        ).to(self.device)
        
        # --- CLIP text encoder (prompt → text embeddings) ---
        self._text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder", torch_dtype=self.dtype
        ).to(self.device)
        
        # --- CLIP tokenizer (text → tokens) ---
        self._tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )
        
        print("[SDEdit] Models loaded successfully.")
    
    def _get_scheduler(self, num_inference_steps: int = 50):
        """Get or create a scheduler for the reverse diffusion process."""
        sched = DiffusersDDIMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        sched.set_timesteps(num_inference_steps)
        return sched

    def _get_scheduler_scratch(self, num_inference_steps: int = 50):
        """Create a hand-rolled DDIM scheduler (for testing)."""
        return DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )
    
    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a text prompt into CLIP embeddings.
        
        Returns both conditional (prompt) and unconditional (empty) embeddings
        for classifier-free guidance.
        
        Args:
            prompt: Text description to guide the editing.
            
        Returns:
            Tuple of (conditional_embeds, unconditional_embeds).
            Each has shape [1, 77, 768] for Stable Diffusion 1.x.
        """
        # Tokenize the prompt
        text_input = self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Encode to embeddings
        text_embeddings = self._text_encoder(
            text_input.input_ids.to(self.device)
        )[0]
        
        # Tokenize empty prompt for unconditional guidance
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self._tokenizer(
            ["", ""],
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self._text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]
        
        return text_embeddings, uncond_embeddings
    
    @torch.no_grad()
    def _encode_image(self, image: Image.Image, size: int = 512) -> torch.Tensor:
        """Preprocess and encode a PIL image into latent space.
        
        Args:
            image: Input PIL image.
            size: Target size (shorter edge resized to this).
            
        Returns:
            Latent tensor [1, 4, H/8, W/8] ready for diffusion.
        """
        # Preprocess: resize, center-crop, normalize
        tensor = utils.preprocess_image(image, size=size).to(self.device, dtype=self.dtype)
        
        # Encode: pixel space → latent space (using VAE encoder)
        # The VAE compresses the image by a factor of 8 in each dimension
        latents = self._vae.encode(tensor).latent_dist.sample()
        latents = latents * self._vae.config.scaling_factor
        
        return latents
    
    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents back into a PIL image.
        
        Args:
            latents: Latent tensor [1, 4, H/8, W/8].
            
        Returns:
            PIL Image in RGB.
        """
        latents = latents / self._vae.config.scaling_factor
        image = self._vae.decode(latents).sample
        return utils.postprocess_image(image)
    
    @torch.no_grad()
    def edit(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.4,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        seed: Optional[int] = None,
        size: int = 512,
    ) -> Image.Image:
        """Run SDEdit: edit an image guided by a text prompt.
        
        This is the main entry point of the SDEdit algorithm.
        
        Algorithm overview:
        1. Encode image → latent space z₀
        2. Pick timestep t₀ based on `strength`
        3. Add noise: z_{t₀} = √(ᾱ_{t₀})·z₀ + √(1-ᾱ_{t₀})·ε
        4. Run reverse denoising from t₀ → 0 with CFG
        5. Decode z₀' → pixel space
        
        Args:
            image: Input PIL image to edit.
            prompt: Text prompt guiding the edit.
            strength: Editing strength [0, 1].
                - 0: No editing (pure reconstruction, no noise added)
                - ~0.3-0.5: Moderate changes, structure preserved
                - ~0.5-0.7: Significant changes, some structure preserved
                - 1: Full generation (pure noise → image from prompt)
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
                Higher = stronger adherence to prompt.
            eta: DDIM stochasticity (0 = deterministic, 1 = full stochastic).
            seed: Random seed for reproducibility.
            size: Image size for processing.
            
        Returns:
            Edited PIL Image.
        """
        # --- Setup ---
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Load models on first call
        if self._vae is None:
            self._load_models()
        
        # Create scheduler
        scheduler = self._get_scheduler(num_inference_steps)

        # --- Step 1: Encode input image to latent space ---
        print("[SDEdit] Encoding input image to latent space...")
        latents = self._encode_image(image, size=size)  # z₀: [1, 4, 64, 64]

        # --- Step 2: Pick timestep t₀ based on strength ---
        # strength maps to [0, T-1], where T is total training timesteps
        # Higher strength = noisier starting point = more editing freedom
        timesteps = scheduler.timesteps.tolist()
        t_start = max(0, len(scheduler.timesteps) - int(len(scheduler.timesteps) * strength))
        edit_timesteps = timesteps[t_start:]

        if len(edit_timesteps) == 0:
            print("[SDEdit] Strength = 0, returning input image unchanged.")
            return image
        
        t_0 = edit_timesteps[0]  # This is our starting timestep
        total_noise_level = scheduler._get_variance(t_0, 0) ** 0.5
        print(f"[SDEdit] Starting from timestep t₀ = {t_0} "
              f"(noise level: {total_noise_level:.3f}, "
              f"editing steps: {len(edit_timesteps)})")
        
        # --- Step 3: Add noise up to timestep t₀ ---
        # This is the key SDEdit step: partially noise the real image
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(
            latents, noise, torch.tensor([t_0], device=self.device)
        )
        
        # --- Step 4: Encode prompt for classifier-free guidance ---
        print(f"[SDEdit] Encoding prompt: \"{prompt}\"")
        text_embeds, uncond_embeds = self._encode_prompt(prompt)
        
        # Concatenate for batch inference: [uncond, cond]
        # UNet will process both at once, saving inference time
        text_embeddings = torch.cat([uncond_embeds[:1], text_embeds[:1]])
        
        # --- Step 5: Reverse denoising loop (t₀ → 0) ---
        current_latents = noisy_latents
        
        print(f"[SDEdit] Running {len(edit_timesteps)} denoising steps...")
        for i, t in enumerate(edit_timesteps):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
            
            # Expand latents for batch inference: [uncond_path, cond_path]
            latent_model_input = torch.cat([current_latents] * 2)
            
            # --- Classifier-Free Guidance (CFG) ---
            # The UNet predicts noise for both paths simultaneously
            noise_pred = self._unet(
                latent_model_input, t_tensor, encoder_hidden_states=text_embeddings
            ).sample
            
            # Split into unconditional and conditional predictions
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # CFG: guided_noise = uncond + guidance_scale * (conditional - uncond)
            # This pushes the result towards the text prompt
            noise_pred_guided = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            
            # Single reverse step
            step_result = scheduler.step(
                noise_pred_guided, t, current_latents, eta=eta
            )
            current_latents = step_result.prev_sample
            
            if (i + 1) % 10 == 0 or i == len(edit_timesteps) - 1:
                progress = (i + 1) / len(edit_timesteps) * 100
                print(f"  [{progress:5.1f}%] Step {i+1}/{len(edit_timesteps)} "
                      f"(timestep {t} → {max(0, t-1)})")
        
        # --- Step 6: Decode latents back to pixel space ---
        print("[SDEdit] Decoding result...")
        result = self._decode_latents(current_latents)
        
        print("[SDEdit] Done!")
        return result
    
    def edit_with_progress(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.4,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        seed: Optional[int] = None,
        size: int = 512,
    ) -> Tuple[Image.Image, List[Image.Image]]:
        """Like edit(), but also returns intermediate results for visualization.
        
        Returns:
            Tuple of (final_image, list_of_intermediate_images).
        """
        # Setup same as edit()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        if self._vae is None:
            self._load_models()
        
        scheduler = self._get_scheduler(num_inference_steps)
        latents = self._encode_image(image, size=size)

        timesteps = scheduler.timesteps.tolist()
        t_start = max(0, len(timesteps) - int(len(timesteps) * strength))
        edit_timesteps = timesteps[t_start:]
        
        if len(edit_timesteps) == 0:
            return image, [image]
        
        t_0 = edit_timesteps[0]
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, torch.tensor([t_0]))
        
        text_embeds, uncond_embeds = self._encode_prompt(prompt)
        text_embeddings = torch.cat([uncond_embeds[:1], text_embeds[:1]])
        
        current_latents = noisy_latents
        intermediates = []
        
        # Record the noisy starting point
        intermediates.append(self._decode_latents(current_latents))
        
        for i, t in enumerate(edit_timesteps):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
            latent_model_input = torch.cat([current_latents] * 2)
            
            noise_pred = self._unet(
                latent_model_input, t_tensor, encoder_hidden_states=text_embeddings
            ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_guided = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            
            current_latents = scheduler.step(
                noise_pred_guided, t, current_latents, eta=eta
            ).prev_sample
            
            # Record intermediate at regular intervals
            if (i + 1) % (len(edit_timesteps) // 5) == 0 or i == len(edit_timesteps) - 1:
                intermediates.append(self._decode_latents(current_latents))
        
        return self._decode_latents(current_latents), intermediates
