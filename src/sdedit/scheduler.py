
"""
DDIM (Denoising Diffusion Implicit Models) Scheduler

Implements the noise scheduling and sampling logic used in the
forward diffusion and reverse denoising processes.

Reference:
    Song et al. "Denoising Diffusion Implicit Models" (2021)
    https://arxiv.org/abs/2010.02502
"""

import torch
from typing import Optional


class DDIMScheduler:
    """DDIM scheduler that defines the noise schedule and sampling steps.
    
    The forward diffusion process adds noise to a clean sample x₀:
        x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε,  ε ~ N(0, I)
    
    where ᾱ_t is the cumulative product of (1 - β_s) for s = 1..t.
    
    The reverse (denoising) step predicts x_{t-1} from x_t:
        x₀_pred = (x_t - √(1 - ᾱ_t) · ε_θ) / √(ᾱ_t)
        x_{t-1} = √(ᾱ_{t-1}) · x₀_pred + √(1 - ᾱ_{t-1}) · ε_θ
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
    ):
        """
        Args:
            num_train_timesteps: Total number of diffusion timesteps.
            beta_start: Starting noise level for the beta schedule.
            beta_end: Ending noise level for the beta schedule.
            beta_schedule: Type of noise schedule ("linear" or "scaled_linear").
            prediction_type: What the model predicts ("epsilon" or "v_prediction").
        """
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        
        # --- Step 1: Build the noise schedule (betas) ---
        if beta_schedule == "scaled_linear":
            # Stable Diffusion uses "scaled_linear" schedule
            # beta increases from sqrt(beta_start) to sqrt(beta_end)
            ramp = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps)
            betas = torch.clamp(ramp ** 2, min=0.0, max=0.999)
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        self.betas = betas
        
        # --- Step 2: Compute alphas and their cumulative product ---
        # alpha_t = 1 - beta_t  (signal retention at each step)
        # alpha_bar_t = prod(alpha_s for s=1..t)  (cumulative signal retention)
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Pre-computed sqrt terms for efficiency
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
    
    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to a clean latent up to a given timestep.
        
        Implements: x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε
        
        This is the core of SDEdit: we take a real image's latent z₀,
        add noise up to timestep t₀, then let the reverse process 
        "reimagine" it guided by a text prompt.
        
        Args:
            latents: Clean latent representation [B, C, H, W] (z₀).
            noise: Random noise tensor [B, C, H, W] (ε ~ N(0, I)).
            timestep: Target timestep(s) to noise up to.
            
        Returns:
            Noisy latents at the given timestep [B, C, H, W].
        """
        # Gather pre-computed terms for given timestep(s)
        sqrt_alpha_prod = self.sqrt_alphas_cumprod.to(latents.device)[timestep]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.to(latents.device)[timestep]
        
        # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
        for _ in range(latents.ndim - 1):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        return sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        latents: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single reverse denoising step: from x_t to x_{t-1}.
        
        DDIM step (deterministic when eta=0):
            1. Predict x₀:  x₀_pred = (x_t - √(1-ᾱ_t) · ε_θ) / √(ᾱ_t)
            2. Predict x_{t-1}: x_{t-1} = √(ᾱ_{t-1}) · x₀_pred + √(1-ᾱ_{t-1}) · ε_θ
        
        With stochasticity (eta > 0), additional noise is added controlled by sigma.
        
        Args:
            model_output: Predicted noise ε_θ from the UNet [B, C, H, W].
            timestep: Current timestep t.
            latents: Current noisy latents x_t [B, C, H, W].
            eta: Stochasticity parameter (0 = deterministic DDIM, 1 = DDPM-like).
            
        Returns:
            Denoised latents x_{t-1} [B, C, H, W].
        """
        device = latents.device
        
        # Gather cumulative alpha products for current timestep
        alpha_prod_t = self.alphas_cumprod[timestep].to(device)
        alpha_prod_t_prev = (
            self.alphas_cumprod[timestep - 1].to(device)
            if timestep > 0
            else torch.tensor(1.0, device=device)
        )
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # --- Step 1: Predict x₀ from current latent and model output ---
        # x₀_pred = (x_t - √(1 - ᾱ_t) · ε_θ) / √(ᾱ_t)
        if self.prediction_type == "epsilon":
            pred_original = (latents - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.prediction_type == "v_prediction":
            # v = ε / √(1-ᾱ) - x₀ / √(ᾱ)  (used in some models like v-prediction)
            pred_original = alpha_prod_t.sqrt() * latents - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        
        # --- Step 2: Compute x_{t-1} ---
        # Direction coefficient pointing towards x₀
        pred_original_coeff = alpha_prod_t_prev.sqrt() * beta_prod_t / beta_prod_t_prev.sqrt()
        
        # Direction coefficient pointing towards ε_θ (the noise direction)
        current_coeff = beta_prod_t_prev.sqrt()
        
        # DDIM with optional stochasticity
        if eta > 0:
            # sigma controls how much random noise to add
            sigma = (
                eta
                * (beta_prod_t_prev / beta_prod_t).sqrt()
                * (1 - alpha_prod_t / alpha_prod_t_prev).sqrt()
            )
            random_noise = torch.randn_like(latents)
        else:
            sigma = 0.0
            random_noise = 0
        
        # Full DDIM step formula:
        # x_{t-1} = √(ᾱ_{t-1}) · x₀_pred 
        #         + √(1 - ᾱ_{t-1} - σ²) · ε_θ
        #         + σ · ε_t
        prev_latents = (
            alpha_prod_t_prev.sqrt() * pred_original
            + (beta_prod_t_prev - sigma ** 2).sqrt() * model_output
            + sigma * random_noise
        )
        
        return prev_latents
    
    def get_timesteps(self, num_inference_steps: int, offset: int = 0) -> list:
        """Get the list of timesteps for the reverse denoising loop.
        
        Skips evenly across the full schedule so we only run 
        `num_inference_steps` steps instead of the full 1000.
        
        Args:
            num_inference_steps: Number of denoising steps to run.
            offset: Starting offset for the sequence.
            
        Returns:
            List of timesteps to iterate through (descending).
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (
            (torch.arange(0, num_inference_steps) * step_ratio)
            .round()
            .long()
            .tolist()
        )
        timesteps = [min(t + offset, self.num_train_timesteps - 1) for t in timesteps]
        return timesteps[::-1]
    
    def get_noise_level(self, timestep: int) -> float:
        """Return the noise level (standard deviation of noise) at a given timestep.
        
        Noise level = √(1 - ᾱ_t), i.e. the fraction of noise in the latent.
        At t=0, noise_level = 0 (clean). At t=T, noise_level ≈ 1 (pure noise).
        
        This is useful for SDEdit's strength parameter: 
        t₀ controls how much editing freedom vs how much structure preservation.
        """
        return self.sqrt_one_minus_alphas_cumprod[timestep].item()
