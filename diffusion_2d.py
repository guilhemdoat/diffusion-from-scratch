"""
Diffusion Model from Scratch - 2D Toy Example
==============================================

This implements a Denoising Diffusion Probabilistic Model (DDPM) on 2D data.
The goal is pedagogical: understanding how diffusion models work and their
connection to statistical physics.

Physics Connections:
-------------------
1. Forward process: This is essentially a discrete-time Ornstein-Uhlenbeck process,
   which describes Brownian motion in a confining potential. As t → ∞, any initial
   distribution converges to a Gaussian (equilibrium/thermal distribution).

2. The forward SDE can be written as:
   dx = -β(t)/2 · x dt + √β(t) dW

   This is Langevin dynamics with a linear drift toward zero and diffusion.

3. Reverse process: Anderson (1982) showed that any diffusion process can be reversed
   if we know the "score function" ∇_x log p(x,t). This is the key insight!

4. The score ∇ log p(x) points toward regions of high probability density.
   In physics terms, it's like -∇U/kT where U is a free energy.

5. Training objective: We learn the score function by "denoising score matching" -
   the network learns to predict the noise that was added, which is equivalent
   to learning the score.

Author: Built with Claude for pedagogical purposes
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# PART 1: TOY DATASET
# =============================================================================

def make_swiss_roll(n_samples: int = 10000) -> torch.Tensor:
    """
    Generate a 2D Swiss roll dataset.

    This is a classic ML toy dataset - a spiral that tests whether
    the model can learn a complex, non-convex distribution.

    The Swiss roll is parametrized as:
        x = t * cos(t)
        y = t * sin(t)
    where t is sampled from some range.
    """
    # Parameter t controls position along the spiral
    t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(n_samples))

    # Spiral coordinates with some noise
    x = t * np.cos(t)
    y = t * np.sin(t)

    # Stack and normalize to roughly [-1, 1] range
    data = np.stack([x, y], axis=1)
    data = data / np.abs(data).max() * 0.8  # Scale to [-0.8, 0.8]

    # Add small Gaussian noise to make it slightly fuzzy
    data += 0.02 * np.random.randn(*data.shape)

    return torch.tensor(data, dtype=torch.float32)


def make_two_moons(n_samples: int = 10000) -> torch.Tensor:
    """
    Generate a two moons dataset - two interleaving half circles.
    Another classic for testing generative models.
    """
    n = n_samples // 2

    # First moon
    theta1 = np.linspace(0, np.pi, n)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)

    # Second moon (shifted and flipped)
    theta2 = np.linspace(0, np.pi, n)
    x2 = 1 - np.cos(theta2)
    y2 = 1 - np.sin(theta2) - 0.5

    # Combine
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    data = np.stack([x, y], axis=1)

    # Center and scale
    data = data - data.mean(axis=0)
    data = data / np.abs(data).max() * 0.8

    # Add noise
    data += 0.03 * np.random.randn(*data.shape)

    return torch.tensor(data, dtype=torch.float32)


# =============================================================================
# PART 2: FORWARD DIFFUSION PROCESS
# =============================================================================

class ForwardDiffusion:
    """
    The forward diffusion process gradually adds noise to data.

    Physics interpretation:
    -----------------------
    This is a discrete approximation of the Ornstein-Uhlenbeck process.
    At each step, we're essentially doing:

        x_{t+1} = √(1-β_t) · x_t + √β_t · ε,  where ε ~ N(0, I)

    This can be rewritten as:
        x_{t+1} - x_t = -(1 - √(1-β_t)) · x_t + √β_t · ε

    For small β_t, this approximates:
        dx = -β/2 · x dt + √β dW

    This is Langevin dynamics! The drift term pulls x toward 0,
    while the noise term adds thermal fluctuations.

    Key insight: After many steps, ANY initial distribution converges
    to N(0, I) - this is thermalization/equilibration.

    Variance Schedule:
    -----------------
    β_t controls the noise level at each step. Common choices:
    - Linear: β_t increases linearly from β_min to β_max
    - Cosine: More gradual, often works better in practice

    We also precompute useful quantities:
    - α_t = 1 - β_t  (signal retention per step)
    - ᾱ_t = ∏_{s=1}^t α_s  (cumulative signal retention)

    The magic formula: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    This lets us jump directly to any timestep t!
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, schedule: str = 'linear'):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Create the noise schedule
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            # Cosine schedule from "Improved DDPM" paper
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos((steps / num_timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # ᾱ_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For the reverse process
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0) - the forward process.

        This uses the "reparametrization trick":
            x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

        where ε ~ N(0, I).

        Args:
            x_0: Original data, shape (batch, dim)
            t: Timesteps, shape (batch,)
            noise: Optional pre-generated noise

        Returns:
            x_t: Noised data
            noise: The noise that was added (needed for training)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get the right coefficients for each sample in the batch
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        # The magic formula!
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

        return x_t, noise

    def visualize_forward_process(self, data: torch.Tensor, timesteps: list = None):
        """
        Visualize how the forward process destroys structure.

        This is like watching a gas expand and equilibrate!
        """
        if timesteps is None:
            timesteps = [0, 50, 100, 250, 500, 999]

        fig, axes = plt.subplots(1, len(timesteps), figsize=(3*len(timesteps), 3))

        for ax, t in zip(axes, timesteps):
            if t == 0:
                x_t = data
            else:
                t_tensor = torch.full((data.shape[0],), t, dtype=torch.long)
                x_t, _ = self.q_sample(data, t_tensor)

            ax.scatter(x_t[:, 0].numpy(), x_t[:, 1].numpy(), s=1, alpha=0.5)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.set_title(f't = {t}')

            # Add variance info
            var = x_t.var().item()
            ax.text(0.05, 0.95, f'var={var:.2f}', transform=ax.transAxes,
                   fontsize=8, verticalalignment='top')

        plt.suptitle('Forward Diffusion: Data → Noise (Thermalization)')
        plt.tight_layout()
        return fig


# =============================================================================
# PART 3: THE NEURAL NETWORK (Score/Noise Predictor)
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for the timestep.

    This is borrowed from the Transformer architecture. The idea is to
    encode the timestep t as a high-dimensional vector that the network
    can easily use.

    Why sinusoidal?
    - Different frequencies capture different scales of t
    - The network can learn to extract "how noisy is this?" information
    - It's smooth and continuous in t
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2

        # Frequencies: exp(-log(10000) * i / half_dim) for i = 0, ..., half_dim-1
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Outer product: (batch,) x (half_dim,) -> (batch, half_dim)
        embeddings = t[:, None] * embeddings[None, :]

        # Concatenate sin and cos
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class NoisePredictor(nn.Module):
    """
    Neural network that predicts the noise ε given (x_t, t).

    Physics interpretation:
    ----------------------
    Remember: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

    If we predict ε, we can recover x_0:
        x_0 = (x_t - √(1-ᾱ_t) · ε_pred) / √ᾱ_t

    Equivalently, predicting ε is related to predicting the SCORE:
        ∇_x log p(x_t) ≈ -ε / √(1-ᾱ_t)

    The score tells us which direction increases the probability density.
    In physics terms: it's like -∇U/(kT), pointing "downhill" in free energy.

    Architecture:
    ------------
    For 2D data, we use a simple MLP. For images, you'd use a U-Net.
    The key is that the network takes (x_t, t) as input and outputs
    a prediction of the same dimension as x_t.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128,
                 time_emb_dim: int = 32, num_layers: int = 4):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main network: processes x_t with time conditioning
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise given noisy data x_t and timestep t.

        Args:
            x: Noisy data, shape (batch, input_dim)
            t: Timesteps, shape (batch,)

        Returns:
            Predicted noise, shape (batch, input_dim)
        """
        # Embed the timestep
        t_emb = self.time_mlp(t.float())  # (batch, hidden_dim)

        # Project input and add time embedding
        h = self.input_proj(x) + t_emb

        # Process through hidden layers with residual connections
        for layer in self.layers:
            h = h + layer(h)  # Residual connection

        # Project to output dimension
        return self.output_proj(h)


# =============================================================================
# PART 4: TRAINING
# =============================================================================

def train_diffusion(model: nn.Module, diffusion: ForwardDiffusion,
                    data: torch.Tensor, num_epochs: int = 500,
                    batch_size: int = 256, lr: float = 1e-3,
                    device: str = 'cpu', save_dir: str = None,
                    sample_every: int = 100) -> list:
    """
    Train the diffusion model using the denoising score matching objective.

    The Training Objective:
    ----------------------
    We want to learn the score function ∇_x log p(x_t).

    The brilliant insight (from Vincent, 2011 and Song et al.) is that we can
    learn this by "denoising score matching":

        L = E_{t, x_0, ε} [ ||ε_θ(x_t, t) - ε||² ]

    where:
    - t is uniformly sampled from {1, ..., T}
    - x_0 is sampled from the data distribution
    - ε ~ N(0, I) is the noise
    - x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    - ε_θ is our neural network

    This is just MSE between predicted and actual noise!

    Why does this work?
    ------------------
    The noise ε is related to the score by:
        ∇_x log p(x_t | x_0) = -(x_t - √ᾱ_t · x_0) / (1 - ᾱ_t)
                             = -ε / √(1 - ᾱ_t)

    So predicting ε is equivalent to predicting the score (up to scaling).
    """
    import time

    model = model.to(device)
    data = data.to(device)

    # Move diffusion coefficients to device
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    losses = []
    start_time = time.time()

    print(f"\n    {'Epoch':<8} {'Loss':<12} {'Time':<10} {'Progress'}")
    print("    " + "-" * 50)

    for epoch in range(num_epochs):
        # Shuffle data
        perm = torch.randperm(data.shape[0])

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, data.shape[0], batch_size):
            batch = data[perm[i:i+batch_size]]

            # Sample random timesteps for each sample in the batch
            t = torch.randint(0, diffusion.num_timesteps, (batch.shape[0],), device=device)

            # Sample noise and create noisy data
            noise = torch.randn_like(batch)
            x_t, _ = diffusion.q_sample(batch, t, noise)

            # Predict the noise
            noise_pred = model(x_t, t)

            # MSE loss between predicted and actual noise
            loss = ((noise_pred - noise) ** 2).mean()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        # Progress logging every 25 epochs
        elapsed = time.time() - start_time
        progress = (epoch + 1) / num_epochs

        if (epoch + 1) % 25 == 0 or epoch == 0:
            bar = "#" * int(20 * progress) + "-" * (20 - int(20 * progress))
            print(f"    {epoch+1:<8} {avg_loss:<12.6f} {elapsed:>6.1f}s    [{bar}] {100*progress:.0f}%")

        # Generate intermediate samples to watch the model learn
        if save_dir and (epoch + 1) % sample_every == 0:
            model.eval()
            with torch.no_grad():
                samples = sample_ddpm_fast(model, diffusion, num_samples=500, device=device)
                samples = samples.cpu()

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=1, alpha=0.5, c='red')
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_aspect('equal')
                ax.set_title(f'Samples at epoch {epoch+1}')
                plt.savefig(f'{save_dir}/samples_epoch_{epoch+1:04d}.png', dpi=100, bbox_inches='tight')
                plt.close()
                print(f"    -> Saved intermediate samples: samples_epoch_{epoch+1:04d}.png")
            model.train()

    total_time = time.time() - start_time
    print("    " + "-" * 50)
    print(f"    Training complete! Total time: {total_time:.1f}s")

    return losses


@torch.no_grad()
def sample_ddpm_fast(model: nn.Module, diffusion: ForwardDiffusion,
                     num_samples: int = 1000, device: str = 'cpu',
                     num_steps: int = 200) -> torch.Tensor:
    """
    Fast sampling with fewer steps (for intermediate visualization).
    Uses stride to skip timesteps - a simple form of accelerated sampling.
    """
    model.eval()

    # Move coefficients to device
    betas = diffusion.betas.to(device)
    alphas = diffusion.alphas.to(device)
    sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    posterior_variance = diffusion.posterior_variance.to(device)

    # Use strided timesteps for faster sampling
    stride = max(1, diffusion.num_timesteps // num_steps)
    timesteps = list(range(0, diffusion.num_timesteps, stride))[::-1]

    # Start from pure noise
    x = torch.randn(num_samples, 2, device=device)

    for t in timesteps:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch)

        alpha_t = alphas[t]
        beta_t = betas[t]

        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / sqrt_one_minus_alphas_cumprod[t]) * eps_pred
        )

        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(posterior_variance[t])
            x = mean + sigma_t * noise
        else:
            x = mean

    return x


# =============================================================================
# PART 5: SAMPLING (Reverse Process)
# =============================================================================

@torch.no_grad()
def sample_ddpm(model: nn.Module, diffusion: ForwardDiffusion,
                num_samples: int = 1000, device: str = 'cpu') -> torch.Tensor:
    """
    Generate samples using the reverse diffusion process (DDPM sampling).

    The Reverse Process:
    -------------------
    We start from pure noise x_T ~ N(0, I) and iteratively denoise.

    The reverse step is:
        x_{t-1} = μ_θ(x_t, t) + σ_t · z,  where z ~ N(0, I)

    where:
        μ_θ(x_t, t) = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t))

    Physics interpretation:
    ----------------------
    This is discretized Langevin dynamics running BACKWARD in time!

    Normal Langevin dynamics: dx = -∇U dt + √(2/β) dW
        - Drift toward low energy regions
        - Noise for exploration

    Here, the neural network provides ∇ log p(x_t) ≈ -ε_θ/√(1-ᾱ_t)
    which plays the role of -∇U. We're following the probability gradient!

    The variance σ_t adds stochasticity, which is crucial:
    - It ensures we sample from the full distribution, not just modes
    - It corresponds to the thermal noise in Langevin dynamics
    """
    model.eval()

    # Move coefficients to device
    betas = diffusion.betas.to(device)
    alphas = diffusion.alphas.to(device)
    alphas_cumprod = diffusion.alphas_cumprod.to(device)
    alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(device)
    sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    posterior_variance = diffusion.posterior_variance.to(device)

    # Start from pure noise (the "equilibrium" distribution)
    x = torch.randn(num_samples, 2, device=device)

    # Store trajectory for visualization
    trajectory = [x.cpu().clone()]

    # Reverse diffusion: t = T-1, T-2, ..., 0
    for t in reversed(range(diffusion.num_timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # Predict the noise
        eps_pred = model(x, t_batch)

        # Compute the mean of p(x_{t-1} | x_t)
        # μ = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ)
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]

        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / sqrt_one_minus_alphas_cumprod[t]) * eps_pred
        )

        # Add noise (except for t=0)
        if t > 0:
            noise = torch.randn_like(x)
            # Use the posterior variance
            sigma_t = torch.sqrt(posterior_variance[t])
            x = mean + sigma_t * noise
        else:
            x = mean

        # Store some timesteps for visualization
        if t % 100 == 0 or t < 10:
            trajectory.append(x.cpu().clone())

    return x, trajectory


# =============================================================================
# PART 6: VISUALIZATION AND MAIN
# =============================================================================

def visualize_samples(original_data: torch.Tensor, generated_samples: torch.Tensor,
                     trajectory: list = None):
    """Compare original data distribution with generated samples."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original data
    axes[0].scatter(original_data[:, 0].numpy(), original_data[:, 1].numpy(),
                   s=1, alpha=0.5, c='blue')
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[0].set_aspect('equal')
    axes[0].set_title('Original Data')

    # Generated samples
    axes[1].scatter(generated_samples[:, 0].numpy(), generated_samples[:, 1].numpy(),
                   s=1, alpha=0.5, c='red')
    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].set_aspect('equal')
    axes[1].set_title('Generated Samples')

    plt.tight_layout()
    return fig


def visualize_reverse_process(trajectory: list):
    """Visualize the reverse diffusion process (denoising)."""

    # Select a subset of timesteps to show
    n_show = min(8, len(trajectory))
    indices = np.linspace(0, len(trajectory)-1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(2.5*n_show, 2.5))

    for ax, idx in zip(axes, indices):
        samples = trajectory[idx]
        ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=1, alpha=0.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f'Step {idx}')

    plt.suptitle('Reverse Diffusion: Noise → Data (Generation)')
    plt.tight_layout()
    return fig


def main():
    """Main function to train and evaluate the diffusion model."""

    print("="*60)
    print("DIFFUSION MODEL FROM SCRATCH - 2D TOY EXAMPLE")
    print("="*60)

    # 1. Create dataset
    print("\n[1] Creating Swiss roll dataset...")
    data = make_swiss_roll(n_samples=10000)
    print(f"    Data shape: {data.shape}")
    print(f"    Data range: [{data.min():.2f}, {data.max():.2f}]")

    # 2. Set up forward diffusion
    print("\n[2] Setting up forward diffusion process...")
    diffusion = ForwardDiffusion(num_timesteps=1000, beta_start=1e-4,
                                  beta_end=0.02, schedule='linear')
    print(f"    Timesteps: {diffusion.num_timesteps}")
    print(f"    beta range: [{diffusion.betas[0]:.6f}, {diffusion.betas[-1]:.6f}]")
    print(f"    Final alpha_bar_T: {diffusion.alphas_cumprod[-1]:.6f}")
    print("    (alpha_bar_T ~ 0 means data is fully destroyed -> pure noise)")

    # Visualize forward process
    print("\n[3] Visualizing forward diffusion...")
    fig_forward = diffusion.visualize_forward_process(data[:2000])
    # plt.savefig('figs',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: forward_process.png")

    # 3. Create and train model
    print("\n[4] Creating noise prediction network...")
    model = NoisePredictor(input_dim=2, hidden_dim=128, time_emb_dim=32, num_layers=4)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {num_params:,}")

    print("\n[5] Training (with intermediate samples every 100 epochs)...")
    save_dir = 'diff_2d'
    losses = train_diffusion(model, diffusion, data, num_epochs=500,
                            batch_size=256, lr=1e-3, device=device,
                            save_dir=save_dir, sample_every=100)

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    # plt.savefig('figs',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: training_loss.png")

    # 4. Generate samples
    print("\n[6] Generating samples via reverse diffusion...")
    model.eval()
    samples, trajectory = sample_ddpm(model, diffusion, num_samples=2000, device=device)
    samples = samples.cpu()
    print(f"    Generated {samples.shape[0]} samples")

    # 5. Visualize results
    print("\n[7] Visualizing results...")

    fig_samples = visualize_samples(data[:2000], samples)
    # plt.savefig('figs',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: comparison.png")

    fig_reverse = visualize_reverse_process(trajectory)
    # plt.savefig('figs',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: reverse_process.png")

    print("\n" + "="*60)
    print("DONE! Check the generated images to see the results.")
    print("="*60)

    return model, diffusion, data, samples


if __name__ == "__main__":
    model, diffusion, data, samples = main()
