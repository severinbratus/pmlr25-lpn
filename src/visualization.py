import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from models.lpn import LPN, make_leave_one_out
from models.encoder import IOPairSetEncoder
from models.decoder import Decoder
from models.utils import ReluNet
from sine_data_generator import SineDataGenerator

class LPNVisualizer:
    def __init__(
        self,
        d_latent: int = 2,
        numstep: int = 1,
        model_path: str = None,
        device: str = None
    ):
        """Initialize the LPN visualizer with model loading and setup.
        
        Args:
            d_latent: Dimension of latent space
            numstep: Number of steps for the model
            model_path: Path to the saved model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.d_latent = d_latent
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize model components
        self.encoder = IOPairSetEncoder(
            phi=ReluNet(2, 16, 32),
            rho_0=ReluNet(32, 16, 16),
            rho_1=ReluNet(16, 8, d_latent)
        )
        
        self.decoder = Decoder(
            d_input=1,
            d_latent=d_latent,
            ds_hidden=[16, 16, 16],
            d_output=1
        )
        
        self.lpn = LPN(
            d_input=1,
            d_output=1,
            d_latent=d_latent,
            encoder=self.encoder,
            decoder=self.decoder
        ).to(self.device)
        
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.lpn.load_state_dict(state_dict, strict=False)
        
        self.lpn.eval()

    def generate_data(self, num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate sine wave data for visualization.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (xs, ys, amplitudes, phases)
        """
        data_gen = SineDataGenerator(num_samples_per_class=num_samples, batch_size=1)
        xs, ys, amp, phase = data_gen.generate()
        # Convert to float32
        return xs, ys, amp, phase
    
    def process_data(self, io_pairs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process input-output pairs for visualization.
        
        Args:
            io_pairs: Input-output pairs tensor
            
        Returns:
            Tuple of (leave_one_out, left_one_out, left_one_out_y_true, z_mu)
        """
        leave_one_out = make_leave_one_out(io_pairs, axis=1)
        left_one_out = io_pairs[:, :, 0].unsqueeze(-1)
        left_one_out_y_true = io_pairs[:, :, 1].unsqueeze(-1)
        z_mu, _ = self.lpn.encoder(leave_one_out)

        aux = self.lpn(left_one_out, debug=True, K=10)
        self.lpn.gradient_ascent(z_sample, leave_one_out, K, debug=debug)
        z_traj = self.lpn.z_traj
        
        return leave_one_out, left_one_out, left_one_out_y_true, z_mu, z_traj

    def create_latent_grid(self, resolution: int = 300) -> torch.Tensor:
        """Create a 2D grid in latent space for visualization.
        
        Args:
            resolution: Resolution of the grid
            
        Returns:
            Grid tensor
        """
        return torch.tensor(
            np.mgrid[-5:5:resolution*1j, -5:5:resolution*1j].reshape(2, 1, -1).T
        ).float().to(self.device)

    def plot_latent_space(
        self,
        X_cat: np.ndarray,
        z_mu: Optional[torch.Tensor] = None,
        sample: int = 0,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """Plot the latent space visualization.
        
        Args:
            X_cat: Concatenated data for plotting
            z_mu: Mean latent vectors
            sample: Sample index to plot
            log_scale: Whether to use log scale for MSE values
            figsize: Figure size
        """
        X = X_cat[:, sample, :2]
        Y = X_cat[:, sample, 2]
        init_z = z_mu[:, sample, :].detach().cpu().numpy() if z_mu is not None else None
        
        plt.figure(figsize=figsize)
        if log_scale:
            plt.scatter(X[:, 0], X[:, 1], c=np.log(Y), s=5)
        else:
            plt.scatter(X[:, 0], X[:, 1], c=Y, s=5)
            
        if z_mu is not None:
            plt.scatter(init_z[:, 0], init_z[:, 1], c='red', s=5, alpha=0.5)
            
        plt.colorbar()
        plt.show()

    def make_samples(
        self,
        num_samples: int = 10,
        resolution: int = 300,
        log_scale: bool = True
    ) -> None:
        """Generate and visualize samples with latent space plots.
        
        Args:
            num_samples: Number of samples to generate
            resolution: Resolution of the latent space grid
            log_scale: Whether to use log scale for MSE values
        """
        # Generate data
        xs, ys, amp, phase = self.generate_data(num_samples)
        io_pairs = torch.cat([xs, ys], dim=-1).to(self.device)
        
        # Process data
        leave_one_out, left_one_out, left_one_out_y_true, z_mu, z_traj = self.process_data(io_pairs)
        
        # Create grid and compute MSE
        grid2d = self.create_latent_grid(resolution)

        # match grid size for cat
        shape_expand = [max(x,y) for x,y in zip(grid2d.size()[:-1], left_one_out.size()[:-1])]
        grid2d = grid2d.expand(shape_expand + [grid2d.size()[-1]])
        left_one_out = left_one_out.expand(shape_expand + [left_one_out.size()[-1]])
        z_xs = torch.cat([grid2d, left_one_out], dim=-1)

        left_one_out_y_pred = self.lpn.decoder(z_xs)
        
        mse_grid = (left_one_out_y_pred - left_one_out_y_true)**2
        
        # Prepare data for plotting
        X_cat = torch.cat([grid2d, mse_grid], dim=-1)
        X_cat = X_cat.detach().cpu().numpy()
        
        # Plot each sample
        self.data = X_cat, z_mu, amp, phase, num_samples, log_scale

    def plot_sample(self, sample) -> None:
        """Plot the generated samples in latent space."""
        X_cat, z_mu, amp, phase, num_samples, log_scale = self.data
        assert 0 <= sample < num_samples, "Sample index out of range"
        self.plot_latent_space(X_cat, z_mu, sample, log_scale)