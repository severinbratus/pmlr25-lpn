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
        device: str = None,
        resolution: int = 300,
    ):
        """Initialize the LPN visualizer with model loading and setup.

        Args:
            d_latent: Dimension of latent space
            numstep: Number of steps for the model
            model_path: Path to the saved model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.d_latent = d_latent
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.resolution = resolution

        # Initialize model components
        self.encoder = IOPairSetEncoder(
            phi=ReluNet(2, 16, 32),
            rho_0=ReluNet(32, 16, 16),
            rho_1=ReluNet(16, 8, d_latent),
        )

        self.decoder = Decoder(
            d_input=1, d_latent=d_latent, ds_hidden=[16, 16, 16], d_output=1
        )

        self.lpn = LPN(
            d_input=1,
            d_output=1,
            d_latent=d_latent,
            encoder=self.encoder,
            decoder=self.decoder,
        ).to(self.device)

        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.lpn.load_state_dict(state_dict, strict=False)

        self.lpn.eval()

    def set_alpha(self, alpha: float) -> None:
        self.lpn.alpha = alpha

    def set_K(self, K: int) -> None:
        self.K = K

    def generate_data(
        self, num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def process_data(
        self, io_pairs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process input-output pairs for visualization.

        Args:
            io_pairs: Input-output pairs tensor

        Returns:
            Tuple of (leave_one_out, left_one_out, left_one_out_y_true, z_mu)
        """
        xy_support = make_leave_one_out(io_pairs, axis=1)
        x_test = io_pairs[:, :, 0].unsqueeze(-1)
        y_test = io_pairs[:, :, 1].unsqueeze(-1)
        z_mu, _ = self.lpn.encoder(xy_support)

        self.lpn.gradient_ascent(z_mu, io_pairs, K=self.K, debug=True)
        z_traj = self.lpn.z_traj

        return xy_support, x_test, y_test, z_mu, z_traj

    def create_latent_grid(self, resolution=None) -> torch.Tensor:
        """Create a 2D grid in latent space for visualization.

        Args:
            resolution: Resolution of the grid

        Returns:
            Grid tensor
        """
        resolution = resolution or self.resolution
        return (
            torch.tensor(
                np.mgrid[-5 : 5 : resolution * 1j, -5 : 5 : resolution * 1j]
                .reshape(2, 1, -1)
                .T
            )
            .float()
            .to(self.device)
        )

    def plot_latent_space(
        self,
        grid2d: torch.Tensor,
        mse_grid: torch.Tensor,
        z_mu: Optional[torch.Tensor] = None,
        z_traj: Optional[List[torch.Tensor]] = None,
        sample: int = 0,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (8, 6),
        title: str = None,
    ) -> None:
        """Plot the latent space visualization.

        Args:
            X_cat: Concatenated data for plotting
            z_mu: Mean latent vectors
            sample: Sample index to plot
            log_scale: Whether to use log scale for MSE values
            figsize: Figure size
        """
        # X_cat = X_cat.detach().cpu().numpy()
        X = grid2d[:, sample, :].detach().cpu().numpy()
        Y = mse_grid[:, sample, :].detach().cpu().numpy()
        # X = X_cat[:, sample, :2]
        # Y = X_cat[:, sample, 2]
        init_z = z_mu[:, sample, :].detach().cpu().numpy() if z_mu is not None else None

        plt.figure(figsize=figsize)
        # Reshape the data into a grid for contour plot
        resolution = int(np.sqrt(X.shape[0]))
        X0 = X[:, 0].reshape(resolution, resolution)
        X1 = X[:, 1].reshape(resolution, resolution)
        Y_grid = Y.reshape(resolution, resolution)

        if log_scale:
            plt.pcolormesh(X0, X1, np.log(Y_grid), shading="auto")
        else:
            plt.pcolormesh(X0, X1, Y_grid, shading="auto")

        if z_traj is not None:
            num_steps = len(z_traj)
            for idx, z in enumerate(z_traj):
                # Compute a color that transitions from blue (0, 0, 1) to red (1, 0, 0)
                ratio = idx / (num_steps - 1) if num_steps > 1 else 0
                color = (ratio, 0, 1 - ratio)
                plt.scatter(
                    z[:, sample, 0].detach().cpu().numpy(),
                    z[:, sample, 1].detach().cpu().numpy(),
                    c=[color],
                    s=5,
                    alpha=0.5,
                    label="Trajectory" if idx == 0 else "",
                )

        if z_mu is not None:
            plt.scatter(
                init_z[:, 0], init_z[:, 1], c="red", s=5, alpha=0.5, label="Initial"
            )

        # add whether log scale or not
        if log_scale:
            plt.colorbar(label="log(MSE)")
        else:
            plt.colorbar(label="MSE")

        plt.title(title or "Latent Space")
        plt.show()

    def make_samples(
        self,
        num_samples: int = 10,
        use_true_mse: bool = False,
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
        xy_support, x_test, y_test, z_mu, z_traj = self.process_data(io_pairs)

        # Create grid and compute MSE
        grid2d = self.create_latent_grid()  # (resolution**2, 1, 2)

        # match grid size for cat
        # grid2d has (resolution**2, 1, 2)
        # x_test has (1, num_samples, 1)
        shape_expand = [
            max(x, y) for x, y in zip(grid2d.size()[:-1], x_test.size()[:-1])
        ]
        grid2d = grid2d.expand(shape_expand + [grid2d.size()[-1]])
        x_test = x_test.expand(shape_expand + [x_test.size()[-1]])
        z_xs = torch.cat([grid2d, x_test], dim=-1)  # (resolution**2, num_samples, 3)

        # Use true MSE
        y_pred = self.lpn.decoder(z_xs)  # (resolution**2, num_samples, 1)
        mse_test = (y_pred - y_test) ** 2

        # Use predicted MSE
        mse_support = self.lpn.nll_fn(grid2d, io_pairs.expand_as(grid2d))
        z_traj_cat = torch.cat(z_traj, dim=0)  # (K, num_samples, d_latent)
        mse_traj = self.lpn.nll_fn(z_traj_cat, io_pairs.expand_as(z_traj_cat))

        # Plot each sample
        self.data = (
            grid2d,
            mse_test,
            mse_support,
            z_mu,
            amp,
            phase,
            num_samples,
            z_traj,
            mse_traj,
        )

    def plot_sample(self, sample=0, log_scale=False) -> None:
        """Plot the generated samples in latent space."""
        (
            grid2d,
            mse_test,
            mse_support,
            z_mu,
            amp,
            phase,
            num_samples,
            z_traj,
            mse_traj,
        ) = self.data
        assert 0 <= sample < num_samples, "Sample index out of range"

        mse_traj_s = mse_traj[:, sample, :].detach().cpu().numpy()
        # print(f"z_traj {z_traj}:")
        print("mse_traj: " + " ".join(map(str, mse_traj_s.flatten())))
        self.plot_latent_space(
            grid2d=grid2d,
            mse_grid=mse_test,
            z_mu=z_mu,
            z_traj=z_traj,
            sample=sample,
            log_scale=log_scale,
            title="MSE Test",
        )

        self.plot_latent_space(
            grid2d=grid2d,
            mse_grid=mse_support,
            z_mu=z_mu,
            z_traj=z_traj,
            sample=sample,
            log_scale=log_scale,
            title="MSE Support",
        )
