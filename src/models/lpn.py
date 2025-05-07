import torch
import torch.nn as nn

from models.utils import make_leave_one_out


class LPN(nn.Module):
    def __init__(self,
                 d_input,
                 d_output,
                 d_latent,
                 encoder,
                 decoder,
                 alpha=4e-4,
                 beta=1e-3):
        """
        Args:
            d_input: Input dimension.
            d_output: Output dimension.
            d_latent: Latent dimension.
            encoder: Encoder module (e.g., IOPairSetEncoder).
            decoder: Decoder module (e.g., Decoder).
            alpha: Step size or learning rate in inner latent optimization.
            beta: "Prior KL Coeff". Weight for the KL divergence term.
        """
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        assert self.d_input == self.d_output == 1, "Unsupported d_input or d_output"
        self.d_latent = d_latent
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()


    def forward(self, pairs, debug=False, K=0, deterministic=False):
        B, N, H = pairs.size(0), pairs.size(1), self.d_latent

        # pairs: (B, N, 2)
        # NOTE: assuming d_input = d_output = 1
        # pairs_one_left_out: (B, N, N-1, 2)
        pairs_one_left_out = make_leave_one_out(pairs, axis=1)
        if debug:
            assert pairs_one_left_out.shape == (B, N, N-1, 2), f"{pairs_one_left_out.shape} != {(B, N, N-1, 2)}"

        # Encode the context IO pairs (one-left-out) into a latent
        # z_mu, z_logvar: (B, N, H)
        z_mu, z_logvar = self.encoder(pairs_one_left_out)
        if debug:
            assert z_mu.shape == z_logvar.shape
            assert z_mu.shape == (B, N, H), f"{z_mu.shape} != {(B, N, H)}"

        # During training, sample from distribution
        # During testing, just use the mean
        if deterministic:
            z_sample = z_mu
        else:
            z_sample = self.sample_latents(z_mu, z_logvar)
        
        kl_loss = self.kl_divergence(z_mu, z_logvar)

        if K:
            z_prime = self.gradient_ascent(z_sample, pairs, K, debug=debug)
        else:
            z_prime = z_sample

        # Decode the (target) inputs with the one-left-out latents into outputs
        xs = pairs[:, :, 0].unsqueeze(-1) # (B, N, 1)
        z_xs = torch.cat([z_prime, xs], dim=-1)

        # outputs_pred: (B, N, 1)
        ys_pred = self.decoder(z_xs)

        ys_true = pairs[:, :, 1].unsqueeze(-1)

        # NOTE: assuming a variance of 1. for intuition see https://chatgpt.com/share/6809ffa6-7710-8000-bef7-b73d0116c0e2
        recon_loss = .5 * self.mse(ys_pred, ys_true)

        # recon_loss is wrt the decoder (p_theta)
        # kl_loss is wrt the encoder (q_phi)
        loss = recon_loss + self.beta * kl_loss

        aux = {
            'z_mu': z_mu.detach().clone(),
            'z_logvar': z_logvar.detach().clone(),
            'z_sample': z_sample.detach().clone(),
            'z_prime': z_prime.detach().clone(),
            'ys_pred': ys_pred.detach().clone(),
            'z_traj': self.z_traj if hasattr(self, 'z_traj') else None,
        }

        return aux, loss


    def decode(self, z, xs):
        """Decode a batch of inputs with a single latent z"""
        # z: (H,)
        # inputs: (B, 1)
        B = xs.shape[0]
        assert xs.shape == (B, 1), f"xs {xs.shape} != {(B, 1)}"
        assert z.shape == (self.d_latent,), f"z {z.shape} != {(self.d_latent,)}"
        z_wide = z.unsqueeze(0).expand(B, -1)  # shape: (B, H)
        z_xs = torch.cat([z_wide, xs], dim=-1)  # shape: (B, H + 1)
        ys_pred = self.decoder(z_xs)
        return ys_pred


    def gradient_ascent(self, z_init, pairs, K, debug=False):
        # z: (B, N, H)
        # pairs: (B, N, 2)

        z_prime = z_init
        if debug:
            self.z_traj = [z_init.detach().clone()]

        for k in range(K):
            # Re-create z as a tensor parameter requiring gradients
            z = z_init.detach().clone().requires_grad_(True)

            # Compute 
            mse = self.nll_fn(z, pairs, debug=debug)
            z_grads = torch.autograd.grad(torch.sum(mse), z)[0]

            z_prime -= self.alpha * z_grads.detach()
        
            if debug:
                assert z_grads.shape == z.shape, f"{z_grads.shape} != {z.shape}"
                self.z_traj.append(z_prime.detach().clone())
        
        return z_prime
    

    def nll_fn(self, z, pairs, debug=False):
        B = pairs.size(0)
        N = pairs.size(1)
        H = z.size(2)

        # NOTE: olo = one left out
        xs = pairs[:, :, 0].unsqueeze(-1) # (B, N, 1)
        xs_olo = make_leave_one_out(xs, axis=1) # (B, N, N-1, 1)
        ys = pairs[:, :, 1].unsqueeze(-1) # (B, N, 1)
        ys_olo = make_leave_one_out(ys, axis=1) # (B, N, N-1, 1)

        z_wide = z.unsqueeze(2).expand(-1, -1, N-1, -1) # (B, N, N-1, H)
        z_xs_olo = torch.cat([z_wide, xs_olo], dim=-1) # (B, N, N-1, H+1)
        ys_hat_olo = self.decoder(z_xs_olo) # (B, N, N-1, 1)
        if debug:
            assert z_wide.shape == (B, N, N-1, H), f"{z_wide.shape} != {(B, N, N-1, H)}"
            assert z_xs_olo.shape == (B, N, N-1, H+1), f"{z_xs_olo.shape} != {(B, N, N-1, H+1)}"
            assert ys_hat_olo.shape == ys_olo.shape
            assert ys_hat_olo.shape == (B, N, N-1, 1), f"{ys_hat_olo.shape} != {(B, N, N-1, 1)}"

        mse = nn.functional.mse_loss(ys_hat_olo, ys_olo, reduction='none').sum(dim=-2) # (B, N, 1)
        if debug:
            print(f"{torch.sum(mse).item()=}")
        return mse


    def sample_latents(self, z_mu, z_logvar):
        std = torch.exp(0.5 * z_logvar)  # (B*, H)
        eps = torch.randn_like(std)      # (B*, H), same shape as std
        z = z_mu + eps * std             # (B*, H)
        return z


    def kl_divergence(self, z_mu, z_logvar):
        # KL divergence between N(z_mu, exp(z_logvar)) and N(0, I)
        kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=-1)  # shape: (B*)
        return kl.mean()


class DeterministicLPN(nn.Module):
    def __init__(self,
                 d_input,
                 d_output,
                 d_latent,
                 encoder,
                 decoder,
                 alpha=4e-4):
        """
        Args:
            d_input: Input dimension.
            d_output: Output dimension.
            d_latent: Latent dimension.
            encoder: Encoder module (e.g., IOPairSetEncoder).
            decoder: Decoder module (e.g., Decoder).
            alpha: Step size or learning rate in inner latent optimization.
        """
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        assert self.d_input == self.d_output == 1, "Unsupported d_input or d_output"
        self.d_latent = d_latent
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.mse = nn.MSELoss()


    def forward(self, pairs, debug=False, K=0):
        # K flag for GA optimization
        B, N, H = pairs.size(0), pairs.size(1), self.d_latent

        # pairs: (B, N, 2)
        # NOTE: assuming d_input = d_output = 1
        # pairs_one_left_out: (B, N, N-1, 2)
        pairs_one_left_out = make_leave_one_out(pairs, axis=1)
        if debug:
            assert pairs_one_left_out.shape == (B, N, N-1, 2), f"{pairs_one_left_out.shape} != {(B, N, N-1, 2)}"

        # Encode the context IO pairs (one-left-out) into a latent
        # z: (B, N, H)
        z = self.encoder(pairs_one_left_out)
        if debug:
            assert z.shape == (B, N, H), f"{z.shape} != {(B, N, H)}"

        if K:
            z_prime = self.gradient_ascent(z, pairs, K, debug=debug)
        else:
            z_prime = z

        # Decode the (target) inputs with the one-left-out latents into outputs
        xs = pairs[:, :, 0].unsqueeze(-1) # (B, N, 1)
        z_xs = torch.cat([z_prime, xs], dim=-1)

        # outputs_pred: (B, N, 1)
        ys_pred = self.decoder(z_xs)

        ys_true = pairs[:, :, 1].unsqueeze(-1)

        # Loss is just MSE now
        loss = self.mse(ys_pred, ys_true)

        aux = {
            'z': z.detach().clone(),
            'z_prime': z_prime.detach().clone(),
            'ys_pred': ys_pred.detach().clone(),
            'z_traj': self.z_traj if hasattr(self, 'z_traj') else None,
        }

        return aux, loss


    def decode(self, z, xs):
        """Decode a batch of inputs with a single latent z"""
        # z: (H,)
        # inputs: (B, 1)
        B = xs.shape[0]
        assert xs.shape == (B, 1), f"xs {xs.shape} != {(B, 1)}"
        assert z.shape == (self.d_latent,), f"z {z.shape} != {(self.d_latent,)}"
        z_wide = z.unsqueeze(0).expand(B, -1)  # shape: (B, H)
        z_xs = torch.cat([z_wide, xs], dim=-1)  # shape: (B, H + 1)
        ys_pred = self.decoder(z_xs)
        return ys_pred


    def gradient_ascent(self, z_init, pairs, K, debug=False):
        # z: (B, N, H)
        # pairs: (B, N, 2)

        z_prime = z_init
        if debug:
            self.z_traj = [z_init.detach().clone()]

        for k in range(K):
            # Re-create z as a tensor parameter requiring gradients
            z = z_init.detach().clone().requires_grad_(True)

            # Compute 
            mse = self.nll_fn(z, pairs, debug=debug)
            z_grads = torch.autograd.grad(torch.sum(mse), z)[0]

            z_prime -= self.alpha * z_grads.detach()
        
            if debug:
                assert z_grads.shape == z.shape, f"{z_grads.shape} != {z.shape}"
                self.z_traj.append(z_prime.detach().clone())
        
        return z_prime
    

    def nll_fn(self, z, pairs, debug=False):
        B = pairs.size(0)
        N = pairs.size(1)
        H = z.size(2)

        # NOTE: olo = one left out
        xs = pairs[:, :, 0].unsqueeze(-1) # (B, N, 1)
        xs_olo = make_leave_one_out(xs, axis=1) # (B, N, N-1, 1)
        ys = pairs[:, :, 1].unsqueeze(-1) # (B, N, 1)
        ys_olo = make_leave_one_out(ys, axis=1) # (B, N, N-1, 1)

        z_wide = z.unsqueeze(2).expand(-1, -1, N-1, -1) # (B, N, N-1, H)
        z_xs_olo = torch.cat([z_wide, xs_olo], dim=-1) # (B, N, N-1, H+1)
        ys_hat_olo = self.decoder(z_xs_olo) # (B, N, N-1, 1)
        if debug:
            assert z_wide.shape == (B, N, N-1, H), f"{z_wide.shape} != {(B, N, N-1, H)}"
            assert z_xs_olo.shape == (B, N, N-1, H+1), f"{z_xs_olo.shape} != {(B, N, N-1, H+1)}"
            assert ys_hat_olo.shape == ys_olo.shape
            assert ys_hat_olo.shape == (B, N, N-1, 1), f"{ys_hat_olo.shape} != {(B, N, N-1, 1)}"

        mse = nn.functional.mse_loss(ys_hat_olo, ys_olo, reduction='none').sum(dim=-2) # (B, N, 1)
        if debug:
            print(f"{torch.sum(mse).item()=}")
        return mse