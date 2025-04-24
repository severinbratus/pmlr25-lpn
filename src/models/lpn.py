import torch
import torch.nn as nn

from src.models.utils import make_leave_one_out


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


    def forward(self, pairs):

        # pairs: (B, N, 2)
        # Note: assuming d_input = d_output = 1
        # pairs_one_left_out: (B, N, N-1, 2)
        ## print(f"{pairs.shape=}")
        pairs_one_left_out = make_leave_one_out(pairs, axis=1)
        ## print(f"{pairs_one_left_out.shape=}")

        # Encode the context IO pairs (one-left-out) into a latent
        # z_mu, z_logvar: (B, N, H)
        z_mu, z_logvar = self.encoder(pairs_one_left_out)
        ## print(f"{z_mu.shape=}")

        assert z_mu.shape == z_logvar.shape
        b, n = pairs.size(0), pairs.size(1)
        assert z_mu.shape == (b, n, self.d_latent), f"{z_mu.shape} != {(b, n, self.d_latent)}"

        z_sample = self.sample_latents(z_mu, z_logvar)
        kl_loss = self.kl_divergence(z_mu, z_logvar)
        
        # TODO latent optimization

        # Decode the (target) inputs with the one-left-out latents into outputs
        inputs = pairs[:, :, 0].unsqueeze(-1) # (B, N, 1)
        z_inputs = torch.cat([z_sample, inputs], dim=-1)

        # outputs_pred: (B, N, 1)
        outputs_pred = self.decoder(z_inputs)

        outputs_true = pairs[:, :, 1].unsqueeze(-1)

        # NOTE: assuming a variance of 1. for intuition see https://chatgpt.com/share/6809ffa6-7710-8000-bef7-b73d0116c0e2
        recon_loss = .5 * self.mse(outputs_pred, outputs_true)

        # recon_loss is wrt the decoder (p_theta)
        # kl_loss is wrt the encoder (q_phi)
        loss = recon_loss + self.beta * kl_loss

        aux = {
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'z_sample': z_sample,
            'outputs_pred': outputs_pred,
        }

        return aux, loss


    def sample_latents(self, z_mu, z_logvar):
        std = torch.exp(0.5 * z_logvar)  # (B*, H)
        eps = torch.randn_like(std)      # (B*, H), same shape as std
        z = z_mu + eps * std             # (B*, H)
        return z


    def kl_divergence(self, z_mu, z_logvar):
        # KL divergence between N(z_mu, exp(z_logvar)) and N(0, I)
        kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=-1)  # shape: (B*)
        return kl.mean()


