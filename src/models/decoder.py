from utils import ReluNet


class Decoder(ReluNet):
    # TODO add a batchnorm

    def __init__(self, d_input=1, d_latent=3, ds_hidden=[32, 32], d_output=1):
        sizes = [d_latent + d_input] + ds_hidden + [d_output]
        super().__init__(*sizes)
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_output = d_output

