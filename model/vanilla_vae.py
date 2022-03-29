import torch
from torch import nn
from torch.nn import functional as F
#from .types_ import *
from util.params import config


class VanillaVAE(nn.Module):
    def __init__(self, args):
        super(VanillaVAE, self).__init__()
        self.args = args
        self.config = config[args.data]
        self.in_channels = self.config['in_channels']
        self.size = self.config['size']
        self.hidden_dims = self.config['generator']['hidden_dims']
        self.kernel_size = self.config['generator']['kernel_size']
        self.kl_weight = self.config['generator']['kl_weight']
        self.z_wh = int(self.size / 2**len(self.hidden_dims))

        if self.config['generator']['final_activation']=='tanh':
            self.final_activaton = nn.Tanh()
        elif self.config['generator']['final_activation']=='sigmoid':
            self.final_activaton = nn.Sigmoid()
        elif self.config['generator']['final_activation']=='identity':
            pass

        modules = []
        in_channels = self.in_channels

        # Build Encoder
        for i, h_dim in enumerate(self.hidden_dims[:-1]):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=self.kernel_size, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Conv2d(self.hidden_dims[-2], out_channels=self.hidden_dims[-1],
                               kernel_size=self.kernel_size, stride=2, padding=1)
        self.fc_var = nn.Conv2d(self.hidden_dims[-2], out_channels=self.hidden_dims[-1],
                               kernel_size=self.kernel_size, stride=2, padding=1)

        # Build Decoder
        modules = []

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims[:-1])):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=self.kernel_size,
                                       stride=2,
                                       padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1], out_channels=self.in_channels,
                                               kernel_size=self.kernel_size,
                                               stride=2, padding=1),
                            self.final_activaton)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result).view(self.args.batch_size, -1)
        log_var = self.fc_var(result).view(self.args.batch_size, -1)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = z.view(-1, self.hidden_dims[0], self.z_wh, self.z_wh)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + self.kl_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self, mu):
        """
        Samples from the mu
        """
        samples = self.decode(mu)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
