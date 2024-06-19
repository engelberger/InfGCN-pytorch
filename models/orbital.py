import math
from typing import List

import torch
import torch.nn as nn
from e3nn import o3
from jaxtyping import Float

from .utils import BroadcastGTOTensor


class GaussianOrbital(nn.Module):
    r"""
    Gaussian-type orbital

    .. math::
        \psi_{n\ell m}(\mathbf{r})=\sqrt{\frac{2(2a_n)^{\ell+3/2}}{\Gamma(\ell+3/2)}}
        \exp(-a_n r^2) r^\ell Y_{\ell}^m(\hat{\mathbf{r}})

    """

    def __init__(self, gauss_start: float, gauss_end: float, num_gauss: int, lmax: int = 7):
        """
        Initialize the GaussianOrbital module.

        Args:
            gauss_start (float): Starting value of the Gaussian exponent.
            gauss_end (float): Ending value of the Gaussian exponent.
            num_gauss (int): Number of Gaussian functions.
            lmax (int): Maximum angular momentum quantum number. Default is 7.
        """
        super(GaussianOrbital, self).__init__()
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.num_gauss = num_gauss
        self.lmax = lmax

        self.lc2lcm = BroadcastGTOTensor(lmax, num_gauss, src='lc', dst='lcm')  # Broadcast tensor from 'lc' to 'lcm'
        self.m2lcm = BroadcastGTOTensor(lmax, num_gauss, src='m', dst='lcm')  # Broadcast tensor from 'm' to 'lcm'
        self.gauss: torch.Tensor  # Gaussian exponents
        self.lognorm: torch.Tensor  # Logarithm of normalization constants

        self.register_buffer('gauss', torch.linspace(gauss_start, gauss_end, num_gauss))  # Register Gaussian exponents as a buffer
        self.register_buffer('lognorm', self._generate_lognorm())  # Register logarithm of normalization constants as a buffer

    def _generate_lognorm(self) -> torch.Tensor:
        """
        Generate the logarithm of normalization constants.

        Returns:
            torch.Tensor: Logarithm of normalization constants of shape (l * c,).
        """
        power = (torch.arange(self.lmax + 1) + 1.5).unsqueeze(-1)  # Shape: (l, 1)
        numerator = power * torch.log(2 * self.gauss).unsqueeze(0) + math.log(2)  # Shape: (l, c)
        denominator = torch.special.gammaln(power)  # Shape: (l, 1)
        lognorm = (numerator - denominator) / 2  # Shape: (l, c)

        return lognorm.view(-1)  # Shape: (l * c,)

    def forward(self, vec: Float[torch.Tensor, '..., 3']) -> Float[torch.Tensor, '..., (l+1)^2 * c']:
        """
        Evaluate the basis functions.

        Args:
            vec (Float[torch.Tensor, '..., 3']): Un-normalized vectors of shape (..., 3).

        Returns:
            Float[torch.Tensor, '..., (l+1)^2 * c']: Basis values of shape (..., (l+1)^2 * c).
        """
        # Spherical part
        device = vec.device  # Get the device of the input tensor
        r = vec.norm(dim=-1) + 1e-8  # Compute the radial distance (add a small value to avoid division by zero)
        spherical = o3.spherical_harmonics(
            list(range(self.lmax + 1)), vec / r[..., None],
            normalize=False, normalization='integral'
        )  # Compute spherical harmonics

        # Radial part
        r = r.unsqueeze(-1)  # Add a new dimension for broadcasting
        lognorm = self.lognorm * torch.ones_like(r)  # Broadcast logarithm of normalization constants to match the shape of r
        exponent = -self.gauss * (r * r)  # Compute the exponent of the Gaussian function
        poly = torch.arange(self.lmax + 1, dtype=torch.float, device=device) * torch.log(r)  # Compute the polynomial part
        log = exponent.unsqueeze(-2) + poly.unsqueeze(-1)  # Combine the exponent and polynomial parts
        radial = torch.exp(log.view(*log.size()[:-2], -1) + lognorm)  # Compute the radial part

        return self.lc2lcm(radial) * self.m2lcm(spherical)  # Combine the radial and spherical parts using broadcast tensors