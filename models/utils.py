import torch
import torch.nn as nn
from jaxtyping import Float


class BroadcastGTOTensor(nn.Module):
    r"""
    Broadcast between spherical tensors of the Gaussian Type Orbitals (GTOs):

    .. math::
        \{a_{clm}, 1\le c\le c_{max}, 0\le\ell\le\ell_{max}, -\ell\le m\le\ell\}

    For efficiency reason, the feature tensor is indexed by l, c, m.
    For example, for lmax = 3, cmax = 2, we have a tensor of 1s2s 1p2p 1d2d 1f2f.
    Currently, we support the following broadcasting:
    lc -> lcm;
    m -> lcm.
    """

    def __init__(self, lmax: int, cmax: int, src: str = 'lc', dst: str = 'lcm'):
        """
        Initialize the BroadcastGTOTensor module.

        Args:
            lmax (int): Maximum angular momentum quantum number.
            cmax (int): Maximum number of Gaussian functions.
            src (str): Source tensor format. Can be 'lc' or 'm'. Default is 'lc'.
            dst (str): Destination tensor format. Currently only supports 'lcm'. Default is 'lcm'.
        """
        super(BroadcastGTOTensor, self).__init__()
        assert src in ['lc', 'm'], "Invalid source tensor format. Must be 'lc' or 'm'."
        assert dst in ['lcm'], "Invalid destination tensor format. Currently only supports 'lcm'."
        self.src = src
        self.dst = dst
        self.lmax = lmax
        self.cmax = cmax

        if src == 'lc':
            self.src_dim = (lmax + 1) * cmax  # Source tensor dimension for 'lc' format
        else:
            self.src_dim = (lmax + 1) ** 2  # Source tensor dimension for 'm' format
        self.dst_dim = (lmax + 1) ** 2 * cmax  # Destination tensor dimension

        if src == 'lc':
            indices = self._generate_lc2lcm_indices()  # Generate indices for 'lc' to 'lcm' broadcasting
        else:
            indices = self._generate_m2lcm_indices()  # Generate indices for 'm' to 'lcm' broadcasting
        self.register_buffer('indices', indices)  # Register the indices as a buffer

    def _generate_lc2lcm_indices(self) -> torch.Tensor:
        r"""
        Generate indices for broadcasting from 'lc' to 'lcm' format.

        lc -> lcm
        .. math::
            1s2s 1p2p → 1s2s 1p_x1p_y1p_z2p_x2p_y2p_z
        [0, 1, 2, 2, 2, 3, 3, 3]

        Returns:
            torch.Tensor: Indices tensor of shape ((lmax+1)^2 * cmax,).
        """
        indices = [
            l * self.cmax + c for l in range(self.lmax + 1)
            for c in range(self.cmax)
            for _ in range(2 * l + 1)
        ]
        return torch.LongTensor(indices)

    def _generate_m2lcm_indices(self) -> torch.Tensor:
        r"""
        Generate indices for broadcasting from 'm' to 'lcm' format.

        m -> lcm
        .. math::
            s p_x p_y p_z → 1s2s 1p_x1p_y1p_z2p_x2p_y2p_z
        [0, 0, 1, 2, 3, 1, 2, 3]

        Returns:
            torch.Tensor: Indices tensor of shape ((lmax+1)^2 * cmax,).
        """
        indices = [
            l * l + m for l in range(self.lmax + 1)
            for _ in range(self.cmax)
            for m in range(2 * l + 1)
        ]
        return torch.LongTensor(indices)

    def forward(self, x: Float[torch.Tensor, '..., src_dim']) -> Float[torch.Tensor, '..., dst_dim']:
        """
        Apply broadcasting to the input tensor.

        Args:
            x (Float[torch.Tensor, '..., src_dim']): Input tensor of shape (..., src_dim).

        Returns:
            Float[torch.Tensor, '..., dst_dim']: Broadcasted tensor of shape (..., dst_dim).
        """
        assert x.size(-1) == self.src_dim, f'Input dimension mismatch! ' \
                                           f'Should be {self.src_dim}, but got {x.size(-1)} instead!'
        if self.src == self.dst:
            return x  # No broadcasting needed if source and destination formats are the same
        return x[..., self.indices]  # Apply broadcasting using the generated indices