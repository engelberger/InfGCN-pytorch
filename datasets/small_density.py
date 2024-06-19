import os
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from jaxtyping import Array, Float, Long

from ._base import register_dataset

# Dictionary mapping molecule names to their atom types
ATOM_TYPES: Dict[str, Array[Long]] = {
    'benzene': torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
    'ethanol': torch.LongTensor([0, 0, 2, 1, 1, 1, 1, 1, 1]),
    'phenol': torch.LongTensor([0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1]),
    'resorcinol': torch.LongTensor([0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1]),
    'ethane': torch.LongTensor([0, 0, 1, 1, 1, 1, 1, 1]),
    'malonaldehyde': torch.LongTensor([2, 0, 0, 0, 2, 1, 1, 1, 1]),
}


@register_dataset('small_density')
class SmallDensityDataset(Dataset):
    def __init__(self, root: str, mol_name: str, split: str):
        """
        Density dataset for small molecules in the MD datasets.
        Note that the validation and test splits are the same.

        Args:
            root (str): Data root directory.
            mol_name (str): Name of the molecule.
            split (str): Data split, can be 'train', 'validation', 'test'.
        """
        super(SmallDensityDataset, self).__init__()
        assert mol_name in ('benzene', 'ethanol', 'phenol', 'resorcinol', 'ethane', 'malonaldehyde')
        self.root = root
        self.mol_name = mol_name
        self.split = split
        if split == 'validation':
            split = 'test'

        self.n_grid = 50  # Number of grid points along each dimension
        self.grid_size = 20.  # Box size in Bohr
        self.data_path = os.path.join(root, mol_name, f'{mol_name}_{split}')

        self.atom_type = ATOM_TYPES[mol_name]  # Get the atom types for the molecule
        self.atom_coords = torch.FloatTensor(np.load(os.path.join(self.data_path, 'structures.npy')))  # Load atom coordinates from file and convert to PyTorch tensor
        self.densities = self._convert_fft(np.load(os.path.join(self.data_path, 'dft_densities.npy')))  # Load density data from file and convert from FFT coefficients
        self.grid_coord = self._generate_grid()  # Generate grid coordinates

    def _convert_fft(self, fft_coeff: Array[Float]) -> Array[Float]:
        """
        Convert FFT coefficients back to real space densities.

        Args:
            fft_coeff (Array[Float]): FFT coefficients.

        Returns:
            Array[Float]: Real space densities.
        """
        print(f'Precomputing {self.split} density from FFT coefficients ...')
        fft_coeff = torch.FloatTensor(fft_coeff).to(torch.complex64)  # Convert FFT coefficients to PyTorch tensor with complex dtype
        density = fft_coeff.view(-1, self.n_grid, self.n_grid, self.n_grid)  # Reshape FFT coefficients to 4D tensor
        half_grid = self.n_grid // 2  # Calculate half of the grid size
        
        # First dimension
        density[:, :half_grid] = (density[:, :half_grid] - density[:, half_grid:] * 1j) / 2  # Perform inverse FFT along first dimension
        density[:, half_grid:] = torch.flip(density[:, 1:half_grid + 1], [1]).conj()  # Flip and conjugate the second half of the first dimension
        density = torch.fft.ifft(density, dim=1)  # Apply inverse FFT along first dimension
        
        # Second dimension
        density[:, :, :half_grid] = (density[:, :, :half_grid] - density[:, :, half_grid:] * 1j) / 2  # Perform inverse FFT along second dimension
        density[:, :, half_grid:] = torch.flip(density[:, :, 1:half_grid + 1], [2]).conj()  # Flip and conjugate the second half of the second dimension
        density = torch.fft.ifft(density, dim=2)  # Apply inverse FFT along second dimension
        
        # Third dimension
        density[..., :half_grid] = (density[..., :half_grid] - density[..., half_grid:] * 1j) / 2  # Perform inverse FFT along third dimension
        density[..., half_grid:] = torch.flip(density[..., 1:half_grid + 1], [3]).conj()  # Flip and conjugate the second half of the third dimension
        density = torch.fft.ifft(density, dim=3)  # Apply inverse FFT along third dimension
        
        return torch.flip(density.real.view(-1, self.n_grid ** 3), [-1]).detach()  # Flip the density tensor, reshape it, and detach from the computation graph

    def _generate_grid(self) -> Array[Float]:
        """
        Generate grid coordinates.

        Returns:
            Array[Float]: Grid coordinates.
        """
        x = torch.linspace(self.grid_size / self.n_grid, self.grid_size, self.n_grid)  # Generate evenly spaced grid points along each dimension
        return torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1).view(-1, 3).detach()  # Create a meshgrid of coordinates and reshape to (n_points, 3)

    def __getitem__(self, item: int) -> Tuple[Data, Array[Float], Array[Float], Dict[str, Array]]:
        """
        Get an item from the dataset.

        Args:
            item (int): Index of the item.

        Returns:
            Tuple[Data, Array[Float], Array[Float], Dict[str, Array]]: Tuple containing atom data, densities, grid coordinates, and additional info.
        """
        info = {
            'cell': torch.eye(3) * self.grid_size,  # Create a 3x3 identity matrix and multiply by the grid size
            'shape': [self.n_grid, self.n_grid, self.n_grid]  # Store the shape of the grid
        }
        return (
            Data(x=self.atom_type, pos=self.atom_coords[item]),  # Create a Data object with atom types and coordinates for the given item
            self.densities[item],  # Get the density data for the given item
            self.grid_coord,  # Get the grid coordinates
            info  # Additional information dictionary
        )

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.atom_coords.shape[0]  # Return the number of items in the dataset based on the first dimension of atom_coords