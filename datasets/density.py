import json
import os
import time

import numpy as np
import torch
from e3nn import o3
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ._base import register_dataset

Bohr = 0.529177  # Bohr radius in angstrom


def pbc_expand(atom_type, atom_coord):
    """
    Expand the atoms by periodic boundary condition to eight directions in the neighboring cells.
    :param atom_type: atom types, tensor of shape (n_atom,)
    :param atom_coord: atom coordinates, tensor of shape (n_atom, 3)
    :return: expanded atom types and coordinates
    """
    exp_type, exp_coord = [], []
    exp_direction = torch.FloatTensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]
    ])  # Define the expansion directions for periodic boundary condition
    for a_type, a_coord in zip(atom_type, atom_coord):
        for direction in exp_direction:
            new_coord = a_coord + direction  # Calculate new coordinates by adding the expansion direction
            if (new_coord <= 1).all():  # Check if the new coordinates are within the unit cell
                exp_type.append(a_type)  # Append the atom type to the expanded types
                exp_coord.append(new_coord)  # Append the new coordinates to the expanded coordinates
    return torch.LongTensor(exp_type), torch.stack(exp_coord, dim=0)  # Return the expanded atom types and coordinates as tensors


def rotate_voxel(shape, cell, density, rotated_grid):
    """
    Rotate the volumetric data using trilinear interpolation.
    :param shape: voxel shape, tensor of shape (3,)
    :param cell: cell vectors, tensor of shape (3, 3)
    :param density: original density, tensor of shape (n_grid,)
    :param rotated_grid: rotated grid coordinates, tensor of shape (n_grid, 3)
    :return: rotated density, tensor of shape (n_grid,)
    """
    density = density.view(1, 1, *shape)  # Reshape the density tensor to (1, 1, *shape)
    rotated_grid = rotated_grid.view(1, *shape, 3)  # Reshape the rotated grid coordinates to (1, *shape, 3)
    shape = torch.FloatTensor(shape)  # Convert the shape to a float tensor
    grid_cell = cell / shape.view(3, 1)  # Calculate the grid cell size by dividing the cell vectors by the shape
    normalized_grid = (2 * rotated_grid @ torch.linalg.inv(grid_cell) - shape + 1) / (shape - 1)  # Normalize the rotated grid coordinates
    return F.grid_sample(density, torch.flip(normalized_grid, [-1]),
                         mode='bilinear', align_corners=False).view(-1)  # Perform trilinear interpolation and reshape the result to (n_grid,)


@register_dataset('density')
class DensityDataset(Dataset):
    def __init__(self, root, split, split_file, atom_file, extension='CHGCAR',
                 compression='lz4', rotate=False, pbc=False):
        """
        The density dataset contains volumetric data of molecules.
        :param root: data root
        :param split: data split, can be 'train', 'validation', 'test'
        :param split_file: the data split file containing file names of the split
        :param atom_file: atom information file
        :param extension: raw data file extension, can be 'CHGCAR', 'cube', 'json'
        :param compression: raw data compression, can be 'lz4', 'xz', or None (no compression)
        :param rotate: whether to rotate the molecule and the volumetric data
        :param pbc: whether the data satisfy the periodic boundary condition
        """
        super(DensityDataset, self).__init__()
        self.root = root
        self.split = split
        self.extension = extension
        self.compression = compression
        self.rotate = rotate
        self.pbc = pbc

        self.file_pattern = f'.{extension}'
        if compression is not None:
            self.file_pattern += f'.{compression}'  # Create the file pattern based on extension and compression
        with open(os.path.join(root, split_file)) as f:
            # reverse the order so that larger molecules are tested first
            self.file_list = list(reversed(json.load(f)[split]))  # Load the file list for the specified split
        with open(atom_file) as f:
            atom_info = json.load(f)  # Load the atom information from the file
        atom_list = [info['name'] for info in atom_info]  # Extract the atom names from the atom information
        self.atom_name2idx = {name: idx for idx, name in enumerate(atom_list)}  # Create a mapping from atom names to indices
        self.atom_name2idx.update({name.encode(): idx for idx, name in enumerate(atom_list)})  # Update the mapping with encoded atom names
        self.atom_num2idx = {info['atom_num']: idx for idx, info in enumerate(atom_info)}  # Create a mapping from atomic numbers to indices
        self.idx2atom_num = {idx: info['atom_num'] for idx, info in enumerate(atom_info)}  # Create a mapping from indices to atomic numbers

        if extension == 'CHGCAR':
            self.read_func = self.read_chgcar  # Set the read function based on the file extension
        elif extension == 'cube':
            self.read_func = self.read_cube
        elif extension == 'json':
            self.read_func = self.read_json
        else:
            raise TypeError(f'Unknown extension {extension}')

        if compression == 'lz4':
            import lz4.frame
            self.open = lz4.frame.open  # Set the open function based on the compression type
        elif compression == 'xz':
            import lzma
            self.open = lzma.open
        else:
            self.open = open

    def __getitem__(self, item):
        file_name = f'{(self.file_list[item])}{self.file_pattern}'  # Get the file name for the specified item
        with self.open(os.path.join(self.root, file_name)) as f:
            g, density, grid_coord, info = self.read_func(f)  # Read the data using the appropriate read function
        info['file_name'] = file_name

        if self.rotate:
            rot = o3.rand_matrix()  # Generate a random rotation matrix
            center = info['cell'].sum(dim=0) / 2  # Calculate the center of the cell
            g.pos = (g.pos - center) @ rot.t() + center  # Rotate the atom positions
            rotated_grid = (grid_coord - center) @ rot + center  # Rotate the grid coordinates
            density = rotate_voxel(info['shape'], info['cell'], density, rotated_grid)  # Rotate the density using trilinear interpolation
            info['rot'] = rot  # Store the rotation matrix in the info dictionary
        return g, density, grid_coord, info

    def __len__(self):
        return len(self.file_list)  # Return the length of the file list

    def read_cube(self, fileobj):
        """Read atoms and data from CUBE file."""
        if self.pbc:
            raise NotImplementedError('PBC not implemented for cube files')

        readline = fileobj.readline
        readline()  # the first comment line
        readline()  # the second comment line

        # Third line contains actual system information:
        line = readline().split()
        n_atom = int(line[0])  # Number of atoms

        # Origin around which the volumetric data is centered
        # (at least in FHI aims):
        origin = torch.FloatTensor([float(x) for x in line[1::]])  # Extract the origin coordinates

        shape = []
        cell = torch.empty(3, 3, dtype=torch.float)
        # the upcoming three lines contain the cell information
        for i in range(3):
            n, x, y, z = [float(s) for s in readline().split()]
            shape.append(int(n))  # Append the grid size along each dimension
            cell[i] = torch.FloatTensor([x, y, z])  # Store the cell vectors
        x_coord = torch.arange(shape[0]).unsqueeze(-1) * cell[0]  # Calculate the x-coordinates of the grid points
        y_coord = torch.arange(shape[1]).unsqueeze(-1) * cell[1]  # Calculate the y-coordinates of the grid points
        z_coord = torch.arange(shape[2]).unsqueeze(-1) * cell[2]  # Calculate the z-coordinates of the grid points
        grid_coord = x_coord.view(-1, 1, 1, 3) + y_coord.view(1, -1, 1, 3) + z_coord.view(1, 1, -1, 3)  # Combine the coordinates to form the grid
        grid_coord = grid_coord.view(-1, 3) - origin  # Subtract the origin from the grid coordinates

        atom_type = torch.empty(n_atom, dtype=torch.long)
        atom_coord = torch.empty(n_atom, 3, dtype=torch.float)
        for i in range(n_atom):
            line = readline().split()
            atom_type[i] = self.atom_num2idx[int(line[0])]  # Map the atomic number to the atom type index
            atom_coord[i] = torch.FloatTensor([float(s) for s in line[2:]])  # Extract the atom coordinates

        g = Data(x=atom_type, pos=atom_coord)  # Create a Data object with atom types and coordinates
        density = torch.FloatTensor([float(s) for s in fileobj.read().split()])  # Read the density values from the file
        return g, density, grid_coord, {'shape': shape, 'cell': cell, 'origin': origin}

    def read_chgcar(self, fileobj):
        """Read atoms and data from CHGCAR file."""
        readline = fileobj.readline
        readline()  # the first comment line
        scale = float(readline())  # the scaling factor (lattice constant)

        # the upcoming three lines contain the cell information
        cell = torch.empty(3, 3, dtype=torch.float)
        for i in range(3):
            cell[i] = torch.FloatTensor([float(s) for s in readline().split()])  # Read the cell vectors
        cell = cell * scale  # Scale the cell vectors by the scaling factor

        # the sixth line specifies the constituting elements
        elements = readline().split()
        # the seventh line supplies the number of atoms per atomic species
        n_atoms = [int(s) for s in readline().split()]
        # the eighth line is always "Direct" in our application
        readline()

        tot_atoms = sum(n_atoms)
        atom_type = torch.empty(tot_atoms, dtype=torch.long)
        atom_coord = torch.empty(tot_atoms, 3, dtype=torch.float)
        # the upcoming lines contains the atomic positions in fractional coordinates
        idx = 0
        for elem, n in zip(elements, n_atoms):
            atom_type[idx:idx + n] = self.atom_name2idx[elem]  # Map the element name to the atom type index
            for _ in range(n):
                atom_coord[idx] = torch.FloatTensor([float(s) for s in readline().split()])  # Read the atom coordinates
                idx += 1
        if self.pbc:
            atom_type, atom_coord = pbc_expand(atom_type, atom_coord)  # Expand the atoms based on periodic boundary conditions
        # the coordinates are fractional, convert them to cartesian
        atom_coord = atom_coord @ cell  # Convert fractional coordinates to Cartesian coordinates
        g = Data(x=atom_type, pos=atom_coord)  # Create a Data object with atom types and coordinates

        readline()  # an empty line
        shape = [int(s) for s in readline().split()]  # grid size
        n_grid = shape[0] * shape[1] * shape[2]
        # the grids are corner-aligned
        x_coord = torch.linspace(0, shape[0] - 1, shape[0]).unsqueeze(-1) / shape[0] * cell[0]  # Calculate the x-coordinates of the grid points
        y_coord = torch.linspace(0, shape[1] - 1, shape[1]).unsqueeze(-1) / shape[1] * cell[1]  # Calculate the y-coordinates of the grid points
        z_coord = torch.linspace(0, shape[2] - 1, shape[2]).unsqueeze(-1) / shape[2] * cell[2]  # Calculate the z-coordinates of the grid points
        grid_coord = x_coord.view(-1, 1, 1, 3) + y_coord.view(1, -1, 1, 3) + z_coord.view(1, 1, -1, 3)  # Combine the coordinates to form the grid
        grid_coord = grid_coord.view(-1, 3)

        # the augmented occupancies are ignored
        density = torch.FloatTensor([float(s) for s in fileobj.read().split()[:n_grid]])  # Read the density values from the file
        # the value stored is the charge within a grid instead of the charge density
        # divide the charge by the grid volume to get the density
        volume = torch.linalg.det(cell).abs()  # Calculate the volume of the cell
        density = density / volume  # Convert charge to density by dividing by the volume
        # CHGCAR file stores the density as Z-Y-X, convert them to X-Y-Z
        density = density.view(shape[2], shape[1], shape[0]).transpose(0, 2).contiguous().view(-1)  # Reshape and transpose the density tensor
        return g, density, grid_coord, {'shape': shape, 'cell': cell}

    def read_json(self, fileobj):
        """Read atoms and data from JSON file."""

        def read_2d_tensor(s):
            return torch.FloatTensor([[float(x) for x in line] for line in s])  # Read a 2D tensor from a list of lists

        data = json.load(fileobj)
        scale = float(data['vector'][0][0])  # Read the scaling factor
        cell = read_2d_tensor(data['lattice'][0]) * scale  # Read the cell vectors and scale them
        elements = data['elements'][0]
        n_atoms = [int(s) for s in data['elements_number'][0]]  # Read the number of atoms for each element

        tot_atoms = sum(n_atoms)
        atom_coord = read_2d_tensor(data['coordinates'][0])  # Read the atom coordinates
        atom_type = torch.empty(tot_atoms, dtype=torch.long)
        idx = 0
        for elem, n in zip(elements, n_atoms):
            atom_type[idx:idx + n] = self.atom_name2idx[elem]  # Map the element name to the atom type index
            idx += n
        if self.pbc:
            atom_type, atom_coord = pbc_expand(atom_type, atom_coord)  # Expand the atoms based on periodic boundary conditions
        atom_coord = atom_coord @ cell  # Convert fractional coordinates to Cartesian coordinates
        g = Data(x=atom_type, pos=atom_coord)  # Create a Data object with atom types and coordinates

        shape = [int(s) for s in data['FFTgrid'][0]]  # Read the grid shape
        x_coord = torch.linspace(0, shape[0] - 1, shape[0]).unsqueeze(-1) / shape[0] * cell[0]  # Calculate the x-coordinates of the grid points
        y_coord = torch.linspace(0, shape[1] - 1, shape[1]).unsqueeze(-1) / shape[1] * cell[1]  # Calculate the y-coordinates of the grid points
        z_coord = torch.linspace(0, shape[2] - 1, shape[2]).unsqueeze(-1) / shape[2] * cell[2]  # Calculate the z-coordinates of the grid points
        grid_coord = x_coord.view(-1, 1, 1, 3) + y_coord.view(1, -1, 1, 3) + z_coord.view(1, 1, -1, 3)  # Combine the coordinates to form the grid
        grid_coord = grid_coord.view(-1, 3)

        n_grid = shape[0] * shape[1] * shape[2]
        n_line = (n_grid + 9) // 10
        density = torch.FloatTensor([
            float(s) if not s.startswith('*') else 0.
            for line in data['chargedensity'][0][:n_line]
            for s in line
        ]).view(-1)[:n_grid]  # Read the density values from the file
        volume = torch.linalg.det(cell).abs()  # Calculate the volume of the cell
        density = density / volume  # Convert charge to density by dividing by the volume
        density = density.view(shape[2], shape[1], shape[0]).transpose(0, 2).contiguous().view(-1)  # Reshape and transpose the density tensor
        return g, density, grid_coord, {'shape': shape, 'cell': cell}

    # TODO: cube files are in unit of Bohr
    def write_cube(self, fileobj, atom_type, atom_coord, density, info):
        """Write a cube file."""
        fileobj.write('Cube file written on ' + time.strftime('%c'))
        fileobj.write("\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")

        cell = info['cell']
        shape = info['shape']
        origin = info.get('origin', np.zeros(3))
        fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(len(atom_type), *origin))  # Write the number of atoms and origin

        for s, c in zip(shape, cell):
            d = c / s
            fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(s, *d))  # Write the grid shape and cell vectors

        for Z, (x, y, z) in zip(atom_type, atom_coord):
            Z = self.idx2atom_num[Z]  # Map the atom type index to the atomic number
            fileobj.write(
                '{0:5}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n'.format(Z, Z, x, y, z)
            )  # Write the atomic number and coordinates
        density.tofile(fileobj, sep='\n', format='%e')  # Write the density values to the file