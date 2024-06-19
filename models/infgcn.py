
"""
1. Embedding layer: Maps atom types to initial features.
2. Spherical harmonics irreducible representations: Defines the spherical harmonics basis functions.
3. Node feature irreducible representations: Defines the node feature representations.
4. InfGCN layers: Performs message passing using the GCNLayer module.
5. Activation function: Applies an activation function (scalar or norm) to the node features.
6. Residue prediction layer: Computes the residue using an additional GCNLayer.
7. Gaussian orbital layer: Computes orbital values using Gaussian basis functions.

The forward pass involves the following steps:

1. Embedding: Embeds atom types to initial features and constructs the edge index based on the cutoff distance.
2. GCN: Performs message passing using the InfGCN layers and applies the activation function.
3. Residue: Computes the residue using the residue prediction layer if enabled.
4. Orbital: Computes orbital values using the Gaussian orbital layer and calculates the density by multiplying orbital values with features and summing.
5. Density: Scatters density values based on batch indices and adds the residue if enabled.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import radius_graph, radius
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Extract, Activation
from jaxtyping import Float, Int

from .orbital import GaussianOrbital
from ._base import register_model


class ScalarActivation(nn.Module):
    """
    Use the invariant scalar features to gate higher order equivariant features.
    Adapted from `e3nn.nn.Gate`.
    """

    def __init__(self, irreps_in: str, act_scalars: nn.Module, act_gates: nn.Module):
        """
        Initialize the ScalarActivation module.

        Args:
            irreps_in (str): Input representations.
            act_scalars (nn.Module): Scalar activation function.
            act_gates (nn.Module): Gate activation function (for higher order features).
        """
        super(ScalarActivation, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.num_spherical = len(self.irreps_in)

        irreps_scalars = self.irreps_in[0:1]
        irreps_gates = irreps_scalars * (self.num_spherical - 1)
        irreps_gated = self.irreps_in[1:]
        self.act_scalars = Activation(irreps_scalars, [act_scalars])
        self.act_gates = Activation(irreps_gates, [act_gates] * (self.num_spherical - 1))
        self.extract = Extract(
            self.irreps_in,
            [irreps_scalars, irreps_gated],
            instructions=[(0,), tuple(range(1, self.irreps_in.lmax + 1))]
        )
        self.mul = o3.ElementwiseTensorProduct(irreps_gates, irreps_gated)

    def forward(self, features: Float[torch.Tensor, '..., irreps_in']) -> Float[torch.Tensor, '..., irreps_out']:
        """
        Apply scalar activation to the input features.

        Args:
            features (Float[torch.Tensor, '..., irreps_in']): Input features.

        Returns:
            Float[torch.Tensor, '..., irreps_out']: Activated features.
        """
        scalars, gated = self.extract(features)
        scalars_out = self.act_scalars(scalars)
        if gated.shape[-1]:
            gates = self.act_gates(scalars.repeat(1, self.num_spherical - 1))
            gated_out = self.mul(gates, gated)
            features = torch.cat([scalars_out, gated_out], dim=-1)
        else:
            features = scalars_out
        return features


class NormActivation(nn.Module):
    """
    Use the norm of the higher order equivariant features to gate themselves.
    Idea from the TFN paper.
    """

    def __init__(self, irreps_in: str, act_scalars: nn.Module = torch.nn.functional.silu, act_vectors: nn.Module = torch.sigmoid):
        """
        Initialize the NormActivation module.

        Args:
            irreps_in (str): Input representations.
            act_scalars (nn.Module): Scalar activation function. Default is torch.nn.functional.silu.
            act_vectors (nn.Module): Vector activation function (for the norm of higher order features). Default is torch.sigmoid.
        """
        super(NormActivation, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.scalar_irreps = self.irreps_in[0:1]
        self.vector_irreps = self.irreps_in[1:]
        self.act_scalars = act_scalars
        self.act_vectors = act_vectors
        self.scalar_idx = self.irreps_in[0].mul

        inner_out = o3.Irreps([(mul, (0, 1)) for mul, _ in self.vector_irreps])
        self.inner_prod = o3.TensorProduct(
            self.vector_irreps, self.vector_irreps, inner_out, [
                (i, i, i, 'uuu', False) for i in range(len(self.vector_irreps))
            ]
        )
        self.mul = o3.ElementwiseTensorProduct(inner_out, self.vector_irreps)

    def forward(self, features: Float[torch.Tensor, '..., irreps_in']) -> Float[torch.Tensor, '..., irreps_out']:
        """
        Apply norm activation to the input features.

        Args:
            features (Float[torch.Tensor, '..., irreps_in']): Input features.

        Returns:
            Float[torch.Tensor, '..., irreps_out']: Activated features.
        """
        scalars = self.act_scalars(features[..., :self.scalar_idx])
        vectors = features[..., self.scalar_idx:]
        norm = torch.sqrt(self.inner_prod(vectors, vectors) + 1e-8)
        act = self.act_vectors(norm)
        vectors_out = self.mul(act, vectors)
        return torch.cat([scalars, vectors_out], dim=-1)


class GCNLayer(nn.Module):
    def __init__(self, irreps_in: str, irreps_out: str, irreps_edge: str, radial_embed_size: int, num_radial_layer: int, radial_hidden_size: int,
                 is_fc: bool = True, use_sc: bool = True, irrep_normalization: str = 'component', path_normalization: str = 'element'):
        r"""
        Initialize a single InfGCN layer for Tensor Product-based message passing.
        If the tensor product is fully connected, we have (for every path)

        .. math::
            z_w=\sum_{uv}w_{uvw}x_u\otimes y_v=\sum_{u}w_{uw}x_u \otimes y

        Else, we have

        .. math::
            z_u=x_u\otimes \sum_v w_{uv}y_v=w_u (x_u\otimes y)

        Here, uvw are radial (channel) indices of the first input, second input, and output, respectively.
        Notice that in our model, the second input is always the spherical harmonics of the edge vector,
        so the index v can be safely ignored.

        Args:
            irreps_in (str): Irreducible representations of input node features.
            irreps_out (str): Irreducible representations of output node features.
            irreps_edge (str): Irreducible representations of edge features.
            radial_embed_size (int): Embedding size of the edge length.
            num_radial_layer (int): Number of hidden layers in the radial network.
            radial_hidden_size (int): Hidden size of the radial network.
            is_fc (bool): Whether to use fully connected tensor product. Default is True.
            use_sc (bool): Whether to use self-connection. Default is True.
            irrep_normalization (str): Representation normalization passed to the `o3.FullyConnectedTensorProduct`. Default is 'component'.
            path_normalization (str): Path normalization passed to the `o3.FullyConnectedTensorProduct`. Default is 'element'.
        """
        super(GCNLayer, self).__init__()
        
        # Store input, output, and edge irreducible representations
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        
        # Store radial embedding size, number of radial layers, and radial hidden size
        self.radial_embed_size = radial_embed_size
        self.num_radial_layer = num_radial_layer
        self.radial_hidden_size = radial_hidden_size
        
        # Store flags for fully connected tensor product and self-connection
        self.is_fc = is_fc
        self.use_sc = use_sc

        # Create tensor product module based on fully connected flag
        if self.is_fc:
            # Use fully connected tensor product
            self.tp = o3.FullyConnectedTensorProduct(
                self.irreps_in, self.irreps_edge, self.irreps_out,
                internal_weights=False, shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        else:
            # Create instruction for tensor product based on input, edge, and output irreps
            instr = [
                (i_1, i_2, i_out, 'uvu', True)
                for i_1, (_, ir_1) in enumerate(self.irreps_in)
                for i_2, (_, ir_edge) in enumerate(self.irreps_edge)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                if ir_out in ir_1 * ir_edge
            ]
            # Use tensor product with specified instruction
            self.tp = o3.TensorProduct(
                self.irreps_in, self.irreps_edge, self.irreps_out, instr,
                internal_weights=False, shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        
        # Create fully connected network for radial embedding
        self.fc = FullyConnectedNet(
            [radial_embed_size] + num_radial_layer * [radial_hidden_size] + [self.tp.weight_numel],
            torch.nn.functional.silu
        )
        
        # Create self-connection linear layer if use_sc flag is set
        self.sc = None
        if self.use_sc:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(self, edge_index: Int[torch.Tensor, '2, num_edges'], node_feat: Float[torch.Tensor, 'num_nodes, irreps_in'], edge_feat: Float[torch.Tensor, 'num_edges, irreps_edge'], edge_embed: Float[torch.Tensor, 'num_edges, radial_embed_size'], dim_size: int = None) -> Float[torch.Tensor, 'num_nodes, irreps_out']:
        """
        Perform message passing on the graph.

        Args:
            edge_index (Int[torch.Tensor, '2, num_edges']): Edge index tensor.
            node_feat (Float[torch.Tensor, 'num_nodes, irreps_in']): Node feature tensor.
            edge_feat (Float[torch.Tensor, 'num_edges, irreps_edge']): Edge feature tensor.
            edge_embed (Float[torch.Tensor, 'num_edges, radial_embed_size']): Edge embedding tensor.
            dim_size (int): Size of the output dimension. Default is None.

        Returns:
            Float[torch.Tensor, 'num_nodes, irreps_out']: Output node feature tensor.
        """
        # Extract source and destination node indices from edge index
        src, dst = edge_index
        
        # Compute radial weights using the fully connected network
        weight = self.fc(edge_embed)
        
        # Perform tensor product between node features and edge features
        out = self.tp(node_feat[src], edge_feat, weight=weight)
        
        # Scatter the tensor product results to destination nodes
        out = scatter(out, dst, dim=0, dim_size=dim_size, reduce='sum')
        
        # Add self-connection if use_sc flag is set
        if self.use_sc:
            out = out + self.sc(node_feat)
        
        return out

def pbc_vec(vec: Float[torch.Tensor, 'num_graphs, num_samples, 3'], cell: Float[torch.Tensor, 'num_graphs, 3, 3']) -> Float[torch.Tensor, 'num_graphs, num_samples, 3']:
    """
    Apply periodic boundary condition to the vector.

    Args:
        vec (Float[torch.Tensor, 'num_graphs, num_samples, 3']): Original vector.
        cell (Float[torch.Tensor, 'num_graphs, 3, 3']): Cell frame.

    Returns:
        Float[torch.Tensor, 'num_graphs, num_samples, 3']: Shortest vector.
    """
    coord = vec @ torch.linalg.inv(cell)
    coord = coord - torch.round(coord)
    pbc_vec = coord @ cell
    return pbc_vec.detach()


@register_model('infgcn')
class InfGCN(nn.Module):
    def __init__(self, n_atom_type: int, num_radial: int, num_spherical: int, radial_embed_size: int, radial_hidden_size: int,
                 num_radial_layer: int = 2, num_gcn_layer: int = 3, cutoff: float = 3.0, grid_cutoff: float = 3.0, is_fc: bool = True,
                 gauss_start: float = 0.5, gauss_end: float = 5.0, activation: str = 'norm', residual: bool = True, pbc: bool = False, **kwargs):
        """
        Initialize the InfGCN model for electron density estimation.

        Args:
            n_atom_type (int): Number of atom types.
            num_radial (int): Number of radial basis.
            num_spherical (int): Maximum number of spherical harmonics for each radial basis,
                number of spherical basis will be (num_spherical + 1)^2.
            radial_embed_size (int): Embedding size of the edge length.
            radial_hidden_size (int): Hidden size of the radial network.
            num_radial_layer (int): Number of hidden layers in the radial network. Default is 2.
            num_gcn_layer (int): Number of InfGCN layers. Default is 3.
            cutoff (float): Cutoff distance for building the molecular graph. Default is 3.0.
            grid_cutoff (float): Cutoff distance for building the grid-atom graph. Default is 3.0.
            is_fc (bool): Whether the InfGCN layer should use fully connected tensor product. Default is True.
            gauss_start (float): Start coefficient of the Gaussian radial basis. Default is 0.5.
            gauss_end (float): End coefficient of the Gaussian radial basis. Default is 5.0.
            activation (str): Activation type for the InfGCN layer, can be ['scalar', 'norm']. Default is 'norm'.
            residual (bool): Whether to use the residue prediction layer. Default is True.
            pbc (bool): Whether the data satisfy the periodic boundary condition. Default is False.
        """
        super(InfGCN, self).__init__()
        self.n_atom_type = n_atom_type
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.radial_embed_size = radial_embed_size
        self.radial_hidden_size = radial_hidden_size
        self.num_radial_layer = num_radial_layer
        self.num_gcn_layer = num_gcn_layer
        self.cutoff = cutoff
        self.grid_cutoff = grid_cutoff
        self.is_fc = is_fc
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.activation = activation
        self.residual = residual
        self.pbc = pbc

        assert activation in ['scalar', 'norm'], "Invalid activation type. Must be 'scalar' or 'norm'."

        # Embedding layer to map atom types to initial features
        self.embedding = nn.Embedding(n_atom_type, num_radial)
        
        # Spherical harmonics irreducible representations
        self.irreps_sh = o3.Irreps.spherical_harmonics(num_spherical, p=1)
        
        # Node feature irreducible representations
        self.irreps_feat = (self.irreps_sh * num_radial).sort().irreps.simplify()
        
        # InfGCN layers
        self.gcns = nn.ModuleList([
            GCNLayer(
                (f'{num_radial}x0e' if i == 0 else self.irreps_feat), self.irreps_feat, self.irreps_sh,
                radial_embed_size, num_radial_layer, radial_hidden_size, is_fc=is_fc, **kwargs
            ) for i in range(num_gcn_layer)
        ])
        
        # Activation function
        if self.activation == 'scalar':
            self.act = ScalarActivation(self.irreps_feat, torch.nn.functional.silu, torch.sigmoid)
        else:
            self.act = NormActivation(self.irreps_feat)
        
        # Residue prediction layer
        self.residue = None
        if self.residual:
            self.residue = GCNLayer(
                self.irreps_feat, '0e', self.irreps_sh,
                radial_embed_size, num_radial_layer, radial_hidden_size,
                is_fc=True, use_sc=False, **kwargs
            )
        
        # Gaussian orbital layer
        self.orbital = GaussianOrbital(gauss_start, gauss_end, num_radial, num_spherical)

    def forward(self, atom_types: Int[torch.Tensor, 'num_atoms'], atom_coord: Float[torch.Tensor, 'num_atoms, 3'], grid: Float[torch.Tensor, 'num_graphs, num_samples, 3'], batch: Int[torch.Tensor, 'num_atoms'], infos: list) -> Float[torch.Tensor, 'num_graphs, num_samples']:
        """
        Perform forward pass of the InfGCN model.

        Args:
            atom_types (Int[torch.Tensor, 'num_atoms']): Atom types.
            atom_coord (Float[torch.Tensor, 'num_atoms, 3']): Atom coordinates.
            grid (Float[torch.Tensor, 'num_graphs, num_samples, 3']): Coordinates at grid points.
            batch (Int[torch.Tensor, 'num_atoms']): Batch index for each node.
            infos (list): List of dictionary containing additional information.

        Returns:
            Float[torch.Tensor, 'num_graphs, num_samples']: Predicted value at each grid point.
        """
        # Embedding
        cell = torch.stack([info['cell'] for info in infos], dim=0).to(batch.device)  # Stack cell information from infos
        feat = self.embedding(atom_types)  # Embed atom types to initial features
        edge_index = radius_graph(atom_coord, self.cutoff, batch, loop=False)  # Construct edge index based on cutoff distance
        src, dst = edge_index
        edge_vec = atom_coord[src] - atom_coord[dst]  # Compute edge vectors
        edge_len = edge_vec.norm(dim=-1) + 1e-8  # Compute edge lengths
        edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical + 1)), edge_vec / edge_len[..., None],
            normalize=False, normalization='integral'
        )  # Compute spherical harmonics edge features
        edge_embed = soft_one_hot_linspace(
            edge_len, start=0.0, end=self.cutoff,
            number=self.radial_embed_size, basis='gaussian', cutoff=False
        ).mul(self.radial_embed_size ** 0.5)  # Compute edge embeddings using soft one-hot encoding

        # GCN
        for i, gcn in enumerate(self.gcns):
            feat = gcn(edge_index, feat, edge_feat, edge_embed, dim_size=atom_types.size(0))  # Perform message passing
            if i != self.num_gcn_layer - 1:
                feat = self.act(feat)  # Apply activation function

        # Residue
        n_graph, n_sample = grid.size(0), grid.size(1)
        if self.residual:
            grid_flat = grid.view(-1, 3)  # Flatten grid coordinates
            grid_batch = torch.arange(n_graph, device=grid.device).repeat_interleave(n_sample)  # Create batch indices for grid points
            grid_dst, node_src = radius(atom_coord, grid_flat, self.grid_cutoff, batch, grid_batch)  # Compute grid-atom distances and indices
            grid_edge = grid_flat[grid_dst] - atom_coord[node_src]  # Compute grid-atom edge vectors
            grid_len = torch.norm(grid_edge, dim=-1) + 1e-8  # Compute grid-atom edge lengths
            grid_edge_feat = o3.spherical_harmonics(
                list(range(self.num_spherical + 1)), grid_edge / (grid_len[..., None] + 1e-8),
                normalize=False, normalization='integral'
            )  # Compute spherical harmonics features for grid-atom edges
            grid_edge_embed = soft_one_hot_linspace(
                grid_len, start=0.0, end=self.grid_cutoff,
                number=self.radial_embed_size, basis='gaussian', cutoff=False
            ).mul(self.radial_embed_size ** 0.5)  # Compute grid-atom edge embeddings using soft one-hot encoding
            residue = self.residue(
                (node_src, grid_dst), feat, grid_edge_feat, grid_edge_embed, dim_size=grid_flat.size(0)
            )  # Compute residue using the residue prediction layer
        else:
            residue = 0.

        # Orbital
        sample_vec = grid[batch] - atom_coord.unsqueeze(-2)  # Compute sample vectors
        if self.pbc:
            sample_vec = pbc_vec(sample_vec, cell[batch])  # Apply periodic boundary condition to sample vectors
        orbital = self.orbital(sample_vec)  # Compute orbital values using Gaussian orbital layer
        density = (orbital * feat.unsqueeze(1)).sum(dim=-1)  # Compute density by multiplying orbital values with features and summing
        density = scatter(density, batch, dim=0, reduce='sum')  # Scatter density values based on batch indices
        if self.residual:
            density = density + residue.view(*density.size())  # Add residue to the density
        return density