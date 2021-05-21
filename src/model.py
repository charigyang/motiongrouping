import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def build_grid(resolution):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.tensor(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


class SoftPositionEmbed(nn.Module):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, hidden_size, resolution):
    """Builds the soft position embedding layer.

    Args:
      hidden_size: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
    super(SoftPositionEmbed, self).__init__()
    self.proj = nn.Linear(4, hidden_size)
    self.grid = build_grid(resolution)

  def forward(self, inputs):
    return inputs + self.proj(self.grid)


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
    grid = einops.repeat(slots, 'b_n i j d -> b_n (tilei i) (tilej j) d', tilei=resolution[0], tilej=resolution[1])
    # `grid` has shape: [batch_size*num_slots, height, width, slot_size].
    return grid


def spatial_flatten(x):
  return torch.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])


def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = einops.rearrange(x, '(b s) c h w -> b s c h w', b=batch_size)
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)
    return channels, masks


class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()

        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        # Parameters for Gaussian init (shared by all slots).
        # self.slots_mu = nn.Parameter(torch.randn(1, 1, encoder_dims))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, encoder_dims))

        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, num_slots=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # random slots initialization,
        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)

        # learnable slots initialization
        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))

        # Multiple rounds of attention.
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SlotAttentionAutoEncoder(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""
    def __init__(self, resolution, num_slots, in_out_channels=3, iters=5):
        """Builds the Slot Attention-based Auto-encoder.

        Args:
            resolution: Tuple of integers specifying width and height of input image
            num_slots: Number of slots in Slot Attention.
            iters: Number of iterations in Slot Attention.
        """
        super(SlotAttentionAutoEncoder, self).__init__()

        self.iters = iters
        self.num_slots = num_slots
        self.resolution = resolution
        self.in_out_channels = in_out_channels

        self.encoder_arch = [64, 'MP', 128, 'MP', 256]
        self.encoder_dims = self.encoder_arch[-1]
        self.encoder_cnn, ratio = self.make_encoder(self.in_out_channels, self.encoder_arch)
        self.encoder_end_size = (int(resolution[0] / ratio), int(resolution[1] / ratio))
        self.encoder_pos = SoftPositionEmbed(self.encoder_dims, self.encoder_end_size)
        self.decoder_initial_size = (int(resolution[0] / 8), int(resolution[1] / 8))
        self.decoder_pos = SoftPositionEmbed(self.encoder_dims, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(self.encoder_dims)

        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dims, self.encoder_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims, self.encoder_dims)
        )

        self.slot_attention = SlotAttention(
            iters=self.iters,
            num_slots=self.num_slots,
            encoder_dims=self.encoder_dims,
            hidden_dim=self.encoder_dims)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_dims, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_out_channels + 1, kernel_size=5, padding=2, stride=1)
        )

    def make_encoder(self, in_channels, encoder_arch):
        layers = []
        down_factor = 0
        for v in encoder_arch:
            if v == 'MP':
                layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                down_factor += 1
            else:
                conv1 = nn.Conv2d(in_channels, v, kernel_size=5, padding=2)
                conv2 = nn.Conv2d(v, v, kernel_size=5, padding=2)

                layers += [conv1, nn.InstanceNorm2d(v, affine=True), nn.ReLU(inplace=True),
                           conv2, nn.InstanceNorm2d(v, affine=True), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers), 2 ** down_factor


    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, height, width].
        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.encoder_pos(x)  # Position embedding.
        x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
        x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # Spatial broadcast decoder.
        x = spatial_broadcast(slots, self.decoder_initial_size)
        # `x` has shape: [batch_size*num_slots, height_init, width_init, slot_size].
        x = self.decoder_pos(x)
        x = einops.rearrange(x, 'b_n h w c -> b_n c h w')
        x = self.decoder_cnn(x)
        # `x` has shape: [batch_size*num_slots, num_channels+1, height, width].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = unstack_and_split(x, batch_size=image.shape[0], num_channels=self.in_out_channels)
        # `recons` has shape: [batch_size, num_slots, num_channels, height, width].
        # `masks` has shape: [batch_size, num_slots, 1, height, width].
        
        # Normalize alpha masks over slots.
        masks = torch.softmax(masks, axis=1)

        recon_combined = torch.sum(recons * masks, axis=1)  # Recombine image.
        # `recon_combined` has shape: [batch_size, num_channels, height, width].
        return recon_combined, recons, masks, slots


