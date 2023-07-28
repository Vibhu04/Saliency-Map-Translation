import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_seq_len, embed_size)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(1)  # Add batch_size dimension

    def forward(self, x, device):
        return x + self.encoding[:x.size(0), :].to(device)


class RegionEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers):
        super(RegionEncoder, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.patch_size = 32
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.embedding = nn.Linear(3 * self.patch_size * self.patch_size, d_model)
        self.trainable_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.fc = nn.Linear(d_model, 64*64*3)
        self.delimiter_token = torch.randn(1, 1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len=100)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, regions, device):

        batch_size = images.shape[0]
        num_img_patches = (images.shape[-2] // self.patch_size) * (images.shape[-1] // self.patch_size)
        num_region_patches = (regions.shape[-2] // self.patch_size) * (regions.shape[-1] // self.patch_size)

        images_patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        images_patches = images_patches.contiguous().view(-1, 3 * self.patch_size * self.patch_size)

        region_patches = regions.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        region_patches = region_patches.contiguous().view(-1, 3 * self.patch_size * self.patch_size)

        img_patch_features = self.embedding(images_patches)
        region_patch_features = self.embedding(region_patches)

        img_patch_features = img_patch_features.reshape((num_img_patches, batch_size, -1))
        region_patch_features = region_patch_features.reshape((num_region_patches, batch_size, -1))

        trainable_token = self.trainable_token.repeat(1, batch_size, 1).to(device)
        delimiter_token = self.delimiter_token.repeat(1, batch_size, 1).to(device)
        final_input = torch.cat((trainable_token, region_patch_features, delimiter_token, img_patch_features), dim=0)

        final_input = self.pos_encoding(final_input, device)

        out = self.transformer_encoder(final_input)

        out = out[0, :, :]

        out = self.fc(out)

        out = self.sigmoid(out)


        return out
