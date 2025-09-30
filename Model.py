import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings


class VisionTransformer(nn.Module):
    def __init__(self, num_classes = 10, channels = 3, patch_shape = (8, 8), image_shape = (256, 256), embedding_dim = 256, num_heads = 8, num_layers = 6):
        super().__init__()
        self.patch_shape = patch_shape
        self.image_shape = image_shape
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.per_head_dim = embedding_dim // num_heads
        self.num_patches = (image_shape[0]//patch_shape[0], image_shape[1]//patch_shape[1])
        self.cls_embedding = nn.Parameter(torch.randn(1,1,self.embedding_dim))
        self.mlp_head = nn.Linear(self.embedding_dim, num_classes)
        self.linear_projection = nn.Linear(self.channels*patch_shape[0]*patch_shape[1] , embedding_dim)
        self.rope = RotaryPositionalEmbeddings(dim =  self.per_head_dim, max_seq_len = 1 + image_shape[0]*image_shape[1]//(patch_shape[0]*patch_shape[1]))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = embedding_dim, nhead = num_heads, batch_first = True, norm_first = True, activation = "gelu")
        self.encoders = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)
    
    def forward(self, batch):
        batch_size, channels, image_height, image_width = batch.shape
        assert image_height == self.image_shape[0] and image_width == self.image_shape[1], f"Image shape wrong, should be of dim (3, {self.image_shape[0]}, {self.image_shape[1]}); is of dim (3, {image_height}, {image_width}) "
        batch = batch.view(batch_size, channels, self.num_patches[0], self.patch_shape[0], self.num_patches[1], self.patch_shape[1]).permute(0,2,4,1,3,5)
        batch = batch.contiguous().view(batch_size, self.num_patches[0], self.num_patches[1], channels*self.patch_shape[0]*self.patch_shape[1])
        batch = batch.view(batch_size, self.num_patches[0]*self.num_patches[1], channels*self.patch_shape[0]*self.patch_shape[1])
        batch = self.linear_projection(batch)
        batch = torch.cat((self.cls_embedding.expand(batch_size, -1, -1), batch), dim = 1)
        batch = batch.view(batch_size, -1, self.num_heads, self.per_head_dim)
        batch = self.rope(batch)
        batch = batch.view(batch_size, -1, self.num_heads*self.per_head_dim)
        batch = self.encoders(batch)
        mlp_head = self.mlp_head(batch[:, 0:1, :])
        return mlp_head


if __name__ == "__main__":
    Model = VisionTransformer(channels = 1)
    x = torch.randn(8, 1, 256, 256)
    output = Model(x)
    breakpoint()