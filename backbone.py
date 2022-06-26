import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, image_size, in_channels = 3, embed_dim = 768):
        super(PatchEmbed, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.n_patches = int(image_size/patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.proj(x) # (batch_size, embed_dim, 1, 1)
        x = x.flatten(2)
        return x.transpose(1,2)

class Attention(nn.Module):
    def __init__(self, dim, n_heads, qvk_bias, attn_p = 0., proj_p = 0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv= nn.Linear(dim, dim*3, bias=qvk_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
    def forward(self, x):
        # x shape '(batch_size, n_patch+1, dim)'
        batch_size, n_tokens, dim = x.shape
        qvk = self.qkv(x) # (batch_size, n_patch+1, dim * 3)
        qvk = qvk.reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        qvk = qvk.permute(2, 0, 3, 1, 4) #(3, batch_size, n_heads, n_tokens, head_dim)
        q, k, v = qvk[0], qvk[1], qvk[2] #(batch_size, n_heads, n_patches + 1, head_dim)
        k_t = k.transpose(-2, -1) #(batch_size, n_heads, head_dim, n_patches+1)
        dp = (q @ k_t) * self.scale #(batch_size, n_heads, n_patches + 1, n_patches + 1) 
        attn = dp.softmax(dim = -1) #(batch_size, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v #(batch_size, n_head, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (batch_size, n_patches + 1, n_head, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (batch_Size, n_patches + 1, dim)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, n_heads, qvk_bias, attn_p = 0., proj_p = 0., mlp_hidden_ratio = 4, mlp_p = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps = 1e-6)
        self.attn = Attention(dim, n_heads, qvk_bias, attn_p, proj_p)
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
        self.mlp = MLP(dim, dim * mlp_hidden_ratio, dim, mlp_p)
    
    def forward(self, x): # (batch_size, n_patches + 1, dim)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size = 224, patch_size = 16, in_channels = 3, embed_dim = 384, depth = 12, n_heads = 12, qvk_bias = True, attn_p=0., proj_p=0., mlp_hidden_ratio = 4, mlp_p = 0.1, n_classes = 1000, p = 0.):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size,image_size=image_size, in_channels= in_channels, embed_dim = embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p)
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, n_heads, qvk_bias, attn_p, proj_p, mlp_hidden_ratio, mlp_p)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.head = nn.Linear(embed_dim, n_classes)
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim = 1) #(batch_size, 1 + n_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # cls_final = x[:, 0] #(batch_size, embed_dim)
        # out = self.head(cls_final)
        return x
def main():
    x = torch.rand(1, 3, 224, 224)
    model = VisionTransformer()
    # model.load_state_dict(torch.load("dino_deitsmall16_pretrain.pth"))
    print(model)
    print(model(x))
# main()