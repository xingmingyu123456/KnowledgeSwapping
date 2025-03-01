import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from torch.nn import Parameter
from IPython import embed
import math
import loralib as lora

MIN_NUM_PATCHES = 16


class Softmax(nn.Module):
    r"""Implement of Softmax (normal classification head):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                   if device_id=None, it will be trained on CPU without model parallel.
    """

    def __init__(self, in_features, out_features, device_id):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, label):
        if self.device_id == None:
            out = F.linear(x, self.weight, self.bias)
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            out = F.linear(temp_x, weight, bias)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                out = torch.cat(
                    (out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1
                )
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()




class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, lora_rank=8):
        super().__init__()
        self.net = nn.Sequential(
            lora.Linear(dim, hidden_dim, r=lora_rank),
            nn.GELU(),
            nn.Dropout(dropout),
            lora.Linear(hidden_dim, dim, r=lora_rank),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, lora_rank=8):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim**-0.5

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv = lora.MergedLinear(
            in_features=dim,
            out_features=inner_dim * 3,
            r=lora_rank,
            enable_lora=[True, True, True],
            bias=False,
        )
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        # embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout,
        lora_rank,
        up=False,
        lora_pos: str = "FFN",
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=dropout,
                                    lora_rank=(
                                        lora_rank if lora_pos == "Attention" else 0
                                    ),
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(
                                dim,
                                FeedForward(
                                    dim,
                                    mlp_dim,
                                    dropout=dropout,
                                    lora_rank=lora_rank if lora_pos == "FFN" else 0,
                                ),
                            )
                        ),
                    ]
                )
            )

        self.up = up
        self.depth = depth

    def forward(self, x, mask=None):
        if self.up:
            for i, (attn, ff) in enumerate(self.layers):
                if i < self.depth // 2:
                    continue
                x = attn(x, mask=mask)
                # embed()
                x = ff(x)
            return x
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            # embed()
            x = ff(x)
        return x


class ViT_bird(nn.Module):
    def __init__(
        self,
        *,
        loss_type,
        GPU_ID,
        num_class,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        lora_rank=8,
        lora_pos: str = "FFN",
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2
        assert (
            num_patches > MIN_NUM_PATCHES
        ), f"your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size"
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, lora_rank, lora_pos=lora_pos
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        if self.loss_type == "None":
            print("no loss for vit_face")
        else:
            if self.loss_type == "Softmax":
                self.loss = Softmax(
                    in_features=dim, out_features=num_class, device_id=self.GPU_ID
                )
            elif self.loss_type == "CosFace":
                self.loss = CosFace(
                    in_features=dim, out_features=num_class, device_id=self.GPU_ID
                )
            elif self.loss_type == "ArcFace":
                self.loss = ArcFace(
                    in_features=dim, out_features=num_class, device_id=self.GPU_ID
                )
            elif self.loss_type == "SFace":
                self.loss = SFaceLoss(
                    in_features=dim, out_features=num_class, device_id=self.GPU_ID
                )

    def forward(self, img, label=None, mask=None):
        """
        :return: output is the feature vector after FFN (or other operations for a specific loss),
        emb is the feature vector after transformer
        """
        p = self.patch_size

        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb



