import os.path as osp
import os
import datetime
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from dassl.data import DataManager
from tensorboard._vendor.webencodings import labels
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform

from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.manifold import TSNE

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root=cfg.MODEL.BACKBONE.PATH)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def ent_loss(logit):
    prob = F.softmax(logit, dim=-1)
    epsilon = 1e-5
    entropy = -prob * torch.log(prob + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    entropy = entropy.mean()

    mean_prob = prob.mean(dim=0)
    log_mean_prob = torch.log(mean_prob + epsilon)
    balance_loss = torch.sum(-mean_prob * log_mean_prob)

    entropy_loss = entropy - balance_loss
    return entropy_loss

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 2)
    feat_var = feat.var(dim=1, keepdim=True) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.mean(dim=1, keepdim=True)
    return feat_mean, feat_std

class PIN(nn.Module):
    def __init__(self, fea_dim):
        super().__init__()
        self.style_mean = nn.Parameter(
            torch.zeros(1, fea_dim),
            requires_grad=True
        )
        self.style_std = nn.Parameter(
            torch.ones(1, fea_dim),
            requires_grad=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, content_feat):
        content_mean, content_std = calc_mean_std(content_feat)
        content_feat_norm = (content_feat - content_mean) / content_std
        target_feat = content_feat_norm * self.style_std.expand_as(content_feat) + self.style_mean.expand_as(content_feat)
        target_feat = self.relu(target_feat)
        return target_feat

class ImageAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        fea_dim = 1024 if cfg.MODEL.BACKBONE.NAME == 'RN50' else 512
        self.down = nn.Linear(in_features=fea_dim, out_features=fea_dim // 4, bias=False)
        self.pin = PIN(fea_dim // 4)
        self.up = nn.Linear(in_features=fea_dim // 4, out_features=fea_dim, bias=False)

    def forward(self, x):
        out = self.down(x)
        out = self.pin(out)
        out = self.up(out)
        return out

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class ResNetImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.visual
        self.attnpool = clip_model.visual.attnpool
        self.num_heads = self.attnpool.num_heads
        self.embed_dim = self.attnpool.k_proj.in_features
        self.spacial_dim = self.encoder.input_resolution // 32
        self.relu = nn.ReLU(inplace=True)
        self.out_indices = (0, 1, 2, 3)

    @autocast()
    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.encoder.conv1, self.encoder.bn1), (self.encoder.conv2, self.encoder.bn2),
                             (self.encoder.conv3, self.encoder.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.encoder.avgpool(x)
            return x

        x = x.type(self.encoder.conv1.weight.dtype)
        x = stem(x)

        outs = []
        x = self.encoder.layer1(x)
        outs.append(x)
        x = self.encoder.layer2(x)
        outs.append(x)
        x = self.encoder.layer3(x)
        outs.append(x)
        x = self.encoder.layer4(x)
        outs.append(x)

        x = self.attnpool(x, spatial=True)
        return outs, x

    def forward_attn(self, x):
        x = self.attnpool(x, spatial=False)
        return x

class VITImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.visual

    @autocast()
    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                              device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        features.append(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        features.append(x)
        x = self.encoder.ln_post(x)
        features.append(x)
        if self.encoder.proj is not None:
            x = x @ self.encoder.proj

        return features, x

class PatchMean(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        pass

    def forward(self, x):
        return x.mean(dim=self.dim)

    pass

class ResNetAdapter(nn.Module):
    def __init__(
            self,
            output_dim=1024,
            num_classes=65
    ):
        super().__init__()
        self.adapter_1 = nn.Sequential(
                nn.Conv2d(256, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(8)
        )

        self.adapter_2 = nn.Sequential(
                nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(4)
        )

        self.adapter_3 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )

        self.adapter_4 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.adapter_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, num_classes, bias=False)
        )

    def forward(self, x):
        x1, x2, x3, x4 = x
        x1a = self.adapter_1(x1)
        x2a = self.adapter_2(x2)
        x3a = self.adapter_3(x3)
        x4a = self.adapter_4(x4)
        x6 = (0.1 * x1a + 0.2 * x2a + 0.3 * x3a + 0.4 * x4a) / 1.0
        return x6

    def forward_mlp(self, x):
        out = self.adapter_mlp(x)
        return out

class TransformerAdapter(nn.Module):
    def __init__(
            self,
            output_dim=512,
            num_classes=65
    ):
        super().__init__()
        self.adapter_1 = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.LayerNorm(768),
            nn.ReLU(inplace=True),
            PatchMean(dim=1)
        )

        self.adapter_2 = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.LayerNorm(768),
            nn.ReLU(inplace=True),
            PatchMean(dim=1)
        )

        self.adapter_3 = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.LayerNorm(768),
            nn.ReLU(inplace=True),
            PatchMean(dim=1)
        )

        self.adapter_4 = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.LayerNorm(768),
            nn.ReLU(inplace=True),
            PatchMean(dim=1)
        )

        self.adapter_mlp = nn.Sequential(
            nn.Linear(output_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes, bias=False)
        )

    def forward(self, x):
        x1, x2, x3, x4 = x
        x1a = self.adapter_1(x1)
        x2a = self.adapter_2(x2)
        x3a = self.adapter_3(x3)
        x4a = self.adapter_4(x4)
        x6 = (0.1 * x1a + 0.2 * x2a + 0.3 * x3a + 0.4 * x4a) / 1.0
        return x6

    def forward_mlp(self, x):
        out = self.adapter_mlp(x)
        return out

class Adapter(nn.Module):
    def __init__(
            self,
            cfg,
            num_classes=65
    ):
        super().__init__()
        output_dim = 1024 if cfg.MODEL.BACKBONE.NAME == 'RN50' else 512

        if cfg.MODEL.BACKBONE.NAME == 'ViT-B/16':
            self.maf = TransformerAdapter(output_dim, num_classes)
        else:
            self.maf = ResNetAdapter(output_dim, num_classes)

    def forward(self, x):
        x = self.maf(x)
        return x

    def forward_mlp(self, x):
        x = self.maf.forward_mlp(x)
        return x

class FeatureFinetuner(nn.Module):
    def __init__(
            self,
            cfg,
    ):
        super().__init__()
        fea_dim = 1024 if cfg.MODEL.BACKBONE.NAME == 'RN50' else 512
        self.adapter1 = nn.Sequential(
            nn.Linear(fea_dim, fea_dim, bias=False),
            nn.BatchNorm1d(fea_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fea_dim, fea_dim, bias=False),
        )
        self.adapter2 = nn.Sequential(
            nn.Linear(fea_dim, fea_dim, bias=False),
            nn.BatchNorm1d(fea_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fea_dim, fea_dim, bias=False),
        )
        self.adapter3 = nn.Sequential(
            nn.Linear(fea_dim, fea_dim, bias=False),
            nn.BatchNorm1d(fea_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fea_dim, fea_dim, bias=False),
        )

    def forward(self, x, y):
        xa = self.adapter1(x)
        ya = self.adapter2(y)
        xa = xa + ya
        xa = self.adapter3(xa)
        return xa

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm, bmkc->bnkc', attn, v).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead=4,
            dropout=0.1,
    ):
        super().__init__()
        self.cross_attn = Attention(d_model, nhead, qkv_bias=True, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        tgt2 = self.cross_attn(x, mem, mem)
        x = x + self.dropout1(tgt2)
        x = self.norm1(x)
        tgt2 = self.mlp(x)
        x = x + self.dropout2(tgt2)
        x = self.norm2(x)
        return x

class PromptGenerator(nn.Module):
    def __init__(
            self,
            cfg,
            transformer_width=256,
            transformer_heads=4,
            transformer_layers=2,
            visual_dim=1024,
            dropout=0.1,
            **kwargs
    ):
        super().__init__()
        visual_dim = 1024 if cfg.MODEL.BACKBONE.NAME == 'RN50' else 512
        transformer_width = visual_dim

        self.norm = nn.LayerNorm(visual_dim)

        self.layer = nn.ModuleList([
            CrossAttentionLayer(transformer_width, transformer_heads, dropout=dropout) for _ in range(transformer_layers)
        ])

        self.g_weight = nn.Sequential(
            nn.Linear(visual_dim, visual_dim, bias=False),
            nn.BatchNorm1d(visual_dim),
            nn.ReLU(inplace=True),
            nn.Linear(visual_dim, 1, bias=False),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, tgt, mem):
        x = tgt
        for layer in self.layer:
            x = layer(x, mem)

        weight = self.g_weight(mem[:, 0, :])
        weight = weight.unsqueeze(2)
        return weight * x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MODEL.N_CTX

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        domainnames = cfg.DATASET.SOURCE_DOMAINS + cfg.DATASET.TARGET_DOMAINS
        domainnames = [
            ", a {} image.".format(domain) for domain in domainnames
        ]

        n_dm = len(cfg.DATASET.SOURCE_DOMAINS) + len(cfg.DATASET.TARGET_DOMAINS)
        n_dmx = cfg.TRAINER.MODEL.N_DMX
        n = n_dmx + n_ctx
        self.n_dm = n_dm
        self.n_dmx = n_dmx

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        naive_prompt_prefix = f'a {cfg.DATASET.TARGET_DOMAINS[0]} photo of a'.replace("_", " ")

        if cfg.TRAINER.MODEL.CSC is True:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing generic contexts")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n)
        print("ctx vector size: {}".format(ctx_vectors.size()))
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        domain_vectors = torch.empty(n_dm, n_dmx, ctx_dim, dtype=dtype)
        nn.init.normal_(domain_vectors, std=0.02)
        self.domain_vectors = nn.Parameter(domain_vectors)  # to be optimized

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of domain context words (tokens): {n_dmx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        naive_prompts = [naive_prompt_prefix + " " + name + "." for name in classnames]

        prompts = [prompt_prefix + " " + name + "" + domain + "."
                   for domain in domainnames for name in classnames
                    ]
        print('Prompts: {}'.format({prompts[0]}))
        print("Naive prompt: {}".format(naive_prompts[0]))

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        naive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in naive_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            naive_embedding = clip_model.token_embedding(naive_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # tokenized_prompts = torch.cat([tokenized_prompts, naive_tokenized_prompts])
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.MODEL.CSC
        self.tokenized_prompts = tokenized_prompts
        self.naive_tokenized_prompts = naive_tokenized_prompts
        self.name_lens = name_lens
        self.naive_embedding = naive_embedding
        self.adapter = Adapter(cfg, n_cls)
        self.finetuner = FeatureFinetuner(cfg)
        self.prompt_generator = PromptGenerator(cfg)
        self.gamma_v = nn.Parameter(torch.ones(1) * 0.01)
        self.gamma_t = nn.Parameter(torch.ones(1) * 0.01)

    @autocast()
    def forward(self):
        ctx = self.ctx
        ctx_dim = ctx.size(-1)
        dmx = self.domain_vectors
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1)    # dm 16 512
            if not self.csc:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)   # dm cls 16 512
        else:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1, -1)    # dm cls 16 512

        dmx = dmx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)   # dm cls 16 512
        ctxdmx = torch.cat([ctx, dmx],
                              dim=2).reshape(self.n_cls * self.n_dm,
                                             self.n_ctx + self.n_dmx, ctx_dim)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,
                ctxdmx,
                suffix
            ],
            dim=1,
        )

        return prompts

class DomainAgnosticPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MODEL.N_CTX

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        n_dm = len(cfg.DATASET.SOURCE_DOMAINS) + len(
            cfg.DATASET.TARGET_DOMAINS)  # number of domains
        n_lencls = cfg.TRAINER.MODEL.N_CLS

        n = n_ctx  # number of learnable tokens
        self.n_dm = n_dm
        self.n_lencls = n_lencls
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        naive_prompt_prefix = f'a {cfg.DATASET.TARGET_DOMAINS[0]} photo of a'.replace("_", " ")
        # print(naive_prompt_prefix_len)
        # define the learnable prompt
        if cfg.TRAINER.MODEL.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        print("ctx vectors size: ".format(ctx_vectors.size()))

        self.gamma_t = nn.Parameter(torch.ones(1) * 0.01)
        self.gamma_v = nn.Parameter(torch.ones(1) * 0.01)
        prompt_prefix = " ".join(["X"] * n)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of cls words (tokens): {n_lencls}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        naive_prompts = [
            naive_prompt_prefix + " " + name + "." for name in classnames
        ]
        prompts = [
            prompt_prefix + " " + name + "." for name in classnames
        ]
        print(f'Prompts: "{prompts[0]}"')
        print(f'Naive Prompts: "{naive_prompts[0]}"')
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        naive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in naive_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # cls, 77, 512
            naive_embedding = clip_model.token_embedding(naive_tokenized_prompts).type(dtype)  # cls, 77, 512

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # tokenized_prompts = torch.cat(
        #     [tokenized_prompts, naive_tokenized_prompts])
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.MODEL.CSC
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.naive_tokenized_prompts = naive_tokenized_prompts
        self.name_lens = name_lens
        self.naive_embedding = naive_embedding
        self.prompt_generator = PromptGenerator(cfg)
        self.image_adapter = ImageAdapter(cfg)

    @autocast()
    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # cls 16 512, broadcast to all classes
        prompts = torch.cat([
            prefix,  # (n_cls, 1, dim)
            ctx,  # (n_cls, n_ctx, dim)
            suffix,  # (n_cls, *, dim)
        ],
            dim=1)

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, update_txt=True, update_vis=True):
        super().__init__()
        self.prompt_learner = DomainAgnosticPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = VITImageEncoder(
            clip_model) if cfg.MODEL.BACKBONE.NAME == 'ViT-B/16' else ResNetImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = self.prompt_learner.n_cls

        self.naive_text_embedding = (self.text_encoder(self.prompt_learner.naive_embedding,
                                                       self.prompt_learner.naive_tokenized_prompts)
                                     .to(torch.device("cuda")))  # naive_text_embeddings
        self.update_txt = update_txt
        self.update_vis = update_vis

    @autocast()
    def forward(self, image, pse=False, pin=False, fea=False):
        multi_features, image_features = self.image_encoder(image.type(self.dtype))
        visual_embeddings = image_features[:, 0, :]  # B, C
        B, HW, C = image_features.shape

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        logit_scale = self.logit_scale.exp()

        text_features = text_features.expand(B, -1, -1)

        if self.update_txt:
            text_diff = self.prompt_learner.prompt_generator(text_features, image_features)
            updated_text_features = text_features + text_diff
        else:
            updated_text_features = text_features

        if self.update_vis:
            visual_diff = self.prompt_learner.image_adapter(visual_embeddings)
            updated_visual_features = visual_embeddings + self.prompt_learner.gamma_v * visual_diff
        else:
            updated_visual_features = visual_embeddings

        visual = updated_visual_features / updated_visual_features.norm(dim=-1, keepdim=True)
        text = F.normalize(updated_text_features, p=2, dim=-1)

        return_all = []

        logits = logit_scale * torch.einsum("bc, bkc->bk", visual, text)
        return_all.append(logits)

        if pse:
            ori_vis = visual_embeddings
            ori_vis = ori_vis / ori_vis.norm(dim=-1, keepdim=True)
            nav_txt = self.naive_text_embedding
            nav_txt = nav_txt / nav_txt.norm(dim=-1, keepdim=True)
            pse_logits = ori_vis @ nav_txt.t()
            pse_logits = logit_scale * pse_logits
            return_all.append(pse_logits)
        if pin:
            pin_logit = torch.einsum('bc, bkc->bk', visual, text).mean(dim=-1)
            pin_loss = (1 - pin_logit).mean()
            return_all.append(pin_loss)
        if fea:
            ori_visual = visual_embeddings / visual_embeddings.norm(dim=-1, keepdim=True)
            return_all.append(ori_visual)
            return_all.append(visual)
        return return_all

@TRAINER_REGISTRY.register()
class MODEL(TrainerXU):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MODEL.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.MODEL.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        self.dm = DataManager(cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
        self.lab2cname = self.dm.lab2cname

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        input_x2 = batch_x["img2"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]
        input_u2 = batch_u["img2"]
        # label_u is used only for evaluating pseudo labels' accuracy
        label_u = batch_u["label"]

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)
        label_u = label_u.to(self.device)

        return input_x, input_x2, label_x, input_u, input_u2, label_u

    def parse_batch_train2(self, batch_u):
        input_u = batch_u["img"]
        input_u2 = batch_u["img2"]
        label_u = batch_u["label"]

        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)
        label_u = label_u.to(self.device)

        return input_u, input_u2, label_u

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MODEL.PREC == "fp32" or cfg.TRAINER.MODEL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, update_txt=cfg.TRAINER.MODEL.TEXT, update_vis=cfg.TRAINER.MODEL.VISUAL)

        self.n_cls = self.model.prompt_learner.n_cls
        self.dtype = clip_model.dtype

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.optim2 = build_optimizer(self.model.prompt_learner, cfg.OPTIM_V)
        self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM_V)

        '''
        register model could be updated. When new module needs to be updated
        register the module before use
        '''
        self.register_model("prompt_learner", self.model.prompt_learner,
                            self.optim, self.sched)
        self.register_model("prompt_learner2", self.model.prompt_learner,
                            self.optim2, self.sched2)

        self.scaler = GradScaler() if cfg.TRAINER.MODEL.PREC == "amp" else None

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def train(self):
        """Generic training loops."""
        self.max_epoch = self.max_epoch + self.cfg.OPTIM_V.MAX_EPOCH
        self.before_train()
        if self.start_epoch < self.max_epoch - self.cfg.OPTIM_V.MAX_EPOCH:
            for self.epoch in range(self.start_epoch, self.max_epoch - self.cfg.OPTIM_V.MAX_EPOCH):
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()
            for self.epoch in range(self.max_epoch - self.cfg.OPTIM_V.MAX_EPOCH, self.max_epoch):
                self.before_epoch()
                self.run_epoch2()
                self.after_epoch()
        elif self.max_epoch - self.cfg.OPTIM_V.MAX_EPOCH <= self.start_epoch <= self.max_epoch:
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch2()
                self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.threshold = self.cfg.TRAINER.MODEL.TAU
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x if self.cfg.DATASET.NAME == "OfficeHome" else 500
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr(names='prompt_learner'),
                    ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch_x, batch_u=None):
        # label_u only used for matric
        image_x, image_x2, label, image_u, image_u2, label_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.MODEL.PREC
        if prec == "amp":
            with (autocast()):
                output_x, pin_loss_x = self.model(image_x, pin=True)
                output_u, pse_logits, pin_loss_u = self.model(image_u, pse=True, pin=True)
                output_u2 = self.model(image_u2)[0]

                pseudo_label = (0.5 * torch.softmax(output_u.reshape(-1, self.n_cls) / self.cfg.TRAINER.MODEL.T,
                                                    dim=-1) + 0.5 * torch.softmax(
                    pse_logits.reshape(-1, self.n_cls) / self.cfg.TRAINER.MODEL.T, dim=-1)).detach()

                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.threshold).float()

                loss_x = F.cross_entropy(output_x, label)
                loss_u = torch.tensor(0.0).cuda() if mask.sum() == 0 else (F.cross_entropy(output_u, label_p,
                                                                                           reduction="none") * mask).sum() / mask.sum()
                loss_u2 = torch.tensor(0.0).cuda() if mask.sum() == 0 else (F.cross_entropy(output_u2, label_p,
                                                                                            reduction="none") * mask).sum() / mask.sum()
                loss_ent = ent_loss(output_u)

                loss = loss_x + self.cfg.TRAINER.MODEL.U * (loss_u + loss_u2) + self.cfg.TRAINER.MODEL.ENT * loss_ent + self.cfg.TRAINER.MODEL.PIN * (pin_loss_x + pin_loss_u)

                self.optim.zero_grad()
                self.optim2.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "acc_x":
            compute_accuracy(output_x, label)[0].item(),
            "gamma": self.model.prompt_learner.gamma_v,
        }

        self.update_lr(names="prompt_learner")
        return loss_summary

    def run_epoch2(self):
        self.threshold = self.cfg.TRAINER.MODEL.TAU
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        len_train_loader_u = len(self.train_loader_u)
        self.num_batches = len_train_loader_u if self.cfg.DATASET.NAME == "OfficeHome" else 500

        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward2(batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr(names='prompt_learner2'),
                    ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward2(self, batch_u):
        image_u, image_u2, label_u = self.parse_batch_train2(batch_u)
        prec = self.cfg.TRAINER.MODEL.PREC
        if prec == "amp":
            with (autocast()):
                output_u, pse_logits, pin_loss_u = self.model(image_u, pse=True, pin=True)
                output_u2 = self.model(image_u2)[0]

                pseudo_label = (0.5 * torch.softmax(output_u.reshape(-1, self.n_cls) / self.cfg.TRAINER.MODEL.T,
                                                    dim=-1) + 0.5 * torch.softmax(
                    pse_logits.reshape(-1, self.n_cls) / self.cfg.TRAINER.MODEL.T, dim=-1)).detach()

                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.threshold).float()

                loss_u = torch.tensor(0.0).cuda() if mask.sum() == 0 else (F.cross_entropy(output_u, label_p, reduction="none") * mask).sum() / mask.sum()
                loss_u2 = torch.tensor(0.0).cuda() if mask.sum() == 0 else (F.cross_entropy(output_u2, label_p, reduction="none") * mask).sum() / mask.sum()
                loss_ent = ent_loss(output_u)

                loss = self.cfg.TRAINER.MODEL.U * (loss_u + loss_u2) + self.cfg.TRAINER.MODEL.ENT * loss_ent

                self.optim.zero_grad()
                self.optim2.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim2)
                self.scaler.update()

        loss_summary = {
            "loss": loss.item(),
            "loss_u": loss_u.item(),
            "loss_u2": loss_u2.item(),
            "loss_ent": loss_ent.item(),
            "acc_u": compute_accuracy(pseudo_label, label_u)[0].item()
        }
        self.update_lr(names="prompt_learner2")
        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch,
                                self.output_dir,
                                model_name="model-best.pth.tar")

            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'MODEL not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)[0]
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all


    # @torch.no_grad()
    # def test(self, split=None):
    #     """A generic testing pipeline."""
    #     self.set_model_mode("eval")
    #     self.evaluator.reset()
    #     if split is None:
    #         split = self.cfg.TEST.SPLIT
    #
    #     data_loader = self.test_loader
    #     print("Do evaluation on test set")
    #
    #     ori_visuals = []
    #     visuals = []
    #     labels = []
    #     for batch_idx, batch in enumerate(data_loader):
    #         input, label = self.parse_batch_test(batch)
    #         output, ori_visual, visual = self.model(input, fea=True)
    #         ori_visuals.extend(ori_visual.cpu().numpy())
    #         visuals.extend(visual.cpu().numpy())
    #         labels.extend(label.cpu().numpy())
    #
    #     ori_visuals = np.array(ori_visuals)
    #     visuals = np.array(visuals)
    #     tsen = TSNE(n_components=2, random_state=42)
    #     clip_tsne = tsen.fit_transform(ori_visuals)
    #     model_tsne = tsen.fit_transform(visuals)
    #
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    #     colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
    #     cmap = ListedColormap(colors)
    #
    #     unique_labels = np.unique(labels)
    #     for label in unique_labels:
    #         mask = labels == label
    #         ax1.scatter(
    #             clip_tsne[mask, 0],
    #             clip_tsne[mask, 1],
    #             label=f'{label}',
    #             color=cmap(label),
    #             alpha=0.7,
    #             s=50,  # 点的大小
    #             edgecolors='w',  # 点边缘颜色
    #             linewidth=0.5
    #         )
    #     ax1.set_title('CLIP', fontsize=40)
    #     ax1.set_xticks([])
    #     ax1.set_yticks([])
    #
    #     for label in unique_labels:
    #         mask = labels == label
    #         ax2.scatter(
    #             model_tsne[mask, 0],
    #             model_tsne[mask, 1],
    #             label=f'{label}',
    #             color=cmap(label),
    #             alpha=0.7,
    #             s=50,
    #             edgecolors='w',
    #             linewidth=0.5
    #         )
    #     ax2.set_title('DualUDA', fontsize=40)
    #     ax2.set_xticks([])
    #     ax2.set_yticks([])
    #
    #     plt.tight_layout()
    #     plt.savefig(f'tsen.pdf', dpi=300)
    #     plt.close()

    # @torch.no_grad()
    # def test(self, split=None):
    #     """A generic testing pipeline."""
    #     self.set_model_mode("eval")
    #     self.evaluator.reset()
    #     if split is None:
    #         split = self.cfg.TEST.SPLIT
    #
    #     data_loader_x = self.train_loader_x
    #     data_loader_u = self.train_loader_u
    #     print("Drawing t-SNE")
    #
    #     src_ori_visuals = []
    #     src_visuals = []
    #     for batch_idx, batch in enumerate(data_loader_x):
    #         input, label = self.parse_batch_test(batch)
    #         output, ori_visual, visual = self.model(input, fea=True)
    #         src_ori_visuals.extend(ori_visual.cpu().numpy())
    #         src_visuals.extend(visual.cpu().numpy())
    #
    #     tar_ori_visuals = []
    #     tar_visuals = []
    #     for batch_idx, batch in enumerate(data_loader_u):
    #         input, label = self.parse_batch_test(batch)
    #         output, ori_visual, visual = self.model(input, fea=True)
    #         tar_ori_visuals.extend(ori_visual.cpu().numpy())
    #         tar_visuals.extend(visual.cpu().numpy())
    #
    #     clip_all_embeddings = np.concatenate([src_ori_visuals, tar_ori_visuals], axis=0)
    #     model_all_embeddings = np.concatenate([src_visuals, tar_visuals], axis=0)
    #
    #     tsen = TSNE(n_components=2, random_state=42)
    #     clip_tsne = tsen.fit_transform(clip_all_embeddings)
    #     model_tsne = tsen.fit_transform(model_all_embeddings)
    #
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    #
    #     src_clip_tsne = clip_tsne[:len(src_ori_visuals)]
    #     tar_clip_tsne = clip_tsne[len(src_ori_visuals):]
    #     ax1.scatter(src_clip_tsne[:, 0], src_clip_tsne[:, 1], c='blue')
    #     ax1.scatter(tar_clip_tsne[:, 0], tar_clip_tsne[:, 1], c='red')
    #     ax1.set_title('CLIP', fontsize=40)
    #     ax1.set_xticks([])
    #     ax1.set_yticks([])
    #
    #     src_model_tsne = model_tsne[:len(src_visuals)]
    #     tar_model_tsne = model_tsne[len(src_visuals):]
    #     ax2.scatter(src_model_tsne[:, 0], src_model_tsne[:, 1], c='blue')
    #     ax2.scatter(tar_model_tsne[:, 0], tar_model_tsne[:, 1], c='red')
    #     ax2.set_title('DualUDA', fontsize=40)
    #     ax2.set_xticks([])
    #     ax2.set_yticks([])
    #
    #     plt.tight_layout()
    #     plt.savefig(f'tsen_da.pdf', dpi=300)
    #     plt.close()

#     @torch.no_grad()
#     def test(self, split=None):
#         """A generic testing pipeline."""
#         self.set_model_mode("eval")
#         self.evaluator.reset()
#
#         all_labels = []
#         all_preds = []
#
#         if split is None:
#             split = self.cfg.TEST.SPLIT
#
#         data_loader = self.test_loader
#         print("Do evaluation on test set")
#
#         for batch_idx, batch in enumerate(data_loader):
#             input, label = self.parse_batch_test(batch)
#             output = self.model_inference(input)[0]
#
#             _, preds = torch.max(output, dim=1)
#             all_labels.extend(label.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#
#             self.evaluator.process(output, label)
#
#         results = self.evaluator.evaluate()
#         for k, v in results.items():
#             tag = "{}/{}".format(split, k)
#             self.write_scalar(tag, v, self.epoch)
#
#         results_all = results["accuracy"]
#         plot_confusion_matrix(all_labels, all_preds, split)
#
#         return results_all
#
#
# def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
#                           max_classes=30, show_values=False, figsize_per_class=0.15):
#     """
#     绘制混淆矩阵（支持多类别场景）
#     y_true: 真实标签列表
#     y_pred: 预测标签列表
#     title: 图表标题
#     max_classes: 最大显示类别数，超过时启用分块显示
#     """
#     classes = sorted(list(set(y_true)))
#     n_classes = len(classes)
#
#     if n_classes == 0:
#         return
#
#     # 生成混淆矩阵
#     cm = confusion_matrix(y_true, y_pred, labels=classes)
#     cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 行归一化
#
#     # 多类别情况：分块绘制或显示关键混淆
#     plot_large_confusion(cm_norm, classes, title, show_values, figsize_per_class)
#
#
# def plot_large_confusion(cm, classes, title, show_values, figsize_per_class):
#     """处理多类别情况的混淆矩阵可视化"""
#     n_classes = len(classes)
#     print(f"类别数较多 ({n_classes} 类)，显示关键混淆信息...")
#
#     # 1. 提取最关键的混淆对（Top 20）
#     key_confusions = extract_key_confusions(cm, classes, top_n=20)
#     print("\n=== Top 20 Class Confusions ===")
#     for i, (true_cls, pred_cls, rate, count) in enumerate(key_confusions):
#         print(f"{i + 1}. True: {true_cls}, Pred: {pred_cls}, "
#               f"Confusion Rate: {rate * 100:.2f}%, Count: {count}")
#
#     # 2. 绘制关键混淆的热力图
#     if len(key_confusions) > 0:
#         plot_key_confusions(key_confusions, title, show_values)
#
#     # 3. 生成简化版完整混淆矩阵（不含数值）
#     plot_simplified_full_confusion(cm, classes, title, figsize_per_class)
#
#
# def extract_key_confusions(cm, classes, top_n=20):
#     """提取最关键的混淆对"""
#     key_confusions = []
#     n_classes = len(classes)
#
#     for i in range(n_classes):
#         for j in range(n_classes):
#             if i != j and cm[i, j] > 0:
#                 # 记录：(真实类别, 预测类别, 混淆率, 混淆次数)
#                 key_confusions.append((classes[i], classes[j], cm[i, j], cm[i, j] * cm[i].sum()))
#
#     # 按混淆率降序排序
#     key_confusions.sort(key=lambda x: x[2], reverse=True)
#     return key_confusions[:top_n]
#
#
# def plot_key_confusions(confusions, title, show_values):
#     """绘制关键混淆对的热力图"""
#     if not confusions:
#         return
#
#     # 提取唯一类别
#     all_classes = sorted(list(set([c[0] for c in confusions] + [c[1] for c in confusions])))
#     class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
#     n_classes = len(all_classes)
#
#     # 构建关键混淆矩阵
#     key_cm = np.zeros((n_classes, n_classes))
#     for true_cls, pred_cls, rate, _ in confusions:
#         i, j = class_to_idx[true_cls], class_to_idx[pred_cls]
#         key_cm[i, j] = rate
#
#     plt.figure(figsize=(min(12, n_classes * 0.8), min(10, n_classes * 0.8)))
#     fmt = '.2f' if show_values else ''
#     annot = key_cm if show_values else False
#
#     ax = sns.heatmap(key_cm, annot=annot, fmt=fmt, cmap='Reds',
#                      xticklabels=all_classes, yticklabels=all_classes,
#                      cbar_kws={'label': 'Confusion Rate'})
#
#     plt.title(f'Key Confusions ({title})', fontsize=15)
#     plt.xlabel('Predicted Label', fontsize=12)
#     plt.ylabel('True Label', fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f'key_confusions_{title}.png', dpi=300)
#     plt.close()
#
#
# def plot_simplified_full_confusion(cm, classes, title, figsize_per_class):
#     """生成简化版完整混淆矩阵（不含数值，适合大量类别）"""
#     n_classes = len(classes)
#     figsize = (min(20, n_classes * figsize_per_class * 1.2),
#                min(18, n_classes * figsize_per_class * 1.2))
#
#     plt.figure(figsize=figsize)
#
#     # 绘制不带数值的热力图
#     ax = sns.heatmap(cm, annot=False, cmap='Blues',
#                      xticklabels=classes, yticklabels=classes,
#                      cbar_kws={'label': 'Normalized Confusion Rate'})
#
#     # 设置标题和标签
#     plt.title(f'Simplified Full Confusion Matrix ({title})', fontsize=15)
#     plt.xlabel('Predicted Label', fontsize=12)
#     plt.ylabel('True Label', fontsize=12)
#
#     # 优化标签旋转和布局
#     plt.xticks(rotation=90, ha='right', rotation_mode='anchor')
#     plt.tight_layout()
#
#     # 保存图像
#     plt.savefig(f'simplified_full_{title}.png', dpi=300)
#     plt.close()