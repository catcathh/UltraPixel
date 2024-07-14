import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from modules.speed_util import checkpoint
class Linear(torch.nn.Linear):
    def reset_parameters(self):
        return None

class Conv2d(torch.nn.Conv2d):
    def reset_parameters(self):
        return None

class AttnBlock_lrfuse_backup(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0, use_checkpoint=True):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            Linear(c_cond, c)
        )
        self.fuse_mapper = nn.Sequential(
            nn.SiLU(),
            Linear(c_cond, c)
        )
        self.use_checkpoint = use_checkpoint
       
    def forward(self, hr, lr):
        return checkpoint(self._forward, (hr, lr), self.paramters(), self.use_checkpoint)
    def _forward(self, hr, lr):
        res = hr
        hr = self.kv_mapper(rearrange(hr, 'b c h w -> b (h w ) c'))
        lr_fuse = self.attention(self.norm(lr), hr, self_attn=False) + lr

        lr_fuse = self.fuse_mapper(rearrange(lr_fuse, 'b c h w -> b (h w ) c'))
        hr = self.attention(self.norm(res), lr_fuse, self_attn=False) + res
        return hr
        
        
class AttnBlock_lrfuse(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0, kernel_size=3, use_checkpoint=True):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            Linear(c_cond, c)
        )
      
       
        self.depthwise = Conv2d(c, c , kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
             
        self.channelwise = nn.Sequential(
                    Linear(c + c, c ),
                    nn.GELU(),
                    GlobalResponseNorm(c ),
                    nn.Dropout(dropout),
                    Linear(c , c)
                )
        self.use_checkpoint = use_checkpoint
    
    
    def forward(self, hr, lr):
        return checkpoint(self._forward, (hr, lr), self.parameters(), self.use_checkpoint)
        
    def _forward(self, hr, lr):
        res = hr
        hr = self.kv_mapper(rearrange(hr, 'b c h w -> b (h w ) c'))
        lr_fuse = self.attention(self.norm(lr), hr, self_attn=False) + lr
        
        lr_fuse = torch.nn.functional.interpolate(lr_fuse.float(), res.shape[2:])
        #print('in line 65', lr_fuse.shape, res.shape)
        media = torch.cat((self.depthwise(lr_fuse), res), dim=1)
        out = self.channelwise(media.permute(0,2,3,1)).permute(0,3,1,2) + res
        
        return out        
        
        


class Attention2D(nn.Module):
    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        if self_attn:
            #print('in line 23 algong self att ', kv.shape, x.shape)

            kv = torch.cat([x, kv], dim=1)
            #if x.shape[1] > 48 * 48 and not self.training:
            #    x = x * math.sqrt(math.log(x.shape[1] , 24*24))
           
        x = self.attn(x, kv, kv, need_weights=False)[0]
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x
class Attention2D_splitpatch(nn.Module):
    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        
        #x = rearrange(x, 'b c h w -> b c (nh wh) (nw ww)', wh=24, ww=24, nh=orig_shape[-2] // 24, nh=orig_shape[-1] // 24,)
        x = rearrange(x, 'b c (nh wh) (nw ww) -> (b nh nw) (wh ww) c', wh=24, ww=24, nh=orig_shape[-2] // 24, nw=orig_shape[-1] // 24,)
        #print('in line 168', x.shape)
        #x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        if self_attn:
            #print('in line 23 algong self att ', kv.shape, x.shape)
            num = (orig_shape[-2] // 24) * (orig_shape[-1] // 24)
            kv = torch.cat([x, kv.repeat(num, 1, 1)], dim=1)
            #if x.shape[1] > 48 * 48 and not self.training:
            #    x = x * math.sqrt(math.log(x.shape[1] / math.sqrt(16), 24*24))
           
        x = self.attn(x, kv, kv, need_weights=False)[0]
        x = rearrange(x, ' (b nh nw) (wh ww) c -> b c (nh wh) (nw ww)', b=orig_shape[0], wh=24, ww=24, nh=orig_shape[-2] // 24, nw=orig_shape[-1] // 24)
        #x = x.permute(0, 2, 1).view(*orig_shape)
        
        return x
class Attention2D_extra(nn.Module):
    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True)

    def forward(self, x, kv, extra_emb=None, self_attn=False):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        num_x = x.shape[1]

       
        if extra_emb is not None:
            ori_extra_shape = extra_emb.shape
            extra_emb = extra_emb.view(extra_emb.size(0), extra_emb.size(1), -1).permute(0, 2, 1)
            x = torch.cat((x, extra_emb), dim=1)  
        if self_attn:
            #print('in line 23 algong self att ', kv.shape, x.shape)
            kv = torch.cat([x, kv], dim=1) 
        x = self.attn(x, kv, kv, need_weights=False)[0]
        img = x[:, :num_x, :].permute(0, 2, 1).view(*orig_shape)
        if extra_emb is not None:
            fix = x[:, num_x:, :].permute(0, 2, 1).view(*ori_extra_shape)
            return img, fix
        else:
            return img
class AttnBlock_extraq(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        #self.norm2 = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention2D_extra(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            Linear(c_cond, c)
        )
    # norm2 initialization in generator in init extra parameter
    def forward(self, x, kv, extra_emb=None):
        #print('in line 84', x.shape, kv.shape, self.self_attn, extra_emb if extra_emb is None else extra_emb.shape)
        #in line 84 torch.Size([1, 1536, 32, 32]) torch.Size([1, 85, 1536]) True None
        #if extra_emb is not None:

        kv = self.kv_mapper(kv)
        if extra_emb is not None:
            res_x, res_extra = self.attention(self.norm(x), kv, extra_emb=self.norm2(extra_emb), self_attn=self.self_attn)
            x = x + res_x
            extra_emb = extra_emb + res_extra
            return x, extra_emb
        else:
            x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn)
            return x
class AttnBlock_latent2ex(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            Linear(c_cond, c)
        )

    def forward(self, x, kv):
        #print('in line 84', x.shape, kv.shape, self.self_attn)
        kv = F.interpolate(kv.float(), x.shape[2:])
        kv = kv.view(kv.size(0), kv.size(1), -1).permute(0, 2, 1)
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn)
        return x

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
class AttnBlock_crossbranch(nn.Module):
    def __init__(self, attnmodule, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()
        self.attn = AttnBlock(c, c_cond, nhead, self_attn, dropout)
        #print('in line 108', attnmodule.device)
        self.attn.load_state_dict(attnmodule.state_dict())
        self.norm1 = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
       
        self.channelwise1 = nn.Sequential(
            Linear(c *2, c ),
            nn.GELU(),
            GlobalResponseNorm(c ),
            nn.Dropout(dropout),
            Linear(c, c)
        )
        self.channelwise2 = nn.Sequential(
            Linear(c *2, c ),
            nn.GELU(),
            GlobalResponseNorm(c ),
            nn.Dropout(dropout),
            Linear(c, c)
        )
        self.c = c
    def forward(self, x, kv, main_x):
        #print('in line 84', x.shape, kv.shape, main_x.shape, self.c)
        
        x = self.channelwise1(torch.cat((x, F.interpolate(main_x.float(), x.shape[2:])), dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x
        x = self.attn(x, kv)
        main_x = self.channelwise2(torch.cat((main_x, F.interpolate(x.float(), main_x.shape[2:])), dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + main_x
        return main_x, x

class GlobalResponseNorm(nn.Module):
    "from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105"
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ResBlock(nn.Module):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0, use_checkpoint =True):  # , num_heads=4, expansion=2):
        super().__init__()
        self.depthwise = Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        #         self.depthwise = SAMBlock(c, num_heads, expansion)
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            Linear(c + c_skip, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            Linear(c * 4, c)
        )
        self.use_checkpoint = use_checkpoint
    def forward(self, x, x_skip=None):
    
        if x_skip is not None:
            return checkpoint(self._forward_skip, (x, x_skip), self.parameters(), self.use_checkpoint)
        else:
            #print('in line 298', x.shape)
            return checkpoint(self._forward_woskip, (x, ), self.parameters(), self.use_checkpoint)
                    
    
    
    def _forward_skip(self, x, x_skip):
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res
    def _forward_woskip(self, x):
        x_res = x
        x = self.norm(self.depthwise(x))
       
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res

class AttnBlock(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0, use_checkpoint=True):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            Linear(c_cond, c)
        )
        self.use_checkpoint = use_checkpoint
    def forward(self, x, kv):
        return checkpoint(self._forward, (x, kv), self.parameters(), self.use_checkpoint)
    def _forward(self, x, kv):
        kv = self.kv_mapper(kv)
        res = self.attention(self.norm(x), kv, self_attn=self.self_attn)
        
        #print(torch.unique(res), torch.unique(x), self.self_attn) 
        #scale = math.sqrt(math.log(x.shape[-2] * x.shape[-1], 24*24))
        x = x + res
     
        return x
class AttnBlock_mytest(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_cond, c)
        )
 
    def forward(self, x, kv):
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn)
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, c, dropout=0.0):
        super().__init__()
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            Linear(c, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            Linear(c * 4, c)
        )

    def forward(self, x):
        x = x + self.channelwise(self.norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class TimestepBlock(nn.Module):
    def __init__(self, c, c_timestep, conds=['sca'], use_checkpoint=True):
        super().__init__()
        self.mapper = Linear(c_timestep, c * 2)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", Linear(c_timestep, c * 2))

        self.use_checkpoint = use_checkpoint
    def forward(self, x, t):
        return checkpoint(self._forward, (x, t), self.parameters(), self.use_checkpoint)
        
    def _forward(self, x, t):
        #print('in line 284', x.shape, t.shape, self.conds)
        #in line 284 torch.Size([4, 2048, 19, 29]) torch.Size([4, 192]) ['sca', 'crp']
        t = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + ac, b + bc
        return x * (1 + a) + b
