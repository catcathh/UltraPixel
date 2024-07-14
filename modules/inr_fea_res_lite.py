import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import models
from modules.common_ckpt import Linear, Conv2d, AttnBlock, ResBlock, LayerNorm2d
#from modules.common_ckpt import AttnBlock,
from einops import rearrange
import torch.fft as fft
from modules.speed_util import checkpoint
def batched_linear_mm(x, wb):
    # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    return torch.matmul(torch.cat([x, one], dim=-1), wb)
def make_coord_grid(shape, range, device=None):
    """
        Args:
            shape: tuple
            range: [minv, maxv] or [[minv_1, maxv_1], ..., [minv_d, maxv_d]] for each dim
        Returns:
            grid: shape (*shape, )
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    return grid
def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()
    
def init_wb_rewrite(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    
    torch.nn.init.xavier_uniform_(weight)

    bias = torch.empty(shape[1], 1)
    torch.nn.init.xavier_uniform_(bias)
   

    return torch.cat([weight, bias], dim=1).t().detach()
class HypoMlp(nn.Module):

    def __init__(self, depth, in_dim, out_dim, hidden_dim, use_pe, pe_dim, out_bias=0, pe_sigma=1024):
        super().__init__()
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.depth = depth
        self.param_shapes = dict()
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        for i in range(depth):  # for each layer the weight
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f'wb{i}'] = (last_dim + 1, cur_dim)
            last_dim = cur_dim
        self.relu = nn.ReLU()
        self.params = None
        self.out_bias = out_bias
        
    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def forward(self, x):
        B, query_shape = x.shape[0], x.shape[1: -1]
        x = x.view(B, -1, x.shape[-1])
        if self.use_pe:
            x = self.convert_posenc(x)
            #print('in line 79 after pos embedding', x.shape)
        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f'wb{i}'])
            if i < self.depth - 1:
                x = self.relu(x)
            else:
                x = x + self.out_bias
        x = x.view(B, *query_shape, -1)
        return x



class Attention(nn.Module):

    def __init__(self, dim, n_head, head_dim, dropout=0.):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q = nn.Sequential(
            nn.SiLU(),
            Linear(dim, inner_dim ))
        self.to_kv = nn.Sequential(
            nn.SiLU(),
            Linear(dim, inner_dim * 2))
        self.scale = head_dim ** -0.5
        # self.to_out = nn.Sequential(
        #     Linear(inner_dim, dim),
        #     nn.Dropout(dropout),
        # )

    def forward(self, fr, to=None):
        if to is None:
            to = fr
        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1) # b h n n
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return out


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim, dropout=0.):
        super().__init__()
       
        self.net = nn.Sequential(
            Linear(dim, ff_dim),
            nn.GELU(),
            #GlobalResponseNorm(ff_dim),
            nn.Dropout(dropout),
            Linear(ff_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


#TransInr(ind=2048, ch=256, n_head=16, head_dim=16, n_groups=64, f_dim=256, time_dim=self.c_r, t_conds = [])
class TransformerEncoder(nn.Module):

    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        for norm_attn, norm_ff in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ff(x)
        return x
class ImgrecTokenizer(nn.Module):

    def __init__(self, input_size=32*32, patch_size=1, dim=768, padding=0, img_channels=16):
        super().__init__()
        
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels, dim)
        
        self.posemb = nn.Parameter(torch.randn(input_size, dim))

    def forward(self, x):
        #print(x.shape)
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding) # (B, C * p * p, L)
        #print('in line 185 after unfoding', x.shape)
        x = x.permute(0, 2, 1).contiguous()
        ttt = self.prefc(x)
        
        x = self.prefc(x) + self.posemb[:x.shape[1]].unsqueeze(0)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class TimestepBlock_res(nn.Module):
    def __init__(self, c, c_timestep, conds=['sca']):
        super().__init__()
        
        self.mapper = Linear(c_timestep, c * 2)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", Linear(c_timestep, c * 2))

    
    
    
    def forward(self, x, t):
        #print(x.shape, t.shape, self.conds, 'in line 269')
        t = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + ac, b + bc
        return x * (1 + a) + b
        
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
        

                
class ScaleNormalize_res(nn.Module):
    def __init__(self, c, scale_c, conds=['sca']):
        super().__init__()
        self.c_r = scale_c
        self.mapping = TimestepBlock_res(c, scale_c, conds=conds)
        self.t_conds = conds
        self.alpha = nn.Conv2d(c, c, kernel_size=1)
        self.gamma = nn.Conv2d(c, c, kernel_size=1)
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        
    
    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb
    def forward(self, x, std_size=24*24):
        scale_val = math.sqrt(math.log(x.shape[-2] * x.shape[-1], std_size))
        scale_val = torch.ones(x.shape[0]).to(x.device)*scale_val
        scale_val_f = self.gen_r_embedding(scale_val)
        for c in self.t_conds:
            t_cond = torch.zeros_like(scale_val)
            scale_val_f = torch.cat([scale_val_f, self.gen_r_embedding(t_cond)], dim=1)
       
        f = self.mapping(x, scale_val_f)
    
        return f + x
        

class TransInr_withnorm(nn.Module):

    def __init__(self, ind=2048, ch=16, n_head=12, head_dim=64, n_groups=64, f_dim=768, time_dim=2048, t_conds=[]):
        super().__init__()
        self.input_layer=  nn.Conv2d(ind, ch, 1)
        self.tokenizer = ImgrecTokenizer(dim=ch, img_channels=ch)
        #self.hyponet = HypoMlp(depth=12, in_dim=2, out_dim=ch, hidden_dim=f_dim, use_pe=True, pe_dim=128)
        #self.transformer_encoder = TransformerEncoder(dim=f_dim, depth=12, n_head=n_head, head_dim=f_dim // n_head, ff_dim=3*f_dim, )

        self.hyponet = HypoMlp(depth=2, in_dim=2, out_dim=ch, hidden_dim=f_dim, use_pe=True, pe_dim=128)
        self.transformer_encoder = TransformerEncoder(dim=f_dim, depth=1, n_head=n_head, head_dim=f_dim // n_head, ff_dim=f_dim)
        #self.transformer_encoder = TransInr( ch=ch, n_head=16, head_dim=16, n_groups=64, f_dim=ch, time_dim=time_dim, t_conds = [])
        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        for name, shape in self.hyponet.param_shapes.items():
            self.base_params[name] = nn.Parameter(init_wb(shape))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(f_dim),
                nn.Linear(f_dim, shape[0] - 1),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, f_dim))
        self.output_layer=  nn.Conv2d(ch, ind, 1)
       
        
        self.mapp_t = TimestepBlock_res( ind, time_dim, conds = t_conds)

        
        self.hr_norm = ScaleNormalize_res(ind, 64, conds=[])
         
        self.normalize_final = nn.Sequential(
            LayerNorm2d(ind, elementwise_affine=False, eps=1e-6),
        )
        
        self.toout = nn.Sequential(
        Linear( ind*2,  ind // 4),
        nn.GELU(),
        Linear( ind // 4,  ind)
        )
        self.apply(self._init_weights)
        
        mask = torch.zeros((1, 1, 32, 32))
        h, w = 32, 32
        center_h, center_w = h // 2, w // 2
        low_freq_h, low_freq_w = h // 4, w // 4  
        mask[:, :, center_h-low_freq_h:center_h+low_freq_h, center_w-low_freq_w:center_w+low_freq_w] = 1
        self.mask = mask
        
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  
        #nn.init.constant_(self.last.weight, 0)
    def adain(self, feature_a, feature_b):
        norm_mean = torch.mean(feature_a, dim=(2, 3), keepdim=True)
        norm_std = torch.std(feature_a, dim=(2, 3), keepdim=True)
        #feature_a = F.interpolate(feature_a, feature_b.shape[2:])
        feature_b = (feature_b - feature_b.mean(dim=(2, 3), keepdim=True)) / (1e-8 + feature_b.std(dim=(2, 3), keepdim=True)) * norm_std + norm_mean
        return  feature_b 
    def forward(self, target_shape, target, dtokens, t_emb):
        #print(target.shape, dtokens.shape, 'in line 290')
        hlr, wlr = dtokens.shape[2:]
        original = dtokens
        
        dtokens = self.input_layer(dtokens)
        dtokens = self.tokenizer(dtokens)
        B = dtokens.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        #print(wtokens.shape, dtokens.shape)
        trans_out = self.transformer_encoder(torch.cat([dtokens, wtokens], dim=1))
        trans_out = trans_out[:, -len(self.wtokens):, :]

        params = dict()
        for name, shape in self.hyponet.param_shapes.items():
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng[name]
            x = self.wtoken_postfc[name](trans_out[:, l: r, :])
            x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
            w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)

            wb = torch.cat([w, b], dim=1)
            params[name] = wb
        coord = make_coord_grid(target_shape[2:], (-1, 1), device=dtokens.device)
        coord = einops.repeat(coord, 'h w d -> b h w d', b=dtokens.shape[0])
        self.hyponet.set_params(params)
        ori_up = F.interpolate(original.float(), target_shape[2:])
        hr_rec = self.output_layer(rearrange(self.hyponet(coord), 'b h w c -> b c h w'))  + ori_up
        #print(hr_rec.shape, target.shape, torch.cat((hr_rec, target), dim=1).permute(0, 2, 3, 1).shape, 'in line 537')
       
        output = self.toout(torch.cat((hr_rec, target), dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #print(output.shape, 'in line 540')
        #output = self.last(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)* 0.3
        output = self.mapp_t(output, t_emb)
        output  = self.normalize_final(output)
        output = self.hr_norm(output)
        #output = self.last(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #output = self.mapp_t(output, t_emb)
        #output = self.weight(output) * output
        
        return output 






class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

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



if __name__ == '__main__':
    #ef __init__(self, ch, n_head, head_dim, n_groups):
    trans_inr = TransInr(16, 24, 32, 64).cuda()
    input = torch.randn((1, 16, 24, 24)).cuda()
    source = torch.randn((1, 16, 16, 16)).cuda()
    t = torch.randn((1, 128)).cuda()
    output, hr = trans_inr(input, t, source)
    
    total_up = sum([ param.nelement()  for param in trans_inr.parameters()])
    print(output.shape, hr.shape, total_up /1e6 )
   
