import torch
from torch import nn
import numpy as np
import math
from modules.common_ckpt import AttnBlock, LayerNorm2d, ResBlock, FeedForwardBlock, TimestepBlock
from .controlnet import ControlNetDeliverer
import torch.nn.functional as F
from modules.inr_fea_res_lite import  TransInr_withnorm as TransInr
from modules.inr_fea_res_lite import ScaleNormalize_res
from einops import rearrange
import torch.fft as fft
import random 
class UpDownBlock2d(nn.Module):
    def __init__(self, c_in, c_out, mode, enabled=True):
        super().__init__()
        assert mode in ['up', 'down']
        interpolation = nn.Upsample(scale_factor=2 if mode == 'up' else 0.5, mode='bilinear',
                                    align_corners=True) if enabled else nn.Identity()
        mapping = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == 'up' else [mapping, interpolation])

    def forward(self, x):
        for block in self.blocks:
            x = block(x.float())
        return x
def ada_in(a, b):
    mean_a = torch.mean(a, dim=(2, 3), keepdim=True)
    std_a = torch.std(a, dim=(2, 3), keepdim=True)
    
    mean_b = torch.mean(b, dim=(2, 3), keepdim=True)
    std_b = torch.std(b, dim=(2, 3), keepdim=True)
    
    return (b - mean_b) / (1e-8 + std_b) * std_a + mean_a
def feature_dist_loss(x1, x2):
    mu1 = torch.mean(x1, dim=(2, 3))
    mu2 = torch.mean(x2, dim=(2, 3))
    
    std1 = torch.std(x1, dim=(2, 3))
    std2 = torch.std(x2, dim=(2, 3))
    std_loss = torch.mean(torch.abs(torch.log(std1+ 1e-8) - torch.log(std2+ 1e-8)))
    mean_loss = torch.mean(torch.abs(mu1 - mu2))
    #print('in line 36', std_loss, mean_loss)
    return std_loss +  mean_loss*0.1
class StageC(nn.Module):
    def __init__(self, c_in=16, c_out=16, c_r=64, patch_size=1, c_cond=2048, c_hidden=[2048, 2048], nhead=[32, 32],
                 blocks=[[8, 24], [24, 8]], block_repeat=[[1, 1], [1, 1]], level_config=['CTA', 'CTA'],
                 c_clip_text=1280, c_clip_text_pooled=1280, c_clip_img=768, c_clip_seq=4, kernel_size=3,
                 dropout=[0.1, 0.1], self_attn=True, t_conds=['sca', 'crp'], switch_level=[False],
                 lr_h=24, lr_w=24):
        super().__init__()
        
        self.lr_h, self.lr_w = lr_h, lr_w
        self.block_repeat = block_repeat
        self.c_in = c_in
        self.c_cond = c_cond
        self.patch_size = patch_size
        self.c_hidden = c_hidden
        self.nhead = nhead
        self.blocks = blocks
        self.level_config = level_config
        self.kernel_size = kernel_size
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)
        self.self_attn = self_attn
        self.dropout = dropout
        self.switch_level = switch_level
        # CONDITIONING
        self.clip_txt_mapper = nn.Linear(c_clip_text, c_cond)
        self.clip_txt_pooled_mapper = nn.Linear(c_clip_text_pooled, c_cond * c_clip_seq)
        self.clip_img_mapper = nn.Linear(c_clip_img, c_cond * c_clip_seq)
        self.clip_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(c_in * (patch_size ** 2), c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6)
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == 'C':
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == 'A':
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == 'F':
                return FeedForwardBlock(c_hidden, dropout=dropout)
            elif block_type == 'T':
                return TimestepBlock(c_hidden, c_r, conds=t_conds)
            else:
                raise Exception(f'Block type {block_type} not supported')

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i - 1], elementwise_affine=False, eps=1e-6),
                    UpDownBlock2d(c_hidden[i - 1], c_hidden[i], mode='down', enabled=switch_level[i - 1])
                ))
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i], self_attn=self_attn[i])
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1))
                self.down_repeat_mappers.append(block_repeat_mappers)



        #extra down blocks


        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i], elementwise_affine=False, eps=1e-6),
                    UpDownBlock2d(c_hidden[i], c_hidden[i - 1], mode='up', enabled=switch_level[i - 1])
                ))
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i],
                                      self_attn=self_attn[i])
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1))
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_hidden[0], c_out * (patch_size ** 2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

        # --- WEIGHT INIT ---
        self.apply(self._init_weights)  # General init
        nn.init.normal_(self.clip_txt_mapper.weight, std=0.02)  # conditionings
        nn.init.normal_(self.clip_txt_pooled_mapper.weight, std=0.02)  # conditionings
        nn.init.normal_(self.clip_img_mapper.weight, std=0.02)  # conditionings
        torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
        nn.init.constant_(self.clf[1].weight, 0)  # outputs

        # blocks
        for level_block in self.down_blocks + self.up_blocks:
            for block in level_block:
                if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
                elif isinstance(block, TimestepBlock):
                    for layer in block.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.constant_(layer.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def _init_extra_parameter(self):
        
       
        
        self.agg_net = nn.ModuleList()
        for _ in range(2):
        
          self.agg_net.append(TransInr(ind=2048, ch=1024, n_head=32, head_dim=32, n_groups=64, f_dim=1024, time_dim=self.c_r, t_conds = []))  #
          
        self.agg_net_up = nn.ModuleList()
        for _ in range(2):
    
          self.agg_net_up.append(TransInr(ind=2048, ch=1024, n_head=32, head_dim=32, n_groups=64, f_dim=1024, time_dim=self.c_r, t_conds = []))  #
          
       
        
        
        
        self.norm_down_blocks = nn.ModuleList()
        for i in range(len(self.c_hidden)):
            
            up_blocks = nn.ModuleList()
            for j in range(self.blocks[0][i]):
                if j % 4 == 0:
                    up_blocks.append(
                      ScaleNormalize_res(self.c_hidden[0], self.c_r, conds=[]))
            self.norm_down_blocks.append(up_blocks)
       
       
        self.norm_up_blocks = nn.ModuleList()
        for i in reversed(range(len(self.c_hidden))):
           
            up_block = nn.ModuleList()
            for j in range(self.blocks[1][::-1][i]):
                if j % 4 == 0:
                    up_block.append(ScaleNormalize_res(self.c_hidden[0], self.c_r, conds=[]))
            self.norm_up_blocks.append(up_block)
         
        
        
        
        self.agg_net.apply(self._init_weights)
        self.agg_net_up.apply(self._init_weights)
        self.norm_up_blocks.apply(self._init_weights)
        self.norm_down_blocks.apply(self._init_weights)
        for block in self.agg_net + self.agg_net_up:
            #for block in level_block:
            if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
            elif isinstance(block, TimestepBlock):
                    for layer in block.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.constant_(layer.weight, 0)
       
       



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

    def gen_c_embeddings(self, clip_txt, clip_txt_pooled, clip_img):
        clip_txt = self.clip_txt_mapper(clip_txt)
        if len(clip_txt_pooled.shape) == 2:
            clip_txt_pool = clip_txt_pooled.unsqueeze(1)
        if len(clip_img.shape) == 2:
            clip_img = clip_img.unsqueeze(1)
        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.c_clip_seq, -1)
        clip_img = self.clip_img_mapper(clip_img).view(clip_img.size(0), clip_img.size(1) * self.c_clip_seq, -1)
        clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
        clip = self.clip_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, clip, cnet=None, require_q=False, lr_guide=None, r_emb_lite=None, guide_weight=1):
        level_outputs = []
        if require_q:
            qs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
        for stage_cnt,  (down_block, downscaler, repmap) in enumerate(block_group):
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for inner_cnt, block in enumerate(down_block):
                   
                    
                    if isinstance(block, ResBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  ResBlock)):
                        if cnet is not None and lr_guide is None:
                        #if cnet is not None :
                            next_cnet = cnet()
                            if next_cnet is not None:
                               
                                x = x + nn.functional.interpolate(next_cnet.float(), size=x.shape[-2:], mode='bilinear',
                                                                  align_corners=True)
                        x = block(x)
                    elif isinstance(block, AttnBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  AttnBlock)):
                        
                        x = block(x, clip)
                        if require_q and (inner_cnt == 2 ):
                            qs.append(x.clone())
                        if lr_guide is not None and (inner_cnt == 2 ) :
                            
                            guide = self.agg_net[stage_cnt](x.shape, x, lr_guide[stage_cnt], r_emb_lite)
                            x = x + guide
                         
                    elif isinstance(block, TimestepBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  TimestepBlock)):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)    # 0 indicate last output
        if require_q:
            return level_outputs, qs
        return level_outputs


    def _up_decode(self, level_outputs, r_embed, clip, cnet=None, require_ff=False, agg_f=None, r_emb_lite=None, guide_weight=1):
        if require_ff:
            agg_feas = []
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    
                    if isinstance(block, ResBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  ResBlock)):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        
                        
                        if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                            x = torch.nn.functional.interpolate(x.float(), skip.shape[-2:], mode='bilinear',
                                                                align_corners=True)
                                       
                        if cnet is not None and agg_f is None:
                            next_cnet = cnet()
                            if next_cnet is not None:
                                
                                x = x + nn.functional.interpolate(next_cnet.float(), size=x.shape[-2:], mode='bilinear',
                                                                  align_corners=True)

                        
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  AttnBlock)):
                        
                           
                        x = block(x, clip)
                        if require_ff and (k == 2 ):
                            agg_feas.append(x.clone())
                        if agg_f is not None and (k == 2 ) :  

                            guide = self.agg_net_up[i](x.shape, x, agg_f[i], r_emb_lite)  # training 1  test 4k 0.8   2k 0.7
                            if not self.training:
                                hw = x.shape[-2] * x.shape[-1]
                                if hw >= 96*96:
                                    guide = 0.7*guide

                                else:
                                
                                    if hw >= 72*72:
                                        guide = 0.5* guide
                                    else:

                                        guide = 0.3* guide
                                      
                            x = x + guide
                      
                           
                    elif isinstance(block, TimestepBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  TimestepBlock)):
                        x = block(x, r_embed)
                        #if require_ff:
                        #    agg_feas.append(x.clone())
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)

        
        if require_ff:
            return x, agg_feas
        
        return x
      

    
        
    def forward(self, x, r,  clip_text, clip_text_pooled, clip_img, lr_guide=None, reuire_f=False, cnet=None, require_t=False, guide_weight=0.5, **kwargs):

        r_embed = self.gen_r_embedding(r)
        
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond)], dim=1)
        clip = self.gen_c_embeddings(clip_text, clip_text_pooled, clip_img)

        # Model Blocks
       
        x = self.embedding(x)
        
       
      
        if cnet is not None:
            cnet = ControlNetDeliverer(cnet)
       
        if not reuire_f:
            level_outputs = self._down_encode(x, r_embed, clip, cnet, lr_guide= lr_guide[0] if lr_guide is not None else None, \
            require_q=reuire_f, r_emb_lite=self.gen_r_embedding(r), guide_weight=guide_weight)
            x = self._up_decode(level_outputs, r_embed, clip, cnet, agg_f=lr_guide[1] if lr_guide is not None else None, \
            require_ff=reuire_f, r_emb_lite=self.gen_r_embedding(r), guide_weight=guide_weight)
        else:
            level_outputs, lr_enc = self._down_encode(x, r_embed, clip, cnet, lr_guide= lr_guide[0] if lr_guide is not None else None, require_q=True)
            x, lr_dec = self._up_decode(level_outputs, r_embed, clip, cnet, agg_f=lr_guide[1] if lr_guide is not None else None, require_ff=True)
          
        if reuire_f and require_t:
            return self.clf(x), r_embed, lr_enc, lr_dec
        if reuire_f:
            return self.clf(x), lr_enc, lr_dec   
        if require_t:
            return self.clf(x), r_embed
        return self.clf(x)
       

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone().to(self_params.device) * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone().to(self_buffers.device) * (1 - beta)



if __name__ == '__main__':
    generator = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]])
    total_ori = sum([ param.nelement()  for param in generator.parameters()])
    generator._init_extra_parameter()
    generator = generator.cuda()
    total = sum([ param.nelement()  for param in generator.parameters()])
    total_down = sum([ param.nelement()  for param in generator.down_blocks.parameters()])

    total_up = sum([ param.nelement()  for param in generator.up_blocks.parameters()])
    total_pro = sum([ param.nelement()  for param in generator.project.parameters()])
    
    
    print(total_ori / 1e6, total / 1e6, total_up / 1e6, total_down / 1e6, total_pro / 1e6)
   
    # for name, module in generator.down_blocks.named_modules():
    #     print(name, module)
    output, out_lr = generator(
        x=torch.randn(1, 16, 24, 24).cuda(), 
        x_lr=torch.randn(1, 16, 16, 16).cuda(), 
        r=torch.tensor([0.7056]).cuda(),
        clip_text=torch.randn(1, 77, 1280).cuda(),
        clip_text_pooled = torch.randn(1, 1, 1280).cuda(),
        clip_img = torch.randn(1, 1, 768).cuda()
    )
    print(output.shape, out_lr.shape)
    # cnt
