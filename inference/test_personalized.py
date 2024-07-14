
import os
import yaml
import torch
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath('./'))
from inference.utils import *
from train import WurstCoreB
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight
from train import WurstCore_personalized as WurstCoreC
import torch.nn.functional as F
import numpy as np
import random
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--height', type=int, default=3072, help='image height')
    parser.add_argument('--width', type=int, default=4096, help='image width')
    parser.add_argument('--dtype', type=str, default='bf16', help=' if bf16 does not work, change it to float32 ')
    parser.add_argument('--seed', type=int, default=23, help='random seed')
    parser.add_argument('--config_c', type=str, 
    default="configs/training/lora_personalization.yaml" ,help='config file for stage c, latent generation')
    parser.add_argument('--config_b', type=str, 
    default='configs/inference/stage_b_1b.yaml' ,help='config file for stage b, latent decoding')
    parser.add_argument( '--prompt', type=str,
     default='A photo of cat [roubaobao] in space suit, high quality, detail rich, 8k', help='text prompt')
    parser.add_argument( '--num_image', type=int, default=4, help='how many images generated')
    parser.add_argument( '--output_dir', type=str, default='figures/personalized/', help='output directory for generated image')
    parser.add_argument( '--stage_a_tiled', action='store_true', help='whther or nor to use tiled decoding for stage a to save memory')
    parser.add_argument( '--pretrained_path_lora', type=str, default='models/lora_cat.safetensors',help='pretrained path of personalized lora parameter')
    parser.add_argument( '--pretrained_path', type=str, default='models/ultrapixel_t2i.safetensors', help='pretrained path of newly added paramter of UltraPixel')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float
    
    
    # SETUP STAGE C
    with open(args.config_c, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
    core = WurstCoreC(config_dict=loaded_config, device=device, training=False)
    
    # SETUP STAGE B
    with open(args.config_b, "r", encoding="utf-8") as file:
        config_file_b = yaml.safe_load(file)    
    core_b = WurstCoreB(config_dict=config_file_b, device=device, training=False)
    
    extras = core.setup_extras_pre()
    models = core.setup_models(extras)
    models.generator.eval().requires_grad_(False)
    print("STAGE C READY")
    
    extras_b = core_b.setup_extras_pre()
    models_b = core_b.setup_models(extras_b, skip_clip=True)
    models_b = WurstCoreB.Models(
       **{**models_b.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
    )
    models_b.generator.bfloat16().eval().requires_grad_(False)
    print("STAGE B READY")
    
    
    batch_size = 1
    captions = [args.prompt] * args.num_image
    height, width = args.height, args.width
    save_dir = args.output_dir
    
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
     
    
    pretrained_pth = args.pretrained_path
    sdd = torch.load(pretrained_pth, map_location='cpu')
    collect_sd = {}
    for k, v in sdd.items():
        collect_sd[k[7:]] = v
    
    models.train_norm.load_state_dict(collect_sd)
    
    
    pretrained_pth_lora = args.pretrained_path_lora
    sdd = torch.load(pretrained_pth_lora, map_location='cpu')
    collect_sd = {}
    for k, v in sdd.items():
        collect_sd[k[7:]] = v
    
    models.train_lora.load_state_dict(collect_sd)
    
    
    models.generator.eval()
    models.train_norm.eval()
    
    
    height_lr, width_lr = get_target_lr_size(height / width, std_size=32)
    stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)
    stage_c_latent_shape_lr, stage_b_latent_shape_lr = calculate_latent_sizes(height_lr, width_lr, batch_size=batch_size)
    
    # Stage C Parameters
    
    extras.sampling_configs['cfg'] = 4
    extras.sampling_configs['shift'] = 1
    extras.sampling_configs['timesteps'] = 20
    extras.sampling_configs['t_start'] = 1.0
    extras.sampling_configs['sampler'] = DDPMSampler(extras.gdf)
    
    
    
    # Stage B Parameters
    
    extras_b.sampling_configs['cfg'] = 1.1
    extras_b.sampling_configs['shift'] = 1
    extras_b.sampling_configs['timesteps'] = 10
    extras_b.sampling_configs['t_start'] = 1.0
    
    
    for cnt, caption in enumerate(captions):
    
        batch = {'captions': [caption] * batch_size}
        conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
        unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
        
        conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
        unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)
        
        
        
        
        for cnt, caption in enumerate(captions):
    
           
            batch = {'captions': [caption] * batch_size}
            conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
            unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
            
            conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
            unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)
            
             
            with torch.no_grad():
        
              
                models.generator.cuda()
                print('STAGE C GENERATION***************************')
                with torch.cuda.amp.autocast(dtype=dtype):
                    sampled_c = generation_c(batch, models, extras, core, stage_c_latent_shape, stage_c_latent_shape_lr, device)
                
                    
                      
                models.generator.cpu()
                torch.cuda.empty_cache()
                
                conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
                unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)
                conditions_b['effnet'] = sampled_c
                unconditions_b['effnet'] = torch.zeros_like(sampled_c)
                print('STAGE B + A DECODING***************************')
        
                with torch.cuda.amp.autocast(dtype=dtype):
                  sampled = decode_b(conditions_b, unconditions_b, models_b, stage_b_latent_shape, extras_b, device, stage_a_tiled=args.stage_a_tiled)

                torch.cuda.empty_cache()
                imgs = show_images(sampled)
                for idx, img in enumerate(imgs):
                    print(os.path.join(save_dir, args.prompt[:20]+'_' + str(cnt).zfill(5) + '.jpg'), idx)
                    img.save(os.path.join(save_dir, args.prompt[:20]+'_' + str(cnt).zfill(5) + '.jpg'))
                    
                
    print('finished! Results at ', save_dir )
                
    
    
