import os
import yaml
import torch
import torchvision
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath('./'))

from inference.utils import *
from core.utils import load_or_fail
from train import WurstCore_control_lrguide, WurstCoreB
from PIL import Image
from core.utils import load_or_fail
import math
import argparse
import time
import random
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--height', type=int, default=3840, help='image height')
    parser.add_argument('--width', type=int, default=2160, help='image width')
    parser.add_argument('--control_weight', type=float, default=0.70, help='[ 0.3, 0.8]')
    parser.add_argument('--dtype', type=str, default='bf16', help=' if bf16 does not work, change it to float32 ')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--config_c', type=str, 
    default='configs/training/cfg_control_lr.yaml' ,help='config file for stage c, latent generation')
    parser.add_argument('--config_b', type=str, 
    default='configs/inference/stage_b_1b.yaml' ,help='config file for stage b, latent decoding')
    parser.add_argument( '--prompt', type=str,
     default='A peaceful lake surrounded by mountain,  white cloud in the sky, high quality,', help='text prompt')
    parser.add_argument( '--num_image', type=int, default=4, help='how many images generated')
    parser.add_argument( '--output_dir', type=str, default='figures/controlnet_results/', help='output directory for generated image')
    parser.add_argument( '--stage_a_tiled', action='store_true', help='whther or nor to use tiled decoding for stage a to save memory')
    parser.add_argument( '--pretrained_path', type=str, default='models/ultrapixel_t2i.safetensors',  help='pretrained path of newly added paramter of UltraPixel')
    parser.add_argument( '--canny_source_url', type=str, default="figures/California_000490.jpg", help='image used to extract canny edge map')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
   
    args = parse_args()
    width = args.width
    height = args.height
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float
    
    
    # SETUP STAGE C
    with open(args.config_c, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
    core = WurstCore_control_lrguide(config_dict=loaded_config, device=device, training=False)
    
    # SETUP STAGE B
    with open(args.config_b, "r", encoding="utf-8") as file:
        config_file_b = yaml.safe_load(file)
        
    core_b = WurstCoreB(config_dict=config_file_b, device=device, training=False)
    
    extras = core.setup_extras_pre()
    models = core.setup_models(extras)
    models.generator.eval().requires_grad_(False)
    print("CONTROLNET READY")
    
    extras_b = core_b.setup_extras_pre()
    models_b = core_b.setup_models(extras_b, skip_clip=True)
    models_b = WurstCoreB.Models(
       **{**models_b.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
    )
    models_b.generator.eval().requires_grad_(False)
    print("STAGE B READY")
    
    batch_size = 1
    save_dir = args.output_dir
    url = args.canny_source_url
    images = resize_image(Image.open(url).convert("RGB")).unsqueeze(0).expand(batch_size, -1, -1, -1)
    batch = {'images': images}
    
    
    
    
    

    cnet_multiplier = args.control_weight # 0.8 0.6 0.3  control strength
    caption_list = [args.prompt] * args.num_image
    height_lr, width_lr = get_target_lr_size(height / width, std_size=32)
    stage_c_latent_shape_lr, stage_b_latent_shape_lr = calculate_latent_sizes(height_lr, width_lr, batch_size=batch_size)
    stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)
    
    
    

    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    
    
    sdd = torch.load(args.pretrained_path, map_location='cpu')
    collect_sd = {}
    for k, v in sdd.items():
       collect_sd[k[7:]] = v
    models.train_norm.load_state_dict(collect_sd, strict=True)
    
    
    
    
    models.controlnet.load_state_dict(load_or_fail(core.config.controlnet_checkpoint_path), strict=True)
    # Stage C Parameters
    extras.sampling_configs['cfg'] = 1
    extras.sampling_configs['shift'] = 2
    extras.sampling_configs['timesteps'] = 20
    extras.sampling_configs['t_start'] = 1.0
    
    # Stage B Parameters
    extras_b.sampling_configs['cfg'] = 1.1
    extras_b.sampling_configs['shift'] = 1
    extras_b.sampling_configs['timesteps'] = 10
    extras_b.sampling_configs['t_start'] = 1.0
    
    # PREPARE CONDITIONS
    
    
    
    
    for out_cnt, caption in enumerate(caption_list):
        with torch.no_grad():
            
                batch['captions'] = [caption + ' high quality'] * batch_size
                conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
                unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
    
                cnet, cnet_input = core.get_cnet(batch, models, extras)
                cnet_uncond = cnet
                conditions['cnet'] = [c.clone() * cnet_multiplier if c is not None else c for c in cnet]
                unconditions['cnet'] = [c.clone() * cnet_multiplier if c is not None else c for c in cnet_uncond]
                edge_images = show_images(cnet_input)
                models.generator.cuda()
                for idx, img in enumerate(edge_images):
                    img.save(os.path.join(save_dir, f"edge_{url.split('/')[-1]}"))
               
                
                print('STAGE C GENERATION***************************')
                with torch.cuda.amp.autocast(dtype=dtype):
                    sampled_c = generation_c(batch, models, extras, core, stage_c_latent_shape, stage_c_latent_shape_lr, device, conditions, unconditions)
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
                    img.save(os.path.join(save_dir, args.prompt[:20]+'_' + str(out_cnt).zfill(5) + '.jpg'))
        print('finished! Results at ', save_dir )
