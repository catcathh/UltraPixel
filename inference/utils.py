import PIL
import torch
import requests
import torchvision
from math import ceil
from io import BytesIO
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import math
from tqdm import tqdm
def download_image(url):
    return PIL.Image.open(requests.get(url, stream=True).raw).convert("RGB")


def resize_image(image, size=768):
    tensor_image = F.to_tensor(image)
    resized_image = F.resize(tensor_image, size, antialias=True)
    return resized_image


def downscale_images(images, factor=3/4):
    scaled_height, scaled_width = int(((images.size(-2)*factor)//32)*32), int(((images.size(-1)*factor)//32)*32)
    scaled_image = torchvision.transforms.functional.resize(images, (scaled_height, scaled_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    return scaled_image



def calculate_latent_sizes(height=1024, width=1024, batch_size=4, compression_factor_b=42.67, compression_factor_a=4.0):
    resolution_multiple = 42.67
    latent_height = ceil(height / compression_factor_b)
    latent_width = ceil(width / compression_factor_b)
    stage_c_latent_shape = (batch_size, 16, latent_height, latent_width)
    
    latent_height = ceil(height / compression_factor_a)
    latent_width = ceil(width / compression_factor_a)
    stage_b_latent_shape = (batch_size, 4, latent_height, latent_width)
    
    return stage_c_latent_shape, stage_b_latent_shape


def get_views(H, W, window_size=64, stride=16):
    '''
    - H, W: height and width of the latent
    '''
    num_blocks_height = (H - window_size) // stride + 1
    num_blocks_width = (W - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views

       

def show_images(images, rows=None, cols=None, **kwargs):
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.size(1) > 3:
        images = images[:, :3]
    
    if rows is None:
        rows = 1
    if cols is None:
        cols = images.size(0) // rows

    _, _, h, w = images.shape

    imgs = []
    for i, img in enumerate(images):
        imgs.append( torchvision.transforms.functional.to_pil_image(img.clamp(0, 1)))
    
    return imgs
    


def decode_b(conditions_b, unconditions_b, models_b, bshape,  extras_b, device, \
    stage_a_tiled=False, num_instance=4, patch_size=256, stride=24):
   
    
    sampling_b = extras_b.gdf.sample(
        models_b.generator.half(), conditions_b,  bshape,
            unconditions_b, device=device,
            **extras_b.sampling_configs,
        )
    models_b.generator.cuda()
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
        sampled_b = sampled_b
    models_b.generator.cpu()
    torch.cuda.empty_cache()
    if stage_a_tiled:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            padding = (stride*2, stride*2, stride*2, stride*2)
            sampled_b = torch.nn.functional.pad(sampled_b, padding, mode='reflect')
            count = torch.zeros((sampled_b.shape[0], 3, sampled_b.shape[-2]*4, sampled_b.shape[-1]*4), requires_grad=False, device=sampled_b.device)
            sampled = torch.zeros((sampled_b.shape[0], 3, sampled_b.shape[-2]*4, sampled_b.shape[-1]*4), requires_grad=False, device=sampled_b.device)
            views = get_views(sampled_b.shape[-2], sampled_b.shape[-1], window_size=patch_size, stride=stride)
           
            for view_idx, (h_start, h_end, w_start, w_end) in enumerate(tqdm(views, total=len(views))):
            
                sampled[:, :, h_start*4:h_end*4, w_start*4:w_end*4] += models_b.stage_a.decode(sampled_b[:, :, h_start:h_end, w_start:w_end]).float()   
                count[:, :, h_start*4:h_end*4, w_start*4:w_end*4] += 1
            sampled /= count    
            sampled = sampled[:, :, stride*4*2:-stride*4*2, stride*4*2:-stride*4*2]
    else:
    
        sampled = models_b.stage_a.decode(sampled_b, tiled_decoding=stage_a_tiled)

    return sampled.float()


def generation_c(batch, models, extras, core, stage_c_latent_shape, stage_c_latent_shape_lr, device, conditions=None, unconditions=None):
    if conditions is None:
        conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
    if unconditions is None:
        unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
    sampling_c = extras.gdf.sample(
                models.generator,  conditions, stage_c_latent_shape, stage_c_latent_shape_lr, 
                unconditions, device=device, **extras.sampling_configs, 
            )
    for idx, (sampled_c, sampled_c_curr, _, _) in enumerate(tqdm(sampling_c, total=extras.sampling_configs['timesteps'])):
                sampled_c = sampled_c
    return sampled_c
    
def get_target_lr_size(ratio, std_size=24):
        w, h = int(std_size / math.sqrt(ratio)), int(std_size * math.sqrt(ratio)) 
        return (h * 32 , w *32 ) 

