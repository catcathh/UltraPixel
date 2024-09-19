import os
import yaml
import torch
import sys
sys.path.append(os.path.abspath('./'))
from inference.utils import *
from train import WurstCoreB
from gdf import DDPMSampler
from train import WurstCore_t2i as WurstCoreC
import numpy as np
import random
import argparse
import gradio as gr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--height', type=int, default=2560, help='image height')
    parser.add_argument('--width', type=int, default=5120, help='image width')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--dtype', type=str, default='bf16', help=' if bf16 does not work, change it to float32 ')
    parser.add_argument('--config_c', type=str, 
    default='configs/training/t2i.yaml' ,help='config file for stage c, latent generation')
    parser.add_argument('--config_b', type=str, 
    default='configs/inference/stage_b_1b.yaml' ,help='config file for stage b, latent decoding')
    parser.add_argument( '--prompt', type=str,
     default='A photo-realistic image of a west highland white terrier in the garden, high quality, detail rich, 8K', help='text prompt')
    parser.add_argument( '--num_image', type=int, default=1, help='how many images generated')
    parser.add_argument( '--output_dir', type=str, default='figures/output_results/', help='output directory for generated image')
    parser.add_argument( '--stage_a_tiled', action='store_true', help='whther or nor to use tiled decoding for stage a to save memory')
    parser.add_argument( '--pretrained_path', type=str, default='models/ultrapixel_t2i.safetensors', help='pretrained path of newly added paramter of UltraPixel')
    args = parser.parse_args()
    return args

def clear_image():
    return None
def load_message(height, width, seed, prompt, args, stage_a_tiled):
    args.height = height
    args.width = width
    args.seed  = seed
    args.prompt = prompt + ' rich detail, 4k, high quality'
    args.stage_a_tiled = stage_a_tiled
    return args

def get_image(height, width, seed, prompt, cfg, timesteps, stage_a_tiled):
    global args
    args = load_message(height, width, seed, prompt,  args, stage_a_tiled)
    torch.manual_seed(args.seed)
    random.seed(args.seed) 
    np.random.seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float

    captions = [args.prompt] * args.num_image
    height, width = args.height, args.width
    batch_size=1 
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

    for _, caption in enumerate(captions):

        
            batch = {'captions': [caption] * batch_size}
            #conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
            #unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
            
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
                #for idx, img in enumerate(imgs):
                    #print(os.path.join(save_dir, args.prompt[:20]+'_' + str(cnt).zfill(5) + '.jpg'), idx)
                    #img.save(os.path.join(save_dir, args.prompt[:20]+'_' + str(cnt).zfill(5) + '.jpg'))
                    
    return imgs[0]           
    #print('finished! Results ')


with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("<h1><center>UltraPixel: Advancing Ultra-High-Resolution Image Synthesis to New Peaks </center></h1>")
        
        with gr.Row():
            prompt = gr.Textbox(
                label="Text Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False
            )
            polish_button = gr.Button("Submit!", scale=0)
        
        output_img = gr.Image(label="Output Image", show_label=False)
        
        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Number(
                label="Random Seed",
                value=123,
                step=1,
                minimum=0,
                #maximum=MAX_SEED
            )
            
            #randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
            
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=1536,
                    maximum=5120,
                    step=32,
                    value=4096
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=1536,
                    maximum=4096,
                    step=32,
                    value=2304
                )
            
            with gr.Row():
                cfg = gr.Slider(
                    label="CFG",
                    minimum=3,
                    maximum=10,
                    step=0.1,
                    value=4
                )
                
                timesteps = gr.Slider(
                    label="Timesteps",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=20
                )
            
            stage_a_tiled = gr.Checkbox(label="Stage_a_tiled", value=False)
        
        clear_button = gr.Button("Clear!")
        
        gr.Examples(
            examples=[
                "A detailed view of a blooming magnolia tree, with large, white flowers and dark green leaves, set against a clear blue sky.",
                "A close-up portrait of a young woman with flawless skin, vibrant red lipstick, and wavy brown hair, wearing a vintage floral dress and standing in front of a blooming garden.",
                "The image features a snow-covered mountain range with a large, snow-covered mountain in the background. The mountain is surrounded by a forest of trees, and the sky is filled with clouds. The scene is set during the winter season, with snow covering the ground and the trees.",
                "Crocodile in a sweater.",
                "A vibrant anime scene of a young girl with long, flowing pink hair, big sparkling blue eyes, and a school uniform, standing under a cherry blossom tree with petals falling around her. The background shows a traditional Japanese school with cherry blossoms in full bloom.",
                "A playful Labrador retriever puppy with a shiny, golden coat, chasing a red ball in a spacious backyard, with green grass and a wooden fence.",
                "A cozy, rustic log cabin nestled in a snow-covered forest, with smoke rising from the stone chimney, warm lights glowing from the windows, and a path of footprints leading to the front door.",
                "A highly detailed, high-quality image of the Banff National Park in Canada. The turquoise waters of Lake Louise are surrounded by snow-capped mountains and dense pine forests. A wooden canoe is docked at the edge of the lake. The sky is a clear, bright blue, and the air is crisp and fresh.",
                "A highly detailed, high-quality image of a Shih Tzu receiving a bath in a home bathroom. The dog is standing in a tub, covered in suds, with a slightly wet and adorable look. The background includes bathroom fixtures, towels, and a clean, tiled floor.",
            ],
            inputs=[prompt],
            outputs=[output_img],
            examples_per_page=5
        )
        
        polish_button.click(get_image, inputs=[height, width, seed, prompt, cfg, timesteps, stage_a_tiled], outputs=output_img)           
        polish_button.click(clear_image, inputs=[], outputs=output_img)
   

if __name__ == "__main__":
   
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config_file = args.config_c
    with open(config_file, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)
    
    core = WurstCoreC(config_dict=loaded_config, device=device, training=False)
    
    # SETUP STAGE B
    config_file_b = args.config_b
    with open(config_file_b, "r", encoding="utf-8") as file:
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
    
    pretrained_path = args.pretrained_path    
    sdd = torch.load(pretrained_path, map_location='cpu')
    collect_sd = {}
    for k, v in sdd.items():
        collect_sd[k[7:]] = v
    
    models.train_norm.load_state_dict(collect_sd)
    models.generator.eval()
    models.train_norm.eval()
    
    
    demo.launch(
            debug=True, share=True, 
        )
