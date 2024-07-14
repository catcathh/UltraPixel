import torch
import json
import yaml
import torchvision
from torch import nn, optim
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from warmup_scheduler import GradualWarmupScheduler
import torch.multiprocessing as mp
import os
import numpy as np
import re
import sys
sys.path.append(os.path.abspath('./'))

from dataclasses import dataclass
from torch.distributed import init_process_group, destroy_process_group, barrier
from gdf import GDF_dual_fixlrt as GDF
from gdf import EpsilonTarget, CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight
from torchtools.transforms import SmartCrop
from fractions import Fraction
from modules.effnet import EfficientNetEncoder
from modules.model_4stage_lite import StageC, ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock
from modules.common_ckpt import GlobalResponseNorm
from modules.previewer import Previewer
from core.data import Bucketeer
from train.base import DataCore, TrainingCore
from tqdm import tqdm
from core import WarpCore
from core.utils import EXPECTED, EXPECTED_TRAIN, load_or_fail

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from contextlib import contextmanager
from train.dist_core import *
import glob
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from core.utils import EXPECTED, EXPECTED_TRAIN, update_weights_ema, create_folder_if_necessary
from core.utils import Base
import torch.nn.functional as F
import functools
import math
import copy
import random
from modules.lora import apply_lora, apply_retoken, LoRA, ReToken

Image.MAX_IMAGE_PIXELS = None
torch.manual_seed(23)
random.seed(23)
np.random.seed(23)
#7978026

class Null_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        pass




def identity(x):
    if isinstance(x, bytes):
        x = x.decode('utf-8')
    return x
def check_nan_inmodel(model, meta=''):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"nan detected in {name}", meta)
                return True
        print('no nan', meta)
        return False  
class mydist_dataset(Dataset):
    def __init__(self, rootpath, tmp_prompt, img_processor=None):

        self.img_pathlist = glob.glob(os.path.join(rootpath, '*.jpg'))
        self.img_pathlist = self.img_pathlist * 100000
        self.img_processor = img_processor
        self.length = len( self.img_pathlist)
        self.caption = tmp_prompt
      
      
    def __getitem__(self, idx):
        
        imgpath = self.img_pathlist[idx]
        txt = self.caption
        
        
        
       
        try:  
          img = Image.open(imgpath).convert('RGB')
          w, h = img.size
          if self.img_processor is not None:
            img = self.img_processor(img)

        except:
          print('exception', imgpath)
          return self.__getitem__(random.randint(0, self.length -1 ) )
        return dict(captions=txt, images=img)
    def __len__(self):
        return self.length
class WurstCore(TrainingCore, DataCore, WarpCore):
    @dataclass(frozen=True)
    class Config(TrainingCore.Config, DataCore.Config, WarpCore.Config):
        # TRAINING PARAMS
        lr: float = EXPECTED_TRAIN
        warmup_updates: int = EXPECTED_TRAIN
        dtype: str = None

        # MODEL VERSION
        model_version: str = EXPECTED  # 3.6B or 1B
        clip_image_model_name: str = 'openai/clip-vit-large-patch14'
        clip_text_model_name: str = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
       
        # CHECKPOINT PATHS
        effnet_checkpoint_path: str = EXPECTED
        previewer_checkpoint_path: str = EXPECTED
        generator_checkpoint_path: str = None
        ultrapixel_path: str = EXPECTED

        # gdf customization
        adaptive_loss_weight: str = None

       # LoRA STUFF
        module_filters: list = EXPECTED
        rank: int = EXPECTED
        train_tokens: list = EXPECTED
        use_ddp: bool=EXPECTED
        tmp_prompt: str=EXPECTED
    @dataclass(frozen=True)
    class Data(Base):
        dataset: Dataset = EXPECTED
        dataloader: DataLoader  = EXPECTED
        iterator: any = EXPECTED
        sampler: DistributedSampler = EXPECTED

    @dataclass(frozen=True)
    class Models(TrainingCore.Models, DataCore.Models, WarpCore.Models):
        effnet: nn.Module = EXPECTED
        previewer: nn.Module = EXPECTED
        train_norm: nn.Module = EXPECTED
        train_lora: nn.Module = EXPECTED

    @dataclass(frozen=True)
    class Schedulers(WarpCore.Schedulers):
        generator: any = None

    @dataclass(frozen=True)
    class Extras(TrainingCore.Extras, DataCore.Extras, WarpCore.Extras):
        gdf: GDF = EXPECTED
        sampling_configs: dict = EXPECTED
        effnet_preprocess: torchvision.transforms.Compose = EXPECTED

    info: TrainingCore.Info
    config: Config

    def setup_extras_pre(self) -> Extras:
        gdf = GDF(
            schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
            input_scaler=VPScaler(), target=EpsilonTarget(),
            noise_cond=CosineTNoiseCond(),
            loss_weight=AdaptiveLossWeight() if self.config.adaptive_loss_weight is True else P2LossWeight(),
        )
        sampling_configs = {"cfg": 5, "sampler": DDPMSampler(gdf), "shift": 1, "timesteps": 20}

        if self.info.adaptive_loss is not None:
            gdf.loss_weight.bucket_ranges = torch.tensor(self.info.adaptive_loss['bucket_ranges'])
            gdf.loss_weight.bucket_losses = torch.tensor(self.info.adaptive_loss['bucket_losses'])

        effnet_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        ])

        clip_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        if self.config.training:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.config.image_size[-1], interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
                SmartCrop(self.config.image_size, randomize_p=0.3, randomize_q=0.2)
            ])
        else:
            transforms = None

        return self.Extras(
            gdf=gdf,
            sampling_configs=sampling_configs,
            transforms=transforms,
            effnet_preprocess=effnet_preprocess,
            clip_preprocess=clip_preprocess
        )

    def get_conditions(self, batch: dict, models: Models, extras: Extras, is_eval=False, is_unconditional=False,
                       eval_image_embeds=False, return_fields=None):
        conditions = super().get_conditions(
            batch, models, extras, is_eval, is_unconditional,
            eval_image_embeds, return_fields=return_fields or ['clip_text', 'clip_text_pooled', 'clip_img']
        )
        return conditions

    def setup_models(self, extras: Extras) -> Models:   # configure model

       
        dtype = getattr(torch, self.config.dtype) if self.config.dtype else torch.bfloat16

        # EfficientNet encoderin
        effnet = EfficientNetEncoder()
        effnet_checkpoint = load_or_fail(self.config.effnet_checkpoint_path)
        effnet.load_state_dict(effnet_checkpoint if 'state_dict' not in effnet_checkpoint else effnet_checkpoint['state_dict'])
        effnet.eval().requires_grad_(False).to(self.device)
        del effnet_checkpoint

        # Previewer
        previewer = Previewer()
        previewer_checkpoint = load_or_fail(self.config.previewer_checkpoint_path)
        previewer.load_state_dict(previewer_checkpoint if 'state_dict' not in previewer_checkpoint else previewer_checkpoint['state_dict'])
        previewer.eval().requires_grad_(False).to(self.device)
        del previewer_checkpoint

        @contextmanager
        def dummy_context():
            yield None

        loading_context = dummy_context if self.config.training else init_empty_weights

        # Diffusion models
        with loading_context():
            generator_ema = None
            if self.config.model_version == '3.6B':
                generator = StageC()
                if self.config.ema_start_iters is not None:  # default setting
                    generator_ema = StageC()
            elif self.config.model_version == '1B':
                print('in line 155 1b light model', self.config.model_version )
                generator = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]])
                
                if self.config.ema_start_iters is not None and self.config.training:
                    generator_ema = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]])
            else:
                raise ValueError(f"Unknown model version {self.config.model_version}")

       
        
        if loading_context is dummy_context:
            generator.load_state_dict( load_or_fail(self.config.generator_checkpoint_path))
        else:
            for param_name, param in load_or_fail(self.config.generator_checkpoint_path).items():
                    set_module_tensor_to_device(generator, param_name, "cpu", value=param)

        generator._init_extra_parameter()
        generator = generator.to(torch.bfloat16).to(self.device)
       
        train_norm = nn.ModuleList()
        
        
        cnt_norm = 0
        for mm in generator.modules():
            if isinstance(mm,  GlobalResponseNorm):
               
                train_norm.append(Null_Model())
                cnt_norm += 1
                
        
        
        
        train_norm.append(generator.agg_net)
        train_norm.append(generator.agg_net_up)
        sdd = torch.load(self.config.ultrapixel_path, map_location='cpu')
        collect_sd = {}
        for k, v in sdd.items():
            collect_sd[k[7:]] = v
        train_norm.load_state_dict(collect_sd)
    
        
        
         # CLIP encoders
        tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
        text_model = CLIPTextModelWithProjection.from_pretrained( self.config.clip_text_model_name).requires_grad_(False).to(dtype).to(self.device)
        image_model = CLIPVisionModelWithProjection.from_pretrained(self.config.clip_image_model_name).requires_grad_(False).to(dtype).to(self.device)
        
         # PREPARE LORA
        train_lora = nn.ModuleList()
        update_tokens = []
        for tkn_regex, aggr_regex in self.config.train_tokens:
            if (tkn_regex.startswith('[') and tkn_regex.endswith(']')) or (tkn_regex.startswith('<') and tkn_regex.endswith('>')):
                # Insert new token
                tokenizer.add_tokens([tkn_regex])
                # add new zeros embedding
                new_embedding = torch.zeros_like(text_model.text_model.embeddings.token_embedding.weight.data)[:1]
                if aggr_regex is not None:  # aggregate embeddings to provide an interesting baseline
                    aggr_tokens = [v for k, v in tokenizer.vocab.items() if re.search(aggr_regex, k) is not None]
                    if len(aggr_tokens) > 0:
                        new_embedding = text_model.text_model.embeddings.token_embedding.weight.data[aggr_tokens].mean(dim=0, keepdim=True)
                    elif self.is_main_node:
                        print(f"WARNING: No tokens found for aggregation regex {aggr_regex}. It will be initialized as zeros.")
                text_model.text_model.embeddings.token_embedding.weight.data = torch.cat([
                    text_model.text_model.embeddings.token_embedding.weight.data, new_embedding
                ], dim=0)
                selected_tokens = [len(tokenizer.vocab) - 1]
            else:
                selected_tokens = [v for k, v in tokenizer.vocab.items() if re.search(tkn_regex, k) is not None]
            update_tokens += selected_tokens
        update_tokens = list(set(update_tokens))  # remove duplicates

        apply_retoken(text_model.text_model.embeddings.token_embedding, update_tokens)
        
        apply_lora(generator, filters=self.config.module_filters, rank=self.config.rank)
        for module in generator.modules():
            if isinstance(module, LoRA) or (hasattr(module, '_fsdp_wrapped_module') and isinstance(module._fsdp_wrapped_module, LoRA)):
                train_lora.append(module)
        
       
        train_lora.append(text_model.text_model.embeddings.token_embedding.parametrizations.weight[0])

        if os.path.exists(os.path.join(self.config.output_path, self.config.experiment_id, 'train_lora.safetensors')):
            sdd = torch.load(os.path.join(self.config.output_path, self.config.experiment_id, 'train_lora.safetensors'), map_location='cpu')
            collect_sd = {}
            for k, v in sdd.items():
                collect_sd[k[7:]] = v
            train_lora.load_state_dict(collect_sd, strict=True)
        
        
        train_norm.to(self.device).train().requires_grad_(True)
       
        if generator_ema is not None:
           
            generator_ema.load_state_dict(load_or_fail(self.config.generator_checkpoint_path))
            generator_ema._init_extra_parameter()
            pretrained_pth = os.path.join(self.config.output_path, self.config.experiment_id, 'generator.safetensors')
            if os.path.exists(pretrained_pth):
              generator_ema.load_state_dict(torch.load(pretrained_pth, map_location='cpu'))

            generator_ema.eval().requires_grad_(False)

        check_nan_inmodel(generator, 'generator')
     
        
        
        if self.config.use_ddp and self.config.training:
           
            train_lora = DDP(train_lora, device_ids=[self.device], find_unused_parameters=True)
           
       

        return self.Models(
            effnet=effnet, previewer=previewer, train_norm = train_norm,
            generator=generator, generator_ema=generator_ema,
            tokenizer=tokenizer, text_model=text_model, image_model=image_model,
             train_lora=train_lora
        )

    def setup_optimizers(self, extras: Extras, models: Models) -> TrainingCore.Optimizers:
        

        params = []
        params += list(models.train_lora.module.parameters())
        optimizer = optim.AdamW(params, lr=self.config.lr) 

        return self.Optimizers(generator=optimizer)

    def ema_update(self, ema_model, source_model, beta):
        for param_src, param_ema in zip(source_model.parameters(), ema_model.parameters()):
            param_ema.data.mul_(beta).add_(param_src.data, alpha = 1 - beta)
            
    def sync_ema(self, ema_model):
        print('sync ema', torch.distributed.get_world_size())
        for param in ema_model.parameters():
            torch.distributed.all_reduce(param.data, op=torch.distributed.ReduceOp.SUM)
            param.data /= torch.distributed.get_world_size()
    def setup_optimizers_backup(self, extras: Extras, models: Models) -> TrainingCore.Optimizers:
       

        optimizer = optim.AdamW(
            models.generator.up_blocks.parameters() , 
        lr=self.config.lr)
        optimizer = self.load_optimizer(optimizer, 'generator_optim',
                                        fsdp_model=models.generator if self.config.use_fsdp else None)
        return self.Optimizers(generator=optimizer)

    def setup_schedulers(self, extras: Extras, models: Models, optimizers: TrainingCore.Optimizers) -> Schedulers:
        scheduler = GradualWarmupScheduler(optimizers.generator, multiplier=1, total_epoch=self.config.warmup_updates)
        scheduler.last_epoch = self.info.total_steps
        return self.Schedulers(generator=scheduler)

    def setup_data(self, extras: Extras) -> WarpCore.Data:
        # SETUP DATASET
        dataset_path = self.config.webdataset_path
       

        dataset = mydist_dataset(dataset_path,  self.config.tmp_prompt,  \
            torchvision.transforms.ToTensor() if self.config.multi_aspect_ratio is not None \
                else extras.transforms)

        # SETUP DATALOADER
        real_batch_size = self.config.batch_size // (self.world_size * self.config.grad_accum_steps)
       
        sampler =  DistributedSampler(dataset, rank=self.process_id, num_replicas = self.world_size, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=real_batch_size, num_workers=4, pin_memory=True,
            collate_fn=identity if self.config.multi_aspect_ratio is not None else None,
            sampler = sampler
        )
        if self.is_main_node:
            print(f"Training with batch size {self.config.batch_size} ({real_batch_size}/GPU)")

        if self.config.multi_aspect_ratio is not None:
            aspect_ratios = [float(Fraction(f)) for f in self.config.multi_aspect_ratio]
            dataloader_iterator = Bucketeer(dataloader, density=[ss*ss for ss in self.config.image_size] , factor=32,
                                            ratios=aspect_ratios, p_random_ratio=self.config.bucketeer_random_ratio,
                                            interpolate_nearest=False)  # , use_smartcrop=True)
        else:
           
            dataloader_iterator = iter(dataloader)

        return self.Data(dataset=dataset, dataloader=dataloader, iterator=dataloader_iterator, sampler=sampler)





    def setup_ddp(self, experiment_id, single_gpu=False, rank=0):

        if not single_gpu:
            local_rank = rank
            process_id = rank
            world_size = get_world_size()

            self.process_id = process_id
            self.is_main_node = process_id == 0
            self.device = torch.device(local_rank)
            self.world_size = world_size
          
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '14443'
            torch.cuda.set_device(local_rank)
            init_process_group(
                backend="nccl",
                rank=local_rank,
                world_size=world_size,
                # init_method=init_method,
            )
            print(f"[GPU {process_id}] READY")
        else:
            self.is_main_node = rank == 0
            self.process_id = rank
            self.device = torch.device('cuda:0')
            self.world_size = 1
            print("Running in single thread, DDP not enabled.")
    # Training loop --------------------------------
    def get_target_lr_size(self, ratio, std_size=24):
        w, h = int(std_size / math.sqrt(ratio)), int(std_size * math.sqrt(ratio)) 
        return (h * 32 , w * 32) 
    def forward_pass(self, data: WarpCore.Data, extras: Extras, models: Models):
       
        batch = data
        ratio = batch['images'].shape[-2] / batch['images'].shape[-1]
        shape_lr = self.get_target_lr_size(ratio)
        with torch.no_grad():
            conditions = self.get_conditions(batch, models, extras)
            
            latents = self.encode_latents(batch, models, extras)
            latents_lr = self.encode_latents(batch, models, extras,target_size=shape_lr)
            
            

            flag_lr = random.random() < 0.5 or self.info.iter <5000
           
            if flag_lr:
                noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents_lr, shift=1, loss_shift=1)
            else:
                noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents, shift=1, loss_shift=1)
            if not flag_lr:    
                noised_lr, noise_lr, target_lr, logSNR_lr, noise_cond_lr, loss_weight_lr = \
                extras.gdf.diffuse(latents_lr, shift=1, loss_shift=1, t=torch.ones(latents.shape[0]).to(latents.device)*0.05, )

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            
            
            if not flag_lr:
                with torch.no_grad():
                    _, lr_enc_guide, lr_dec_guide = models.generator(noised_lr, noise_cond_lr, reuire_f=True, **conditions)
            
            
            pred = models.generator(noised, noise_cond, reuire_f=False, lr_guide=(lr_enc_guide, lr_dec_guide) if not flag_lr else None , **conditions)             
            loss = nn.functional.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3]) 
           
            loss_adjusted = (loss * loss_weight ).mean() / self.config.grad_accum_steps 
            

        if isinstance(extras.gdf.loss_weight, AdaptiveLossWeight):
            extras.gdf.loss_weight.update_buckets(logSNR, loss)
        return loss,  loss_adjusted

    def backward_pass(self, update, loss_adjusted, models: Models, optimizers: TrainingCore.Optimizers, schedulers: Schedulers):
        
        if update:
            
            torch.distributed.barrier()
            loss_adjusted.backward()
            
            grad_norm = nn.utils.clip_grad_norm_(models.train_lora.module.parameters(), 1.0)
            optimizers_dict = optimizers.to_dict()
            for k in optimizers_dict:
                if k != 'training':
                    optimizers_dict[k].step()
            schedulers_dict = schedulers.to_dict()
            for k in schedulers_dict:
                if k != 'training':
                    schedulers_dict[k].step()
            for k in optimizers_dict:
                if k != 'training':
                    optimizers_dict[k].zero_grad(set_to_none=True)
            self.info.total_steps += 1
        else:
            
            loss_adjusted.backward()
            grad_norm = torch.tensor(0.0).to(self.device)
        
        return grad_norm

    def models_to_save(self):
        return ['generator', 'generator_ema', 'trans_inr', 'trans_inr_ema']

    def encode_latents(self, batch: dict, models: Models, extras: Extras, target_size=None) -> torch.Tensor:
        
        images = batch['images'].to(self.device)
        if target_size is not None:
          images = F.interpolate(images, target_size)
         
        return models.effnet(extras.effnet_preprocess(images))

    def decode_latents(self, latents: torch.Tensor, batch: dict, models: Models, extras: Extras) -> torch.Tensor:
        return models.previewer(latents)

    def __init__(self, rank=0, config_file_path=None, config_dict=None, device="cpu", training=True, world_size=1, ):
       
        self.is_main_node = (rank == 0)
        self.config: self.Config = self.setup_config(config_file_path, config_dict, training)
        self.setup_ddp(self.config.experiment_id, single_gpu=world_size <= 1, rank=rank)
        self.info: self.Info = self.setup_info()
        print('in line 292', self.config.experiment_id, rank, world_size <= 1)
        p = [i for i in range( 2 * 768 // 32)]
        p = [num / sum(p) for num in p]
        self.rand_pro = p
        self.res_list = [o for o in range(800, 2336, 32)]
        
      
        
    def __call__(self, single_gpu=False):
       
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.is_main_node:
            print()
            print("**STARTIG JOB WITH CONFIG:**")
            print(yaml.dump(self.config.to_dict(), default_flow_style=False))
            print("------------------------------------")
            print()
            print("**INFO:**")
            print(yaml.dump(vars(self.info), default_flow_style=False))
            print("------------------------------------")
            print()
        print('in line 308', self.is_main_node, self.is_main_node, self.process_id, self.device  )
        # SETUP STUFF
        extras = self.setup_extras_pre()
        assert extras is not None, "setup_extras_pre() must return a DTO"



        data = self.setup_data(extras)
        assert data is not None, "setup_data() must return a DTO"
        if self.is_main_node:
            print("**DATA:**")
            print(yaml.dump({k:type(v).__name__ for k, v in data.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()

        models = self.setup_models(extras)
        assert models is not None, "setup_models() must return a DTO"
        if self.is_main_node:
            print("**MODELS:**")
            print(yaml.dump({
                k:f"{type(v).__name__} - {f'trainable params {sum(p.numel() for p in v.parameters() if p.requires_grad)}' if isinstance(v, nn.Module) else 'Not a nn.Module'}" for k, v in models.to_dict().items()
            }, default_flow_style=False))
            print("------------------------------------")
            print()



        optimizers = self.setup_optimizers(extras, models)
        assert optimizers is not None, "setup_optimizers() must return a DTO"
        if self.is_main_node:
            print("**OPTIMIZERS:**")
            print(yaml.dump({k:type(v).__name__ for k, v in optimizers.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()

        schedulers = self.setup_schedulers(extras, models, optimizers)
        assert schedulers is not None, "setup_schedulers() must return a DTO"
        if self.is_main_node:
            print("**SCHEDULERS:**")
            print(yaml.dump({k:type(v).__name__ for k, v in schedulers.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()

        post_extras =self.setup_extras_post(extras, models, optimizers, schedulers)
        assert post_extras is not None, "setup_extras_post() must return a DTO"
        extras = self.Extras.from_dict({ **extras.to_dict(),**post_extras.to_dict() })
        if self.is_main_node:
            print("**EXTRAS:**")
            print(yaml.dump({k:f"{v}" for k, v in extras.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()
        # -------

        # TRAIN
        if self.is_main_node:
            print("**TRAINING STARTING...**")
        self.train(data, extras, models, optimizers, schedulers)

        if single_gpu is False:
            barrier()
            destroy_process_group()
        if self.is_main_node:
            print()
            print("------------------------------------")
            print()
            print("**TRAINING COMPLETE**")
            if self.config.wandb_project is not None:
                wandb.alert(title=f"Training {self.info.wandb_run_id} finished", text=f"Training {self.info.wandb_run_id} finished")


    def train(self, data: WarpCore.Data, extras: WarpCore.Extras, models: Models, optimizers: TrainingCore.Optimizers,
              schedulers: WarpCore.Schedulers):
        start_iter = self.info.iter + 1
        max_iters = self.config.updates * self.config.grad_accum_steps
        if self.is_main_node:
            print(f"STARTING AT STEP: {start_iter}/{max_iters}")

       
        if self.is_main_node:
            create_folder_if_necessary(f'{self.config.output_path}/{self.config.experiment_id}/')
        if 'generator' in self.models_to_save():
            models.generator.train()
        
        iter_cnt = 0
        epoch_cnt = 0
        models.train_norm.train()
        while True:
          epoch_cnt += 1
          if self.world_size > 1:
            
            data.sampler.set_epoch(epoch_cnt)  
          for ggg in range(len(data.dataloader)):
              iter_cnt += 1
              # FORWARD PASS
             
              loss, loss_adjusted = self.forward_pass(next(data.iterator), extras, models)
              
             
              # # BACKWARD PASS
    
              grad_norm = self.backward_pass(
                        iter_cnt % self.config.grad_accum_steps == 0 or iter_cnt == max_iters, loss_adjusted,
                        models, optimizers, schedulers
                      )
              
              
              
              self.info.iter = iter_cnt
              

              self.info.ema_loss = loss.mean().item() if self.info.ema_loss is None else self.info.ema_loss * 0.99 + loss.mean().item() * 0.01
  
            
              if self.is_main_node and  np.isnan(loss.mean().item()) or np.isnan(grad_norm.item()):
                      print(f"gggg NaN value encountered in training run {self.info.wandb_run_id}", \
                      f"Loss {loss.mean().item()} - Grad Norm {grad_norm.item()}. Run {self.info.wandb_run_id}")
  
              if self.is_main_node:
                  logs = {
                      'loss': self.info.ema_loss,
                      'backward_loss': loss_adjusted.mean().item(),
                      
                      'ema_loss': self.info.ema_loss,
                      'raw_ori_loss': loss.mean().item(),
                     
                      'grad_norm': grad_norm.item(),
                      'lr': optimizers.generator.param_groups[0]['lr'] if optimizers.generator is not None else 0,
                      'total_steps': self.info.total_steps,
                  }
                 
                        
                  print(iter_cnt, max_iters, logs, epoch_cnt, )
                
                 
  
            
                  
              
              if iter_cnt == 1 or iter_cnt % (self.config.save_every  ) == 0 or iter_cnt == max_iters:
              
                  if np.isnan(loss.mean().item()):
                      if self.is_main_node and self.config.wandb_project is not None:
                          print(f"NaN value encountered in training run {self.info.wandb_run_id}", \
                          f"Loss {loss.mean().item()} - Grad Norm {grad_norm.item()}. Run {self.info.wandb_run_id}")
                     
                  else:
                      if isinstance(extras.gdf.loss_weight, AdaptiveLossWeight):
                          self.info.adaptive_loss = {
                              'bucket_ranges': extras.gdf.loss_weight.bucket_ranges.tolist(),
                              'bucket_losses': extras.gdf.loss_weight.bucket_losses.tolist(),
                          }
                     
                      
                      if self.is_main_node and iter_cnt % (self.config.save_every * self.config.grad_accum_steps) == 0:
                          print('save model', iter_cnt, iter_cnt % (self.config.save_every * self.config.grad_accum_steps), self.config.save_every, self.config.grad_accum_steps )
                          torch.save(models.train_lora.state_dict(), \
                          f'{self.config.output_path}/{self.config.experiment_id}/train_lora.safetensors')
                          
                         
                          torch.save(models.train_lora.state_dict(), \
                              f'{self.config.output_path}/{self.config.experiment_id}/train_lora_{iter_cnt}.safetensors')
                          
                       
              if iter_cnt == 1 or iter_cnt % (self.config.save_every* self.config.grad_accum_steps) == 0 or iter_cnt == max_iters:
                  
                  if self.is_main_node:
                   
                     self.sample(models, data, extras)
              if False:
                param_changes = {name: (param - initial_params[name]).norm().item() for name, param in models.train_norm.named_parameters()}
                threshold = sorted(param_changes.values(), reverse=True)[int(len(param_changes) * 0.1)]  # top 10%
                important_params = [name for name, change in param_changes.items() if change > threshold]
                print(important_params, threshold, len(param_changes), self.process_id)
                json.dump(important_params, open(f'{self.config.output_path}/{self.config.experiment_id}/param.json', 'w'), indent=4)     
                  
         
          if self.info.iter >= max_iters:
            break
            
    def sample(self, models: Models, data: WarpCore.Data, extras: Extras):
       
       
        models.generator.eval()
        models.train_norm.eval()
        with torch.no_grad():
            batch = next(data.iterator)
            ratio = batch['images'].shape[-2] / batch['images'].shape[-1]
            
            shape_lr = self.get_target_lr_size(ratio)
            conditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
            unconditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)

            latents = self.encode_latents(batch, models, extras)
            latents_lr = self.encode_latents(batch, models, extras, target_size = shape_lr)
            
            if self.is_main_node:
               
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                   
                    *_, (sampled, _, _, sampled_lr) = extras.gdf.sample(
                        models.generator, conditions,
                        latents.shape, latents_lr.shape, 
                        unconditions, device=self.device, **extras.sampling_configs
                    )
    
                  
                    sampled_ema = sampled
                    sampled_ema_lr = sampled_lr
           
            
            if self.is_main_node:
                print('sampling results hr latent shape ', latents.shape, 'lr latent shape', latents_lr.shape, )
                noised_images = torch.cat(
                    [self.decode_latents(latents[i:i + 1].float(), batch, models, extras) for i in range(len(latents))], dim=0)
                
                sampled_images = torch.cat(
                    [self.decode_latents(sampled[i:i + 1].float(), batch, models, extras) for i in range(len(sampled))], dim=0)
                sampled_images_ema = torch.cat(
                    [self.decode_latents(sampled_ema[i:i + 1].float(), batch, models, extras) for i in range(len(sampled_ema))],
                    dim=0)
                    
                noised_images_lr = torch.cat(
                    [self.decode_latents(latents_lr[i:i + 1].float(), batch, models, extras) for i in range(len(latents_lr))], dim=0)
                
                sampled_images_lr = torch.cat(
                    [self.decode_latents(sampled_lr[i:i + 1].float(), batch, models, extras) for i in range(len(sampled_lr))], dim=0)
                sampled_images_ema_lr = torch.cat(
                    [self.decode_latents(sampled_ema_lr[i:i + 1].float(), batch, models, extras) for i in range(len(sampled_ema_lr))],
                    dim=0)

                images = batch['images']
                if images.size(-1) != noised_images.size(-1) or images.size(-2) != noised_images.size(-2):
                    images = nn.functional.interpolate(images, size=noised_images.shape[-2:], mode='bicubic')
                    images_lr = nn.functional.interpolate(images, size=noised_images_lr.shape[-2:], mode='bicubic')

                collage_img = torch.cat([
                    torch.cat([i for i in images.cpu()], dim=-1),
                    torch.cat([i for i in noised_images.cpu()], dim=-1),
                    torch.cat([i for i in sampled_images.cpu()], dim=-1),
                    torch.cat([i for i in sampled_images_ema.cpu()], dim=-1),
                ], dim=-2)
                
                collage_img_lr = torch.cat([
                    torch.cat([i for i in images_lr.cpu()], dim=-1),
                    torch.cat([i for i in noised_images_lr.cpu()], dim=-1),
                    torch.cat([i for i in sampled_images_lr.cpu()], dim=-1),
                    torch.cat([i for i in sampled_images_ema_lr.cpu()], dim=-1),
                ], dim=-2)

                torchvision.utils.save_image(collage_img, f'{self.config.output_path}/{self.config.experiment_id}/{self.info.total_steps:06d}.jpg')
                torchvision.utils.save_image(collage_img_lr, f'{self.config.output_path}/{self.config.experiment_id}/{self.info.total_steps:06d}_lr.jpg')
                
                captions = batch['captions']
                if self.config.wandb_project is not None:
                    log_data = [
                        [captions[i]] + [wandb.Image(sampled_images[i])] + [wandb.Image(sampled_images_ema[i])] + [
                            wandb.Image(images[i])] for i in range(len(images))]
                    log_table = wandb.Table(data=log_data, columns=["Captions", "Sampled", "Sampled EMA", "Orig"])
                    wandb.log({"Log": log_table})

                    if isinstance(extras.gdf.loss_weight, AdaptiveLossWeight):
                        plt.plot(extras.gdf.loss_weight.bucket_ranges, extras.gdf.loss_weight.bucket_losses[:-1])
                        plt.ylabel('Raw Loss')
                        plt.ylabel('LogSNR')
                        wandb.log({"Loss/LogSRN": plt})

           
            models.generator.train()
            models.train_norm.train()
            print('finish sampling')
    
    
    
    def sample_fortest(self, models: Models, extras: Extras, hr_shape, lr_shape, batch, eval_image_embeds=False):
       
       
        models.generator.eval()
        models.trans_inr.eval()
        with torch.no_grad():
           
            if self.is_main_node:
                conditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=eval_image_embeds)
                unconditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)
             
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    
                    *_, (sampled, _, _, sampled_lr) = extras.gdf.sample(
                        models.generator, conditions,
                        hr_shape, lr_shape, 
                        unconditions, device=self.device, **extras.sampling_configs
                    )
    
                    if models.generator_ema is not None:
                        
                        *_, (sampled_ema, _, _, sampled_ema_lr) = extras.gdf.sample(
                            models.generator_ema,   conditions,
                            latents.shape, latents_lr.shape, 
                            unconditions, device=self.device, **extras.sampling_configs
                        )
                       
                    else:
                        sampled_ema = sampled
                        sampled_ema_lr = sampled_lr
           
            
        return sampled, sampled_lr
def main_worker(rank, cfg):
    print("Launching Script in main worker")
    warpcore = WurstCore(
        config_file_path=cfg, rank=rank, world_size = get_world_size()
    )
    # core.fsdp_defaults['sharding_strategy'] = ShardingStrategy.NO_SHARD

    # RUN TRAINING
    warpcore(get_world_size()==1)

if __name__ == '__main__':
    
    if get_master_ip() == "127.0.0.1":
      
        mp.spawn(main_worker, nprocs=get_world_size(), args=(sys.argv[1] if len(sys.argv) > 1 else None, ))
    else:
        main_worker(0, sys.argv[1] if len(sys.argv) > 1 else None, )
