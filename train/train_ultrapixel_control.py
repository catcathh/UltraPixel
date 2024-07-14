import torch
import json
import yaml
import torchvision
from torch import nn, optim
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from warmup_scheduler import GradualWarmupScheduler
import torch.multiprocessing as mp
import numpy as np
import sys

import os
from dataclasses import dataclass
from torch.distributed import init_process_group, destroy_process_group, barrier
from gdf import GDF_dual_fixlrt as GDF
from gdf import EpsilonTarget, CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight
from torchtools.transforms import SmartCrop
from fractions import Fraction
from modules.effnet import EfficientNetEncoder

from modules.model_4stage_lite import StageC

from modules.model_4stage_lite import ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock
from modules.common_ckpt import GlobalResponseNorm
from modules.previewer import Previewer
from core.data import Bucketeer
from train.base import DataCore, TrainingCore
from tqdm import tqdm
from core import WarpCore
from core.utils import EXPECTED, EXPECTED_TRAIN, load_or_fail
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy
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
from modules.common import LayerNorm2d
import torch.nn.functional as F
import functools
import math
import copy
import random
from modules.lora import apply_lora, apply_retoken, LoRA, ReToken
from modules import ControlNet, ControlNetDeliverer
from modules import controlnet_filters

Image.MAX_IMAGE_PIXELS = None
torch.manual_seed(8432)
random.seed(8432)
np.random.seed(8432)
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
        #trans_inr_ckpt: str = EXPECTED
        generator_checkpoint_path: str = None
        controlnet_checkpoint_path: str = EXPECTED
        
        # controlnet settings
        controlnet_blocks: list = EXPECTED
        controlnet_filter: str = EXPECTED
        controlnet_filter_params: dict = None
        controlnet_bottleneck_mode: str = None


        # gdf customization
        adaptive_loss_weight: str = None

        #module_filters: list = EXPECTED
        #rank: int = EXPECTED
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
        train_norm_ema: nn.Module = EXPECTED
        controlnet: nn.Module = EXPECTED

    @dataclass(frozen=True)
    class Schedulers(WarpCore.Schedulers):
        generator: any = None

    @dataclass(frozen=True)
    class Extras(TrainingCore.Extras, DataCore.Extras, WarpCore.Extras):
        gdf: GDF = EXPECTED
        sampling_configs: dict = EXPECTED
        effnet_preprocess: torchvision.transforms.Compose = EXPECTED
        controlnet_filter: controlnet_filters.BaseFilter = EXPECTED

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
        controlnet_filter = getattr(controlnet_filters, self.config.controlnet_filter)(
            self.device,
            **(self.config.controlnet_filter_params if self.config.controlnet_filter_params is not None else {})
        )

        return self.Extras(
            gdf=gdf,
            sampling_configs=sampling_configs,
            transforms=transforms,
            effnet_preprocess=effnet_preprocess,
            clip_preprocess=clip_preprocess,
            controlnet_filter=controlnet_filter
        )
    def get_cnet(self, batch: dict, models: Models, extras: Extras, cnet_input=None, target_size=None, **kwargs):
        images = batch['images']
        if target_size is not None:
            images = Image.resize(images, target_size)
        with torch.no_grad():
            if cnet_input is None:
                cnet_input = extras.controlnet_filter(images, **kwargs)
            if isinstance(cnet_input, tuple):
                cnet_input, cnet_input_preview = cnet_input
            else:
                cnet_input_preview = cnet_input
            cnet_input, cnet_input_preview = cnet_input.to(self.device), cnet_input_preview.to(self.device)
        cnet = models.controlnet(cnet_input)
        return cnet, cnet_input_preview

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
        
       

        
        if os.path.exists(os.path.join(self.config.output_path, self.config.experiment_id, 'train_norm.safetensors')):
            sdd = torch.load(os.path.join(self.config.output_path, self.config.experiment_id, 'train_norm.safetensors'), map_location='cpu')
            collect_sd = {}
            for k, v in sdd.items():
                collect_sd[k[7:]] = v
            train_norm.load_state_dict(collect_sd, strict=True)
        
        
        train_norm.to(self.device).train().requires_grad_(True)
        train_norm_ema = copy.deepcopy(train_norm)
        train_norm_ema.to(self.device).eval().requires_grad_(False)
        if generator_ema is not None:
           
            generator_ema.load_state_dict(load_or_fail(self.config.generator_checkpoint_path))
            generator_ema._init_extra_parameter()
           
            pretrained_pth = os.path.join(self.config.output_path, self.config.experiment_id, 'generator.safetensors')
            if os.path.exists(pretrained_pth):
              print(pretrained_pth, 'exists')
              generator_ema.load_state_dict(torch.load(pretrained_pth, map_location='cpu'))
          
            generator_ema.eval().requires_grad_(False)
          
        check_nan_inmodel(generator, 'generator')
     
        
        
        if self.config.use_fsdp and self.config.training:
            train_norm = DDP(train_norm, device_ids=[self.device], find_unused_parameters=True)

       
        # CLIP encoders
        tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
        text_model = CLIPTextModelWithProjection.from_pretrained(self.config.clip_text_model_name).requires_grad_(False).to(dtype).to(self.device)
        image_model = CLIPVisionModelWithProjection.from_pretrained(self.config.clip_image_model_name).requires_grad_(False).to(dtype).to(self.device)

        controlnet = ControlNet(
            c_in=extras.controlnet_filter.num_channels(),
            proj_blocks=self.config.controlnet_blocks,
            bottleneck_mode=self.config.controlnet_bottleneck_mode
        )
        controlnet = controlnet.to(dtype).to(self.device)
        controlnet = self.load_model(controlnet, 'controlnet')
        controlnet.backbone.eval().requires_grad_(True)
        
        
        return self.Models(
            effnet=effnet, previewer=previewer, train_norm = train_norm,
            generator=generator, generator_ema=generator_ema,
            tokenizer=tokenizer, text_model=text_model, image_model=image_model,
            train_norm_ema=train_norm_ema, controlnet =controlnet
        )

    def setup_optimizers(self, extras: Extras, models: Models) -> TrainingCore.Optimizers:
        
#
      
        params = []
        params += list(models.train_norm.module.parameters())

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
        print('in line 96', dataset_path, type(dataset_path))

        dataset = mydist_dataset(dataset_path, \
            torchvision.transforms.ToTensor() if self.config.multi_aspect_ratio is not None \
                else extras.transforms)

        # SETUP DATALOADER
        real_batch_size = self.config.batch_size // (self.world_size * self.config.grad_accum_steps)
        print('in line 119', self.process_id, real_batch_size)
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
            os.environ['MASTER_PORT'] = '41443'
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
        #batch = next(data.iterator)
        batch = data
        ratio = batch['images'].shape[-2] / batch['images'].shape[-1]
        shape_lr = self.get_target_lr_size(ratio)

        with torch.no_grad():
            conditions = self.get_conditions(batch, models, extras)
            
            latents = self.encode_latents(batch, models, extras)
            latents_lr = self.encode_latents(batch, models, extras,target_size=shape_lr)
           
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents, shift=1, loss_shift=1)
            noised_lr, noise_lr, target_lr, logSNR_lr, noise_cond_lr, loss_weight_lr = extras.gdf.diffuse(latents_lr, shift=1, loss_shift=1, t=torch.ones(latents.shape[0]).to(latents.device)*0.05, )

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            
            require_cond = True
          
            with torch.no_grad():
                _, lr_enc_guide, lr_dec_guide = models.generator(noised_lr, noise_cond_lr, reuire_f=True, **conditions)
            
            
            pred = models.generator(noised, noise_cond, reuire_f=False, lr_guide=(lr_enc_guide, lr_dec_guide) if require_cond else None , **conditions)             
            loss = nn.functional.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3]) 
           
            loss_adjusted = (loss * loss_weight ).mean() / self.config.grad_accum_steps 
            #
        if isinstance(extras.gdf.loss_weight, AdaptiveLossWeight):
            extras.gdf.loss_weight.update_buckets(logSNR, loss)
       
        return loss,  loss_adjusted

    def backward_pass(self, update, loss_adjusted, models: Models, optimizers: TrainingCore.Optimizers, schedulers: Schedulers):
     
        if update:
        
            torch.distributed.barrier()
            loss_adjusted.backward()
            
            
            grad_norm = nn.utils.clip_grad_norm_(models.train_norm.module.parameters(), 1.0)
          
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
            #print('in line 457', loss_adjusted)
            loss_adjusted.backward()
            #torch.distributed.barrier()
            grad_norm = torch.tensor(0.0).to(self.device)
        
        return grad_norm

    def models_to_save(self):
        return ['generator', 'generator_ema', 'trans_inr', 'trans_inr_ema']

    def encode_latents(self, batch: dict, models: Models, extras: Extras, target_size=None) -> torch.Tensor:
        
        images = batch['images'].to(self.device)
        if target_size is not None:
          images = F.interpolate(images, target_size)
          #images = apply_degradations(images)
        return models.effnet(extras.effnet_preprocess(images))

    def decode_latents(self, latents: torch.Tensor, batch: dict, models: Models, extras: Extras) -> torch.Tensor:
        return models.previewer(latents)

    def __init__(self, rank=0, config_file_path=None, config_dict=None, device="cpu", training=True, world_size=1, ):
        # Temporary setup, will be overriden by setup_ddp if required
        # self.device = device
        # self.process_id = 0
        # self.is_main_node = True
        # self.world_size = 1
        # ----
        # self.world_size = world_size
        # self.process_id = rank
        # self.device=device
        self.is_main_node = (rank == 0)
        self.config: self.Config = self.setup_config(config_file_path, config_dict, training)
        self.setup_ddp(self.config.experiment_id, single_gpu=world_size <= 1, rank=rank)
        self.info: self.Info = self.setup_info()
        print('in line 292', self.config.experiment_id, rank, world_size <= 1)
        p = [i for i in range( 2 * 768 // 32)]
        p = [num / sum(p) for num in p]
        self.rand_pro = p
        self.res_list = [o for o in range(800, 2336, 32)]
        
        #[32, 40, 48]
        #in line 292 stage_c_3b_finetuning False
        
    def __call__(self, single_gpu=False):
         # this will change the device to the CUDA rank
        #self.setup_wandb()
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
        #initial_params = {name: param.clone() for name, param in models.train_norm.named_parameters()}
        iter_cnt = 0
        epoch_cnt = 0
        models.train_norm.train()
        while True:
          epoch_cnt += 1
          if self.world_size > 1:
            print('sampler set epoch', epoch_cnt)
            data.sampler.set_epoch(epoch_cnt)  
          for ggg in range(len(data.dataloader)):
              iter_cnt += 1
              # FORWARD PASS
              #print('in line 414 before forward', iter_cnt, batch['captions'][0], self.process_id)
              #loss, loss_adjusted, loss_extra = self.forward_pass(batch, extras, models)
              loss, loss_adjusted = self.forward_pass(next(data.iterator), extras, models)
              
              #print('in line 416', loss, iter_cnt)
              # # BACKWARD PASS
    
              grad_norm = self.backward_pass(
                        iter_cnt % self.config.grad_accum_steps == 0 or iter_cnt == max_iters, loss_adjusted,
                        models, optimizers, schedulers
                      )
              
              
              
              self.info.iter = iter_cnt
              
              # UPDATE EMA
              if  iter_cnt % self.config.ema_iters == 0:
                 
                  with torch.no_grad():
                      print('in line 890 ema update', self.config.ema_iters, iter_cnt)
                      self.ema_update(models.train_norm_ema, models.train_norm, self.config.ema_beta)
                      #generator.module.agg_net.
                      #self.ema_update(models.generator_ema.agg_net, models.generator.module.agg_net, self.config.ema_beta)
                      #self.ema_update(models.generator_ema.agg_net_up, models.generator.module.agg_net_up, self.config.ema_beta)
                      
              # UPDATE LOSS METRICS
              self.info.ema_loss = loss.mean().item() if self.info.ema_loss is None else self.info.ema_loss * 0.99 + loss.mean().item() * 0.01
  
              #print('in line 666 after ema loss', grad_norm, loss.mean().item(), iter_cnt, self.info.ema_loss)
              if self.is_main_node and  np.isnan(loss.mean().item()) or np.isnan(grad_norm.item()):
                      print(f"gggg NaN value encountered in training run {self.info.wandb_run_id}", \
                      f"Loss {loss.mean().item()} - Grad Norm {grad_norm.item()}. Run {self.info.wandb_run_id}")
  
              if self.is_main_node:
                  logs = {
                      'loss': self.info.ema_loss,
                      'backward_loss': loss_adjusted.mean().item(),
                      #'raw_extra_loss': loss_extra.mean().item(),
                      'ema_loss': self.info.ema_loss,
                      'raw_ori_loss': loss.mean().item(),
                      #'raw_rec_loss': loss_rec.mean().item(),
                      #'raw_lr_loss': loss_lr.mean().item(),
                      #'reg_loss':loss_reg.item(),
                      'grad_norm': grad_norm.item(),
                      'lr': optimizers.generator.param_groups[0]['lr'] if optimizers.generator is not None else 0,
                      'total_steps': self.info.total_steps,
                  }
                  if iter_cnt % (self.config.save_every) == 0:
                        
                      print(iter_cnt, max_iters, logs, epoch_cnt, )
                  #pbar.set_postfix(logs)
                 
  
              #if iter_cnt % 10 == 0:
                  
              
              if iter_cnt == 1 or iter_cnt % (self.config.save_every  ) == 0 or iter_cnt == max_iters:
              #if True:
                  # SAVE AND CHECKPOINT STUFF
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
                      #self.save_checkpoints(models, optimizers)
           
                      #torch.save(models.trans_inr.module.state_dict(), \
                      #f'{self.config.output_path}/{self.config.experiment_id}/trans_inr.safetensors')
                      #torch.save(models.trans_inr_ema.state_dict(), \
                      #f'{self.config.output_path}/{self.config.experiment_id}/trans_inr_ema.safetensors')
                      
                      
                      if self.is_main_node and iter_cnt % (self.config.save_every * self.config.grad_accum_steps) == 0:
                          print('save model', iter_cnt, iter_cnt % (self.config.save_every * self.config.grad_accum_steps), self.config.save_every, self.config.grad_accum_steps )
                          torch.save(models.train_norm.state_dict(), \
                          f'{self.config.output_path}/{self.config.experiment_id}/train_norm.safetensors')
                          
                          #self.sync_ema(models.train_norm_ema)
                          torch.save(models.train_norm_ema.state_dict(), \
                          f'{self.config.output_path}/{self.config.experiment_id}/train_norm_ema.safetensors')
                          #if self.is_main_node and iter_cnt % (4 * self.config.save_every * self.config.grad_accum_steps) == 0:
                          torch.save(models.train_norm.state_dict(), \
                              f'{self.config.output_path}/{self.config.experiment_id}/train_norm_{iter_cnt}.safetensors')
                          
                       
              if iter_cnt == 1 or iter_cnt % (self.config.save_every* self.config.grad_accum_steps) == 0 or iter_cnt == max_iters:
                  
                  if self.is_main_node:
                     #check_nan_inmodel(models.generator, 'generator')
                     #check_nan_inmodel(models.generator_ema,  'generator_ema')
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
       
        #if 'generator' in self.models_to_save():
        models.generator.eval()
        models.train_norm.eval()
        with torch.no_grad():
            batch = next(data.iterator)
            ratio = batch['images'].shape[-2] / batch['images'].shape[-1]
            #batch['images'] = batch['images'].to(torch.float16)
            shape_lr = self.get_target_lr_size(ratio)
            conditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
            unconditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)
            cnet, cnet_input = self.get_cnet(batch, models, extras)
            conditions, unconditions = {**conditions, 'cnet': cnet}, {**unconditions, 'cnet': cnet}
            
            latents = self.encode_latents(batch, models, extras)
            latents_lr = self.encode_latents(batch, models, extras, target_size = shape_lr)
           
            if self.is_main_node:
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    #print('in line 366 on v100 switch to tf16')
                    *_, (sampled, _, _, sampled_lr) = extras.gdf.sample(
                        models.generator, models.trans_inr, conditions,
                        latents.shape, latents_lr.shape, 
                        unconditions, device=self.device, **extras.sampling_configs
                    )
    
                  
                       
                    #else:
                    sampled_ema = sampled
                    sampled_ema_lr = sampled_lr
            
            
            if self.is_main_node:
                print('sampling results', latents.shape, latents_lr.shape, )
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
                #torchvision.utils.save_image(collage_img, f'{self.config.experiment_id}_latest_output.jpg')

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

            #if 'generator' in self.models_to_save():
            models.generator.train()
            models.train_norm.train()
            print('finishe sampling in line 901')
    
    
    
    def sample_fortest(self, models: Models, extras: Extras, hr_shape, lr_shape, batch, eval_image_embeds=False):
       
        #if 'generator' in self.models_to_save():
        models.generator.eval()
        models.trans_inr.eval()
        models.controlnet.eval()
        with torch.no_grad():
           
            if self.is_main_node:
                conditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=eval_image_embeds)
                unconditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)
                cnet, cnet_input = self.get_cnet(batch, models, extras, target_size = lr_shape)
                conditions, unconditions = {**conditions, 'cnet': cnet}, {**unconditions, 'cnet': cnet}
                
                #print('in line 885', self.is_main_node)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    #print('in line 366 on v100 switch to tf16')
                    *_, (sampled, _, _, sampled_lr) = extras.gdf.sample(
                        models.generator, models.trans_inr, conditions,
                        hr_shape, lr_shape, 
                        unconditions, device=self.device, **extras.sampling_configs
                    )
    
                    if models.generator_ema is not None:
                        
                        *_, (sampled_ema, _, _, sampled_ema_lr) = extras.gdf.sample(
                            models.generator_ema,  models.trans_inr_ema,  conditions,
                            latents.shape, latents_lr.shape, 
                            unconditions, device=self.device, **extras.sampling_configs
                        )
                       
                    else:
                        sampled_ema = sampled
                        sampled_ema_lr = sampled_lr
            #x0, x, epsilon, x0_lr, x_lr, pred_lr)
            #sampled, _ = models.trans_inr(sampled, None, sampled)
            #sampled_lr, _ = models.trans_inr(sampled, None, sampled_lr)
            
        return sampled, sampled_lr
def main_worker(rank, cfg):
    print("Launching Script in main worker")
    print('in line 467', rank)
    warpcore = WurstCore(
        config_file_path=cfg, rank=rank, world_size = get_world_size()
    )
    # core.fsdp_defaults['sharding_strategy'] = ShardingStrategy.NO_SHARD

    # RUN TRAINING
    warpcore(get_world_size()==1)

if __name__ == '__main__':
    print('launch multi process')
    # os.environ["OMP_NUM_THREADS"] = "1" 
    # os.environ["MKL_NUM_THREADS"] = "1" 
    #dist.init_process_group(backend="nccl")
    #torch.backends.cudnn.benchmark = True
#train/train_c_my.py
    #mp.set_sharing_strategy('file_system')
    print('in line 481', sys.argv[1] if len(sys.argv) > 1 else None)
    print('in line 481',get_master_ip(), get_world_size() )
    print('in line 484', get_world_size())
    if get_master_ip() == "127.0.0.1":
        # manually launch distributed processes
        mp.spawn(main_worker, nprocs=get_world_size(), args=(sys.argv[1] if len(sys.argv) > 1 else None, ))
    else:
        main_worker(0, sys.argv[1] if len(sys.argv) > 1 else None, )
