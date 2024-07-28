# UltraPixel

This is the implementation for **UltraPixel: Advancing Ultra-High-Resolution Image Synthesis to New Peaks**.

UltraPixel is designed to create exceptionally high-quality, detail-rich images at various resolutions, pushing the boundaries of ultra-high-resolution image synthesis. For more details and to see more stunning images, please visit the [Project Page](https://jingjingrenabc.github.io/ultrapixel/). The [arXiv version](https://arxiv.org/abs/2407.02158) of the paper contains compressed images, while the [full paper](https://drive.google.com/file/d/1X18HH9kj7ltAnZorrkD84RJEdsJu4gDF/view?usp=sharing) features uncompressed, high-quality images.

![teaser](figures/teaser.jpg)

## Getting Started
**1.** Install dependency by running:
```
pip install -r requirements.txt
```
**2.** Download pre-trained models from [StableCascade model downloading instructions](https://github.com/Stability-AI/StableCascade/tree/master/models). Small-big models (the small model for stage b and the big model for stage with bfloat16 format are used.) The big-big setting is also supported, while small-big favors more efficiency.

**3.**  Download newly added parameters of UltraPixel from [here](https://huggingface.co/roubaofeipi/UltraPixel).

**Note**: All model downloading urls are provided [here](./models/models_checklist.txt). They should be put in the directory [models](./models).

## Inference
### Text-guided Image Generation
Generate an image by running:
```
CUDA_VISIBLE_DEVICES=0 python inference/test_t2i.py
```
**Tips**: To generate aesthetic images, use detailed prompts with specific descriptions. It's recommended to include elements such as the subject, background, colors, lighting, and mood, and enhance your prompts with high-quality modifiers like "high quality", "rich detail", "8k", "photo-realistic", "cinematic", and "perfection". For example, use "A breathtaking sunset over a serene mountain range, with vibrant orange and purple hues in the sky, high quality, rich detail, 8k, photo-realistic, cinematic lighting, perfection". Be concise but detailed, specific and clear, and experiment with different word combinations for the best results.

Several example prompts are provided [here](./prompt_list.txt).

It is recommended to add "--stage_a_tiled" for decoding in stage a to save memory.

The table below  show memory requirements and running times on different GPUs. For the A100 with 80GB memory, tiled decoding is not necessary.

**On 80G A100:**
| Resolution          | Stage C  | Stage B | Stage A |
|---------------------|----------|---------|--------|
|2048*2048            |15.9G / 12s  | 14.5G / 4s    |**w/o tiled**: 11.2G / 1s  |
|4096*4096            |18.7G / 52s  | 19.7G / 26s   |**w/o tiled**: 45.3G / 2s, **tiled**: 9.3G / 128s|

**On 32G V100** (only works using float32 on Stages C and B):
| Resolution                    | Stage C  | Stage B |           Stage A             |
|---------------------|----------|---------|-------------------------------|
|2048*2048            |16.7G / 83s    | 11.7G / 22s   |**w/o tiled**: 10.1G / 2s |
|4096*4096            |18.0G / 287s   | 22.7G / 172s  |**w/o tiled**: OOM, **tiled**: 9.0G / 305s|

**On 24G RTX4090:**
| Resolution                    | Stage C  | Stage B |           Stage A             |
|---------------------|----------|---------|-------------------------------|
|2048*2048            |15.5G / 83s   |  13.2G / 22s  |**w/o tiled**: 11.3G / 1s |
|4096*4096            |19.9G / 153s   | 23.4G / 44s  |**w/o tiled**: OOM, **tiled**: 11.3G / 114s |

### Personalized Image Generation
The repo provides a personalized model of a cat. Download the personalized model [here](https://huggingface.co/roubaofeipi/UltraPixel/blob/main/lora_cat.safetensors) and run the following command to generate personalized results. Note that in the text command you need to use identifier "cat [roubaobao]" to indicate the cat.
```
CUDA_VISIBLE_DEVICES=0 python inference/test_personalized.py
```
### Controlnet Image Generation
Download Canny [ControlNet](https://huggingface.co/stabilityai/stable-cascade/resolve/main/controlnet/canny.safetensors) provided by StableCascade and run the command:
```
CUDA_VISIBLE_DEVICES=0 python inference/test_controlnet.py
```
Note that ControlNet is used without further fine-tuning, so the supported highest resolution is 4K, e.g., 3840 * 2160, 2048 * 2048.


## T2I Training
Put all your images and captions into a folder. Here's an example training dataset [here](./figures/example_dataset) for reference.
Start training by running:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train/train_t2i.py configs/training/t2i.yaml
```


## Personalized Training
Put all your images into a folder. Here's an expample training dataset [here](./figures/example_dataset). The training prompt can be described as: a photo of a cat [roubaobao].

Start training by running:
```
CUDA_VISIBLE_DEVICES=0,1 python train/train_personalized.py \
configs/training/lora_personalization.yaml
```

## Citation
```bibtex
@article{ren2024ultrapixel,
  title={UltraPixel: Advancing Ultra-High-Resolution Image Synthesis to New Peaks},
  author={Ren, Jingjing and Li, Wenbo and Chen, Haoyu and Pei, Renjing and Shao, Bin and Guo, Yong and Peng, Long and Song, Fenglong and Zhu, Lei},
  journal={arXiv preprint arXiv:2407.02158},
  year={2024}
}
```
## Contact Information
To reach out to the paper’s authors, please refer to the contact information provided on the [project page](https://jingjingrenabc.github.io/ultrapixel/).

## Acknowledgements
This project is build upon [StableCascade](https://github.com/Stability-AI/StableCascade) and [Trans-inr](https://github.com/yinboc/trans-inr). Thanks for their code sharing ：）
