from diffusers import StableDiffusionPipeline
import os
import torch
import pandas as pd

device="cuda:0"

CACHE_DIR = f'{os.environ["HOME"]}/common_model/huggingface/'
print(CACHE_DIR)
model_id_or_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path, cache_dir=CACHE_DIR, torch_dtype=torch.float32)
pipe = pipe.to(device)

pipe.safety_checker = None

import matplotlib.pyplot as plt
def display_img(pil_img):
    plt.imshow(pil_img)
    plt.axis('off') 
    plt.show()


def ff_net_hook_0(module, input, output):
    global t, erased_channels_num
    with torch.no_grad():
        new_output = output.clone()
        top_score_channels = [
            5035, 1761, 4499,  185, 4920,         # nudity        √   threshold=1.0
        ]
        
        vectors = [
            1.0318, -0.8551, -0.8290,  0.7260, -0.4764,       # nudity
        ]
        
        new_output[:, :, top_score_channels] = -3 * new_output[:, :, top_score_channels]

    return new_output

modules = [
    pipe.unet.up_blocks[1].attentions[0].transformer_blocks[0].ff.net[0],
]

hooks = []
hooks.append(modules[0].register_forward_hook(ff_net_hook_0))

