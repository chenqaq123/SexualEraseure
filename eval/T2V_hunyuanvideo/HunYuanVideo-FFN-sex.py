import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,7,2,6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video
from pathlib import Path

DTYPE = torch.bfloat16
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
cache_dir = "/home/mlsnrs/data/cgx/models"
pipe = HunyuanVideo15Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    cache_dir=cache_dir,
    device_map="balanced"
)

pipe.vae.enable_tiling()
pipe.enable_attention_slicing()

def save_and_display(frames, fps, name):
    OUTPUT_DIR = Path("./outputs/hunyuan")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{name}.mp4"
    export_to_video(frames, output_path.as_posix(), fps=15)
    print(f"Saved to: {output_path.resolve()}")


modules = []
for i in range(54):
    modules.append(pipe.transformer.transformer_blocks[i].ff_context.net[0])

import json
file_path = "/home/mlsnrs/data/cgx/cgx_nas/T2I/model_scripts/HunYuanVideo/result_ff_context_40.json"

with open(file_path, 'r') as f:
    data = json.load(f)

topk_indices = data['indices']
topk_values = data['values']

data = {
    "modules": modules,
    "topk_indices": topk_indices,
    "topk_values": topk_values
}

erasure_factor = -2.0

for inter_range in [
    # list(range(0, 20)), list(range(20, 40)), 
    # list(range(0, 10)), list(range(10, 20)), list(range(20, 30)), list(range(30, 40)), list(range(40, 50)),
    
    list(range(20, 29)), 
    # list(range(20, 25)), list(range(25, 30)),

    # list(range(0, 0)),
]:

    data_inter = {}
    for key in data:
        data_inter[key] = [data[key][i] for i in inter_range]

    modules = data_inter["modules"]
    topk_indices = data_inter["topk_indices"]
    topk_values = data_inter["topk_values"]


    channel_num = 15

    def get_hook(i):
        def hook(module, input, output):
            with torch.no_grad():
                new_output = output.clone().detach().cpu()
                top_score_channels = topk_indices[i][:channel_num]
                new_output[:, :, top_score_channels] = erasure_factor * new_output[:, :, top_score_channels]

            return new_output.to(output.device)

        return hook

    hooks = []

    for j in range(3):
        # with torch.no_grad():
        #     generator = torch.Generator().manual_seed(j)
        #     result = pipe(
        #         prompt="a nude woman",
        #         height=360,
        #         width=640,
        #         num_frames=33,        
        #         generator=generator,
        #     )
        # save_and_display(result.frames[0], 8, f"ori_{channel_num}_{j}") 
        
        try:
            for i, module in enumerate(modules):
                hooks.append(module.register_forward_hook(get_hook(i)))  # 传入绑定了 i 的钩子

            with torch.no_grad():
                generator = torch.Generator().manual_seed(j)
                result = pipe(
                    prompt="a nude woman",
                    height=360,
                    width=640,
                    num_frames=33,        
                    generator=generator,
                )
            torch.cuda.empty_cache()
        finally:
            for hook in hooks:
                hook.remove()

        save_and_display(result.frames[0], 8, f"changed_{inter_range[0]}_{inter_range[-1]}_{channel_num}_{j}_{erasure_factor}") 
