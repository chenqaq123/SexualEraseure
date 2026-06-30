import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7,6,5,4"

from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    cache_dir="/home/chenguanxu/common_model/huggingface/hub",
    torch_dtype=torch.bfloat16,
    device_map="balanced",  
)

import matplotlib.pyplot as plt
def display_img(pil_img):
    plt.imshow(pil_img)
    plt.axis('off') 
    plt.show()

D_forget = [
    "an airliner flying in the sky",
] * 10

D_retain = [
    "a bird flying in the sky",
] * 10

D_forget = [
    "a tree in the style of Van Gogh",
    "a cat in the style of Van Gogh",
    "a field in the style of Van Gogh",
] * 5

D_retain = [
    "a tree in the style of realistic",
    "a cat in the style of realistic",
    "a field in the style of realistic",
] * 5

modules = [
    pipe.transformer.transformer_blocks[0].ff.net[0],
    pipe.transformer.transformer_blocks[1].ff.net[0],
    pipe.transformer.transformer_blocks[2].ff.net[0],
    pipe.transformer.transformer_blocks[3].ff.net[0],
    pipe.transformer.transformer_blocks[4].ff.net[0],
    pipe.transformer.transformer_blocks[5].ff.net[0],
    pipe.transformer.transformer_blocks[6].ff.net[0],
    pipe.transformer.transformer_blocks[7].ff.net[0],
    pipe.transformer.transformer_blocks[8].ff.net[0],
    pipe.transformer.transformer_blocks[9].ff.net[0],
    pipe.transformer.transformer_blocks[10].ff.net[0],
    pipe.transformer.transformer_blocks[11].ff.net[0],
    pipe.transformer.transformer_blocks[12].ff.net[0],
    pipe.transformer.transformer_blocks[13].ff.net[0],
    pipe.transformer.transformer_blocks[14].ff.net[0],
    pipe.transformer.transformer_blocks[15].ff.net[0],
    pipe.transformer.transformer_blocks[16].ff.net[0],
    pipe.transformer.transformer_blocks[17].ff.net[0],
    pipe.transformer.transformer_blocks[18].ff.net[0],
    pipe.transformer.transformer_blocks[0].ff_context.net[0],
    pipe.transformer.transformer_blocks[1].ff_context.net[0],
    pipe.transformer.transformer_blocks[2].ff_context.net[0],
    pipe.transformer.transformer_blocks[3].ff_context.net[0],
    pipe.transformer.transformer_blocks[4].ff_context.net[0],
    pipe.transformer.transformer_blocks[5].ff_context.net[0],
    pipe.transformer.transformer_blocks[6].ff_context.net[0],
    pipe.transformer.transformer_blocks[7].ff_context.net[0],
    pipe.transformer.transformer_blocks[8].ff_context.net[0],
    pipe.transformer.transformer_blocks[9].ff_context.net[0],
    pipe.transformer.transformer_blocks[10].ff_context.net[0],
    pipe.transformer.transformer_blocks[11].ff_context.net[0],
    pipe.transformer.transformer_blocks[12].ff_context.net[0],
    pipe.transformer.transformer_blocks[13].ff_context.net[0],
    pipe.transformer.transformer_blocks[14].ff_context.net[0],
    pipe.transformer.transformer_blocks[15].ff_context.net[0],
    pipe.transformer.transformer_blocks[16].ff_context.net[0],
    pipe.transformer.transformer_blocks[17].ff_context.net[0],
    pipe.transformer.transformer_blocks[18].ff_context.net[0],
    pipe.transformer.single_transformer_blocks[0].proj_mlp,
    pipe.transformer.single_transformer_blocks[1].proj_mlp,
    pipe.transformer.single_transformer_blocks[2].proj_mlp,
    pipe.transformer.single_transformer_blocks[3].proj_mlp,
    pipe.transformer.single_transformer_blocks[4].proj_mlp,
    pipe.transformer.single_transformer_blocks[5].proj_mlp,
    pipe.transformer.single_transformer_blocks[6].proj_mlp,
    pipe.transformer.single_transformer_blocks[7].proj_mlp,
    pipe.transformer.single_transformer_blocks[8].proj_mlp,
    pipe.transformer.single_transformer_blocks[9].proj_mlp,
    pipe.transformer.single_transformer_blocks[10].proj_mlp,
    pipe.transformer.single_transformer_blocks[11].proj_mlp,
    pipe.transformer.single_transformer_blocks[12].proj_mlp,
    pipe.transformer.single_transformer_blocks[13].proj_mlp,
    pipe.transformer.single_transformer_blocks[14].proj_mlp,
    pipe.transformer.single_transformer_blocks[15].proj_mlp,
    pipe.transformer.single_transformer_blocks[16].proj_mlp,
    pipe.transformer.single_transformer_blocks[17].proj_mlp,
    pipe.transformer.single_transformer_blocks[18].proj_mlp,
    pipe.transformer.single_transformer_blocks[19].proj_mlp,
    pipe.transformer.single_transformer_blocks[20].proj_mlp,
    pipe.transformer.single_transformer_blocks[21].proj_mlp,
    pipe.transformer.single_transformer_blocks[22].proj_mlp,
    pipe.transformer.single_transformer_blocks[23].proj_mlp,
    pipe.transformer.single_transformer_blocks[24].proj_mlp,
    pipe.transformer.single_transformer_blocks[25].proj_mlp,
    pipe.transformer.single_transformer_blocks[26].proj_mlp,
    pipe.transformer.single_transformer_blocks[27].proj_mlp,
    pipe.transformer.single_transformer_blocks[28].proj_mlp,
    pipe.transformer.single_transformer_blocks[29].proj_mlp,
    pipe.transformer.single_transformer_blocks[30].proj_mlp,
    pipe.transformer.single_transformer_blocks[31].proj_mlp,
    pipe.transformer.single_transformer_blocks[32].proj_mlp,
    pipe.transformer.single_transformer_blocks[33].proj_mlp,
    pipe.transformer.single_transformer_blocks[34].proj_mlp,
    pipe.transformer.single_transformer_blocks[35].proj_mlp,
    pipe.transformer.single_transformer_blocks[36].proj_mlp,
    pipe.transformer.single_transformer_blocks[37].proj_mlp,
]

topk_indices_list = []
topk_values_list = []

for n, module in enumerate(modules):
    D_forget = [
        "a nude woman",
        "a naked woman",
    ] * 5

    D_retain = [
        "a dressed woman",
        "a dressed woman",
    ] * 5

    i = 0
    forget_tensors = []
    def save_forget_hook(module, input, output):
        global forget_tensors
        forget_tensors.append(output.cpu().detach())
        
    save_forget_hook_handle = module.register_forward_hook(save_forget_hook)

    try:
        for _, p in enumerate(D_forget):
            with torch.no_grad():
                generator = torch.Generator().manual_seed(i)
                image = pipe(p, generator=generator).images[0]
                # display_img(image)
            i += 1
    finally:
        save_forget_hook_handle.remove()
        
    forget_tensors = torch.stack(forget_tensors, dim=0)

    i = 0
    retain_tensors = []
    def save_retain_hook(module, input, output):
        global retain_tensors
        retain_tensors.append(output.cpu().detach())
        
    save_retain_hook_handle = module.register_forward_hook(save_retain_hook)

    try:
        for _, p in enumerate(D_retain):
            with torch.no_grad():
                generator = torch.Generator().manual_seed(i)
                image = pipe(p, generator=generator).images[0]
                # display_img(image)
            i += 1
    finally:
        save_retain_hook_handle.remove()

    retain_tensors = torch.stack(retain_tensors, dim=0)

    forget_tensors_flatten = forget_tensors.reshape(-1, forget_tensors.shape[-1])
    retain_tensors_flatten = retain_tensors.reshape(-1, retain_tensors.shape[-1])
    # print(forget_tensors_flatten.shape, retain_tensors_flatten.shape)

    forget_score = forget_tensors_flatten.mean(dim=0)
    retain_score = retain_tensors_flatten.mean(dim=0)
    # print(forget_score.shape, retain_score.shape)

    importance_score = torch.abs(forget_score) / torch.maximum(retain_score, torch.tensor(5e-2).to(retain_score.device))
    topk_indices = torch.topk(importance_score, k=512).indices
    mask = torch.zeros(forget_score.shape, dtype=torch.bool)
    mask[topk_indices] = 1

    masked_forget_score = torch.where(mask, forget_score, torch.tensor(0.0).to(forget_score))
    masked_retain_score = torch.where(mask, retain_score, torch.tensor(0.0).to(forget_score))

    vector = masked_forget_score - masked_retain_score
    k = 200
    topk_indices = torch.topk(torch.abs(vector), k=k).indices
    topk_values = vector[topk_indices]

    topk_indices_list.append(topk_indices)
    topk_values_list.append(topk_values)

topk_indices_list = [indices.numpy().tolist() for indices in topk_indices_list]
topk_values_list = [values.float().numpy().tolist() for values in topk_values_list]

import json
data = {
    "indices": topk_indices_list,
    "values": topk_values_list
}
with open('result_van_gogh.json', 'w') as f:
    json.dump(data, f, indent=2)