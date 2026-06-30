import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "/home/mlsnrs/common_model/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44",
    torch_dtype=torch.bfloat16,
    device_map="balanced",  
)

# 加载 LoRA 权重
pipe.load_lora_weights("/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval/T2I_flux/EraseAnything_lora_weights.safetensors")

results_dir = '/home/mlsnrs/data/cgx/SexualErasure_CCS/eval/T2I_flux/results/EraseAnything/nudity'
os.makedirs(results_dir, exist_ok=True)

# # 1. i2p
# import pandas as pd
# df_i2p = pd.read_csv(f'/root/autodl-fs/prompt_data/i2p.csv')
# df_i2p = df_i2p[df_i2p['categories'].str.contains('sexual', case=False, na=False)].reset_index(drop=True)

# save_dir = f'{results_dir}/i2p/'
# os.makedirs(save_dir, exist_ok=True)
# for i, (p, s, g) in enumerate(zip(df_i2p['prompt'], df_i2p['evaluation_seed'], df_i2p['evaluation_guidance'])):
#     save_path = save_dir + str(i+1) + '.png'
#     if os.path.exists(save_path):
#         continue
#     print(p, s, g)
#     generator=torch.Generator().manual_seed(s)
#     image = pipe(p, generator=generator, height=512, width=512, num_images_per_prompt=1, guidance_scale=g).images
    
#     for j, img in enumerate(image):
#         print(save_path)
#         img.save(save_path)

# # 2. mma
# df_mma = pd.read_csv("/root/autodl-fs/prompt_data/mma.csv")

# save_dir = f'{results_dir}/mma/'
# os.makedirs(save_dir, exist_ok=True)
# for i, p in enumerate(df_mma['sanitized_adv_prompt']):
#     save_path = save_dir + str(i+1) + '.png'
#     if os.path.exists(save_path):
#         continue
#     t = [0, 0, 0]
#     generator=torch.Generator().manual_seed(int(i))
#     image = pipe(p, generator=generator, height=512, width=512, num_images_per_prompt=1).images
    
#     for j, img in enumerate(image):
#         img.save(save_path)

# # 3. sp
# df_sp = pd.read_csv("/root/autodl-fs/prompt_data/sp.csv")
# save_dir = f'{results_dir}/sp/'
# os.makedirs(save_dir, exist_ok=True)

# for i, p in enumerate(df_sp['perturbed_text'].fillna(df_sp['original_text'])):
#     generator=torch.Generator().manual_seed(int(i))
#     image = pipe(p, generator=generator, height=512, width=512, num_images_per_prompt=1).images
    
#     for j, img in enumerate(image):
#         save_path = save_dir + str(i+1) + '.png'
#         print(save_path)
#         img.save(save_path)

# # 4. rab
# df_sp = pd.read_csv("/root/autodl-fs/prompt_data/rab.csv")
# save_dir = f'{results_dir}/sp/'
# os.makedirs(save_dir, exist_ok=True)

# for i, p in enumerate(df_sp['perturbed_text'].fillna(df_sp['original_text'])):
#     generator=torch.Generator().manual_seed(int(i))
#     image = pipe(p, generator=generator, height=512, width=512, num_images_per_prompt=1).images
    
#     for j, img in enumerate(image):
#         save_path = save_dir + str(i+1) + '.png'
#         print(save_path)
#         img.save(save_path)

# # 5. mmd
# df_mmd = pd.read_csv(f'/root/autodl-fs/prompt_data/mmd.csv')

# save_dir = f'{results_dir}/mmd/'
# os.makedirs(save_dir, exist_ok=True)
# for i, (p, s, g) in enumerate(zip(df_mmd['prompt'], df_mmd['evaluation_seed'], df_mmd['evaluation_guidance'])):
#     save_path = save_dir + str(i+1) + '.png'
#     if os.path.exists(save_path):
#         continue
#     print(p, s, g)
#     generator=torch.Generator().manual_seed(s)
#     image = pipe(p, generator=generator, height=512, width=512, num_images_per_prompt=1, guidance_scale=g).images
    
#     for j, img in enumerate(image):
#         print(save_path)
#         img.save(save_path)

# 6. coco
import json
annotation_file = '/home/mlsnrs/common_dataset/coco-2017-dataset/coco2017/annotations/captions_val2017.json'  # 替换为实际路径
with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

print(coco_data.keys())  # 输出 JSON 文件的顶层结构
image_captions = {}  # 用于存储每张图片对应的描述
for item in coco_data['annotations']:
    image_id = item['image_id']
    caption = item['caption']
    
    if image_id not in image_captions:
        image_captions[image_id] = []
    image_captions[image_id].append(caption)

save_dir = f'{results_dir}/coco/'
os.makedirs(save_dir, exist_ok=True)
for i, p in enumerate(image_captions.values()):
    if os.path.exists(os.path.join(save_dir, f"{i+1}.png")):
        continue
    generator=torch.Generator().manual_seed(int(i))
    image = pipe(p[0], generator=generator, num_images_per_prompt=1).images
    
    for j, img in enumerate(image):
        save_path = save_dir + str(i+1) + '.png'
        print(save_path)
        img.save(save_path)