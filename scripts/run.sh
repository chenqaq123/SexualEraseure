export HF_HOME=/home/chenguanxu/common_model/huggingface
export TRANSFORMERS_CACHE=/home/chenguanxu/common_model/huggingface/hub
export HF_DATASETS_CACHE=/home/chenguanxu/common_model/huggingface/datasets

CUDA_VISIBLE_DEVICES=4 python scripts/nudity_guard.py build \
    --model_type sd3 \
    --model_id  stabilityai/stable-diffusion-3-medium-diffusers \
    --positive_prompts prompt_sets/nudity_positive.txt \
    --negative_prompts prompt_sets/nudity_negative.txt \
    --artifact artifacts/sd3_nudity_guard.pt