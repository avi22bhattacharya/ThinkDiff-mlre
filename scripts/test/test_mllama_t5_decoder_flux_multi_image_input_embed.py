"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import thinkdiff.tasks as tasks
from thinkdiff.common.config import Config
from thinkdiff.common.dist_utils import get_rank, init_distributed_mode
from thinkdiff.common.logger import setup_logger
from thinkdiff.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from thinkdiff.common.registry import registry
from thinkdiff.common.utils import now

# imports modules for registration
from thinkdiff.datasets.builders import *
from thinkdiff.models import *
from thinkdiff.processors import *
from thinkdiff.runners import *
from thinkdiff.tasks import *
from torch.utils.data import DataLoader
from PIL import Image
import requests
from thinkdiff.datasets.datasets.cc_sbu_dataset_mllama_vllm_process import llava_brief_instructions
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
from thinkdiff.models.flux_prompt import FluxPipelineRewritePrompt
# import gc
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    model = runner.model
    # dataset = datasets[list(datasets.keys())[0]]["train"]
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     num_workers=1,
    #     pin_memory=True,
    #     collate_fn=dataset.collater
    # )
    model.eval()

    device = model.device
    use_amp = True
    amp_dtype = torch.bfloat16
    ckpt_id = os.path.basename(cfg.model_cfg["ckpt"])


    diffusion_id = "black-forest-labs/FLUX.1-dev"


    diffusion_pipe = FluxPipelineRewritePrompt.from_pretrained(
        diffusion_id,
        # device_map="balanced",
        torch_dtype=torch.bfloat16
    )

    diffusion_pipe.enable_model_cpu_offload()

    output_dir = cfg.run_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # question = "I give you several words and pictures. First, please analyse what the next picture is. Then give me a detailed diffusion prompt to describe the next picture. Please only provide me the detailed prompt and start the answer with 'Create an image'.\n\n"
    
    # placeholders = []

    # placeholders.append({"type": "text", "text": question})
    # placeholders.append({"type": "text", "text": "Word 1: white, "})
    # placeholders.append({"type": "image", "image": "/HOME/yt_ust_danxu/yt_ust_danxu_5/HDD_POOL/zmiaa/datasets/minigpt4/cobsat/datasets/color_car/white_car.jpg"})
    # placeholders.append({"type": "text", "text": "\n\nWord 2: blue, "})
    # placeholders.append({"type": "image", "image": "/HOME/yt_ust_danxu/yt_ust_danxu_5/HDD_POOL/zmiaa/datasets/minigpt4/cobsat/datasets/color_car/blue_car.jpg"})
    # placeholders.append({"type": "text", "text": "\n\nWord 3: red, "})
    # image_names = ["car_white_blue_red"]


    # question = "Create a image generation prompt by expanding this prompt: 'A motorcycle with its brake extended standing outside'.\n\n"
    # question = "Create a image generation prompt by based on this prompt: 'A motorcycle with its brake extended standing outside'. Start your answer with 'Create an image'\n\n"
    # question = "Repeat this 'A motorcycle with its brake extended standing outside'. If the sentence is too short, just pad white space after it.\n\n"
    # question = "Repeat this 'A motorcycle with its brake extended standing outside'. If the sentence is too short, just pad white space after it.\n\n"
    # question = "Repeat this by 10 times: 'A motorcycle with its brake extended standing outside'. If the sentence is too short, just pad white space after it.\n\n"
    # question = "Repeat this infinitely: 'A woman and another woman waiting at a stop.\n'"
    # question = "Repeat this sentence: 'A woman and another woman waiting at a stop.'\n\n"
    # prompt = 'A woman and another woman waiting at a stop.'
    # prompt = 'A motorcycle with its brake extended standing outside.'
    # image_names = ["000000179765"]
    prompt = "This is a picture of an extremely fancy desert."
    image_names = ["000000182417"]

    # prompt = "a photo of a blue pizza and a yellow baseball glove"
    # image_names = ["pizza_baseball_glove"]
    # question = f"Repeat this: \'{prompt}\'."

    # question = "Create a image generation prompt by expanding this prompt to about 64 words: 'a photo of a blue pizza and a yellow baseball glove'.\n\n"
    # question = "Create a image generation prompt by expanding this prompt: 'a photo of a blue pizza and a yellow baseball glove'.\n\n"
    # question = "I give you a prompt. You should create a image generation prompt by expanding this prompt. The final prompt should retain the main meaning of the original one.\n\nPrompt: 'a photo of a blue pizza and a yellow baseball glove'."
    # question = "Create a image generation prompt based on this prompt: 'a photo of a blue pizza and a yellow baseball glove'.\n\n"
    # prompt = "a photo of a blue pizza and a yellow baseball glove"
    # question = f"Repeat this: \'{prompt}\'."
    # prompt = "a photo of a blue pizza and a yellow baseball glove"
    # image_names = ["pizza_baseball_glove"]
    # prompt = "a photo of a bicycle"
    # image_names = ["bicycle"]
    # prompt = "a photo of a dog"
    # image_names = ["dog"]
    # question = f"Create a image generation prompt by expanding this prompt: '{prompt}'.\n\n"

    # prompt = "a photo of a couch and a horse"
    # image_names = ["couch_horse"]
    prompt = "a photo of a toilet and a computer mouse"
    image_names = ["toilet_mouse"]

    prompt = "a photo of a brown giraffe and a white stop sign"
    image_names = ["giraffe_stop_sign"]

    prompt = "a photo of a pink skateboard"
    image_names = ["skateboard"]

    # prompt = "a photo of a microwave and a truck"
    # image_names = ["microwave_truck"]
    

    question = prompt

    
    
    placeholders = []

    placeholders.append({"type": "text", "text": question})


    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            *placeholders,
        ],
    }]

    if getattr(model, "module", None) is None:
        prompt = model.mllama_processor.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            add_vision_id=True)
    else:
        prompt = model.module.mllama_processor.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            add_vision_id=True)
    print(prompt)

    # image_data, _ = process_vision_info(messages)

    sample =  {
            "prompt": prompt,
            # "multi_modal_data": {
            #     "image": image_data
            # },
        }

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            embedding_type = cfg.model_cfg.get("embedding_type", "output_embed")
            if getattr(model, "module", None) is None:
                language_model_inputs, mllama_generated_text = model.get_embed(sample, embedding_type=embedding_type, max_new_tokens=128, need_process=False)
            else:
                language_model_inputs, mllama_generated_text = model.module.get_embed(sample, embedding_type=embedding_type, max_new_tokens=128, need_process=False)
            # language_model_inputs, mllama_generated_text = model.module.get_embed(sample, embedding_type=embedding_type, max_new_tokens=64, min_new_tokens=64)
            for i in range(len(mllama_generated_text)):
                print(language_model_inputs[i].shape)
                # print(sample["answers"][i])
                print(mllama_generated_text[i])
            pass

    guidance_scale = 3.5
    for img_i in range(1):

        image_name = image_names[img_i]
        with torch.no_grad():

            language_model_inputs_i = language_model_inputs[img_i]
            language_model_inputs_i = language_model_inputs_i.unsqueeze(0)
            # pad or cut to max length
            max_length = cfg.run_cfg.get("max_tokens", None)
            if max_length is not None:
                if language_model_inputs_i.shape[1] > max_length:
                    language_model_inputs_i = language_model_inputs_i[:, :max_length]
                elif language_model_inputs_i.shape[1] < max_length:
                    padding = torch.zeros((language_model_inputs_i.shape[0], max_length - language_model_inputs_i.shape[1], language_model_inputs_i.shape[2]), dtype=language_model_inputs_i.dtype, device=language_model_inputs_i.device)
                    language_model_inputs_i = torch.cat([language_model_inputs_i, padding], dim=1)

            prompt_embeds, pooled_prompt_embeds, _ = \
                diffusion_pipe.encode_prompt(
                    prompt="",
                    prompt_2=None,
                    prompt_embeds=language_model_inputs_i
                )

        diffusion_pipe.set_progress_bar_config(disable=False)
        setup_seeds(cfg)

        images = diffusion_pipe(
            prompt_embeds=prompt_embeds.to(torch.bfloat16),
            pooled_prompt_embeds=pooled_prompt_embeds.to(torch.bfloat16),
            num_images_per_prompt=1,
            height=512,
            width=512,
            # height=1024,
            # width=1024,
            num_inference_steps=28,
            guidance_scale=guidance_scale,
        ).images

        # output_dir = f"logs/original_blip2_flux_cfg_0"
        for image_i, image in enumerate(images):
            image.save(f"{output_dir}/{image_name}_edit_4_flux_{embedding_type}_{image_i}_{ckpt_id}.png", format="PNG", compress_level=1)
            print(f"Saved image to {output_dir}/{image_name}_edit_4_flux_{embedding_type}_{image_i}_{ckpt_id}.png")

if __name__ == "__main__":
    main()
