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

from PIL import Image
import requests
from transformers import Blip2Processor, T5EncoderModel, T5TokenizerFast, CLIPVisionModel, AutoProcessor
import torch
from thinkdiff.models.flux_prompt import FluxPipelineRewritePrompt
import os
from torch import nn
import re
from torch.nn import functional as F
import json
import tqdm
import omegaconf

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
    seed = cfg.run_cfg.seed + get_rank()

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    model = model.eval().to(cfg.run_cfg.device, torch.bfloat16)
    
    diffusion_id = "black-forest-labs/FLUX.1-dev"

    diffusion_pipe = FluxPipelineRewritePrompt.from_pretrained(
        diffusion_id,
        # device_map="balanced",
        torch_dtype=torch.bfloat16
    ).to(cfg.run_cfg.device)

    # diffusion_pipe.enable_model_cpu_offload()
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")

    output_dir = cfg.run_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if cfg.run_cfg.get("img_folder", None):
        print("Loading image urls from folder")
        img_folder = cfg.run_cfg["img_folder"]
        img_urls = [os.path.join(img_folder, img_name) for img_name in os.listdir(img_folder)]
        img_urls = [img_url for img_url in img_urls if os.path.isfile(img_url) and (img_url.endswith(".png") or img_url.endswith(".jpg"))]
    elif cfg.run_cfg.get("img_json", None):
        img_json = cfg.run_cfg["img_json"]
        print(f"Loading image urls from {img_json}")
        with open(img_json) as f:
            img_urls = json.load(f)
    else:
        img_urls = cfg.run_cfg["img_urls"]
    if cfg.run_cfg.get("img_urls_len", None):
        img_urls = img_urls[:cfg.run_cfg["img_urls_len"]]
    
    # get image names
    if cfg.run_cfg.get("image_names", None) is None:
        image_names = []
        for img_url in img_urls:
            if type(img_url) == list:
                image_names.append("_".join([sud_img_url.split("/")[-1].split(".")[0] for sud_img_url in img_url]))
            else:
                image_names.append(img_url.split("/")[-1].split(".")[0])
    else:
        image_names = cfg.run_cfg["image_names"]
    if cfg.run_cfg.get("prompt_json", None):
        prompt_json = cfg.run_cfg["prompt_json"]
        print(f"Loading prompt json from {prompt_json}")
        with open(prompt_json) as f:
            questions = json.load(f)
        questions_names = None

    else:
        questions = cfg.run_cfg["questions"]
        questions_names = cfg.run_cfg["questions_names"]
        prompt_json = None

    guidance_scale = cfg.run_cfg["guidance_scale"]
    flux_height = cfg.run_cfg["flux_height"]
    flux_width = cfg.run_cfg["flux_width"]
    flux_num_inference_steps = cfg.run_cfg["flux_num_inference_steps"]
    flux_max_sequence_length = cfg.run_cfg["flux_max_sequence_length"]

    print(img_urls)
    print(questions)

    for img_i, img_url in enumerate(tqdm.tqdm(img_urls)):
        if prompt_json is not None:
            image_name = image_names[img_i]
            prompt = questions[image_name]

            if cfg.run_cfg.get("use_image_name_as_output_name", False):
                output_path = f"{output_dir}/{image_name}.png"
            elif cfg.run_cfg.get("use_image_name_and_prompt_as_output_name", False):
                prompt_name = re.sub(r'[^\w\s-]', '', prompt)
                prompt_name = re.sub(r'\s+', '_', prompt_name)
                output_path = f"{output_dir}/{image_name}_{prompt_name}.png"
            else:
                output_path = f"{output_dir}/{image_name}_clip_t5_flux_seed_{seed}.png"

            if os.path.exists(output_path):
                print(f"Image already exists at {output_path}")
                continue

            if type(img_url) == list or type(img_url) == omegaconf.listconfig.ListConfig:
                prompt_embeds_list = []
                for sub_img_url in img_url:
                    raw_image = Image.open(sub_img_url)
                    inputs = processor(raw_image, prompt, return_tensors="pt").to("cuda", torch.bfloat16)
                    with torch.no_grad():

                        vision_outputs = model.forward_encoder(
                            **inputs,
                        )

                        t5_prompt_embeds = vision_outputs
                        prompt_embeds_list.append(t5_prompt_embeds)
                        
                        # prompt_embeds = torch.cat([t5_prompt_embeds, prompt_embeds], dim=1)
                
                prompt_embeds, pooled_prompt_embeds, _ = \
                    diffusion_pipe.encode_prompt(
                        prompt=prompt,
                        prompt_2=None,
                        max_sequence_length=flux_max_sequence_length
                        # prompt_embeds=t5_prompt_embeds
                    )
                prompt_embeds_list.append(prompt_embeds)
                prompt_embeds = torch.cat(prompt_embeds_list, dim=1)

            else:
                raw_image = Image.open(img_url)

                inputs = processor(raw_image, prompt, return_tensors="pt").to("cuda", torch.bfloat16)
                with torch.no_grad():

                    vision_outputs = model.forward_encoder(
                        **inputs,
                    )

                    t5_prompt_embeds = vision_outputs

                    prompt_embeds, pooled_prompt_embeds, _ = \
                        diffusion_pipe.encode_prompt(
                            prompt=prompt,
                            prompt_2=None,
                            max_sequence_length=flux_max_sequence_length
                            # prompt_embeds=t5_prompt_embeds
                        )
                    
                    prompt_embeds = torch.cat([t5_prompt_embeds, prompt_embeds], dim=1)

            diffusion_pipe.set_progress_bar_config(disable=True)

            images = diffusion_pipe(
                prompt_embeds=prompt_embeds.to(torch.bfloat16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(torch.bfloat16),
                num_images_per_prompt=1,
                height=flux_height,
                width=flux_width,
                num_inference_steps=flux_num_inference_steps,
                guidance_scale=guidance_scale,
            ).images

            # output_dir = f"logs/original_blip2_flux_cfg_0"
            # for image_i, image in enumerate(images):
                # output_path = f"{output_dir}/{image_name}_clip_t5_flux_{name}_seed_{seed}_{image_i}.png"
            images[0].save(output_path, format="PNG", compress_level=1)
            print(f"Image saved to {output_path}")
        else:
            for prompt_i, prompt in enumerate(questions):
                image_name = image_names[img_i]
                name = questions_names[prompt_i]
                # output_path = f"{output_dir}/{image_name}_clip_t5_flux_{name}_0.png"
                output_path = f"{output_dir}/{image_name}_clip_t5_flux_{name}_seed_{seed}.png"

                if os.path.exists(output_path):
                    print(f"Image already exists at {output_path}")
                    continue

                if type(img_url) == list or type(img_url) == omegaconf.listconfig.ListConfig:
                    prompt_embeds_list = []
                    for sub_img_url in img_url:
                        raw_image = Image.open(sub_img_url)
                        inputs = processor(raw_image, prompt, return_tensors="pt").to("cuda", torch.bfloat16)
                        with torch.no_grad():

                            vision_outputs = model.forward_encoder(
                                **inputs,
                            )

                            t5_prompt_embeds = vision_outputs
                            prompt_embeds_list.append(t5_prompt_embeds)
                            
                            # prompt_embeds = torch.cat([t5_prompt_embeds, prompt_embeds], dim=1)
                    
                    prompt_embeds, pooled_prompt_embeds, _ = \
                        diffusion_pipe.encode_prompt(
                            prompt=prompt,
                            prompt_2=None,
                            max_sequence_length=flux_max_sequence_length
                            # prompt_embeds=t5_prompt_embeds
                        )
                    prompt_embeds_list.append(prompt_embeds)
                    prompt_embeds = torch.cat(prompt_embeds_list, dim=1)

                else:
                    raw_image = Image.open(img_url)

                    inputs = processor(raw_image, prompt, return_tensors="pt").to("cuda", torch.bfloat16)
                    with torch.no_grad():

                        vision_outputs = model.forward_encoder(
                            **inputs,
                        )

                        t5_prompt_embeds = vision_outputs

                        prompt_embeds, pooled_prompt_embeds, _ = \
                            diffusion_pipe.encode_prompt(
                                prompt=prompt,
                                prompt_2=None,
                                max_sequence_length=flux_max_sequence_length
                                # prompt_embeds=t5_prompt_embeds
                            )
                        
                        prompt_embeds = torch.cat([t5_prompt_embeds, prompt_embeds], dim=1)

                diffusion_pipe.set_progress_bar_config(disable=True)

                images = diffusion_pipe(
                    prompt_embeds=prompt_embeds.to(torch.bfloat16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(torch.bfloat16),
                    num_images_per_prompt=1,
                    height=flux_height,
                    width=flux_width,
                    num_inference_steps=flux_num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images

                # output_dir = f"logs/original_blip2_flux_cfg_0"
                # for image_i, image in enumerate(images):
                    # output_path = f"{output_dir}/{image_name}_clip_t5_flux_{name}_seed_{seed}_{image_i}.png"
                images[0].save(output_path, format="PNG", compress_level=1)
                print(f"Image saved to {output_path}")


if __name__ == "__main__":
    main()
