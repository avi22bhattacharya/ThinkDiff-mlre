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
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
from thinkdiff.models.flux_prompt import FluxPipelineRewritePrompt
# import gc

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
    model.eval()

    device = model.device
    use_amp = True
    amp_dtype = torch.bfloat16


    diffusion_id = "black-forest-labs/FLUX.1-dev"


    diffusion_pipe = FluxPipelineRewritePrompt.from_pretrained(
        diffusion_id,
        # device_map="balanced",
        torch_dtype=torch.bfloat16
    )

    diffusion_pipe.enable_model_cpu_offload()

    output_dir = cfg.run_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    urls = []
    urls.append("assets/dot_image.jpeg")

    # edit_0
    answers = []
    answers.append("Create an diffusion prompt for a dog in the style of this picture. Do not use 'similar to the image'")

    images = []
    # answers = []
    image_names = []

    for i in range(len(urls)):
        url = urls[i]
        if ":" in url:
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        else:
            image = Image.open(url).convert("RGB")
        images.append([image])
        image_names.append(url.split("/")[-1].split(".")[0])

    sample = {
        "images": images,
        "answers": answers
    }

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # for sample in data_loader:
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v.to(device)
            embedding_type = cfg.model_cfg.get("embedding_type", "output_embed")
            language_model_inputs, mllama_generated_text = model.module.get_embed(sample, embedding_type=embedding_type, max_new_tokens=128)
            # language_model_inputs, mllama_generated_text = model.module.get_embed(sample, embedding_type=embedding_type, max_new_tokens=64, min_new_tokens=64)
            for i in range(len(mllama_generated_text)):
                print(language_model_inputs[i].shape)
                print(sample["answers"][i])
                print(mllama_generated_text[i])
            pass
    
    print(urls)

    guidance_scale = 3.5
    for img_i in range(len(urls)):

        image_name = image_names[img_i]
        with torch.no_grad():

            language_model_inputs_i = language_model_inputs[img_i]
            language_model_inputs_i = language_model_inputs_i.unsqueeze(0)
            prompt_embeds, pooled_prompt_embeds, _ = \
                diffusion_pipe.encode_prompt(
                    prompt="",
                    prompt_2=None,
                    prompt_embeds=language_model_inputs_i
                )

        diffusion_pipe.set_progress_bar_config(disable=False)

        images = diffusion_pipe(
            prompt_embeds=prompt_embeds.to(torch.bfloat16),
            pooled_prompt_embeds=pooled_prompt_embeds.to(torch.bfloat16),
            num_images_per_prompt=1,
            # height=512,
            # width=512,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=guidance_scale,
        ).images

        for image_i, image in enumerate(images):
            image.save(f"{output_dir}/{image_name}_output_embed_flux_{image_i}.png", format="PNG", compress_level=1)
            print(f"Saved image to {output_dir}/{image_name}_output_embed_flux_{image_i}.png")

if __name__ == "__main__":
    main()
