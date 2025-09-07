"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from thinkdiff.common.registry import registry
from thinkdiff.processors.base_processor import BaseProcessor
from thinkdiff.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor


@registry.register_processor("mllama_image_train")
class MllamaImageTrainProcessor(BaseProcessor):
    def __init__(self, model_id):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model_id)

    def __call__(self, images, texts):
        input_text = self.processor.apply_chat_template(texts, add_generation_prompt=True)
        inputs = self.processor(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )

        return inputs

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        model_id = cfg.get("model_id", "meta-llama/Llama-3.2-11B-Vision-Instruct")

        return cls(
            model_id=model_id
        )

