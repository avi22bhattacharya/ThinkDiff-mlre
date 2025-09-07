from diffusers import FluxPipeline
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.loaders import FluxLoraLoaderMixin

from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)


class FluxPipelineRewritePrompt(FluxPipeline):
    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
    ):
        super().__init__(
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            text_encoder_2,
            tokenizer_2,
            transformer
        )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        # if prompt is not None:
        #     batch_size = len(prompt)
        # else:
        #     batch_size = prompt_embeds.shape[0]

        if pooled_prompt_embeds is None:
            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        # text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        # text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids



    # def encode_prompt(
    #     self,
    #     prompt: Union[str, List[str]],
    #     prompt_2: Union[str, List[str]],
    #     device: Optional[torch.device] = None,
    #     num_images_per_prompt: int = 1,
    #     prompt_embeds: Optional[torch.FloatTensor] = None,
    #     pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     max_sequence_length: int = 512,
    #     lora_scale: Optional[float] = None,
    # ):
    #     r"""

    #     Args:
    #         prompt (`str` or `List[str]`, *optional*):
    #             prompt to be encoded
    #         prompt_2 (`str` or `List[str]`, *optional*):
    #             The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
    #             used in all text-encoders
    #         device: (`torch.device`):
    #             torch device
    #         num_images_per_prompt (`int`):
    #             number of images that should be generated per prompt
    #         prompt_embeds (`torch.FloatTensor`, *optional*):
    #             Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
    #             provided, text embeddings will be generated from `prompt` input argument.
    #         pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
    #             Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
    #             If not provided, pooled text embeddings will be generated from `prompt` input argument.
    #         lora_scale (`float`, *optional*):
    #             A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
    #     """
    #     device = device or self._execution_device

    #     # set lora scale so that monkey patched LoRA
    #     # function of text encoder can correctly access it
    #     if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
    #         self._lora_scale = lora_scale

    #         # dynamically adjust the LoRA scale
    #         if self.text_encoder is not None and USE_PEFT_BACKEND:
    #             scale_lora_layers(self.text_encoder, lora_scale)
    #         if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
    #             scale_lora_layers(self.text_encoder_2, lora_scale)

    #     prompt = [prompt] if isinstance(prompt, str) else prompt

    #     if prompt_embeds is None:
    #         prompt_2 = prompt_2 or prompt
    #         prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    #         # We only use the pooled prompt output from the CLIPTextModel
    #         pooled_prompt_embeds = self._get_clip_prompt_embeds(
    #             prompt=prompt,
    #             device=device,
    #             num_images_per_prompt=num_images_per_prompt,
    #         )
    #         prompt_embeds = self._get_t5_prompt_embeds(
    #             prompt=prompt_2,
    #             num_images_per_prompt=num_images_per_prompt,
    #             max_sequence_length=max_sequence_length,
    #             device=device,
    #         )

    #     if self.text_encoder is not None:
    #         if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
    #             # Retrieve the original scale by scaling back the LoRA layers
    #             unscale_lora_layers(self.text_encoder, lora_scale)

    #     if self.text_encoder_2 is not None:
    #         if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
    #             # Retrieve the original scale by scaling back the LoRA layers
    #             unscale_lora_layers(self.text_encoder_2, lora_scale)

    #     dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
    #     text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    #     return prompt_embeds, pooled_prompt_embeds, text_ids