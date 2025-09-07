from transformers import (
    PreTrainedModel, 
    PretrainedConfig, 
    CLIPVisionConfig,
    CLIPVisionModel, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    Blip2ForConditionalGeneration,
    Blip2Config,
    T5Config,
    Blip2VisionModel,
    Blip2QFormerModel,
    Blip2Processor,
    MllamaConfig,
    MllamaForConditionalGeneration,
    MllamaProcessor,
    AutoTokenizer,
    AutoProcessor,
    AutoConfig
)
from transformers.utils import logging, ModelOutput
from transformers.models.auto import CONFIG_MAPPING, TOKENIZER_MAPPING
import torch
from torch import nn
import re
from typing import Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from thinkdiff.common.registry import registry
from thinkdiff.models.base_model import BaseModel
from torch.nn import functional as F
import random
from transformers.utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput, Blip2PreTrainedModel
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG, T5LayerNorm

from thinkdiff.models.model_utils import (
    EmptyConfig,
    IdentityMap,
    )
import warnings
from torch.nn import CrossEntropyLoss
from functools import partial
import time
import os

logger = logging.get_logger(__name__)

from vllm import LLM, SamplingParams


def build_vision_projector(config):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if "t5_norm" in projector_type:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu_t5_norm$', projector_type)
    else:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)

    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            if "t5_norm" in projector_type:
                layer_norm = T5LayerNorm(config.hidden_size)
            else:
                layer_norm = nn.LayerNorm(config.hidden_size)
            # with torch.no_grad():
            #     layer_norm.bias.fill_(0.0)
            #     layer_norm.weight.fill_(0.1)
            modules.append(layer_norm)
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


class BlipT5DecoderConfig(Blip2Config):
    def __init__(self, max_txt_len=32, mm_projector_type="mlp2x_gelu", vision_downsample_factor=None, **kwargs):
        super().__init__(**kwargs)
        self.max_txt_len = max_txt_len
        self.mm_projector_type = mm_projector_type
        self.vision_downsample_factor = vision_downsample_factor


class T5ForDecoder(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        extra_attention_mask: Optional[torch.FloatTensor] = None,
        extra_encoder_outputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
            if extra_attention_mask is not None:
                extra_attention_mask = extra_attention_mask.to(self.decoder.first_device)
            if extra_encoder_outputs_embeds is not None:
                extra_encoder_outputs_embeds = extra_encoder_outputs_embeds.to(self.decoder.first_device)

        
        if extra_attention_mask is not None:
            attention_mask = torch.cat([extra_attention_mask, attention_mask], dim=1)
        if extra_encoder_outputs_embeds is not None:
            hidden_states = torch.cat([extra_encoder_outputs_embeds, hidden_states], dim=1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def random_split_string(s):
    words = s.split(" ")  # Split the string into words
    if len(words) <= 1:
        return "", s  # If there's only one word or none, return the string as is
    split_point = random.randint(1, len(words) - 1)  # Randomly choose a split point
    part1 = ' '.join(words[:split_point])  # First part
    part2 = ' '.join(words[split_point:])  # Second part
    return part1, part2


@dataclass
class MllamaVllmT5DecoderForConditionalGenerationModelOutput(ModelOutput):

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )

class MllamaVllmT5DecoderConfig(PretrainedConfig):
    model_type = "mllama-t5-decoder"


    def __init__(
        self,
        # mllama_config=None,
        # text_config=None,
        # processor_config=None,
        mllama_model_id=None,
        text_model_id=None,
        max_txt_len=32,
        mm_projector_type="mlp2x_gelu",
        vision_downsample_factor=None,
        use_decoder_only_language_model=False,
        mllama_batch_step=None,
        vllm_config=None,
        text_input_key=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.mllama_config = MllamaConfig(**mllama_config)

        # text_model_type = text_config["model_type"] if "model_type" in text_config else "t5"
        # self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        # self.processor_config = processor_config


        self.mllama_model_id = mllama_model_id
        self.text_model_id = text_model_id

        self.max_txt_len = max_txt_len
        self.mm_projector_type = mm_projector_type
        self.vision_downsample_factor = vision_downsample_factor
        self.use_decoder_only_language_model = use_decoder_only_language_model

        self.mllama_batch_step = mllama_batch_step
        self.vllm_config = vllm_config
        self.text_input_key = text_input_key

# Class to store embeddings and manage hook
class EmbeddingExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.embeddings = []
        self.register_hook()

    # Hook function to capture embeddings from the final layer
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            self.embeddings.append(output[0])
        else:
            self.embeddings.append(output)

    # Register the hook
    def register_hook(self):
        layer = dict([*self.model.named_modules()])[self.layer_name]
        layer.register_forward_hook(self.hook_fn)

    # Retrieve embeddings after forward pass
    def get_embeddings(self):
        return self.embeddings
    
    # Retrieve embeddings after forward pass
    def reset_embeddings(self):
        self.embeddings = []

@registry.register_model("mllama-vllm-generate-1")
class MllamaVllmGenerate_1(PreTrainedModel, BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_mllama_vllm_generate_1": "configs/models/mllama_vllm_generate_1.yaml",
    }
    config_class = MllamaVllmT5DecoderConfig
    main_input_name = "pixel_values"

    def __init__(self, config: MllamaVllmT5DecoderConfig):
        super().__init__(config)

        # self.mllama = MllamaForConditionalGeneration._from_config(
        #     config.mllama_config,
        #     # attn_implementation=config._attn_implementation
        # )

        # self.mllama_processor = MllamaProcessor(*config.processor_config)

        if "return_hidden_states" in self.config.vllm_config:
            self.mllama = LLM(
                model=config.mllama_model_id,
                # max_model_len=2048,
                max_model_len=self.config.vllm_config["max_model_len"],
                max_num_batched_tokens=self.config.vllm_config["max_num_batched_tokens"],
                max_num_seqs=self.config.vllm_config["max_num_seqs"],
                enforce_eager=self.config.vllm_config["enforce_eager"],
                gpu_memory_utilization=self.config.vllm_config["gpu_memory_utilization"],
                tensor_parallel_size=self.config.vllm_config["tensor_parallel_size"],
                return_hidden_states=self.config.vllm_config["return_hidden_states"],
                trust_remote_code=True
            )
        else:    
            self.mllama = LLM(
                model=config.mllama_model_id,
                # max_model_len=2048,
                max_model_len=self.config.vllm_config["max_model_len"],
                max_num_batched_tokens=self.config.vllm_config["max_num_batched_tokens"],
                max_num_seqs=self.config.vllm_config["max_num_seqs"],
                enforce_eager=self.config.vllm_config["enforce_eager"],
                gpu_memory_utilization=self.config.vllm_config["gpu_memory_utilization"],
                tensor_parallel_size=self.config.vllm_config["tensor_parallel_size"],
                trust_remote_code=True
            )
        self.mllama_sampling_params = SamplingParams(
            temperature=self.config.vllm_config["temperature"],
            top_p=self.config.vllm_config["top_p"],
            max_tokens=self.config.vllm_config["max_tokens"],
            min_tokens=self.config.vllm_config["min_tokens"],
            ignore_eos=self.config.vllm_config["ignore_eos"],
            stop_token_ids=None)

        # assert self.config.vllm_config["max_tokens"] == self.config.vllm_config["min_tokens"]
        # assert self.config.vllm_config["ignore_eos"] == True
        
        # if self.config.mllama_model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-72B-Instruct"]:
        #     self.mllama_embedding_extractor = EmbeddingExtractor(
        #         self.mllama.llm_engine.model_executor.driver_worker.model_runner.model.model, 
        #         self.config.vllm_config["embedding_layer_name"])  # Adjust layer name
        # else:
        #     self.mllama_embedding_extractor = EmbeddingExtractor(
        #         self.mllama.llm_engine.model_executor.driver_worker.model_runner.model.language_model, 
        #         self.config.vllm_config["embedding_layer_name"])  # Adjust layer name

        # self.mllama_embedding_extractors = []
        # for embedding_layer_name in self.config.vllm_config["embedding_layer_names"]:
        #     self.mllama_embedding_extractors.append(
        #         EmbeddingExtractor(
        #             self.mllama.llm_engine.model_executor.driver_worker.model_runner.model, 
        #             embedding_layer_name
        #         )
        #     )  # Adjust layer name
        #         # "language_model.model.layers.38.post_attention_layernorm"

        self.config.mllama_config = AutoConfig.from_pretrained(config.mllama_model_id, trust_remote_code=True)

        # self.mllama_tokenizer = self.mllama.get_tokenizer()
        # self.mllama_tokenizer.encode = partial(self.mllama_tokenizer.encode, add_special_tokens=False)

        self.mllama_processor = AutoProcessor.from_pretrained(config.mllama_model_id, trust_remote_code=True)


        if self.config.mllama_model_id in ["llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llama3-llava-next-8b-hf", "llava-hf/llava-1.5-7b-hf", "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-72B-Instruct", "OpenGVLab/InternVL2_5-8B"]:
            self.add_special_tokens = True
        else:
            self.add_special_tokens = False

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + BLIP-2 + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def forward_inner(
        self,
        mllama_inputs,
        # pixel_values: torch.FloatTensor,
        # input_ids: torch.FloatTensor,
        # llama_input_images: torch.FloatTensor,
        # llama_input_texts: List[str],
        # attention_mask: Optional[torch.LongTensor] = None,
        # decoder_input_ids: Optional[torch.LongTensor] = None,
        # decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        # interpolate_pos_encoding: bool = False,
        # nonzero_qformer_output_token_num: Optional[int] = None,
        # nonzero_qformer_input_token_num: Optional[int] = None
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2Model
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        with torch.no_grad():

            if self.config.text_input_key is None:
                llama_input_texts = mllama_inputs["answers"]
            else:
                llama_input_texts = mllama_inputs[self.config.text_input_key]
            llama_input_images = mllama_inputs["images"]
            batch_size = len(llama_input_texts)
            llama_input_texts_formatted = []
            for i, prompt in enumerate(llama_input_texts):
                if self.config.mllama_model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-72B-Instruct"]:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": llama_input_images[i],
                                    # "min_pixels": 224 * 224,
                                    # "max_pixels": 1280 * 28 * 28,
                                },
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ]
                elif "InternVL" in self.config.mllama_model_id:
                    messages = [{'role': 'user', 'content': f"<image>\n{prompt}"}]
                else:
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]}
                    ]

                llama_input_texts_formatted.append(messages)
            llama_input_texts_formatted = self.mllama_processor.apply_chat_template(llama_input_texts_formatted, tokenize=False, add_generation_prompt=True)

            mllama_inputs = [{
                "prompt": llama_input_texts_formatted[i],
                "multi_modal_data": {
                    "image": llama_input_images[i]
                },
            } for i in range(batch_size)]

            mllama_outputs = self.mllama.generate(mllama_inputs, sampling_params=self.mllama_sampling_params, use_tqdm=True)
            # generate_end_time = time.time()
            # generate_elapsed_time = generate_end_time - generate_start_time
            # print(f"Generate elapsed time: {generate_elapsed_time} seconds")

            mllama_output_token_dict = {}

            mllama_output_token_dict["input_prompt"] = []
            mllama_output_token_dict["input_prompt_token_ids"] = []
            mllama_output_token_dict["output_text"] = []
            mllama_output_token_dict["output_token_ids"] = []

            for mllama_output in mllama_outputs:
                mllama_output_token_dict["input_prompt"].append(mllama_output.prompt)
                mllama_output_token_dict["input_prompt_token_ids"].append(mllama_output.prompt_token_ids)
                mllama_output_token_dict["output_text"].append(mllama_output.outputs[0].text)
                mllama_output_token_dict["output_token_ids"].append(mllama_output.outputs[0].token_ids)

            mllama_generated_text = [o.outputs[0].text for o in mllama_outputs]

            if self.config.mllama_model_id in ["llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-1.5-7b-hf"]:
                for text_i, text in enumerate(mllama_generated_text):
                    if text[0] == " ":
                        mllama_generated_text[text_i] = text.replace(" ", "", 1)

            
            mllama_output_embeddings_dict = {}
            mllama_output_embeddings = {}
            mllama_output_embeddings["output_embed"] = [mllama_outputs[i].outputs[0].hidden_states for i in range(batch_size)]
            mllama_output_embeddings["input_embed"] = [mllama_outputs[i].prompt_hidden_states for i in range(batch_size)]

            mllama_output_embeddings_dict[self.config.vllm_config["embedding_layer_name"]] = mllama_output_embeddings
        
        return {
            "generated_text": mllama_generated_text,
            "generated_token": mllama_output_token_dict,
            "generated_embed": mllama_output_embeddings_dict
        }

    
    def forward(self, samples, reduction='mean'):
        # pixel_values = samples["image"]
        
        # text_input = []
        # batch_size = len(pixel_values)
        # for i in range(batch_size):
        #     text_input.append(random.choice(llava_brief_instructions))
        #     pixel_values[i] = [pixel_values[i]]

        samples_input = {k: v for k, v in samples.items() if k not in ["epoch", "num_iters_per_epoch", "iters"]}

        outputs = self.forward_inner(
            mllama_inputs=samples_input
        )

        return outputs
    
    @classmethod
    def from_config(cls, cfg):

        dtype = cfg.get("dtype", "float32")
        if dtype == "float16":
            dtype = torch.float16
        elif dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        mllama_pretrained_model_name_or_path = cfg.get("mllama_pretrained_model_name_or_path", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        # mllama = MllamaForConditionalGeneration.from_pretrained(mllama_pretrained_model_name_or_path)
        # mllama_config = mllama.config
        # mllama_processor = MllamaProcessor.from_pretrained(mllama_pretrained_model_name_or_path)
        # processor_config = MllamaProcessor._get_arguments_from_pretrained(mllama_pretrained_model_name_or_path)

        text_pretrained_model_name_or_path = cfg.get("text_pretrained_model_name_or_path", "google/flan-t5-xxl")
        # language_model = T5ForDecoder.from_pretrained(text_pretrained_model_name_or_path, torch_dtype=dtype)
        # text_config = language_model.config
        
        model_config = MllamaVllmT5DecoderConfig(
            mllama_model_id=mllama_pretrained_model_name_or_path,
            text_model_id=text_pretrained_model_name_or_path,
            max_txt_len=cfg.get("max_txt_len", 32),
            mm_projector_type=cfg.get("mm_projector_type", "mlp2x_gelu"),
            vision_downsample_factor=cfg.get("vision_downsample_factor", None),
            use_decoder_only_language_model=cfg.get("use_decoder_only_language_model", False),
            mllama_batch_step=cfg.get("mllama_batch_step", None),
            vllm_config=cfg["vllm_config"],
            text_input_key=cfg.get("text_input_key", None)
        )

        model = MllamaVllmGenerate_1(model_config)

        # tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model_name_or_path)
        # model.tokenizer = tokenizer

        # model.mllama = mllama
        # model.mllama_processor = mllama_processor
        # model.language_model = language_model

        # if cfg.get("layer_norm_reinit_weight_with_language_encoder", False):
        #     with torch.no_grad():
        #         for module in model.mm_projector.modules():
        #             if isinstance(module, T5LayerNorm):
        #                 # get the last layer norm weight from the language model encoder
        #                 module.load_state_dict(model.language_model.encoder.final_layer_norm.state_dict())
        #                 print("Reinit T5LayerNorm with language encoder")

        # del model.language_model.encoder
        # model.mllama.to(dtype=dtype)
        # model.language_model.to(dtype=dtype)

        # if cfg.get("layer_norm_reinit_weight", None):
        #     with torch.no_grad():
        #         for module in model.mm_projector.modules():
        #             if isinstance(module, nn.LayerNorm):
        #                 module.weight.fill_(cfg.layer_norm_reinit_weight)
        #                 module.bias.fill_(0.0)
        
        # if cfg.get("freeze_mllama", True):
        #     for param in model.mllama.parameters():
        #         param.requires_grad = False

        # if cfg.get("freeze_language", True):
        #     for param in model.language_model.parameters():
        #         param.requires_grad = False

        # ckpt_path = cfg.get("ckpt", "")  # load weights of ClipT5
        # if ckpt_path:
        #     print("Load Checkpoint: {}".format(ckpt_path))
        #     ckpt = torch.load(ckpt_path, map_location="cpu")
        #     msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
