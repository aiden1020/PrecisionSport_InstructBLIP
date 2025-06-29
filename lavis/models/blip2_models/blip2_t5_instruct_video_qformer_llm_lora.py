"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import string
import random
import copy
import os

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

import transformers
from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import lavis.models.blip2_models.Qformer_lora as Qformer_lora 
from lavis.common.utils import is_url
from lavis.common.dist_utils import download_cached_file
from lavis.models.blip2_models.Qformer_lora import lora, custom_lora, mark_only_lora_as_trainable, check_lora_application


@registry.register_model("blip2_t5_instruct_video_qformer_llm_lora")
class Blip2T5InstructVideoQformerLLMLoRA(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - flant5xl
        - flant5xxl
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5_instruct", "flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "flant5xl": "configs/models/blip2/blip2_instruct_flant5xl_qformer_lora.yaml",
        "flant5xxl": "configs/models/blip2/blip2_instruct_flant5xxl_qformer_lora.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        max_clips=16,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        num_few_shot_examples=0,
        few_shot_prob=0,
        qformer_text_input=True,
        llm_lora_r=8,
        llm_lora_apply="attn",
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
            
        if self.visual_encoder.num_features !=1408:
            self.projector = nn.Linear(self.visual_encoder.num_features, 1408)
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, 1408
            )
        else:
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        # Train only the Qformer LoRA
        
        mark_only_lora_as_trainable(self.Qformer)
        
        check_lora_application(self.Qformer)
        
        num_params = sum([p.numel() for p in self.Qformer.parameters() if p.requires_grad])
        print(f"Number of trainable parameters in Qformer: {num_params}")
        
        self.t5_tokenizer        = T5TokenizerFast.from_pretrained(t5_model, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')

        special = {"additional_special_tokens": ["<thinking>", "</thinking>", "<answer>", "</answer>"]}
        n1 = self.t5_tokenizer.add_special_tokens(special)
        n2 = self.t5_output_tokenizer.add_special_tokens(special)
        
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )
        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False 
            param.data = param.data.float()
        
        def _find_all_linear_names(model):
            cls = torch.nn.Linear
            lora_module_names = set()
            module_names = set()
            for name, module in model.named_modules():
                print(f"all print :{type(module)}")
                module_names.add(name)
                if isinstance(module, cls):
                    print(name)
                    names = name.split('.')
                    lora_module_names.add('.'+names[0] if len(names) == 1 else '.'+names[-1])
            print(f"1st val {list(module_names)}")
            print(f"2nd val {list(lora_module_names)}")
            # if 'lm_head' in lora_module_names: # needed for 16-bit
            #     lora_module_names.remove('lm_head')
            return list(lora_module_names)
        
        target_modules = []
        if llm_lora_apply == "attn":
            target_modules = ['q','v'] 
        elif llm_lora_apply == "ffn":
            target_modules = ["wi", "wo", "wi_1", "wi_0"]
        elif llm_lora_apply == "all":
            target_modules = ['q', 'v', "wi", "wo", "wi_1", "wi_0"] 
        else: 
            print("Wrong llm_lora_apply value in yaml!!")
        print(f"applying llm lora on {llm_lora_apply}")
        lora_config = LoraConfig(
            r=llm_lora_r,
            lora_alpha=8,
            target_modules=target_modules, #_find_all_linear_names(self.t5_model),
            # lora_dropout=training_args.lora_dropout,
            # bias=training_args.lora_bias,
            task_type="SEQ_2_SEQ_LM",
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        self.t5_model.print_trainable_parameters()
        
        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )
        for param in self.t5_proj.parameters():
            param.requires_grad = False
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        self.clip_segment_embeddings = nn.Embedding(max_clips, self.t5_model.config.hidden_size)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_prob = few_shot_prob

        self.qformer_text_input = qformer_text_input

    def forward(self, samples):
        images    = samples["images"]     # [B, K, T, C, H, W]
        clip_mask = samples["clip_mask"]  # [B, K], dtype=torch.bool
        B, K, T, C, H, W = images.shape
        # 1) 視覺編碼
        images_flat = images.view(B*K, T, C, H, W)
        with self.maybe_autocast():
            feat_flat = (
                self.projector(self.visual_encoder(images_flat))
                if self.visual_encoder.num_features != 1408
                else self.visual_encoder(images_flat)
            )
            feat_flat = self.ln_vision(feat_flat).float()  # [B*K, N, Dv]

        # 2) 重塑、並用 clip_mask 生成 image_atts
        _, N, Dv = feat_flat.shape
        image_embeds = feat_flat.view(B, K, N, Dv)

        image_atts = clip_mask.unsqueeze(2).expand(B, K, N).long()  
        flat_embeds = image_embeds.view(B*K, N, Dv)
        flat_atts   = image_atts.view(B*K, N)  # padding clip 行全 0

        # 3) 準備 query tokens
        Q, Dq = self.query_tokens.size(-2), self.query_tokens.size(-1)
        base_qt = self.query_tokens.view(1, Q, Dq)
        flat_qt = base_qt.expand(B*K, Q, Dq)

        # 4) 用 clip_mask 生成 flat_qat（padding clip 的 query 全 mask 掉）
        flat_clip_mask = clip_mask.view(B*K)            # [B*K], 1=valid 0=pad
        flat_qat = flat_clip_mask.unsqueeze(1)           # [B*K, 1]
        flat_qat = flat_qat.expand(B*K, Q).long()        # [B*K, Q]
        # 5) Q-Former forward（仍帶入 flat_attn、flat_atts）
        if self.qformer_text_input:
            txt = self.tokenizer(
                samples["Qformer_instruction"],
                padding='longest', truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(flat_embeds.device)
            flat_ids  = txt.input_ids.repeat_interleave(K, dim=0)
            flat_mask = txt.attention_mask.repeat_interleave(K, dim=0)
            qformer_attn = torch.cat([flat_qat, flat_mask], dim=1)
            query_out = self.Qformer.bert(
                flat_ids,
                attention_mask=qformer_attn,
                query_embeds=flat_qt,
                encoder_hidden_states=flat_embeds,
                encoder_attention_mask=flat_atts,
                return_dict=True,
            )
        else:
            query_out = self.Qformer.bert(
                query_embeds=flat_qt,
                encoder_hidden_states=flat_embeds,
                encoder_attention_mask=flat_atts,
                return_dict=True,
            )

        # 6) 擷取前 Q token、投影到 T5
        flat_seq = query_out.last_hidden_state       # [B*K, S, Dq]
        flat_query = flat_seq[:, :Q, :]              # [B*K, Q, Dq]
        inputs_t5 = self.t5_proj(flat_query)      
        inputs_t5 = inputs_t5.reshape(B, K*Q, self.t5_model.config.hidden_size)

        clip_ids = torch.arange(K, device=inputs_t5.device)                  # [K]
        clip_ids = clip_ids.unsqueeze(0).repeat(B, 1)              # [B, K]
        clip_ids_flat = clip_ids.repeat_interleave(Q, dim=1)       # [B, K*Q]
        segment_embeds = self.clip_segment_embeddings(clip_ids_flat)
        inputs_t5 = inputs_t5 + segment_embeds                    # [B, K*Q, D_t5]

        # 7) 用 clip_mask 生成 atts_t5（padding clip 對應 positions 全 mask 掉）
        atts_t5 = flat_clip_mask.unsqueeze(1)        # [B*K,1]
        atts_t5 = atts_t5.expand(B*K, Q).reshape(B, K*Q).long()
        # 8) 文本 tokenizer & few-shot
        # print(f"samples['text_input'] : {samples['text_input']}")
        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest", truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(inputs_t5.device)
            output_tokens = self.t5_output_tokenizer(
                samples["text_output"],
                padding="longest", truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt"
            ).to(inputs_t5.device)
            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
            inputs_embeds = torch.cat([
                inputs_t5,
                self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            ], dim=1)
            # 9) T5 forward
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )

            return {"loss": outputs.loss}

    def prepare_few_shot_embeds(self, samples):
        this_n_fs = random.choices(
            list(range(self.num_few_shot_examples + 1)),
            weights=[1 - self.few_shot_prob] + [self.few_shot_prob / self.num_few_shot_examples] * self.num_few_shot_examples
        )[0]

        if this_n_fs == 0:
            return None, None

        images = []
        text_input = []
        for sample in samples:
            for n in range(this_n_fs):
                images.append(sample['image'][n])
                text_input.append(sample['text_input'][n])
        images = torch.stack(images, dim=0)

        image = images

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                text_input,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask = Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        if this_n_fs > 1:
            encoder_atts = encoder_atts.reshape(encoder_atts.size(0) // this_n_fs, encoder_atts.size(1) * this_n_fs)
            inputs_embeds = inputs_embeds.reshape(inputs_embeds.size(0) // this_n_fs, inputs_embeds.size(1) * this_n_fs, inputs_embeds.size(2))

        return inputs_embeds, encoder_atts

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        prompt = samples.get("prompt", self.prompt)
        images = samples["images"]  # [B, K, T, C, H, W]
        clip_mask = samples["clip_mask"]  # [B, K], dtype=torch.bool
        B, K, T, C, H, W = images.shape

        if isinstance(prompt, str):
            prompt = [prompt] * B
        else:
            assert len(prompt) == B

        images_flat = images.view(B * K, T, C, H, W)

        if self.visual_encoder.num_features != 1408:
            with self.maybe_autocast():
                feat_flat = self.projector(self.visual_encoder(images_flat))
        else:
            with self.maybe_autocast():
                feat_flat = self.visual_encoder(images_flat)
        with self.maybe_autocast():
            feat_flat = self.ln_vision(feat_flat).float()

        _, N, Dv = feat_flat.shape
        image_embeds = feat_flat.reshape(B, K, N, Dv)
        image_atts = clip_mask.unsqueeze(2).expand(B, K, N).long()  

        flat_embeds = image_embeds.reshape(B*K, N, Dv)
        flat_atts   = image_atts.view(B*K, N) 

        Q, Dq = self.query_tokens.size(-2), self.query_tokens.size(-1)
        base_qt = self.query_tokens.view(1, Q, Dq)
        flat_qt = base_qt.expand(B*K, Q, Dq)                    

        flat_clip_mask = clip_mask.view(B*K)           
        flat_qat = flat_clip_mask.unsqueeze(1)         
        flat_qat = flat_qat.expand(B*K, Q).long()     
        if self.qformer_text_input:
            text_tokens = self.tokenizer(
                samples["Qformer_instruction"],
                padding="longest",truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image_embeds.device)
            flat_ids  = text_tokens.input_ids.repeat_interleave(K, dim=0)
            flat_mask = text_tokens.attention_mask.repeat_interleave(K, dim=0)
            qformer_attn = torch.cat([flat_qat, flat_mask], dim=1)  
            query_out = self.Qformer.bert(
                flat_ids,
                attention_mask=qformer_attn,
                query_embeds=flat_qt,
                encoder_hidden_states=flat_embeds,
                encoder_attention_mask=flat_atts,
                return_dict=True,
            )
        else:
            query_out = self.Qformer.bert(
                query_embeds=flat_qt,
                encoder_hidden_states=flat_embeds,
                encoder_attention_mask=flat_atts,
                return_dict=True,
            )
        flat_seq = query_out.last_hidden_state       # [B*K, S, Dq]
        flat_query = flat_seq[:, :Q, :]              # [B*K, Q, Dq]
        inputs_t5 = self.t5_proj(flat_query)      
        inputs_t5 = inputs_t5.reshape(B, K*Q, self.t5_model.config.hidden_size)
        
        clip_ids = torch.arange(K, device=inputs_t5.device)                  # [K]
        clip_ids = clip_ids.unsqueeze(0).repeat(B, 1)              # [B, K]
        clip_ids_flat = clip_ids.repeat_interleave(Q, dim=1)       # [B, K*Q]
        segment_embeds = self.clip_segment_embeddings(clip_ids_flat)
        inputs_t5 = inputs_t5 + segment_embeds                    # [B, K*Q, D_t5]

        atts_t5 = flat_clip_mask.unsqueeze(1)       
        atts_t5 = atts_t5.expand(B*K, Q).reshape(B, K*Q).long()
        with self.maybe_autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                prompt,
                padding="longest",
                return_tensors="pt"
            ).to(inputs_t5.device)
            text_mask   = input_tokens.attention_mask                        

            encoder_atts   = torch.cat([atts_t5,   text_mask],  dim=1)
            input_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids) 
            inputs_embeds = torch.cat([inputs_t5, input_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode( outputs, skip_special_tokens=False)
        cleaned_output_text = []
        for txt in output_text:
            txt = txt.replace(self.t5_tokenizer.eos_token, "")
            txt = txt.replace(self.t5_tokenizer.pad_token, "")
            txt = txt.strip()
            cleaned_output_text.append(txt)
        return cleaned_output_text




    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0)
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                # if 'context' in samples.keys():
                #     this_sample['context'] = [samples["context"][i]]

                # if 'history' in samples.keys():
                #     this_sample['history'] = [samples["history"][i]]

                # if 'caption' in samples.keys():
                #     this_sample['caption'] = [samples["caption"][i]]

                

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                # results = [[a], [b], [c]] -> [a, b, c]
                results = [res[0] for res in results]
                pass

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - prompt: the instruction
            candidates:
                (list): A list of candidate class names;
            n_segments:
                (int): Split the candidates into n_segments and predict one by one. This is useful when the number of candidates is too large.
        Returns:
            output_class: predicted class index
        """

        image = samples["image"]
        # prompt = samples["prompt"]

        bs = image.size(0)

        # if isinstance(prompt, str):
        #     prompt = [prompt] * bs
        # else:
        #     assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # if "text_input" in samples.keys():
        #     if type(samples["text_input"][0]) == list:
        #         prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
        #     else:
        #         prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # # scienceqa
        # if 'context' in samples.keys() and samples['context'] != '':
        #     prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # # visual dialog
        # if 'history' in samples.keys() and samples['history'][0] != '':
        #     prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        # if 'caption' in samples.keys() and samples['caption'][0] != '':
        #     prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]
        
        prompt = [samples['text_input'][i] for i in range(len(image))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_t5, atts_t5 = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_t5 = self.t5_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_t5 = self.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)
        output_tokens = self.t5_tokenizer(
            candidates, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        n_cands = len(candidates)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
            )

            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                # this_encoder_outputs = copy.deepcopy(encoder_outputs)
                this_encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0].clone(),
                )

                this_encoder_outputs['last_hidden_state'] = this_encoder_outputs[0].repeat_interleave(seg_len, dim=0)
                this_encoder_atts = encoder_atts.repeat_interleave(seg_len, dim=0)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                this_output_tokens_ids = output_tokens.input_ids[start_i:end_i].repeat(bs, 1)
                this_output_tokens_atts = output_tokens.attention_mask[start_i:end_i].repeat(bs, 1)

                this_targets = this_output_tokens_ids.masked_fill(this_output_tokens_ids == self.t5_tokenizer.pad_token_id, -100)

                outputs = self.t5_model(
                    encoder_outputs=this_encoder_outputs,
                    attention_mask=this_encoder_atts,
                    decoder_attention_mask=this_output_tokens_atts,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )
                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)
            top_predicted_classes = [candidates[idx] for idx in output_class_ranks[:, 0].tolist()]

            # encoder_outputs['last_hidden_state'] = encoder_outputs[0].repeat_interleave(n_cands, dim=0)
            # encoder_atts = encoder_atts.repeat_interleave(n_cands, dim=0)
            # output_tokens.input_ids = output_tokens.input_ids.repeat(bs, 1)
            # output_tokens.attention_mask = output_tokens.attention_mask.repeat(bs, 1)

            # # compute the LM loss for each candidate (sum logprob across all tokens) and select the highest
            # targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100)

            # outputs = self.t5_model(
            #     encoder_outputs=encoder_outputs,
            #     attention_mask=encoder_atts,
            #     decoder_attention_mask=output_tokens.attention_mask,
            #     return_dict=True,
            #     labels=targets,
            #     reduction="none",
            # )
            # loss = outputs.loss

            # loss = loss.reshape(bs, n_cands)
            # output_class_ranks = torch.argsort(loss, dim=-1) # (bs, num_candidates)

        return top_predicted_classes


    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        num_few_shot_examples = cfg.get("num_few_shot_examples", 0)
        few_shot_prob = cfg.get("few_shot_prob", 0.0)

        qformer_text_input = cfg.get("qformer_text_input", True)
        
        # TODO: if you want to control PEFT by config, you should add some varaibles here
        llm_lora_r = cfg.get("llm_lora_r", 8)
        llm_lora_apply = cfg.get("llm_lora_apply", "attn") 
        r = cfg.get("lora_r", 8)
        alpha = cfg.get("lora_alpha", 16)
        dropout = cfg.get("lora_dropout", 0.05)
        
        self_attention_qv_lora = cfg.get("self_attention_qv_lora", False)
        self_attention_output_lora = cfg.get("self_attention_output_lora", False)
        ffn_lora = cfg.get("ffn_lora", False)
        
        qformer_crossattention_lora_q = cfg.get("qformer_crossattention_lora_q", False)
        qformer_crossattention_lora_k = cfg.get("qformer_crossattention_lora_k", False)
        qformer_crossattention_lora_v = cfg.get("qformer_crossattention_lora_v", False)
        qformer_crossattention_lora_o = cfg.get("qformer_crossattention_lora_o", False)

        with lora(r, alpha, dropout, enabled=self_attention_qv_lora, qkv=[qformer_crossattention_lora_q, qformer_crossattention_lora_k, qformer_crossattention_lora_v]), custom_lora(r, alpha, dropout, enabled=(self_attention_output_lora or qformer_crossattention_lora_o), type="BertSelfOutput", sc=[self_attention_output_lora, qformer_crossattention_lora_o]), custom_lora(r, alpha, dropout, enabled=ffn_lora, type="BertOutput"):
            model = cls(
                vit_model=vit_model,
                img_size=img_size,
                drop_path_rate=drop_path_rate,
                use_grad_checkpoint=use_grad_checkpoint,
                vit_precision=vit_precision,
                freeze_vit=freeze_vit,
                num_query_token=num_query_token,
                t5_model=t5_model,
                prompt=prompt,
                max_txt_len=max_txt_len,
                max_output_txt_len=max_output_txt_len,
                apply_lemmatizer=apply_lemmatizer,
                num_few_shot_examples=num_few_shot_examples,
                few_shot_prob=few_shot_prob,
                qformer_text_input=qformer_text_input,
                llm_lora_r=llm_lora_r,
                llm_lora_apply=llm_lora_apply
            )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # strict=False for peft layers
        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg
