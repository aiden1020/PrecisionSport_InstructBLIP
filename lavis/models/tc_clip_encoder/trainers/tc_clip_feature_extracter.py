"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn

# from lavis.models.tc_clip_encoder.trainers.tc_clip_text_encoder import VPTextEncoder

class TCCLIP_encoder(nn.Module):
    def __init__(self, cfg, clip_model, logger):
        super().__init__()
        self.image_encoder = clip_model.visual
        # self.text_encoder = VPTextEncoder(cfg, clip_model, logger)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype if cfg.opt_level != 'O0' else torch.float32
        # self.prompt_generation_layer_level = self.text_encoder.prompt_generation_layer_level
        self.return_layer_num = [11]
        self.num_features = 512
        # if 11 not in self.return_layer_num:
        #     self.return_layer_num.append(11)
        logger.info(f"Using context tokens from vision layer {self.return_layer_num}")

    def forward(self, image, return_attention=False, return_source=False):

        # Encode visual features
        image_features, _, _ = self.image_encoder(image.type(self.dtype),
                                                                          return_layer_num=self.return_layer_num,
                                                                          return_attention=return_attention,
                                                                          return_source=return_source)
        # Now take the mean along the temporal direction with last layer cls tokens
        # image_features_mean = image_features[:, -1, ...].mean(dim=1, keepdim=False)
        image_features_mean = image_features.mean(dim=1, keepdim=False)
        # image_features_mean = image_features_mean / image_features_mean.norm(dim=-1, keepdim=True)  # [b, 512]
        return image_features_mean
    def get_num_layer(self):
        return self.image_encoder.get_num_layer()