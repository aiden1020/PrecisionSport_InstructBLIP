import torch
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from lavis.models.tc_clip_encoder.datasets.pipeline import Compose


class TCClipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [123.675, 116.28, 103.53]
        if std is None:
            std = [58.395, 57.12, 57.375]
        
        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("tc_clip_image_eval")
class TCClipImageEvalProcessor(TCClipImageBaseProcessor):
    def __init__(self, image_size=224, num_frames=16, num_crop=1, num_clip=1, mean=None, std=None):
        super().__init__(mean=mean, std=std)
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

        scale_resize = int(256 / 224 * image_size)
        collect_keys = ['imgs']
        
        self.pipeline = Compose([
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_frames, test_mode=False),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, scale_resize)),
            dict(type='CenterCrop', crop_size=image_size),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=collect_keys, meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ])
        
        if num_crop == 3:
            self.pipeline[3] = dict(type='Resize', scale=(-1, image_size))
            self.pipeline[4] = dict(type='ThreeCrop', crop_size=image_size)
        if num_clip > 1:
            self.pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=num_frames, multiview=num_clip)
    
    def __call__(self, item):
        return self.pipeline({'filename': item, 'tar': False, 'modality': 'RGB', 'start_index': 0})['imgs']
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        return cls(
            image_size=cfg.get("image_size", 224),
            num_frames=cfg.get("num_frames", 16),
            num_crop=cfg.get("num_crop", 1),
            num_clip=cfg.get("num_clip", 1),
            mean=cfg.get("mean", None),
            std=cfg.get("std", None),
        )

