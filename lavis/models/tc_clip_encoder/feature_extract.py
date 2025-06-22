import torch
from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize, compose
from tc_clip_encoder.datasets.pipeline import Compose
from tc_clip_encoder.trainers.build_trainer import returnCLIP
from tc_clip_encoder.utils.logger import create_logger
from tc_clip_encoder.utils.tools import load_checkpoint


class FeatureExtractor:
    def __init__(self, config_path="configs", output="workspace/inference",
                 tc_clip_model_path="workspace/expr/model.pth", device_id=0, disable_logger=True):
        self.device = torch.device(f"cuda:{device_id}")
        self.output = output
        self.tc_clip_model_path = tc_clip_model_path

        self.config = self._load_config(config_path)
        
        self.logger = self._setup_logger()

        self.model = self._build_model()

        self.pipeline = self._build_pipeline()

    def _load_config(self, config_path):
        overrides = [
            f"output={self.output}",
            "eval=test",
            "trainer=tc_clip",
            f"resume={self.tc_clip_model_path}"
        ]
        with initialize(version_base=None, config_path=config_path):
            config = compose(config_name="zero_shot.yaml", overrides=overrides)
        OmegaConf.set_struct(config, False)
        Path(config.output).mkdir(parents=True, exist_ok=True)
        return config

    def _setup_logger(self):
        self.config.trainer_name = "TCCLIP_encoder"
        logger = create_logger(output_dir=self.config.output, dist_rank=0, name=f"{self.config.trainer_name}")
        logger.disabled = True
        return logger

    def _build_model(self):
        model = returnCLIP(self.config, self.logger).to(self.device)
        if self.config.resume:
            load_checkpoint(self.config, model, None, None, self.logger, model_only=True)
        return model

    def _build_pipeline(self):
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False
        )
        scale_resize = int(256 / 224 * self.config.input_size)
        collect_keys = ['imgs']

        val_pipeline = [
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=self.config.num_frames, test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, scale_resize)),
            dict(type='CenterCrop', crop_size=self.config.input_size),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=collect_keys, meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]

        if self.config.num_crop == 3:
            print("yes")
            val_pipeline[3] = dict(type='Resize', scale=(-1, self.config.input_size))
            val_pipeline[4] = dict(type='ThreeCrop', crop_size=self.config.input_size)
        if self.config.num_clip > 1:
            val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1,
                                   num_clips=self.config.num_frames, multiview=self.config.num_clip)

        return Compose([p for p in val_pipeline if p is not None])

    def encode_video(self, video_paths):
        tensors = self._process_videos(video_paths)
        batch_tensor = torch.cat(tensors, dim=0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                features = self.model(batch_tensor)
        return features

    def _process_videos(self, video_paths):
        data_dicts = [{'filename': vp, 'tar': False, 'modality': 'RGB', 'start_index': 0} for vp in video_paths]
        tensors = [self.pipeline(data)['imgs'].unsqueeze(0).to(self.device).float() for data in data_dicts]
        return tensors
