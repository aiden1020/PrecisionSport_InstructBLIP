import torch
from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize, compose
from lavis.models.tc_clip_encoder.datasets.pipeline import Compose
from lavis.models.tc_clip_encoder.trainers.build_trainer import returnCLIP
from lavis.models.tc_clip_encoder.utils.logger import create_logger
from lavis.models.tc_clip_encoder.utils.tools import load_checkpoint
import os
import torch.distributed as dist
class FeatureExtractor:
    def __init__(self, config_path="configs", output="workspace/inference",
                 tc_clip_model_path="pipeline/models/tc_clip_encoder/weight/fully_supervised_combined_stroke_22_87.pth",
                 device_id=None):
        
        if device_id is None and "LOCAL_RANK" in os.environ:
            device_id = int(os.environ["LOCAL_RANK"])
        elif device_id is None:
            device_id = 0
        
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

    def encode_videos_distributed(self, all_paths):
        if not dist.is_initialized():
            raise RuntimeError("need to initialize torch.distributed")

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if rank == 0:
            obj_list = [all_paths]
        else:
            obj_list = [None]
        dist.broadcast_object_list(obj_list, src=0)
        all_paths = obj_list[0]

        total_videos = len(all_paths)

        per_rank_paths = all_paths[rank::world_size]

        local_features = []
        for i, path in enumerate(per_rank_paths):
            if i % 20 == 0:
                print(f"[Rank {rank}] Processing {i+1}/{len(per_rank_paths)} videos...")
            feat = self.encode_video([path])
            vec = feat.squeeze(0).cpu().numpy()
            local_features.append(vec)

        gathered_features = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_features, local_features)

        all_features = []
        for feats in gathered_features:
            all_features.extend(feats)

        assert len(all_features) == total_videos, \
            f"features {len(all_features)} != {total_videos}"

        if rank == 0:
            print(f"[Rank 0] feature extract finished {len(all_features)} ")

        return all_features

    def _process_videos(self, video_paths):
        data_dicts = [{'filename': vp, 'tar': False, 'modality': 'RGB', 'start_index': 0} for vp in video_paths]
        tensors = [self.pipeline(data)['imgs'].unsqueeze(0).to(self.device).float() for data in data_dicts]
        return tensors
