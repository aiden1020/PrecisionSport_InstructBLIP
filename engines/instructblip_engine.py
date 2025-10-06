import os
import argparse
from typing import List, Optional
from collections import OrderedDict
import torch

from lavis.common.config import Config
import lavis.tasks as tasks
from lavis.common.dist_utils import init_distributed_mode
from lavis.common.registry import registry


class InstructBLIPBadmintonVQAEngine:
    
    GENERAL_FORMATTING_GUIDANCE = (
        "Let's think step by step. First, provide your reasoning within `<thinking>` tags. "
        "Then, provide the final answer within `<answer>` tags."
    )
    MULTI_TASK_INSTRUCTIONS = {
        "temporal_grounding": (
            "<Video> This video has {n} strokes. "
            "Your task is to identify the exact strokes where a specific event occurs. "
            f"{GENERAL_FORMATTING_GUIDANCE} "
            "If the event occurs, your answer must be in the format: `The event happens at strokes i,j,…`. "
            "If the event does not occur, your answer must be: `The event does not occur`."
        ),
        "action_classification": (
            "<Video> This video has {n} strokes. "
            "Your task is to classify a specific action based on visual evidence. "
            f"{GENERAL_FORMATTING_GUIDANCE} "
            "Your final answer should be a concise sentence describing the identified action."
        ),
        "action_count": (
            "<Video> This video has {n} strokes. "
            "Your task is to count the total number of times a specific action occurs. "
            f"{GENERAL_FORMATTING_GUIDANCE} "
            "Your final answer should be a sentence stating the total count, for example: `The player performed 3 smashes.`"
        ),
        "summarisation": (
            "<Video> This video has {n} strokes. "
            "Your task is to summarize the tactical patterns and objectives based on the sequence of shots. "
            f"{GENERAL_FORMATTING_GUIDANCE} "
            "Your final answer should be a concise paragraph summarizing the strategy."
        ),
        "default": ( # Fallback instruction
            "<Video> This video has {n} strokes. Please answer the following question based on the video. "
            f"{GENERAL_FORMATTING_GUIDANCE}"
        )
    }

    def __init__(
        self,
        cfg_path: str = "lavis/projects/instructblip/inference/inference_instructblip_badminton_qa_coT_3.yaml",
        device: str = "cuda",
        use_dist: Optional[bool] = None,
        amp: bool = True,
        clip_cache_size: int = 0,
        num_beams: int = 5,
        max_len: int = 300,
        min_len: int = 30,
        length_penalty: float = 0.0,
    ):
        self.device = device
        self.amp = amp and torch.cuda.is_available() and device.startswith("cuda")
        self.dtype = torch.float16 if (self.amp and device.startswith("cuda")) else torch.float32

        self.gen_params = dict(
            num_beams=num_beams,
            inference_method="generate",
            max_len=max_len,
            min_len=min_len,
            length_penalty=length_penalty,
        )

        args = argparse.Namespace(cfg_path=cfg_path, options=None, cfg_options=None)
        self.cfg = Config(args)

        if use_dist is None:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            use_dist = world_size > 1
        if use_dist:
            init_distributed_mode(self.cfg.run_cfg)

        self.task = tasks.setup_task(self.cfg)
        self.model = self.task.build_model(self.cfg).to(self.device).eval()

        proc_cfg = self.cfg.datasets_cfg["badminton_qa"].vis_processor.eval
        self.vis_processor = registry.get_processor_class(proc_cfg.name).from_config(proc_cfg)

        txt_proc_cfg = self.cfg.datasets_cfg["badminton_qa"].text_processor.eval
        self.text_processor = registry.get_processor_class(txt_proc_cfg.name).from_config(txt_proc_cfg)

        self.qformer_instruction_proc = self.text_processor("<Video> A short video description:")

        self._cache_max = max(0, int(clip_cache_size))
        self._clip_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def _get_clip_tensor(self, path: str) -> torch.Tensor:
        if self._cache_max > 0 and path in self._clip_cache:
            t = self._clip_cache.pop(path)
            self._clip_cache[path] = t  
            return t

        clip_tensor = self.vis_processor(path)  
        if (not isinstance(clip_tensor, torch.Tensor)) or clip_tensor.dim() != 4:
            raise ValueError(
                f"Processor output must be [T,C,H,W], but {path}"
                f"{None if not isinstance(clip_tensor, torch.Tensor) else clip_tensor.shape}"
            )

        if self._cache_max > 0:
            self._clip_cache[path] = clip_tensor.cpu()
            while len(self._clip_cache) > self._cache_max:
                self._clip_cache.popitem(last=False)
        return clip_tensor

    def _build_samples(self, items: List[List[str]], question: str, task_id: str):
        per_sample_clips = []
        text_inputs, qformer_instructions = [], []
        for clip_paths in items:
            clip_tensors = [self._get_clip_tensor(p) for p in clip_paths]
            T, C, H, W = clip_tensors[0].shape
            for t in clip_tensors:
                if tuple(t.shape) != (T, C, H, W):
                    raise ValueError(
                        f"shape error: {t.shape} vs {(T,C,H,W)}"
                    )
            per_sample_clips.append(torch.stack(clip_tensors, dim=0).cpu())

            n = len(clip_paths)
            
            instruction_template = self.MULTI_TASK_INSTRUCTIONS.get(task_id, self.MULTI_TASK_INSTRUCTIONS["default"])
            instruction = instruction_template.format(n=n)
            
            processed_q = self.text_processor(f"{instruction} Question: {question} Answer:")
            text_inputs.append(processed_q)
            qformer_instructions.append(self.qformer_instruction_proc)

        ks = [x.size(0) for x in per_sample_clips]
        K_max = max(ks) if ks else 0

        padded, clip_masks = [], []
        for img, k_i in zip(per_sample_clips, ks):
            mask = torch.tensor([1] * k_i + [0] * (K_max - k_i), dtype=torch.bool)
            clip_masks.append(mask)

            if k_i < K_max:
                T, C, H, W = img.shape[1:]
                pad = torch.zeros((K_max - k_i, T, C, H, W), dtype=img.dtype)
                img = torch.cat([img, pad], dim=0)
            padded.append(img)
        
        batch_images = torch.stack(padded, dim=0).to(self.device, non_blocking=True) if padded else torch.empty(0).to(self.device)
        clip_mask = torch.stack(clip_masks, dim=0).to(self.device, non_blocking=True) if clip_masks else torch.empty(0).to(self.device)

        samples = {
            "images": batch_images,
            "clip_mask": clip_mask,
            "text_input": text_inputs,      
            "Qformer_instruction": qformer_instructions,
        }
        return samples

    @torch.no_grad()
    def predict(self, items: List[List[str]], question: str, task_id="temporal_grounding"):
        samples = self._build_samples(items, question, task_id)
        
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.amp else torch.cuda.amp.autocast(enabled=False)
        )
        with torch.inference_mode(), autocast_ctx:
            outputs = self.model.predict_answers(samples, **self.gen_params)
        return outputs, samples



class InstructBLIPBadmintonCaptioningEngine:
    def __init__(
        self,
        cfg_path: str = "lavis/projects/instructblip/inference/inference_instructblip_badminton_caption.yaml",
        device: str = "cuda",
        amp: bool = True,
        max_len: Optional[int] = None,
        num_beams: Optional[int] = None,
        prompt: str = "<video> a short video description",
    ):
        self.device = device
        self.amp = bool(amp)
        self.prompt = prompt

        args = argparse.Namespace(cfg_path=cfg_path, options=None, cfg_options=None)
        self.cfg = Config(args)

        self.task = tasks.setup_task(self.cfg)

        ds_key = "badminton_caption"
        proc_cfg = self.cfg.datasets_cfg[ds_key].vis_processor.eval
        self.vis_processor = registry.get_processor_class(proc_cfg.name).from_config(proc_cfg)

        self.model = self.task.build_model(self.cfg).to(self.device).eval()

        self.max_len = max_len if max_len is not None else getattr(self.cfg.run_cfg, "max_len", 30)
        self.num_beams = num_beams if num_beams is not None else getattr(self.cfg.run_cfg, "num_beams", 5)

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device, non_blocking=True)

    def _encode_video(self, video_path: str) -> torch.Tensor:
        clip = self.vis_processor(video_path)
        if not isinstance(clip, torch.Tensor) or clip.ndim != 4:
            raise ValueError(f"Expect [T,C,H,W] tensor from processor, got: {None if not isinstance(clip, torch.Tensor) else clip.shape}")
        return clip

    @torch.no_grad()
    def generate_one(self, video_path: str, prompt: Optional[str] = None) -> str:
        clip = self._encode_video(video_path)
        batch = self._to_device(clip.unsqueeze(0))  # [1,T,C,H,W]

        samples = {
            "image": batch, 
            "prompt": [prompt or self.prompt],
        }

        ctx = torch.autocast(device_type="cuda") if (self.amp and self.device.startswith("cuda")) else torch.cuda.amp.autocast(enabled=False)
        with torch.inference_mode(), ctx:
            captions = self.model.generate(
                samples,
                max_length=self.max_len,
                num_beams=self.num_beams,
            )
        return captions[0]

    @torch.no_grad()
    def generate_batch(self, video_paths: List[str], prompts: Optional[List[str]] = None) -> List[str]:
        if prompts is not None and len(prompts) != len(video_paths):
            raise ValueError("prompts 長度需與 video_paths 一致，或傳入 None 使用預設 prompt")

        clips = [self._encode_video(p) for p in video_paths]                 # list[[T,C,H,W]]
        batch = torch.stack([c for c in clips], dim=0)                        # [B,T,C,H,W]
        batch = self._to_device(batch)

        prompt_list = prompts if prompts is not None else [self.prompt] * len(video_paths)

        samples = {
            "image": batch,
            "prompt": prompt_list,
        }

        ctx = torch.autocast(device_type="cuda") if (self.amp and self.device.startswith("cuda")) else torch.cuda.amp.autocast(enabled=False)
        with torch.inference_mode(), ctx:
            captions = self.model.generate(
                samples,
                max_length=self.max_len,
                num_beams=self.num_beams,
            )
        return list(captions)