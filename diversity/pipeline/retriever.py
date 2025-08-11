import os
import glob
import re
import json
import argparse
import warnings
from typing import List, Tuple, Optional,Dict, Any
from collections import OrderedDict

import torch
import pandas as pd
import torch.distributed as dist

from lavis.common.config import Config
import lavis.tasks as tasks
from lavis.common.dist_utils import init_distributed_mode
from lavis.common.registry import registry

warnings.filterwarnings("ignore", category=FutureWarning)

INSTRUCTION = (
    "<Video> This video has {n} strokes. "
    "You must answer based only on the strokes you see—do not invent or hallucinate any events. "
    "Let's think step by step. "
    "If the event occurs, output exactly “The event happens at strokes i,j,…” to list the stroke indices"
    "If the event does not occur, output exactly “The event does not occur”"
)
def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        
def dump_json(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def merge_shards(
    in_dir: str,
    out_path: str,
    prefer: str = "first"  # "first" | "last"
) -> List[str]:
    cand = sorted(glob.glob(os.path.join(in_dir, "rank*.json")))
    if not cand:
        raise FileNotFoundError(f"在 {in_dir} 找不到 rank*.json")

    all_rows: List[Dict[str, Any]] = []
    for p in cand:
        rows = load_json(p)
        all_rows.extend(rows)

    by_idx: Dict[int, Dict[str, Any]] = {}
    for r in all_rows:
        gidx = int(r["global_idx"])
        if gidx in by_idx:
            if prefer == "last":
                by_idx[gidx] = r
        else:
            by_idx[gidx] = r

    merged = [by_idx[k] for k in sorted(by_idx.keys())]

    dump_json(out_path, merged)

    uniq = len(merged)
    dup = len(all_rows) - uniq
    
    all_video_paths = []
    for row in merged:
        if "video_paths" in row and isinstance(row["video_paths"], list):
            all_video_paths.extend(row["video_paths"])
    print("Merged")
    return all_video_paths
    
def extract_ans(text: str) -> Tuple[List[int], str]:
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL)
    thinking_str = thinking_match.group(1).strip() if thinking_match else ""

    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if m:
        nums = re.findall(r"\b\d+\b", m.group(1))
        return list(map(int, nums)), thinking_str

    m2 = re.search(r"The event happens at strokes? ([\d,]+)", text)
    if m2:
        nums = m2.group(1).split(",")
        return [int(n) for n in nums], thinking_str

    return [], thinking_str

def shard_indices(n: int, world_size: int, rank: int) -> List[int]:
    return [i for i in range(n) if (i % world_size) == rank]


def is_main_process():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

class InstructBLIPBadmintonEngine:
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

    def _build_samples(self, items: List[List[str]], question: str):
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
            processed_q = self.text_processor(f"{INSTRUCTION.format(n=n)} Question: {question} Answer:")
            text_inputs.append(processed_q)
            qformer_instructions.append(self.qformer_instruction_proc)

        ks = [x.size(0) for x in per_sample_clips]
        K_max = max(ks)

        padded, clip_masks = [], []
        for img, k_i in zip(per_sample_clips, ks):
            mask = torch.tensor([1] * k_i + [0] * (K_max - k_i), dtype=torch.bool)
            clip_masks.append(mask)

            if k_i < K_max:
                T, C, H, W = img.shape[1:]
                pad = torch.zeros((K_max - k_i, T, C, H, W), dtype=img.dtype)
                img = torch.cat([img, pad], dim=0)
            padded.append(img)

        batch_images = torch.stack(padded, dim=0).to(self.device, non_blocking=True)
        clip_mask = torch.stack(clip_masks, dim=0).to(self.device, non_blocking=True)

        samples = {
            "images": batch_images,               # [B,K,T,C,H,W]
            "clip_mask": clip_mask,              # [B,K]  
            "text_input": text_inputs,      
            "Qformer_instruction": qformer_instructions,
        }
        return samples

    @torch.no_grad()
    def predict(self, items: List[List[str]], question: str):
        samples = self._build_samples(items, question)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.amp else torch.cuda.amp.autocast(enabled=False)
        )
        with torch.inference_mode(), autocast_ctx:
            outputs = self.model.predict_answers(samples, **self.gen_params)
        return outputs, samples

class StrokeCsvRetriever:
    def __init__(self, csv_path: str, chunk_size: int, game_id: str = "game1",
                 cfg_path: str = "lavis/projects/instructblip/inference/inference_instructblip_badminton_qa_coT_3.yaml",
                 out_dir: str = "output/results/inference",
                 device: str = "cuda",
                 amp: bool = True,
                 clip_cache_size: int = 0,
                 num_beams: int = 5,
                 max_len: int = 300,
                 min_len: int = 30,
                 length_penalty: float = 0.0):
        self.game_id = game_id
        self.chunk_size = chunk_size
        self.csv_path = csv_path
        self.out_dir = out_dir
        self.engine = InstructBLIPBadmintonEngine(
            cfg_path=cfg_path,
            device=device,
            amp=amp,
            clip_cache_size=clip_cache_size,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            length_penalty=length_penalty,
        )
        os.makedirs(out_dir, exist_ok=True)


    def chunk_game_video(self):
        df = pd.read_csv(self.csv_path)
        df['game'] = df['id'].apply(lambda x: x.split('_')[0])
        df['set']  = df['id'].apply(lambda x: x.split('_')[1])

        if 'relabel_hit_area' in df.columns:
            df['relabel_hit_area'] = pd.to_numeric(df['relabel_hit_area'], errors='coerce')
            df.dropna(subset=['relabel_hit_area'], inplace=True)

        game_df = df[df['game'] == self.game_id]

        chunks = []
        for (game, set_num, rally_id), rally_group in game_df.groupby(['game', 'set', 'rally']):
            if len(rally_group) < 2:
                continue
            rally_chunks = [
                rally_group['id'].iloc[i:i+self.chunk_size].tolist()
                for i in range(0, len(rally_group), self.chunk_size)
            ]
            rally_path_chunks = [[f"{vid}.mp4" for vid in chunk] for chunk in rally_chunks]
            chunks.extend(rally_path_chunks)
        return chunks

    def retrieve(self, question: str, batch_size: int, video_root: str,
                 rank: int, world_size: int):
        
        chunks = self.chunk_game_video()
        idxs = shard_indices(len(chunks), world_size, rank)

        results = []
        for start in range(0, len(idxs), batch_size):
            batch_ids = idxs[start:start + batch_size]
            batch_chunks = [chunks[i] for i in batch_ids]
            items = [[os.path.join(video_root, it) for it in chunk] for chunk in batch_chunks]

            outputs, _ = self.engine.predict(items, question)
            for i, output in enumerate(outputs):
                ans, thinking = extract_ans(output)
                video_paths = [items[i][j] for j in ans if 0 <= j < len(items[i])]
                results.append({
                    "global_idx": int(batch_ids[i]),
                    "rank": rank,
                    "thinking": thinking,
                    "video_paths": video_paths,
                    "raw": output
                })
        out_path = os.path.join(self.out_dir, f"rank{rank:02d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        merged_results = []
        if is_main_process():
            merged_results = merge_shards(
                in_dir=self.out_dir,
                out_path=os.path.join(self.out_dir, "merged_results.json")
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        return merged_results if is_main_process() else []

