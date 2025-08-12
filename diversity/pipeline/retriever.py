import os
import glob
import re
import json
import warnings
from typing import List, Tuple,Dict, Any

import pandas as pd
import torch.distributed as dist

from lavis.common.registry import registry
from engines.instructblip_engine import InstructBLIPBadmintonEngine
warnings.filterwarnings("ignore", category=FutureWarning)

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

