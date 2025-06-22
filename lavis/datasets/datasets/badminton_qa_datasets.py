import os
import json
import random
import torch
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.vqa_datasets import VQAEvalDataset


class __BadmintonMixin:
    """Mixin for inspecting a sample."""
    def displ_item(self, index):
        sample = self[index]
        return OrderedDict({
            "images":        sample["images"],
            "question":      sample["question"],
            "question_id":   sample["question_id"],
            "answer":        sample["answer"],
            "is_impossible": sample["is_impossible"],
            "chunk_id":      sample["chunk_id"],
        })

INSTRUCTION = (
    "<Video> This video has {n} strokes. "
    "You must answer based only on the strokes you see—do not invent or hallucinate any events. "
    "Let's think step by step. "
    "If the event occurs, output exactly “The event happens at strokes i,j,…” to list the stroke indices"
    "If the event does not occur, output exactly “The event does not occur”"
    )
class BadmintonQADataset(BaseDataset, __BadmintonMixin):
    """
    Dataset for chunk-based badminton QA.
    Receives `ann_paths: list` from builder.
    """
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root: str,
        ann_paths: list,
        train_samples_portion="all"
    ):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        ann_path = ann_paths[0]
        with open(ann_path, 'r', encoding='utf-8') as f:
            self.annotation = json.load(f)

        if isinstance(train_samples_portion, int) and train_samples_portion > 0:
            self.annotation = random.sample(self.annotation, train_samples_portion)
        elif train_samples_portion != "all":
            raise ValueError("train_samples_portion must be a positive int or 'all'")

    def __len__(self):
        return len(self.annotation)

    @staticmethod
    def get_text_input(ann: dict) -> str:
        n = len(ann["image"])
        instruction = INSTRUCTION
        return f"{instruction} Question: {ann['question']} Answer:"

    def __getitem__(self, index):
        ann = self.annotation[index]
        # 1) 處理多支 .mp4
        clips = []
        for fname in ann["image"]:
            video_path = os.path.join(self.vis_root, fname)
            clip_tensor = self.vis_processor(video_path) 
            clips.append(clip_tensor)
        images_tensor = torch.stack(clips, dim=0)  

        # 2) 處理文字
        raw_text   = self.get_text_input(ann)
        text_input = self.text_processor(raw_text)
        Qformer_instruction = self.text_processor("<Video> A short video description:")

        return {
            "images":        images_tensor,
            "text_input":    text_input,
            "Qformer_instruction": Qformer_instruction,
            "text_output":   ann["answer"],
            "answer":        ann["answer"],
            "is_impossible": ann.get("is_impossible", False),
            "chunk_id":      ann.get("chunk_id", ""),
            "question_id":   ann.get("question_id", index),
            "question":      ann["question"],
        }

    def collater(self, samples):
        images_list, text_inputs,qformer_instruction, text_outputs, answers, qids, chunks, impossibles = ([] for _ in range(8))
        for s in samples:
            images_list.append(s["images"])         # tensor [k_i, T, C, H, W]
            text_inputs .append(s["text_input"])
            qformer_instruction.append(s["Qformer_instruction"])
            text_outputs.append(s["text_output"])
            answers     .append(s["answer"])
            qids        .append(s["question_id"])
            chunks      .append(s["chunk_id"])
            impossibles .append(s["is_impossible"])

        # 找本 batch 裡最大的 clip 數
        ks = [img.size(0) for img in images_list]
        K_max = max(ks)

        # pad 每個 sample 到 K_max 支 clip
        padded, clip_masks = [], []
        for img, k_i in zip(images_list, ks):
            # 建 clip mask: [K_max], 前 k_i 為 True，後面 pad 部分為 False
            mask = torch.tensor([1]*k_i + [0]*(K_max-k_i), dtype=torch.bool)
            clip_masks.append(mask)

            # pad 影像
            if k_i < K_max:
                T,C,H,W = img.shape[1:]
                pad = torch.zeros((K_max-k_i, T, C, H, W),
                                dtype=img.dtype, device=img.device)
                img = torch.cat([img, pad], dim=0)
            padded.append(img)

        # 最後在 batch 維度 stack
        batch_images = torch.stack(padded, dim=0)  # [B, K_max, T, C, H, W]
        clip_mask    = torch.stack(clip_masks, dim=0)  # [B, K_max]

        return {
            "images":        batch_images,
            "clip_mask":     clip_mask,
            "text_input":    text_inputs,
            "Qformer_instruction": qformer_instruction,
            "text_output":   text_outputs,
            "answer":        answers,
            "question_id":   qids,
            "chunk_id":      chunks,
            "is_impossible": impossibles,
        }



class BadmintonQAEvalDataset(VQAEvalDataset, __BadmintonMixin):
    """
    Evaluation dataset for chunk-based badminton QA.
    Receives `ann_paths: list` from builder.
    """
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root: str,
        ann_paths: list
    ):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        ann_path = ann_paths[0]
        with open(ann_path, 'r', encoding='utf-8') as f:
            self.annotation = json.load(f)

    def __len__(self):
        return len(self.annotation)

    @staticmethod
    def get_text_input(ann: dict) -> str:

        n = len(ann["image"])
        instruction = INSTRUCTION
        return f"{instruction} Question: {ann['question']} Answer:"

    def __getitem__(self, index):
        ann = self.annotation[index]
        clips = []
        for fname in ann["image"]:
            video_path = os.path.join(self.vis_root, fname)
            clip_tensor = self.vis_processor(video_path)
            clips.append(clip_tensor)
        images_tensor = torch.stack(clips, dim=0)

        raw_text   = self.get_text_input(ann)
        text_input = self.text_processor(raw_text)
        Qformer_instruction = self.text_processor("<Video> A short video description:")
        return {
            "images":        images_tensor,
            "text_input":    text_input,
            "Qformer_instruction": Qformer_instruction,
            "question_id":   ann.get("question_id", index),
            "chunk_id":      ann.get("chunk_id", ""),
            "is_impossible": ann.get("is_impossible", False),
            "answer":        ann.get("answer", ""),
            "question":      ann["question"],
        }

    def collater(self, samples):
        images_list, text_inputs,qformer_instruction, qids, chunks, impossibles, answers = ([] for _ in range(7))

        for s in samples:
            images_list.append(s["images"])
            text_inputs.append(s["text_input"])
            qformer_instruction.append(s["Qformer_instruction"])
            qids.append(s["question_id"])
            chunks.append(s["chunk_id"])
            impossibles.append(s["is_impossible"])
            answers.append(s["answer"])

        # 找本 batch 裡最大的 clip 數
        ks = [img.size(0) for img in images_list]
        K_max = max(ks)

        # pad 每個 sample 到 K_max 支 clip
        padded, clip_masks = [], []
        for img, k_i in zip(images_list, ks):
            # 建 clip mask: [K_max], 前 k_i 為 True，後面 pad 部分為 False
            mask = torch.tensor([1]*k_i + [0]*(K_max-k_i), dtype=torch.bool)
            clip_masks.append(mask)

            # pad 影像
            if k_i < K_max:
                T,C,H,W = img.shape[1:]
                pad = torch.zeros((K_max-k_i, T, C, H, W),
                                dtype=img.dtype, device=img.device)
                img = torch.cat([img, pad], dim=0)
            padded.append(img)

        # 最後在 batch 維度 stack
        batch_images = torch.stack(padded, dim=0)  # [B, K_max, T, C, H, W]
        clip_mask    = torch.stack(clip_masks, dim=0)  # [B, K_max]

        return {
            "images":        batch_images,
            "clip_mask":     clip_mask,
            "text_input":    text_inputs,
            "Qformer_instruction": qformer_instruction,
            "question_id":   qids,
            "chunk_id":      chunks,
            "is_impossible": impossibles,
            "answer":        answers,
        }
