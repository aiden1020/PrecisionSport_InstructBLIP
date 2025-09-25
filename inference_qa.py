import os
import torch
import argparse
from lavis.common.config import Config
import lavis.tasks as tasks
from lavis.common.dist_utils import init_distributed_mode
from lavis.common.registry import registry

INSTRUCTION = (
    "<Video> This video has {n} strokes. "
    "You must answer based only on the strokes you see—do not invent or hallucinate any events. "
    "Let's think step by step. "
    "If the event occurs, output exactly “The event happens at strokes i,j,…” to list the stroke indices"
    "If the event does not occur, output exactly “The event does not occur”"
    )
@torch.no_grad()
def single_inference(
    clip_paths,   # List[str] clip 路徑
    question: str,
    cfg_path: str = "lavis/projects/instructblip/inference/inference_instructblip_badminton_qa_coT_3.yaml",
    device: str = "cuda"
):
    # 1. 載入 config 與 task
    args = argparse.Namespace(cfg_path=cfg_path, options=None, cfg_options=None)
    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    task = tasks.setup_task(cfg)

    # 2. 建 processor
    proc_cfg = cfg.datasets_cfg["badminton_qa"].vis_processor.eval
    vis_processor = registry.get_processor_class(proc_cfg.name).from_config(proc_cfg)
    txt_proc_cfg = cfg.datasets_cfg["badminton_qa"].text_processor.eval
    text_processor = registry.get_processor_class(txt_proc_cfg.name).from_config(txt_proc_cfg)

    # 3. 處理 clips -> [K, T, C, H, W]
    clip_tensors = []
    for p in clip_paths:
        clip_tensor = vis_processor(p)
        if clip_tensor.dim() != 4:
            raise ValueError(f"Processor output must be [T,C,H,W], got {clip_tensor.shape}")
        clip_tensors.append(clip_tensor)

    K = len(clip_tensors)
    T, C, H, W = clip_tensors[0].shape
    for t in clip_tensors:
        if t.shape != (T, C, H, W):
            raise ValueError(f"Inconsistent clip shapes: {t.shape} vs {(T,C,H,W)}")

    images_tensor = torch.stack(clip_tensors, dim=0).unsqueeze(0).to(device)  # [1, K, T, C, H, W]
    clip_mask = torch.ones(1, K, dtype=torch.bool, device=device)             # [1, K]

    # 4. text_input & Qformer_instruction
    text_input = text_processor(f"{INSTRUCTION} Question: {question} Answer:")
    qformer_instruction = text_processor("<Video> A short video description:")

    # 5. 組 samples
    samples = {
        "images": images_tensor,
        "clip_mask": clip_mask,
        "text_input": [text_input],
        "Qformer_instruction": [qformer_instruction],
    }
    model = task.build_model(cfg).to(device)
    model.eval()
    # 6. 推論
    outputs = model.predict_answers(
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=64,
        min_len=1,
        length_penalty=0.0
    )

    return outputs, samples


if __name__ == "__main__":
    video_root = "lavis/configs/datasets/badminton_caption/input/images"
    clips = [
      "game1_set1_26511.mp4",
      "game1_set1_26531.mp4",
      "game1_set1_26570.mp4",
      "game1_set1_26591.mp4",
      "game1_set1_26612.mp4"
    ]
    clip_paths = [os.path.join(video_root, clip) for clip in clips]
    question = "What happened after the smash?"
    outputs, samples = single_inference(clip_paths, question)
    print("Predicted:", outputs)
