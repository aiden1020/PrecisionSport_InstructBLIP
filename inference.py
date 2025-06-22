import argparse
import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import init_distributed_mode
from lavis.common.registry import registry


def inference(video_path: str,
              cfg_path: str = "lavis/projects/instructblip/inference/inference_instructblip_badminton_caption.yaml",
              device: str = "cuda"):

    args = argparse.Namespace(cfg_path=cfg_path, options=None, cfg_options=None)
    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)

    task = tasks.setup_task(cfg)

    proc_cfg = cfg.datasets_cfg["badminton_caption"].vis_processor.eval
    processor = registry.get_processor_class(proc_cfg.name).from_config(proc_cfg)

    clip = processor(video_path)
    clip_batch = clip.unsqueeze(0).to(device)

    model = task.build_model(cfg).to(device)
    model.eval()

    inputs = {
        "image": clip_batch,
        "prompt": ["<video> a short video description"],
    }
    captions = model.generate(
            inputs,
            max_length=cfg.run_cfg.max_len,
            num_beams=cfg.run_cfg.num_beams
        )

    return captions[0]


def main():
    video_path = "lavis/configs/datasets/badminton_caption/input/images/game1_set1_16364.mp4"
    caption = inference(
        video_path=video_path,
        cfg_path="lavis/projects/instructblip/eval/finetune_instructblip_badminton_caption_3.yaml",
    )
    print(" Generated caption:\n", caption)

if __name__ == "__main__":
    main()
