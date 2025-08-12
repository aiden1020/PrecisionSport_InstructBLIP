import os
import random
from typing import List, Optional
import numpy as np
import json
import torch

from diversity.pipeline.retriever       import StrokeCsvRetriever 
from diversity.pipeline.feature_extract import FeatureExtractor
from diversity.pipeline.sampler         import VideoDiversitySampler
from diversity.pipeline.visualise       import FeatureVisualizer

from moviepy import VideoFileClip, concatenate_videoclips
from sklearn.metrics.pairwise import cosine_distances
import torch.distributed as dist

def main():
    csv_path = "diversity/label/relabel_encoder_data.csv"
    game_id = "game1"
    chunk_size = 10
    batch_size = 2
    query = "What stroke happens in the middle?"
    video_root = "lavis/configs/datasets/badminton_caption/input/images"
    cfg_path = "lavis/projects/instructblip/inference/inference_instructblip_badminton_qa_coT_3.yaml"
    amp = False
    clip_cache = 0
    num_threads = 1
    num_select  = 10
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, num_threads))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    retriever = StrokeCsvRetriever(
        csv_path=csv_path,
        chunk_size=chunk_size,
        game_id=game_id,
        cfg_path=cfg_path,
        device=device,
        amp=amp,
        clip_cache_size=clip_cache,
    )

    all_paths = retriever.retrieve(
        question=query,
        batch_size=batch_size,
        video_root=video_root,
        rank=rank,
        world_size=world_size
    )
    visualizer = FeatureVisualizer(output_dir="result")

    output_dir  = os.path.join("lavis/models/tc_clip_encoder/workspace/inference")
    model_path  = os.path.join("lavis/models/tc_clip_encoder/weight/fully-supervised-combined-22-86.pth")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    extractor = FeatureExtractor(
        output=output_dir,
        tc_clip_model_path=model_path
    )

    all_features = extractor.encode_videos_distributed(all_paths)

    if dist.get_rank() == 0:
        sampler = VideoDiversitySampler()
        sampler.distance_fn = sampler._euclidean_dist
        ordering, score_list = sampler.run(num_select, all_features)

        print(f"Running diversity sampling (selecting {num_select} from {len(all_features)})")
        ordering, score_list = sampler.run(num_select, all_features)
        sampler = VideoDiversitySampler()
        sampler.distance_fn = sampler._euclidean_dist


        print(f"Running diversity sampling (selecting {num_select} from {len(all_paths)})")
        ordering, score_list = sampler.run(num_select , all_features)
        selected_paths = [all_paths[i] for i in ordering]
        print("Selected Videos by Diversity Sampling")
        for order, (path, score) in enumerate(zip(selected_paths, score_list), start=1):
            print(f"{order:02d}. {os.path.basename(path)} (score: {score:.4f})")

        diversity_indices = []
        for selected_path in selected_paths:
            try:
                idx = all_paths.index(selected_path)
                diversity_indices.append(idx)
            except ValueError:
                print(f"Warning: Selected path not found in full set: {selected_path}")

        print(f"Creating random uniform sampling control group")
        random_indices = random.sample(range(len(all_paths)), num_select)
        random_paths = [all_paths[i] for i in random_indices]
        
        print("Selected Videos by Random Sampling (Control Group)")
        for order, (idx, path) in enumerate(zip(random_indices, random_paths), start=1):
            print(f"{order:02d}. {os.path.basename(path)} (index: {idx})")

        print(f"Selected {len(diversity_indices)} diverse videos from {len(all_paths)} total")
        print(f"Selected {len(random_indices)} random videos from {len(all_paths)} total")

        print("Generating diversity sampling visualization")
        visualizer.visualize_diversity(
            features=all_features,
            video_paths=all_paths,
            query="diversity",
            selected_indices=diversity_indices
        )
        
        print("Generating random sampling visualization")
        visualizer.visualize_diversity(
            features=all_features,
            video_paths=all_paths,
            query="random",
            selected_indices=random_indices
        )
        print("Generating combined comparison visualization")
        visualizer.visualize_comparison(
            features=all_features,
            query=query,
            diversity_indices=diversity_indices,
            random_indices=random_indices
        )
        print("="*60)
        print("DIVERSITY SAMPLING STATISTICS")
        print("="*60)
        visualizer.print_diversity_stats(
            features=all_features,
            video_paths=all_paths,
            selected_indices=diversity_indices
        )
        
        print("="*60)
        print("RANDOM SAMPLING STATISTICS (Control Group)")
        print("="*60)
        visualizer.print_diversity_stats(
            features=all_features,
            video_paths=all_paths,
            selected_indices=random_indices
        )
        
        def calculate_diversity_metrics(features, indices):
            if len(indices) < 2:
                return None

            processed_features = []
            for idx in indices:
                feat = np.asarray(features[idx])
                if feat.ndim > 1:
                    feat = feat.flatten()
                processed_features.append(feat)
            feature_matrix = np.stack(processed_features)

            distances = cosine_distances(feature_matrix)

            upper_tri = np.triu_indices_from(distances, k=1)
            pairwise_distances = distances[upper_tri]

            return {
                'mean_distance': np.mean(pairwise_distances),
                'std_distance': np.std(pairwise_distances),
                'min_distance': np.min(pairwise_distances),
                'max_distance': np.max(pairwise_distances)
            }
        
        diversity_metrics = calculate_diversity_metrics(all_features, diversity_indices)
        random_metrics = calculate_diversity_metrics(all_features, random_indices)
        
        print("="*60)
        print("COMPARATIVE ANALYSIS")
        print("="*60)
        if diversity_metrics and random_metrics:
            print(f"Diversity Sampling - Mean Distance: {diversity_metrics['mean_distance']:.4f}")
            print(f"Random Sampling   - Mean Distance: {random_metrics['mean_distance']:.4f}")
            
            improvement = (diversity_metrics['mean_distance'] - random_metrics['mean_distance']) / random_metrics['mean_distance'] * 100
            print(f"Improvement: {improvement:+.2f}% over random sampling")
            
            print(f"Diversity Sampling - Distance Range: [{diversity_metrics['min_distance']:.4f}, {diversity_metrics['max_distance']:.4f}]")
            print(f"Random Sampling   - Distance Range: [{random_metrics['min_distance']:.4f}, {random_metrics['max_distance']:.4f}]")
        
        diversity_output_file = os.path.join("result", f"selected_diversity.txt")
        with open(diversity_output_file, 'w') as f:
            f.write(f"Diversity Sampling Results for '{query}' stroke\n")
            f.write(f"Selected {len(selected_paths)} videos from {len(all_paths)} total\n\n")
            for order, (path, score) in enumerate(zip(selected_paths, score_list), start=1):
                f.write(f"{order:02d}. {os.path.basename(path)} (score: {score:.4f})\n")
            
            if diversity_metrics:
                f.write(f"\nDiversity Metrics:\n")
                f.write(f"Mean Distance: {diversity_metrics['mean_distance']:.4f}\n")
                f.write(f"Std Distance: {diversity_metrics['std_distance']:.4f}\n")
                f.write(f"Distance Range: [{diversity_metrics['min_distance']:.4f}, {diversity_metrics['max_distance']:.4f}]\n")
        
        random_output_file = os.path.join("result", f"selected_random.txt")
        with open(random_output_file, 'w') as f:
            f.write(f"Random Sampling Results for '{query}' stroke (Control Group)\n")
            f.write(f"Selected {len(random_paths)} videos from {len(all_paths)} total\n\n")
            for order, (idx, path) in enumerate(zip(random_indices, random_paths), start=1):
                f.write(f"{order:02d}. {os.path.basename(path)} (index: {idx})\n")
            
            if random_metrics:
                f.write(f"\nDiversity Metrics:\n")
                f.write(f"Mean Distance: {random_metrics['mean_distance']:.4f}\n")
                f.write(f"Std Distance: {random_metrics['std_distance']:.4f}\n")
                f.write(f"Distance Range: [{random_metrics['min_distance']:.4f}, {random_metrics['max_distance']:.4f}]\n")
        
        print(f"Diversity sampling results saved to: {diversity_output_file}")
        print(f"Random sampling results saved to: {random_output_file}")


        def get_clip_fps_and_size(clip_path: str) -> tuple[float, tuple[int,int]]:
            clip = VideoFileClip(clip_path)
            fps = clip.fps
            size = (clip.w, clip.h)
            clip.reader.close()
            if clip.audio:
                clip.audio.reader.close_proc()
            return fps, size

        def concatenate_clips(
            clip_paths: List[str],
            output_path: str,
            fps: Optional[float] = None,
            size: Optional[tuple[int,int]] = None,
            codec: str = "libx264",
            audio: bool = False
        ):

            if not clip_paths:
                print("[Warning] No clips to concatenate.")
                return

            if fps is None or size is None:
                first_fps, first_size = get_clip_fps_and_size(clip_paths[0])
                fps  = fps or first_fps
                size = size or first_size

            clips = []
            for path in clip_paths:
                try:
                    clip = VideoFileClip(path)
                    if (clip.w, clip.h) != size:
                        clip = clip.resize(newsize=size)
                    clips.append(clip)
                except Exception as e:
                    print(f"[Warning] failed to load {path}: {e}")

            if not clips:
                print("[Warning] No valid clips to concatenate.")
                return

            final = concatenate_videoclips(clips, method="compose")
            final.write_videofile(
                output_path,
                codec=codec,
                fps=fps,
                audio=audio
            )
            print(f"Concatenated video saved to: {output_path}")

            final.close()
            for clip in clips:
                clip.close()
        diversity_out = os.path.join("result", f"farthest_first.mp4")
        concatenate_clips(
            clip_paths=selected_paths,
            output_path=diversity_out,
            codec="libx264",
            audio=False
        )

        random_out = os.path.join("result", f"uniform_sampling.mp4")
        concatenate_clips(
            clip_paths=random_paths,
            output_path=random_out,
            codec="libx264",
            audio=False
        )



if __name__ == "__main__":
    main()
