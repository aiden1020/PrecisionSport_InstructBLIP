# Badminton Video Analysis and Diversity Sampling

A deep learning pipeline for badminton video analysis and diversity sampling. This project leverages the InstructBLIP model to retrieve relevant video clips based on natural language queries and then uses a TC-CLIP encoder to extract features for diversity sampling.

## Overview

This project implements a sophisticated video analysis system that:
- **Retrieves** badminton video clips by understanding natural language questions about the content of the videos via an InstructBLIP model.
- **Extracts** deep learning features using a TC-CLIP encoder model.
- **Samples** a diverse set of video clips using a farthest-first sampling algorithm.
- **Analyzes** and **visualizes** the diversity of the selected clips compared to a random baseline.
- **Generates** summary videos and detailed reports of the sampling results.

## Features

### 🏸 Intelligent Video Retrieval
- Uses **InstructBLIP** for powerful video question-answering capabilities.
- Retrieves clips based on complex, natural language queries about strokes, player positions, and events.
- Processes long videos by chunking them into manageable segments for analysis.

### 🎯 Diversity Sampling
- **Farthest-First Algorithm**: Selects a maximally diverse set of video clips from the retrieved set.
- **Cosine & Euclidean Distance**: Supports multiple metrics to measure feature similarity between clips.
- **Comparative Analysis**: Benchmarks the diversity sampling against a random sampling control group to demonstrate effectiveness.

### 📊 Visualization and Reporting
- **PCA Visualization**: Generates PCA plots to visualize the feature space of video clips, highlighting the selected diverse and random sets.
- **Statistical Reports**: Outputs detailed statistics, including mean, min, and max pairwise distances between features of selected clips.
- **Video Summaries**: Creates concatenated video files from the selected clips for easy review.

## Project Structure

The main components of the project are organized as follows:

```
.
├── run_pipeline.py                  # Main execution script
├── diversity/
│   ├── pipeline/
│   │   ├── retriever.py             # InstructBLIP-based video retrieval
│   │   ├── feature_extract.py       # TC-CLIP feature extraction
│   │   ├── sampler.py               # Diversity sampling algorithm
│   │   └── visualise.py             # Visualization tools
│   └── label/
│       └── relabel_encoder_data.csv # Stroke annotations
├── lavis/
│   ├── projects/instructblip/       # InstructBLIP model configuration
│   └── models/tc_clip_encoder/      # TC-CLIP model weights and config
└── result/                          # Output for analysis, visualizations, and videos
```

## Usage

### Running the Pipeline

To run the complete pipeline with default settings, execute the main script from the root directory of the project:

```bash
python run_pipeline.py
```

The script is configured for distributed data processing, so it's best to run it with `torchrun` for multi-GPU execution:

```bash
torchrun --nproc_per_node=<num_gpus> run_pipeline.py
```

### Configuration

The main parameters for the pipeline are set within the `main()` function in `run_pipeline.py`:

| Parameter | Description | Default Value |
|-----------|-------------|---------|
| `csv_path` | Path to the annotation CSV file. | `"diversity/label/relabel_encoder_data.csv"` |
| `game_id` | The identifier for the game to be analyzed. | `"game1"` |
| `chunk_size` | The number of video clips to group into a single chunk for the VQA model. | `10` |
| `batch_size` | The number of chunks to process in a single batch. | `2` |
| `query` | The natural language question for retrieving relevant clips. | `"What stroke happens in the middle?"` |
| `video_root` | The root directory containing the video files. | `"lavis/configs/datasets/badminton_caption/input/images"` |
| `cfg_path` | Path to the InstructBLIP inference configuration file. | `"lavis/projects/instructblip/inference/inference_instructblip_badminton_qa_coT_3.yaml"` |
| `num_select` | The number of diverse clips to select. | `10` |
| `model_path` | Path to the pre-trained TC-CLIP encoder weights. | `"lavis/models/tc_clip_encoder/weight/fully-supervised-combined-22-86.pth"` |

## Algorithm Details

### 1. Retrieval Phase
- The `StrokeCsvRetriever` reads video metadata from the CSV.
- It groups video clips from the specified `game_id` into chunks.
- The `InstructBLIPBadmintonEngine` processes each chunk with the user's `query`.
- It identifies and returns the paths of video clips that answer the query.

### 2. Feature Extraction Phase
- The `FeatureExtractor` takes the list of retrieved video paths.
- Each video is processed through the pre-trained TC-CLIP model to get a high-dimensional feature vector.
- This phase is distributed across available GPUs for efficiency.

### 3. Diversity Sampling Phase
- The `VideoDiversitySampler` receives the feature vectors for all retrieved clips.
- It applies a farthest-first sampling algorithm to select `num_select` clips that are most distant from each other in the feature space.
- A control group of randomly selected clips is also generated for comparison.

### 4. Analysis and Output
- The `FeatureVisualizer` generates PCA plots to show the distribution of features and the positions of the selected clips.
- It calculates and prints diversity metrics (e.g., mean cosine distance) for both the diversity-sampled and randomly-sampled sets.
- The results, including selected video lists and summary videos, are saved to the `result/` directory.

## Model Information

The pipeline uses two main models:

1.  **InstructBLIP**: A vision-language model used for its powerful VQA capabilities to retrieve relevant video clips based on textual queries.
2.  **TC-CLIP**: A Transformer-based video-text model, pre-trained for stroke classification, used here as a feature encoder to generate rich representations of video clips.

## Results and Analysis

The pipeline outputs several artifacts to the `result/` directory for analysis:

- **`selected_diversity.txt` / `selected_random.txt`**: Text files listing the paths of the selected videos for each sampling method.
- **`farthest_first.mp4` / `uniform_sampling.mp4`**: Concatenated videos of the selected clips.
- **PCA Visualization Images**: PNG files showing the feature distributions.
- **Console Output**: Detailed logs, including comparative analysis and diversity metrics showing the improvement of diversity sampling over random selection.

### Example Output (in `selected_diversity.txt`)
```
Diversity Sampling Results for 'What stroke happens in the middle?' stroke
Selected 10 videos from 150 total

01. game1_set2_12345.mp4 (score: 1.0000)
02. game1_set1_67890.mp4 (score: 0.1875)
...

Diversity Metrics:
Mean Distance: 0.8543
Std Distance: 0.0512
Distance Range: [0.7654, 0.9321]
```