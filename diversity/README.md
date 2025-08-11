# Precision Sport Diversity

A deep learning pipeline for badminton video analysis and diversity sampling, designed to extract meaningful features from sports videos and select diverse representative clips for analysis.

## Overview

This project implements a sophisticated video analysis system that:
- **Retrieves** badminton video clips based on stroke type queries
- **Extracts** deep learning features using a TC-CLIP encoder model
- **Samples** diverse video clips using farthest-first diversity sampling
- **Analyzes** badminton strokes including serves, smashes, drops, and net shots

## Features

### 🏸 Stroke Recognition
- Supports multiple badminton stroke types: serve, smash, drop shot, net lift, etc.
- Handles both Chinese and English stroke descriptions
- Processes video clips with temporal and spatial feature extraction

### 🎯 Diversity Sampling
- **Farthest-First Algorithm**: Selects maximally diverse video clips
- **Cosine Similarity**: Measures feature similarity between clips
- **Smart Selection**: Avoids redundant similar clips in final selection

## Project Structure

```
PrecisionSportDiversity/
├── Dataset/
│   ├── label/
│   │   └── filtered_encoder_data.csv    # Stroke annotations
│   └── video/                           # Video clips (.mp4)
├── pipeline/
│   ├── run_pipeline.py                  # Main execution script
│   ├── retriever.py                     # CSV-based video retrieval
│   ├── feature_extract.py               # TC-CLIP feature extraction
│   ├── sampler.py                       # Diversity sampling algorithm
│   └── models/
│       └── tc_clip_encoder/             # Pre-trained model weights
├── result/                              # Output analysis results
└── Archive/                             # Legacy code
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- ffmpeg for video processing

### Dependencies
```bash
pip install -r pipeline/models/tc_clip_encoder/requirements.txt
```

Key dependencies include:
- `torch` - Deep learning framework
- `mmcv-full==1.7.0` - Computer vision library
- `decord==0.6.0` - Video decoding
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `hydra-core` - Configuration management

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```bash
cd /path/to/PrecisionSportDiversity
python run_pipeline.py
```


### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `query` | Stroke type to search for | "smash" |
| `num_select` | Number of diverse clips to select | 5 |
| `device_id` | GPU device ID | 0 |
| `csv_path` | Path to annotation CSV | "Dataset/label/filtered_encoder_data.csv" |
| `video_dir` | Directory containing video files | "Dataset/video" |

## Dataset Format

### Video Files
- Format: MP4
- Naming: `game{X}_set{Y}_{timestamp}.mp4`
- Content: Short badminton stroke clips

### Annotation CSV
The `filtered_encoder_data.csv` contains:

| Column | Description |
|--------|-------------|
| `id` | Video clip identifier |
| `stroke_name` | English stroke name (serve, smash, drop, etc.) |
| `type` | Chinese stroke description |
| `hit_area` | Court position (1-9 grid system) |
| `player` | Player position (upper/bottom) |
| `backhand` | Backhand indicator (0/1) |
| `stroke_LLM` | Detailed stroke description |

## Algorithm Details

### Diversity Sampling Algorithm

1. **Retrieval Phase**: Query CSV for stroke-specific video IDs
2. **Feature Extraction**: Process videos through TC-CLIP encoder
3. **Diversity Selection**: Apply farthest-first sampling
   - Start with random seed clip
   - Iteratively select clips most dissimilar to already selected ones
   - Use cosine similarity as distance metric

### Feature Extraction Pipeline

1. **Video Decoding**: Extract frames using Decord
2. **Preprocessing**: Resize, crop, normalize frames
3. **Encoding**: Pass through TC-CLIP model
4. **Aggregation**: Average temporal features to single vector

## Model Information

The project uses a pre-trained TC-CLIP encoder:
- **Architecture**: Transformer-based video-text model
- **Training**: Fully supervised on combined stroke dataset
- **Performance**: 87% accuracy on stroke classification
- **Input**: RGB video clips
- **Output**: 512-dimensional feature vectors

## Results and Analysis

The pipeline outputs:
- **Selected Video Paths**: Diverse representative clips
- **Diversity Scores**: Quantitative measure of selection quality
- **Feature Visualizations**: PCA analysis of stroke features (see `result/pca_smash_features.png`)

### Example Output
```
=== Selected Videos (order, path, score) ===
01. Dataset/video/game1_set1_17566.mp4 (score: 1.0000)
02. Dataset/video/game1_set1_23504.mp4 (score: 0.1234)
03. Dataset/video/game1_set1_20288.mp4 (score: 0.0987)
04. Dataset/video/game1_set1_17291.mp4 (score: 0.0756)
05. Dataset/video/game1_set1_25199.mp4 (score: 0.0623)
```

## Applications

### Sports Analysis
- **Technique Comparison**: Analyze diverse stroke execution styles
- **Training Aid**: Select representative examples for coaching
- **Performance Metrics**: Quantify stroke diversity and consistency

### Research Applications
- **Dataset Curation**: Create balanced training sets
- **Behavioral Analysis**: Study player strategy patterns
- **Video Summarization**: Generate diverse highlight reels
