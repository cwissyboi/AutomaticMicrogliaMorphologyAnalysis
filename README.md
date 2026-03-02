# Automatic Microglia Morphology Analysis

An automated pipeline for detecting, segmenting, and analyzing microglia cells in histological images. This pipeline extracts quantitative morphological features from microscopy images to enable systematic analysis of microglia morphology across different experimental conditions.

## Overview

Microglia are the resident immune cells of the central nervous system and exhibit distinct morphological states that reflect their functional activity. This pipeline automates the traditionally manual and time-consuming process of microglia morphology analysis.

### Pipeline Architecture

The pipeline consists of four main stages:

```
Input Image
    ↓
[1] Object Detection (YOLO)
    ↓
[2] Segmentation (SAM + Gaussian)
    ↓
[3] Morphology Extraction
    ↓
[4] CSV Output with 17 Features
```

### What It Does

1. **Object Detection**: Identifies individual microglia cells using YOLOv5
2. **Segmentation**: 
   - Full cell segmentation using SAM (Segment Anything Model)
   - Soma detection using Gaussian filtering on red channel
3. **Morphology Analysis**: Extracts 17 quantitative features per cell:
   - **Skeleton features** (3): process length, branch count, connected components
   - **Soma features** (3): area, perimeter, circularity
   - **Cell features** (7): area, perimeter, convex hull area/perimeter, solidity, convexity, circularity
   - **Bounding box** (4): xmin, ymin, xmax, ymax
4. **Batch Processing**: Processes multiple images and generates a consolidated CSV output

## Features

- ✅ **Fully automated** - no manual annotation required at inference time
- ✅ **GPU accelerated** - uses CUDA when available
- ✅ **Batch processing** - handles multiple images in a folder
- ✅ **Comprehensive metrics** - 17 morphological features per cell
- ✅ **Unique cell tracking** - global cell IDs across images
- ✅ **Modular design** - easy to swap or add segmentation methods

## Installation

### Prerequisites

- Python 3.10 or 3.12
- CUDA-capable GPU (recommended) or CPU
- ~2GB disk space for model weights

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AutomaticMicrogliaMorphologyAnalysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r initial_pipeline_requirements.txt
```

4. Download model weights:
   - Place `yolov3_weights.pt` in the root directory
   - Place `sam_vit_b_01ec64.pth` in `sam_files/` directory

## Usage

### Basic Usage

```bash
cd initial_pipeline
python main.py --input_folder_path ../All_scans_tiled_subset --output_name All_scans_tiled_subset
python yolo_inference_only.py --input_folder_path ../../../All_scans_tiled_subset --output_name yolo_for_semi_supervised
```


python segmentation_training.py --loss_type cldice

python segmentation_training.py --loss_type bce_cldice --cldice_alpha 0.5
python main.py --input_folder_path ../tiles/slide_4_512/ --output_name slide_4_512

### Arguments

- `--input_folder_path`: Path to folder containing input images (required)
- `--output_name`: Name for output files and folders (required)

### Input Format

- **Image format**: PNG, JPG, or TIFF
- **Location**: All images in the specified input folder will be processed
- **Naming**: Images can have any filename

### Output Format

The pipeline generates a CSV file at:
```
initial_pipeline/morphology/morphology_outputs/{output_name}.csv
```

#### CSV Columns

| Column | Description | Units |
|--------|-------------|-------|
| `cell_id` | Cell ID within image | - |
| `image_name` | Source image filename | - |
| `global_cell_id` | Unique ID across all images | - |
| `length_pixels` | Skeleton length (processes only) | pixels |
| `num_branches` | Number of branch points | count |
| `num_components` | Disconnected skeleton pieces | count |
| `soma_area` | Soma region area | pixels² |
| `soma_perimeter` | Soma boundary length | pixels |
| `soma_circularity` | Soma shape circularity (4πA/P²) | 0-1 |
| `cell_area` | Total cell area | pixels² |
| `cell_perimeter` | Total cell boundary length | pixels |
| `cell_convex_hull_area` | Convex hull area | pixels² |
| `cell_convex_hull_perimeter` | Convex hull perimeter | pixels |
| `cell_solidity` | Cell area / convex hull area | 0-1 |
| `cell_convexity` | Convex hull perimeter / cell perimeter | 0-1 |
| `cell_circularity` | Cell shape circularity (4πA/P²) | 0-1 |
| `xmin`, `ymin`, `xmax`, `ymax` | Bounding box coordinates | pixels |

### Example Workflow

```bash
# Process a single slide
cd initial_pipeline
python main.py --input_folder_path ../data/slide_001/ --output_name experiment_001

# Process multiple slides by running multiple times
python main.py --input_folder_path ../data/slide_002/ --output_name experiment_002
python main.py --input_folder_path ../data/slide_003/ --output_name experiment_003

# The CSV outputs can then be combined for analysis
```

## Project Structure

```
AutomaticMicrogliaMorphologyAnalysis/
├── initial_pipeline/
│   ├── main.py                          # Main pipeline script
│   ├── helpers.py                       # Utility functions
│   │
│   ├── object_detection/
│   │   └── yolo_pretrained/
│   │       └── yolo_inference.py        # YOLO detection
│   │
│   ├── segmentation/
│   │   ├── sam/
│   │   │   └── sam_inference.py         # SAM segmentation
│   │   ├── soma_segmentation/
│   │   │   └── gaussian_filter.py       # Gaussian soma detection
│   │   └── custom_segmentation/         # U-Net training (experimental)
│   │       ├── segmentation_training.py
│   │       ├── segmentation_preprocessing.py
│   │       └── data_utils.py
│   │
│   ├── morphology/
│   │   └── morphology_features.py       # Feature extraction
│   │
│   └── analysis/
│       └── basic_analysis.ipynb         # Clustering and visualization
│
├── sam_files/                           # SAM model weights
├── tiles/                               # Input image tiles
├── yolov3_weights.pt                    # YOLO model weights
└── initial_pipeline_requirements.txt    # Python dependencies
```

## Advanced Usage

### Custom Segmentation Model (Experimental)

A custom U-Net model is in development for improved segmentation:

```bash
cd initial_pipeline/segmentation/custom_segmentation

# Train model
python segmentation_training.py

# Preprocess annotations
python segmentation_preprocessing.py
```

**Note**: Custom segmentation is not yet integrated into the main pipeline.

### Analysis Notebooks

Jupyter notebooks are provided for exploratory analysis:

- `initial_pipeline/analysis/basic_analysis.ipynb` - Clustering and visualization
- `initial_pipeline/segmentation/custom_segmentation/segmentation_inference.ipynb` - U-Net inference

### Output Intermediate Results

The pipeline supports saving intermediate outputs for debugging:

Edit the `output_to_file` parameters in `main.py`:

```python
yolo_boxes = yolo_inference(yolo, image_path, output_name=output_name, output_to_file=True)
sam_masks = sam_inference(sam_predictor, yolo_boxes, image_path, output_name=output_name, 
                          image_rgb=image_rgb, output_to_file=True)
```

## Morphological Features Explained

### Skeleton Features
- **Length**: Total length of cell processes (soma excluded)
- **Branch Count**: Number of points where processes split (≥3 neighbors)
- **Components**: Disconnected skeleton pieces (indicates fragmentation)

### Soma Features
- **Area**: Cell body size
- **Perimeter**: Cell body boundary length
- **Circularity**: Shape roundness (1.0 = perfect circle)

### Cell Features
- **Area/Perimeter**: Overall cell size and boundary
- **Convex Hull**: Smallest convex shape containing the cell
- **Solidity**: How "filled" the cell is (cell area / convex hull area)
- **Convexity**: Boundary smoothness (convex perimeter / cell perimeter)
- **Circularity**: Overall shape roundness

### Biological Interpretation

- **Ramified (Resting)**: High branch count, long processes, high convexity
- **Amoeboid (Activated)**: Low branch count, short processes, low convexity, high soma circularity
- **Intermediate States**: Values between ramified and amoeboid

## Requirements

Key dependencies:
- `torch>=2.0` - Deep learning framework
- `torchvision` - Vision models
- `segment-anything==1.0` - SAM model
- `ultralytics>=8.0` - YOLO implementation
- `opencv-python` - Image processing
- `scikit-image` - Morphology operations
- `pandas` - Data management
- `numpy`, `scipy` - Numerical computing

See `initial_pipeline_requirements.txt` for complete list.

## Hardware Requirements

### Minimum
- CPU: Any modern processor
- RAM: 8GB
- Storage: 5GB

### Recommended
- CPU: Multi-core processor (4+ cores)
- GPU: NVIDIA GPU with 6GB+ VRAM
- RAM: 16GB
- Storage: 10GB

**Processing Speed** (approximate):
- With GPU: ~1-2 seconds per cell
- CPU only: ~5-10 seconds per cell

## Known Limitations

1. **Hardcoded model paths**: Model paths are currently hardcoded in `main.py` (see TODO at line 22)
2. **Limited error handling**: Pipeline may fail silently on corrupted images
3. **No validation metrics**: No ground truth comparison in main pipeline
4. **Single scale**: Processes only the provided image resolution
5. **Disconnected processes**: Some cells may have disconnected skeleton components due to segmentation challenges

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size or use CPU
sam.to("cpu")
```

### Import Errors
```bash
# Ensure you're in the correct directory
cd initial_pipeline
python main.py ...
```

### Model Weight Errors
- Verify model files exist at the specified paths
- Check file permissions
- Re-download model weights if corrupted

### Empty Output
- Check that input folder contains valid image files
- Verify YOLO confidence threshold (may be too high)
- Ensure output directory exists and is writable

## Future Development

Planned improvements:
- [ ] Integrate custom U-Net segmentation into main pipeline
- [ ] Configuration file system (YAML/JSON)
- [ ] Validation metrics against ground truth
- [ ] Multi-scale analysis
- [ ] Automated quality control
- [ ] Statistical comparison tools
- [ ] Docker containerization

## Citation

If you use this pipeline in your research, please cite:

```
[Add citation information here]
```

## License

[Add license information here]

## Contact

For questions or issues, please contact:
- [Your name/email]
- Or open an issue on GitHub

## Acknowledgments

This pipeline uses:
- [YOLOv5](https://github.com/ultralytics/yolov5) for object detection
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) for segmentation
- QuPath for manual annotations and training data preparation
