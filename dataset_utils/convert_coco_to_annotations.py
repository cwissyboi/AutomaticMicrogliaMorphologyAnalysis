"""
Convert COCO format back to AnnotationsData format.

Supports two input structures:

1. Original export:
   coco_dataset/
       images/
           {scan_name}_{class_name}_{filename}.png
       instances_default.json

2. CVAT export:
   SegmentationAnnotationsAdjusted/
       images/default/
           {scan_name}_{class_name}_{filename}.png
       annotations/
           instances_default.json
    
Output:
AnnotationsData/Segmentations/
    images/{scan_name}/{class_name}/ann-{id}__x-{x}_y-{y}_w-{w}_h-{h}__ds-{ds}.png
    masks/{scan_name}/{class_name}/ann-{id}__x-{x}_y-{y}_w-{w}_h-{h}__ds-{ds}.png
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Dict, List
import shutil


def rle_to_mask(rle: Dict, height: int, width: int) -> np.ndarray:
    """
    Convert COCO RLE format to binary mask.
    
    Args:
        rle: Dict with 'counts' (list of run lengths) and 'size' [height, width]
        height: Image height
        width: Image width
        
    Returns:
        Binary mask array (H, W) with values 0 or 255
    """
    if isinstance(rle, dict) and 'counts' in rle:
        counts = rle['counts']
        h, w = rle.get('size', [height, width])
    else:
        raise ValueError("RLE must be a dict with 'counts' key")
    
    # Decode RLE
    mask = np.zeros(h * w, dtype=np.uint8)
    current_pos = 0
    current_value = 0
    
    for count in counts:
        if current_value == 1:
            mask[current_pos:current_pos + count] = 255
        current_pos += count
        current_value = 1 - current_value
    
    # Reshape to image dimensions (Fortran order)
    mask = mask.reshape((h, w), order='F')
    
    return mask


def polygon_to_mask(polygons: List, height: int, width: int) -> np.ndarray:
    """
    Convert COCO polygon format to binary mask.
    
    Args:
        polygons: List of polygon coordinates [[x1,y1,x2,y2,...], ...]
        height: Image height
        width: Image width
        
    Returns:
        Binary mask array (H, W) with values 0 or 255
    """
    from PIL import Image, ImageDraw
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for polygon in polygons:
        # Convert [x1,y1,x2,y2,...] to [(x1,y1), (x2,y2), ...]
        coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        draw.polygon(coords, fill=255)
    
    return np.array(mask)


def create_filename(metadata: Dict) -> str:
    """
    Create AnnotationsData filename from metadata.
    
    Args:
        metadata: Dict with ann_id, x, y, width, height, downsample
        
    Returns:
        Filename like: ann-{id}__x-{x}_y-{y}_w-{w}_h-{h}__ds-{ds}.png
    """
    return (
        f"ann-{metadata['ann_id']}"
        f"__x-{metadata['x']}_y-{metadata['y']}"
        f"_w-{metadata['width']}_h-{metadata['height']}"
        f"__ds-{metadata['downsample']}.png"
    )


def convert_from_coco(
    coco_dir: str,
    output_dir: str,
    annotations_file: str = "instances_default.json"
):
    """
    Convert COCO format back to AnnotationsData format.
    
    Automatically detects structure:
    - Original export: images/ and instances_default.json at root
    - CVAT export: images/default/ and annotations/instances_default.json
    
    Args:
        coco_dir: Path to COCO dataset directory
        output_dir: Path to output AnnotationsData/Segmentations directory
        annotations_file: Name of COCO annotations JSON file (relative to coco_dir or annotations/)
    """
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    
    # Try to find annotations file
    # Option 1: Root level (original export)
    annotations_path = coco_dir / annotations_file
    
    # Option 2: In annotations/ folder (CVAT export)
    if not annotations_path.exists():
        annotations_path = coco_dir / "annotations" / annotations_file
    
    # Option 3: User specified full path
    if not annotations_path.exists():
        annotations_path = Path(annotations_file)
    
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Cannot find annotations file. Tried:\n"
            f"  - {coco_dir / annotations_file}\n"
            f"  - {coco_dir / 'annotations' / annotations_file}\n"
            f"  - {annotations_file}"
        )
    
    print(f"Loading annotations from: {annotations_path}")
    
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output structure
    images_output_dir = output_dir / "images"
    masks_output_dir = output_dir / "masks"
    
    # Build category ID to name mapping
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Build image ID to image info mapping
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Build image ID to annotations mapping
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    
    for image_id, image_info in image_map.items():
        ann_id = None  # Initialize
        
        # Extract metadata
        if 'original_scan' in image_info and 'original_class' in image_info:
            # Metadata preserved from conversion
            scan_name = image_info['original_scan']
            class_name = image_info['original_class']
            original_coords = image_info.get('original_coords', {})
            downsample = image_info.get('downsample', 1)
        else:
            # Try to parse from filename
            # Format: {scan_name}_{class_name}_ann-{id}__x-{x}_y-{y}_w-{w}_h-{h}__ds-{ds}.png
            filename = image_info['file_name']
            
            # Use regex to extract components
            import re
            
            # Look for the annotation pattern at the end
            ann_pattern = r'_?(MG_whole|MG_cell_body|Cluster|Unclassified|Dirt|microglia)_(ann-\d+__x-\d+_y-\d+_w-\d+_h-\d+__ds-\d+\.png)$'
            match = re.search(ann_pattern, filename)
            
            if not match:
                print(f"Warning: Cannot parse filename {filename}, skipping")
                skipped_count += 1
                continue
            
            class_name = match.group(1)
            ann_filename = match.group(2)
            scan_name = filename[:match.start()].rstrip('_')
            
            # Parse annotation filename for coordinates
            coord_pattern = r'ann-(\d+)__x-(\d+)_y-(\d+)_w-(\d+)_h-(\d+)__ds-(\d+)\.png'
            coord_match = re.match(coord_pattern, ann_filename)
            
            if coord_match:
                ann_id = int(coord_match.group(1))
                original_coords = {
                    'x': int(coord_match.group(2)),
                    'y': int(coord_match.group(3)),
                    'width': int(coord_match.group(4)),
                    'height': int(coord_match.group(5))
                }
                downsample = int(coord_match.group(6))
            else:
                # Use bbox as fallback
                ann_id = None  # Will be set from annotation data later
                if image_id in annotations_by_image and annotations_by_image[image_id]:
                    bbox = annotations_by_image[image_id][0]['bbox']
                    original_coords = {
                        'x': int(bbox[0]),
                        'y': int(bbox[1]),
                        'width': int(bbox[2]),
                        'height': int(bbox[3])
                    }
                else:
                    original_coords = {
                        'x': 0,
                        'y': 0,
                        'width': image_info['width'],
                        'height': image_info['height']
                    }
                downsample = 1
        
        # Get annotations for this image
        if image_id not in annotations_by_image:
            print(f"Warning: No annotations for image {image_info['file_name']}, skipping")
            skipped_count += 1
            continue
        
        # For now, assume one annotation per image (can be modified for multiple)
        annotation = annotations_by_image[image_id][0]
        
        # Get annotation ID (use parsed value if available, otherwise from annotation data)
        if ann_id is None:
            ann_id = annotation.get('id', image_id)
        
        # Create filename
        metadata = {
            'ann_id': ann_id,
            'x': original_coords.get('x', 0),
            'y': original_coords.get('y', 0),
            'width': original_coords.get('width', image_info['width']),
            'height': original_coords.get('height', image_info['height']),
            'downsample': downsample
        }
        output_filename = create_filename(metadata)
        
        # Create output directories
        image_output_path = images_output_dir / scan_name / class_name
        mask_output_path = masks_output_dir / scan_name / class_name
        image_output_path.mkdir(parents=True, exist_ok=True)
        mask_output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy/load image
        # Try multiple possible image locations
        possible_paths = [
            coco_dir / "images" / image_info['file_name'],  # Original export
            coco_dir / "images" / "default" / image_info['file_name'],  # CVAT export
        ]
        
        source_image_path = None
        for path in possible_paths:
            if path.exists():
                source_image_path = path
                break
        
        if source_image_path is None:
            print(f"Warning: Image not found: {image_info['file_name']}")
            print(f"  Tried:")
            for path in possible_paths:
                print(f"    - {path}")
            skipped_count += 1
            continue
        
        shutil.copy2(source_image_path, image_output_path / output_filename)
        
        # Convert segmentation to mask
        segmentation = annotation['segmentation']
        height = image_info['height']
        width = image_info['width']
        
        # Check if RLE or polygon format
        if isinstance(segmentation, dict) and 'counts' in segmentation:
            # RLE format
            mask = rle_to_mask(segmentation, height, width)
        elif isinstance(segmentation, list):
            # Polygon format
            mask = polygon_to_mask(segmentation, height, width)
        else:
            print(f"Warning: Unknown segmentation format for image {image_id}")
            skipped_count += 1
            continue
        
        # Save mask
        Image.fromarray(mask).save(mask_output_path / output_filename)
        
        processed_count += 1
    
    print(f"\nConversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"Processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert COCO format back to AnnotationsData format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="coco_dataset",
        help="Path to COCO dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="AnnotationsData_restored/Segmentations",
        help="Path to output AnnotationsData/Segmentations directory"
    )
    parser.add_argument(
        "--annotations-file",
        type=str,
        default="instances_default.json",
        help="Name of COCO annotations JSON file (in root or annotations/ folder)"
    )
    
    args = parser.parse_args()
    
    convert_from_coco(
        coco_dir=args.input,
        output_dir=args.output,
        annotations_file=args.annotations_file
    )
