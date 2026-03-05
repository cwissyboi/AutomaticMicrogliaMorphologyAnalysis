"""
Convert AnnotationsData format to COCO format for CVAT upload.

Directory structure:
AnnotationsData/Segmentations/
    images/{scan_name}/{class_name}/ann-{id}__x-{x}_y-{y}_w-{w}_h-{h}__ds-{ds}.png
    masks/{scan_name}/{class_name}/ann-{id}__x-{x}_y-{y}_w-{w}_h-{h}__ds-{ds}.png
    
Output:
coco_dataset/
    images/
        {scan_name}_{class_name}_{filename}.png
    annotations.json
"""

import json
import re
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import shutil


def parse_filename(filename: str) -> Dict:
    """
    Parse annotation filename to extract metadata.
    
    Format: ann-{id}__x-{x}_y-{y}_w-{w}_h-{h}__ds-{ds}.png
    
    Returns:
        dict with keys: ann_id, x, y, width, height, downsample
    """
    pattern = r'ann-(\d+)__x-(\d+)_y-(\d+)_w-(\d+)_h-(\d+)__ds-(\d+)\.png'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Filename {filename} doesn't match expected pattern")
    
    return {
        'ann_id': int(match.group(1)),
        'x': int(match.group(2)),
        'y': int(match.group(3)),
        'width': int(match.group(4)),
        'height': int(match.group(5)),
        'downsample': int(match.group(6))
    }


def mask_to_rle(mask: np.ndarray) -> Dict:
    """
    Convert binary mask to COCO RLE (Run Length Encoding) format.
    
    Args:
        mask: Binary mask array (H, W) with values 0 or 255
        
    Returns:
        RLE dict with 'counts' and 'size'
    """
    # Ensure binary
    binary_mask = (mask > 127).astype(np.uint8)
    
    # Flatten in Fortran order (column-major)
    pixels = binary_mask.flatten(order='F')
    
    # Run length encoding
    runs = []
    current_value = 0
    current_count = 0
    
    for pixel in pixels:
        if pixel == current_value:
            current_count += 1
        else:
            runs.append(current_count)
            current_value = pixel
            current_count = 1
    runs.append(current_count)
    
    return {
        'counts': runs,
        'size': list(mask.shape)
    }


def mask_to_polygon(mask: np.ndarray, tolerance: float = 2.0) -> List[List[float]]:
    """
    Convert binary mask to polygon format.
    
    Args:
        mask: Binary mask array (H, W)
        tolerance: Simplification tolerance for polygon
        
    Returns:
        List of polygon coordinates [[x1,y1,x2,y2,...], ...]
    """
    from skimage import measure
    
    # Ensure binary
    binary_mask = (mask > 127).astype(np.uint8)
    
    # Find contours
    contours = measure.find_contours(binary_mask, 0.5)
    
    polygons = []
    for contour in contours:
        # Flip from (row, col) to (x, y)
        contour = np.flip(contour, axis=1)
        
        # Simplify polygon
        if len(contour) < 3:
            continue
            
        # Flatten to [x1, y1, x2, y2, ...]
        segmentation = contour.ravel().tolist()
        
        # COCO requires at least 6 coordinates (3 points)
        if len(segmentation) >= 6:
            polygons.append(segmentation)
    
    return polygons


def calculate_bbox_from_mask(mask: np.ndarray) -> List[int]:
    """
    Calculate bounding box from binary mask.
    
    Returns:
        [x, y, width, height] in COCO format
    """
    binary_mask = (mask > 127).astype(np.uint8)
    
    # Find non-zero pixels
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    return [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)]


def convert_to_coco(
    annotations_dir: str,
    output_dir: str,
    use_rle: bool = True,
    copy_images: bool = True,
    mask_source: str = "calculated_masks",
    exclude_classes: List[str] = None
):
    """
    Convert AnnotationsData format to COCO format.
    
    Args:
        annotations_dir: Path to AnnotationsData/Segmentations
        output_dir: Path to output COCO dataset directory
        use_rle: If True, use RLE encoding; if False, use polygon
        copy_images: If True, copy images to output dir
        mask_source: Folder name for masks ('masks' or 'calculated_masks')
        exclude_classes: List of class names to exclude (e.g., ['Unclassified'])
    """
    if exclude_classes is None:
        exclude_classes = []
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    
    # Create output structure
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Microglia Morphology Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Build categories from discovered classes
    category_map = {}
    category_id = 1
    
    images_dir = annotations_dir / "images"
    masks_dir = annotations_dir / mask_source
    
    # Verify mask directory exists
    if not masks_dir.exists():
        raise FileNotFoundError(
            f"Mask directory '{mask_source}' not found at {masks_dir}. "
            f"Available options: 'masks' or 'calculated_masks'"
        )
    
    image_id = 1
    annotation_id = 1
    
    # Iterate through scans
    for scan_dir in sorted(images_dir.iterdir()):
        if not scan_dir.is_dir():
            continue
            
        scan_name = scan_dir.name
        
        # Iterate through classes
        for class_dir in sorted(scan_dir.iterdir()):
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            
            # Skip excluded classes
            if class_name in exclude_classes:
                print(f"Skipping excluded class: {class_name}")
                continue
            
            # Add category if not exists
            if class_name not in category_map:
                category_map[class_name] = category_id
                coco_data["categories"].append({
                    "id": category_id,
                    "name": class_name,
                    "supercategory": "microglia"
                })
                category_id += 1
            
            # Process each image
            for img_path in sorted(class_dir.glob("*.png")):
                # Parse filename
                try:
                    metadata = parse_filename(img_path.name)
                except ValueError as e:
                    print(f"Warning: Skipping {img_path.name}: {e}")
                    continue
                
                # Check if corresponding mask exists FIRST
                mask_path = masks_dir / scan_name / class_name / img_path.name
                
                if not mask_path.exists():
                    print(f"Warning: Mask not found for {img_path.name}, skipping image")
                    continue
                
                # Load image
                image = Image.open(img_path)
                img_width, img_height = image.size
                
                # Create unique filename for output
                output_filename = f"{scan_name}_{class_name}_{img_path.name}"
                
                # Copy image if requested
                if copy_images:
                    shutil.copy2(img_path, images_output_dir / output_filename)
                
                # Add image entry
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": output_filename,
                    "width": img_width,
                    "height": img_height,
                    "date_captured": "",
                    "license": 0,
                    "coco_url": "",
                    "flickr_url": "",
                    # Store original metadata
                    "original_scan": scan_name,
                    "original_class": class_name,
                    "original_coords": {
                        "x": metadata['x'],
                        "y": metadata['y'],
                        "width": metadata['width'],
                        "height": metadata['height']
                    },
                    "downsample": metadata['downsample']
                })
                
                # Load mask (we already know it exists)
                mask = np.array(Image.open(mask_path))
                
                # Calculate bbox
                bbox = calculate_bbox_from_mask(mask)
                area = int(np.sum(mask > 127))
                
                # Convert mask to segmentation format
                if use_rle:
                    segmentation = mask_to_rle(mask)
                else:
                    segmentation = mask_to_polygon(mask)
                
                # Add annotation entry
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_map[class_name],
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })
                
                annotation_id += 1
                image_id += 1
    
    # Save COCO JSON with CVAT-compatible naming
    # CVAT expects 'instances_default.json' or 'instances_*.json'
    output_json = output_dir / "instances_default.json"
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"Annotations file: instances_default.json (CVAT-compatible)")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {[cat['name'] for cat in coco_data['categories']]}")
    print(f"\nTo upload to CVAT:")
    print(f"  1. Create new task and upload images from: {images_output_dir}")
    print(f"  2. Actions → Upload annotations → Select: instances_default.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert AnnotationsData to COCO format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="AnnotationsData/Segmentations",
        help="Path to AnnotationsData/Segmentations directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="coco_dataset",
        help="Path to output COCO dataset directory"
    )
    parser.add_argument(
        "--use-polygon",
        action="store_true",
        help="Use polygon format instead of RLE"
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Don't copy images to output directory"
    )
    parser.add_argument(
        "--mask-source",
        type=str,
        default="calculated_masks",
        choices=["masks", "calculated_masks"],
        help="Mask folder to use: 'masks' (original) or 'calculated_masks' (improved)"
    )
    parser.add_argument(
        "--exclude-classes",
        type=str,
        nargs="+",
        default=[],
        help="Class names to exclude (e.g., --exclude-classes Unclassified Cluster)"
    )
    
    args = parser.parse_args()
    
    convert_to_coco(
        annotations_dir=args.input,
        output_dir=args.output,
        use_rle=not args.use_polygon,
        copy_images=not args.no_copy_images,
        mask_source=args.mask_source,
        exclude_classes=args.exclude_classes
    )
