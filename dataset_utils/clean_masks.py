"""
Keep only the largest connected component in segmentation masks.

This script processes all masks in AnnotationsData format and removes small 
disconnected components, keeping only the largest connected component per mask.

Useful for cleaning up segmentation masks that may have noise or small 
artifacts from annotation tools.

Input/Output:
AnnotationsData/Segmentations/
    masks/{scan_name}/{class_name}/ann-{id}__x-{x}_y-{y}_w-{w}_h-{h}__ds-{ds}.png
"""

import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage
import shutil
from typing import Optional
import argparse


def get_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    
    Args:
        mask: Binary mask array (H, W) with values 0 or 255
        
    Returns:
        Binary mask with only largest connected component
    """
    # Ensure binary
    binary_mask = (mask > 127).astype(np.uint8)
    
    # Check if mask is empty
    if not binary_mask.any():
        return mask  # Return original if empty
    
    # Label connected components
    labeled, num_features = ndimage.label(binary_mask)
    
    # If only one component, return original
    if num_features <= 1:
        return mask
    
    # Find largest component
    component_sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))
    largest_component = component_sizes[1:].argmax() + 1  # Skip background (0)
    
    # Create mask with only largest component
    largest_mask = (labeled == largest_component).astype(np.uint8) * 255
    
    return largest_mask


def count_connected_components(mask: np.ndarray) -> int:
    """Count number of connected components in a binary mask."""
    binary_mask = (mask > 127).astype(np.uint8)
    if not binary_mask.any():
        return 0
    labeled, num_features = ndimage.label(binary_mask)
    return num_features


def process_masks(
    masks_dir: str,
    output_dir: Optional[str] = None,
    backup: bool = True,
    dry_run: bool = False,
    min_components: int = 2
):
    """
    Process all masks to keep only largest connected component.
    
    Args:
        masks_dir: Path to masks directory (e.g., AnnotationsData/Segmentations/masks)
        output_dir: Path to output directory (if None, modifies in-place)
        backup: If True and output_dir is None, create backup before modifying
        dry_run: If True, only report what would be done without modifying
        min_components: Only process masks with at least this many components
    """
    masks_dir = Path(masks_dir)
    
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    # Setup output directory
    in_place = output_dir is None
    if in_place:
        output_dir = masks_dir
        if backup and not dry_run:
            backup_dir = masks_dir.parent / f"{masks_dir.name}_backup"
            if backup_dir.exists():
                print(f"Backup already exists: {backup_dir}")
                response = input("Overwrite backup? (y/n): ")
                if response.lower() != 'y':
                    print("Aborting.")
                    return
                shutil.rmtree(backup_dir)
            print(f"Creating backup: {backup_dir}")
            shutil.copytree(masks_dir, backup_dir)
    else:
        output_dir = Path(output_dir)
    
    # Statistics
    stats = {
        'total': 0,
        'processed': 0,
        'single_component': 0,
        'multi_component': 0,
        'empty': 0,
        'components_removed': 0
    }
    
    # Process all masks
    print(f"\nScanning masks in: {masks_dir}")
    print(f"Output directory: {output_dir}")
    if dry_run:
        print("DRY RUN MODE - No files will be modified\n")
    
    for mask_path in masks_dir.rglob("*.png"):
        stats['total'] += 1
        
        # Load mask
        mask = np.array(Image.open(mask_path))
        
        # Count components
        num_components = count_connected_components(mask)
        
        if num_components == 0:
            stats['empty'] += 1
            continue
        elif num_components == 1:
            stats['single_component'] += 1
            # Copy to output if different directory
            if not in_place and not dry_run:
                output_path = output_dir / mask_path.relative_to(masks_dir)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mask_path, output_path)
            continue
        elif num_components < min_components:
            # Not enough components to process
            if not in_place and not dry_run:
                output_path = output_dir / mask_path.relative_to(masks_dir)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mask_path, output_path)
            continue
        
        # Multiple components - process
        stats['multi_component'] += 1
        stats['components_removed'] += (num_components - 1)
        
        rel_path = mask_path.relative_to(masks_dir)
        print(f"Processing {rel_path}: {num_components} components → 1 component")
        
        if not dry_run:
            # Get largest component
            cleaned_mask = get_largest_connected_component(mask)
            
            # Save
            output_path = output_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cleaned_mask).save(output_path)
            
            stats['processed'] += 1
        else:
            stats['processed'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total masks scanned:           {stats['total']}")
    print(f"Masks with 1 component:        {stats['single_component']}")
    print(f"Masks with multiple components: {stats['multi_component']}")
    print(f"Empty masks:                   {stats['empty']}")
    print(f"Masks processed:               {stats['processed']}")
    print(f"Total components removed:      {stats['components_removed']}")
    
    if dry_run:
        print("\nDRY RUN - No files were modified")
        print(f"Run without --dry-run to apply changes")
    elif in_place and backup:
        print(f"\nOriginal masks backed up to: {masks_dir.parent / f'{masks_dir.name}_backup'}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Keep only largest connected component in segmentation masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would happen
  python clean_masks.py --input AnnotationsData_Adjusted/Segmentations/masks --dry-run
  
  # Process in-place with backup
  python clean_masks.py --input AnnotationsData_Adjusted/Segmentations/masks
  
  # Process to new directory
  python clean_masks.py --input AnnotationsData_Adjusted/Segmentations/masks --output AnnotationsData_Cleaned/Segmentations/masks
  
  # Process in-place without backup (dangerous!)
  python clean_masks.py --input AnnotationsData_Adjusted/Segmentations/masks --no-backup
  
  # Only process masks with 3+ components
  python clean_masks.py --input AnnotationsData_Adjusted/Segmentations/masks --min-components 3
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to masks directory (e.g., AnnotationsData/Segmentations/masks)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (if not specified, modifies in-place)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup when modifying in-place (use with caution!)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually modifying files"
    )
    
    parser.add_argument(
        "--min-components",
        type=int,
        default=2,
        help="Only process masks with at least this many components (default: 2)"
    )
    
    args = parser.parse_args()
    
    process_masks(
        masks_dir=args.input,
        output_dir=args.output,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        min_components=args.min_components
    )


if __name__ == "__main__":
    main()
