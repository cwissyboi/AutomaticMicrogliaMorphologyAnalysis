"""
Import helper for custom_segmentation module.

This module sets up sys.path to enable imports from initial_pipeline.
Import this at the top of any file in custom_segmentation/ that needs
to access initial_pipeline modules (helpers, morphology, etc.).

Usage:
    from setup_imports import setup_initial_pipeline_path
    setup_initial_pipeline_path()
    
    # Now you can import from initial_pipeline
    from helpers import get_file_name
    from morphology.morphology_features import compute_junction_count
"""

import sys
from pathlib import Path


def setup_initial_pipeline_path():
    """
    Add initial_pipeline/ to sys.path if not already present.
    
    This allows importing from:
    - helpers.py (in initial_pipeline/)
    - morphology/ modules (in initial_pipeline/morphology/)
    - any other initial_pipeline modules
    
    Call this at the start of any file that needs these imports.
    """
    # Get path to initial_pipeline/
    # Current file: initial_pipeline/segmentation/custom_segmentation/setup_imports.py
    # Go up 3 levels: custom_segmentation -> segmentation -> initial_pipeline
    initial_pipeline_path = Path(__file__).resolve().parent.parent.parent
    
    # Add to sys.path if not already there
    if str(initial_pipeline_path) not in sys.path:
        sys.path.insert(0, str(initial_pipeline_path))
    
    return initial_pipeline_path


# Auto-run on import for convenience
setup_initial_pipeline_path()
