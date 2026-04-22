#!/usr/bin/env python3
"""
Test script to check WebDataset API for distributed training
"""
import sys
try:
    import webdataset as wds
    print("WebDataset version:", wds.__version__ if hasattr(wds, '__version__') else 'unknown')
    
    # Create a dummy dataset to inspect methods
    ds = wds.WebDataset([])
    
    # Look for methods related to distributed training
    methods = [m for m in dir(ds) if any(x in m.lower() for x in ['split', 'node', 'worker', 'rank', 'shard'])]
    print("Available distributed methods:", methods)
    
    # Test the split_by_node method
    if hasattr(ds, 'split_by_node'):
        print("split_by_node method exists")
        help_text = ds.split_by_node.__doc__
        if help_text:
            print("split_by_node docstring:", help_text[:200])
    else:
        print("split_by_node method does NOT exist")
        
    # Check for nodesplitter
    if hasattr(wds, 'split_by_node'):
        print("wds.split_by_node function exists")
    if hasattr(wds, 'nodesplitter'):
        print("wds.nodesplitter function exists")
        
except ImportError as e:
    print(f"Cannot import webdataset: {e}")
    sys.exit(1)