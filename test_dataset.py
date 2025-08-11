#!/usr/bin/env python3
"""Test script to check if the HuggingFace dataset exists and its structure."""

try:
    from datasets import load_dataset
    print("Testing dataset loading...")
    
    # Try to load the dataset
    ds = load_dataset('Amod/mental_health_counseling_conversations')
    print("Dataset loaded successfully")
    print(f"Dataset structure: {ds}")
    
    # Get the first sample
    if "train" in ds:
        first_sample = ds["train"][0]
    else:
        first_key = list(ds.keys())[0]
        first_sample = ds[first_key][0]
    
    print(f"First sample: {first_sample}")
    print(f"Sample keys: {list(first_sample.keys())}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print(f"Error type: {type(e)}")
