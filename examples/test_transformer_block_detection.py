#!/usr/bin/env python3
"""
Demonstration script showing that transformer block detection works
correctly for different model architectures.
"""

import torch
import torch.nn as nn


def get_transformer_blocks(model):
    """
    Extract transformer blocks using the same logic as compute_ica_masks_for_model.
    This is the fixed version that supports all model architectures.
    """
    if hasattr(model, "transformer"):
        blocks = getattr(model.transformer, "h", None) or getattr(
            model.transformer, "blocks", None
        )
    elif hasattr(model, "model"):
        blocks = getattr(model.model, "layers", None) or getattr(
            model.model, "decoder", None
        )
    else:
        blocks = None
    return blocks


def create_gpt_style_model():
    """Create a mock GPT-style model (model.transformer.h)."""
    model = nn.Module()
    model.transformer = nn.Module()
    model.transformer.h = nn.ModuleList([
        nn.Module() for _ in range(12)  # 12 layers like GPT-2
    ])
    return model


def create_transformer_style_model():
    """Create a mock Transformer-style model (model.transformer.blocks)."""
    model = nn.Module()
    model.transformer = nn.Module()
    model.transformer.blocks = nn.ModuleList([
        nn.Module() for _ in range(6)  # 6 layers
    ])
    return model


def create_llama_style_model():
    """Create a mock Llama/Mistral-style model (model.model.layers)."""
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([
        nn.Module() for _ in range(32)  # 32 layers like Llama
    ])
    return model


def create_qwen_style_model():
    """Create a mock Qwen-style model (model.model.layers)."""
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([
        nn.Module() for _ in range(24)  # 24 layers
    ])
    return model


def create_decoder_style_model():
    """Create a mock decoder-style model (model.model.decoder)."""
    model = nn.Module()
    model.model = nn.Module()
    model.model.decoder = nn.ModuleList([
        nn.Module() for _ in range(8)  # 8 layers
    ])
    return model


def main():
    """Test transformer block detection for different architectures."""
    print("Testing Transformer Block Detection")
    print("=" * 50)
    
    # Test GPT-style models
    print("\n1. GPT-style model (model.transformer.h):")
    gpt_model = create_gpt_style_model()
    gpt_blocks = get_transformer_blocks(gpt_model)
    if gpt_blocks is not None:
        print(f"   ✅ Found {len(gpt_blocks)} transformer blocks")
    else:
        print("   ❌ No transformer blocks found")
    
    # Test Transformer-style models
    print("\n2. Transformer-style model (model.transformer.blocks):")
    transformer_model = create_transformer_style_model()
    transformer_blocks = get_transformer_blocks(transformer_model)
    if transformer_blocks is not None:
        print(f"   ✅ Found {len(transformer_blocks)} transformer blocks")
    else:
        print("   ❌ No transformer blocks found")
    
    # Test Llama-style models
    print("\n3. Llama/Mistral-style model (model.model.layers):")
    llama_model = create_llama_style_model()
    llama_blocks = get_transformer_blocks(llama_model)
    if llama_blocks is not None:
        print(f"   ✅ Found {len(llama_blocks)} transformer blocks")
    else:
        print("   ❌ No transformer blocks found")
    
    # Test Qwen-style models
    print("\n4. Qwen-style model (model.model.layers):")
    qwen_model = create_qwen_style_model()
    qwen_blocks = get_transformer_blocks(qwen_model)
    if qwen_blocks is not None:
        print(f"   ✅ Found {len(qwen_blocks)} transformer blocks")
    else:
        print("   ❌ No transformer blocks found")
    
    # Test decoder-style models
    print("\n5. Decoder-style model (model.model.decoder):")
    decoder_model = create_decoder_style_model()
    decoder_blocks = get_transformer_blocks(decoder_model)
    if decoder_blocks is not None:
        print(f"   ✅ Found {len(decoder_blocks)} transformer blocks")
    else:
        print("   ❌ No transformer blocks found")
    
    # Test unsupported model
    print("\n6. Unsupported model architecture:")
    unsupported_model = nn.Linear(10, 10)
    unsupported_blocks = get_transformer_blocks(unsupported_model)
    if unsupported_blocks is None:
        print("   ✅ Correctly identified as unsupported (returned None)")
    else:
        print("   ❌ Should have returned None for unsupported architecture")
    
    # Test priority handling
    print("\n7. Priority test (transformer.h vs transformer.blocks):")
    priority_model = nn.Module()
    priority_model.transformer = nn.Module()
    priority_model.transformer.h = nn.ModuleList([nn.Module()])  # 1 block
    priority_model.transformer.blocks = nn.ModuleList([nn.Module(), nn.Module()])  # 2 blocks
    priority_blocks = get_transformer_blocks(priority_model)
    if priority_blocks is not None and len(priority_blocks) == 1:
        print("   ✅ Correctly prioritized transformer.h over transformer.blocks")
    else:
        print("   ❌ Priority handling failed")
    
    print("\n" + "=" * 50)
    print("All tests completed! The transformer block detection now supports:")
    print("• GPT-2 style models: model.transformer.h")
    print("• Transformer style models: model.transformer.blocks")
    print("• Llama/Mistral style models: model.model.layers")
    print("• Qwen style models: model.model.layers")
    print("• Decoder style models: model.model.decoder")
    print("• Proper priority handling and unsupported model detection")


if __name__ == "__main__":
    main()
