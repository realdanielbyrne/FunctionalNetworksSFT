"""
Test transformer block detection for different model architectures.
"""

import unittest
import torch
import torch.nn as nn


def _get_transformer_blocks(model):
    """Extract the block detection logic for testing."""
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


class TestTransformerBlockDetection(unittest.TestCase):
    """Test that transformer blocks are correctly detected for different architectures."""
    
    def test_gpt_model_detection(self):
        """Test that GPT-style models (model.transformer.h) are detected correctly."""
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList([nn.Module(), nn.Module()])
        
        blocks = _get_transformer_blocks(model)
        
        self.assertIsNotNone(blocks)
        self.assertEqual(len(blocks), 2)
    
    def test_transformer_model_detection(self):
        """Test that Transformer-style models (model.transformer.blocks) are detected correctly."""
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.blocks = nn.ModuleList([nn.Module(), nn.Module()])
        
        blocks = _get_transformer_blocks(model)
        
        self.assertIsNotNone(blocks)
        self.assertEqual(len(blocks), 2)
    
    def test_llama_model_detection(self):
        """Test that Llama/Mistral-style models (model.model.layers) are detected correctly."""
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        
        blocks = _get_transformer_blocks(model)
        
        self.assertIsNotNone(blocks)
        self.assertEqual(len(blocks), 2)
    
    def test_qwen_model_detection(self):
        """Test that Qwen-style models (model.model.layers) are detected correctly."""
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        
        blocks = _get_transformer_blocks(model)
        
        self.assertIsNotNone(blocks)
        self.assertEqual(len(blocks), 2)
    
    def test_decoder_model_detection(self):
        """Test that decoder-style models (model.model.decoder) are detected correctly."""
        model = nn.Module()
        model.model = nn.Module()
        model.model.decoder = nn.ModuleList([nn.Module(), nn.Module()])
        
        blocks = _get_transformer_blocks(model)
        
        self.assertIsNotNone(blocks)
        self.assertEqual(len(blocks), 2)
    
    def test_unsupported_model_detection(self):
        """Test that unsupported model architectures return None."""
        # Create a model with no recognizable transformer structure
        model = nn.Linear(10, 10)
        blocks = _get_transformer_blocks(model)
        
        # Should return None for unsupported architecture
        self.assertIsNone(blocks)
    
    def test_transformer_h_priority(self):
        """Test that transformer.h takes priority over transformer.blocks."""
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList([nn.Module()])
        model.transformer.blocks = nn.ModuleList([nn.Module(), nn.Module()])
        
        blocks = _get_transformer_blocks(model)
        
        # Should return h (1 block) not blocks (2 blocks)
        self.assertEqual(len(blocks), 1)
    
    def test_model_layers_priority(self):
        """Test that model.layers takes priority over model.decoder."""
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Module()])
        model.model.decoder = nn.ModuleList([nn.Module(), nn.Module()])
        
        blocks = _get_transformer_blocks(model)
        
        # Should return layers (1 block) not decoder (2 blocks)
        self.assertEqual(len(blocks), 1)


if __name__ == "__main__":
    unittest.main()
