#!/usr/bin/env python3
"""
Test script to debug the gradient computation issue with ICA masking and PEFT.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def test_gradient_flow():
    """Test gradient flow with and without hooks."""
    print("Testing gradient flow with hooks...")

    # Create a simple model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float32, device_map="auto"
    )

    # Apply PEFT
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Get a down projection layer
    layer = model.base_model.model.model.layers[0]
    print(f"Layer type: {type(layer)}")
    print(f"Layer attributes: {dir(layer)}")

    # Find the MLP down projection
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        down_proj = layer.mlp.down_proj
    else:
        print("Could not find down_proj, trying alternative paths...")
        return

    print(f"Down proj type: {type(down_proj)}")
    print(f"Has base_layer: {hasattr(down_proj, 'base_layer')}")

    if hasattr(down_proj, "base_layer"):
        base_layer = down_proj.base_layer
        print(f"Base layer type: {type(base_layer)}")
        print(f"Base layer in_features: {base_layer.in_features}")
        print(f"Base layer out_features: {base_layer.out_features}")

        # Create a simple mask
        mask = torch.ones(base_layer.in_features, dtype=torch.float32)
        mask[100:200] = 0.0  # Mask some neurons
        mask.requires_grad_(False)

        def pre_hook(mod, inp):
            print(f"Hook called on {type(mod)}")
            if isinstance(inp, tuple):
                x = inp[0]
                print(f"Input shape: {x.shape}, requires_grad: {x.requires_grad}")
                mask_device = mask.to(device=x.device, dtype=x.dtype)
                mask_device.requires_grad_(False)
                masked_x = x * mask_device
                print(f"Masked output requires_grad: {masked_x.requires_grad}")
                return (masked_x,) + inp[1:]
            else:
                print(
                    f"Single input shape: {inp.shape}, requires_grad: {inp.requires_grad}"
                )
                mask_device = mask.to(device=inp.device, dtype=inp.dtype)
                mask_device.requires_grad_(False)
                masked_inp = inp * mask_device
                print(f"Masked output requires_grad: {masked_inp.requires_grad}")
                return masked_inp

        # Test 1: Hook on PEFT wrapper
        print("\n=== Test 1: Hook on PEFT wrapper ===")
        handle1 = down_proj.register_forward_pre_hook(pre_hook)

        # Create test input
        test_input = torch.randn(
            1, 10, base_layer.in_features, requires_grad=True, device=model.device
        )

        try:
            output1 = down_proj(test_input)
            loss1 = output1.sum()
            print(f"Forward pass successful, loss requires_grad: {loss1.requires_grad}")
            loss1.backward()
            print("Backward pass successful with PEFT wrapper hook")
        except Exception as e:
            print(f"Error with PEFT wrapper hook: {e}")
        finally:
            handle1.remove()

        # Test 2: Hook on base layer
        print("\n=== Test 2: Hook on base layer ===")
        handle2 = base_layer.register_forward_pre_hook(pre_hook)

        # Create test input
        test_input = torch.randn(
            1, 10, base_layer.in_features, requires_grad=True, device=model.device
        )

        try:
            output2 = down_proj(test_input)
            loss2 = output2.sum()
            print(f"Forward pass successful, loss requires_grad: {loss2.requires_grad}")
            loss2.backward()
            print("Backward pass successful with base layer hook")
        except Exception as e:
            print(f"Error with base layer hook: {e}")
        finally:
            handle2.remove()


if __name__ == "__main__":
    test_gradient_flow()
