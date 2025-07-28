#!/usr/bin/env python3
"""
Script to check HuggingFace token configuration and access to gated models.
"""

import os
from pathlib import Path
from huggingface_hub import whoami, login
from transformers import AutoTokenizer


def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        print(f"📁 Loading environment variables from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    os.environ[key] = value
                    if key == "HF_TOKEN":
                        print(
                            f"✅ Loaded HF_TOKEN from .env file: {value[:8]}...{value[-8:]}"
                        )
    else:
        print("📁 No .env file found")


def check_hf_token():
    """Check if HF_TOKEN is properly configured and can access gated models."""

    # Check environment variable
    hf_token = os.getenv("HF_TOKEN")
    auth_method = None

    if hf_token:
        print(f"✅ HF_TOKEN found: {hf_token[:8]}...{hf_token[-8:]}")
        auth_method = "environment"
    else:
        print("❌ HF_TOKEN environment variable is not set")
        # Check for cached credentials
        try:
            user_info = whoami()  # This will use cached token if available
            print(
                f"✅ Found cached HuggingFace credentials for user: {user_info['name']}"
            )
            auth_method = "cached"
        except Exception as e:
            print(f"❌ No cached credentials found: {e}")
            print("\nTo fix this:")
            print("1. Get your token from https://huggingface.co/settings/tokens")
            print("2. Set it as an environment variable:")
            print("   export HF_TOKEN=hf_your_token_here")
            print("3. Or run: huggingface-cli login")
            return False

    # Test authentication
    if auth_method == "environment":
        try:
            user_info = whoami(token=hf_token)
            print(f"✅ Authentication successful for user: {user_info['name']}")
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    # For cached credentials, we already tested with whoami() above

    # Test access to the specific gated model
    try:
        print("\n🔍 Testing access to meta-llama/Llama-3.2-1B-Instruct...")
        if auth_method == "environment":
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct", token=hf_token
            )
        else:  # cached credentials
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct"
                # No token parameter needed - will use cached credentials
            )
        print("✅ Successfully accessed the gated model!")
        return True
    except Exception as e:
        print(f"❌ Failed to access gated model: {e}")
        print("\nPossible solutions:")
        print(
            "1. Make sure you've requested access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
        )
        print("2. Wait for approval (can take some time)")
        print("3. Check if your token has the correct permissions")
        return False


def main():
    """Main entry point for the check-hf-token command."""
    print("🔍 Checking HuggingFace token configuration...\n")

    # Load .env file first
    load_env_file()
    print()

    success = check_hf_token()

    if not success:
        print("\n💡 Quick fix commands:")
        print("# Set token for current session:")
        print("export HF_TOKEN=hf_your_token_here")
        print("\n# Or login interactively:")
        print("huggingface-cli login")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
