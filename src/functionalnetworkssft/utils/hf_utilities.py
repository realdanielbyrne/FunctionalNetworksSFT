#!/usr/bin/env python3
"""
HuggingFace Hub utilities for FunctionalNetworksSFT.

This module contains utilities for interacting with the HuggingFace Hub,
including model uploading and authentication handling.

Author: Daniel Byrne
License: MIT
"""

import os
import logging
from typing import Optional
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from huggingface_hub import HfApi, login, whoami
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError

logger = logging.getLogger(__name__)


def upload_to_hub(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    push_adapter_only: bool = False,
    use_peft: Optional[bool] = None,
) -> None:
    """
    Upload a fine-tuned model to the Hugging Face Hub.

    Args:
        model_path: Path to the saved model directory
        tokenizer: The tokenizer used with the model
        repo_id: Repository ID in format 'username/repository-name'
        commit_message: Optional commit message for the upload
        private: Whether to create a private repository
        token: HuggingFace authentication token
        push_adapter_only: Whether to upload only LoRA adapter files
        use_peft: Whether the model uses PEFT (auto-detected if None)

    Raises:
        ValueError: If inputs are invalid
        RepositoryNotFoundError: If repository cannot be found or created
        HfHubHTTPError: If there are HTTP errors during upload
    """
    try:
        logger.info(f"Starting upload to Hugging Face Hub: {repo_id}")

        # Validate inputs
        if not repo_id or "/" not in repo_id:
            raise ValueError(
                "repo_id must be in format 'username/repository-name' or 'organization/repository-name'"
            )

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        # Auto-detect training mode if not specified
        if use_peft is None:
            use_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
            logger.info(
                f"Auto-detected training mode: {'PEFT' if use_peft else 'Full fine-tuning'}"
            )

        # Set default commit message based on training mode
        if commit_message is None:
            if use_peft:
                commit_message = "Upload fine-tuned model with LoRA adapters"
            else:
                commit_message = "Upload full fine-tuned model"

        # Handle authentication
        if token is None:
            token = os.getenv("HF_TOKEN")

        if token is None:
            logger.info(
                "No HF_TOKEN found in environment. Attempting to use cached credentials..."
            )
            try:
                # Check if user is already logged in
                user_info = whoami(token=token)
                logger.info(f"Using cached credentials for user: {user_info['name']}")
            except Exception:
                logger.info(
                    "No cached credentials found. Please log in to Hugging Face Hub..."
                )
                login()
        else:
            logger.info("Using provided authentication token")

        # Initialize HF API
        api = HfApi(token=token)

        # Check if repository exists, create if it doesn't
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
            logger.info(f"Repository {repo_id} exists")
        except RepositoryNotFoundError:
            logger.info(f"Creating new repository: {repo_id}")
            api.create_repo(
                repo_id=repo_id, repo_type="model", private=private, exist_ok=True
            )

        # Determine which files to upload based on training mode and push_adapter_only flag
        files_to_upload = []

        if push_adapter_only or (use_peft and not push_adapter_only):
            # Upload only LoRA adapter files (either explicitly requested or PEFT mode)
            adapter_files = [
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",  # fallback for older format
            ]

            for file_name in adapter_files:
                file_path = os.path.join(model_path, file_name)
                if os.path.exists(file_path):
                    files_to_upload.append(file_name)

            if not files_to_upload:
                if use_peft:
                    raise ValueError(
                        f"No LoRA adapter files found in {model_path}. Model may not be a PEFT model."
                    )
                else:
                    logger.warning(
                        f"No LoRA adapter files found in {model_path}, falling back to full model upload"
                    )
                    push_adapter_only = False

            if files_to_upload:
                logger.info(f"Uploading LoRA adapter files: {files_to_upload}")

        if not push_adapter_only:
            # Upload all model files (full model or full model + adapters)
            if use_peft:
                logger.info("Uploading full PEFT model (base model + adapters)")
            else:
                logger.info("Uploading full fine-tuned model")

        # Upload tokenizer first
        logger.info("Uploading tokenizer...")
        if hasattr(tokenizer, "push_to_hub"):
            tokenizer.push_to_hub(
                repo_id=repo_id,
                commit_message=f"{commit_message} - tokenizer",
                token=token,
                private=private,
            )

        # Upload model files
        if push_adapter_only:
            # Upload individual adapter files
            for file_name in files_to_upload:
                file_path = os.path.join(model_path, file_name)
                logger.info(f"Uploading {file_name}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_name,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"{commit_message} - {file_name}",
                    token=token,
                )
        else:
            # Upload entire model directory
            logger.info("Uploading model files...")

            # Load and upload the model using transformers
            try:
                # Try to load as PEFT model first
                from peft import PeftModel, AutoPeftModelForCausalLM

                # Check if this is a PEFT model
                if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                    logger.info("Detected PEFT model, uploading with PEFT support...")
                    model = AutoPeftModelForCausalLM.from_pretrained(model_path)
                    model.push_to_hub(
                        repo_id=repo_id,
                        commit_message=commit_message,
                        token=token,
                        private=private,
                    )
                else:
                    # Regular model upload
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    model.push_to_hub(
                        repo_id=repo_id,
                        commit_message=commit_message,
                        token=token,
                        private=private,
                    )
            except Exception as e:
                logger.warning(f"Failed to upload using transformers: {e}")
                logger.info("Falling back to file-by-file upload...")

                # Fallback: upload directory contents
                api.upload_folder(
                    folder_path=model_path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=commit_message,
                    token=token,
                )

        logger.info(f"Successfully uploaded model to: https://huggingface.co/{repo_id}")

    except RepositoryNotFoundError as e:
        logger.error(f"Repository not found and could not be created: {e}")
        raise
    except HfHubHTTPError as e:
        if "401" in str(e):
            logger.error(
                "Authentication failed. Please check your token or run 'huggingface-cli login'"
            )
        elif "403" in str(e):
            logger.error(
                "Permission denied. Check if you have write access to the repository"
            )
        elif "404" in str(e):
            logger.error(
                "Repository not found. Make sure the repository name is correct"
            )
        else:
            logger.error(f"HTTP error during upload: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise


def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        print(f"Loading environment variables from {env_file}")
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
        print("No .env file found")


def check_hf_token():
    """Check if HF_TOKEN is properly configured and can access gated models."""

    # Check environment variable
    hf_token = os.getenv("HF_TOKEN")
    auth_method = None

    if hf_token:
        print(f"HF_TOKEN found: {hf_token[:8]}...{hf_token[-8:]}")
        auth_method = "environment"
    else:
        print("HF_TOKEN environment variable is not set")
        # Check for cached credentials
        try:
            user_info = whoami()  # This will use cached token if available
            print(
                f"✅ Found cached HuggingFace credentials for user: {user_info['name']}"
            )
            auth_method = "cached"
        except Exception as e:
            print(f"No cached credentials found: {e}")
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
            print(f"Authentication successful for user: {user_info['name']}")
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False
    # For cached credentials, we already tested with whoami() above

    # Test access to the specific gated model
    try:
        print("\nTesting access to meta-llama/Llama-3.2-1B-Instruct...")
        if auth_method == "environment":
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct", token=hf_token
            )
        else:  # cached credentials
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct"
                # No token parameter needed - will use cached credentials
            )
        print("Successfully accessed the gated model!")
        return True
    except Exception as e:
        print(f"Failed to access gated model: {e}")
        print("\nPossible solutions:")
        print(
            "1. Make sure you've requested access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
        )
        print("2. Wait for approval (can take some time)")
        print("3. Check if your token has the correct permissions")
        return False


def main():
    """Main entry point for the check-hf-token command."""
    print("Checking HuggingFace token configuration...\n")

    # Load .env file first
    load_env_file()
    print()

    success = check_hf_token()

    if not success:
        print("\n Quick fix commands:")
        print("# Set token for current session:")
        print("export HF_TOKEN=hf_your_token_here")
        print("\n# Or login interactively:")
        print("huggingface-cli login")
        return 1

    return 0
