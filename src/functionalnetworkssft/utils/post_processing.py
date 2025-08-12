"""
Post-processing utilities for the functionalnetworkssft system.

This module provides functionality for post-processing steps that run after
pipeline execution, including Hugging Face Hub upload and GGUF conversion.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, login, whoami
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def upload_to_hub(
    model_path: str,
    tokenizer_path: str,
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    push_adapter_only: bool = False,
) -> None:
    """
    Upload fine-tuned model to Hugging Face Hub.

    This method handles uploading both the base model and LoRA adapters to the
    Hugging Face Hub, with proper authentication and error handling.

    Args:
        model_path (str): Path to the saved model directory
        tokenizer_path (str): Path to the saved tokenizer directory
        repo_id (str): Repository name/ID on Hugging Face Hub (e.g., "username/model-name")
        commit_message (Optional[str]): Commit message for the upload.
            Defaults to "Upload fine-tuned model"
        private (bool): Whether to create a private repository. Defaults to False
        token (Optional[str]): Hugging Face authentication token. If None, will check
            HF_TOKEN environment variable or prompt for login
        push_adapter_only (bool): If True, only push LoRA adapter files.
            Defaults to False (push full model)

    Raises:
        ValueError: If repo_id is invalid or model_path doesn't exist
        HfHubHTTPError: If there are authentication or network issues
        RepositoryNotFoundError: If the repository doesn't exist and can't be created

    Examples:
        # Upload full model to public repository
        upload_to_hub(
            model_path="./outputs/final_model",
            tokenizer_path="./outputs/final_model",
            repo_id="myusername/my-fine-tuned-llama"
        )

        # Upload only LoRA adapters to private repository
        upload_to_hub(
            model_path="./outputs/final_model",
            tokenizer_path="./outputs/final_model",
            repo_id="myusername/my-lora-adapters",
            private=True,
            push_adapter_only=True,
            commit_message="Upload LoRA adapters for Llama-7B"
        )
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

        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer path does not exist: {tokenizer_path}")

        # Set default commit message
        if commit_message is None:
            commit_message = "Upload fine-tuned model with LoRA adapters"

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

        # Determine which files to upload
        files_to_upload = []

        if push_adapter_only:
            # Only upload LoRA adapter files
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
                raise ValueError(f"No LoRA adapter files found in {model_path}")

            logger.info(f"Uploading LoRA adapter files: {files_to_upload}")
        else:
            # Upload all model files
            logger.info("Uploading full model (base model + adapters)")

        # Load and upload tokenizer first
        logger.info("Uploading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if hasattr(tokenizer, "push_to_hub"):
                tokenizer.push_to_hub(
                    repo_id=repo_id,
                    commit_message=f"{commit_message} - tokenizer",
                    token=token,
                    private=private,
                )
        except Exception as e:
            logger.warning(f"Failed to upload tokenizer: {e}")

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

        logger.info(
            f"✅ Successfully uploaded model to: https://huggingface.co/{repo_id}"
        )

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
