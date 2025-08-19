# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for tokenization."""

import warnings
import os
import sys


def _clear_telechat_cache_safe():
    """Safely clean up telechat-related transformers caches to avoid race conditions."""
    print(">>>>>> _clear_telechat_cache_safe >>>>>> ")
    import shutil
    import glob
    
    cache_patterns = [
        "/root/.cache/huggingface/modules/transformers_modules/checkpoint-*",
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/checkpoint-*"),
    ]
    
    cleared = False
    for pattern in cache_patterns:
        for path in glob.glob(pattern):
            # Check whether this checkpoint directory contains telechat files
            telechat_file = os.path.join(path, "tokenization_telechat3.py")
            if os.path.exists(telechat_file):
                try:
                    shutil.rmtree(path)
                    print(f"[VERL] Clean up the telechat cache directory: {path}")
                    cleared = True
                except Exception as e:
                    print(f"[VERL] Clean up failed: {path} ({e})")

    # Clean up imported telechat modules
    modules_to_remove = [m for m in sys.modules.keys() if 'telechat' in m.lower()]
    for module_name in modules_to_remove:
        try:
            del sys.modules[module_name]
            print(f"[VERL] Clean up module: {module_name}")
        except:
            pass
    
    if not cleared:
        print("[VERL] No telechat cache found to clean up")

    # Force refresh importlib cache
    import importlib
    importlib.invalidate_caches()


def _ensure_fresh_telechat_load(name_or_path, **kwargs):
    """Ensure a fresh loading of the telechat tokenizer to avoid caching issues."""
    from transformers import AutoTokenizer
    
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"[VERL] Retrying load (attempt {attempt+1})...")
                # Clean up before retrying
                _clear_telechat_cache_safe()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
            print(f"[VERL] telechat tokenizer loaded successfully (attempt {attempt+1})")
            return tokenizer
            
        except FileNotFoundError as e:
            last_error = e
            print(f"[VERL] Load failed on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                print("[VERL] Cleaning cache and retrying...")
                continue
            else:
                print("[VERL] All retries failed")
                break
        except Exception as e:
            # Other types of errors, do not retry
            print(f"[VERL] Load failed (non-retryable error): {e}")
            raise
    
    raise last_error


__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn("Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1)
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    if "telechat" in name_or_path:
        # Clean telechat cache before loading, and use a retry mechanism to ensure successful loading
        _clear_telechat_cache_safe()
        print(f"> Loading telechat tokenizer from {name_or_path} with kwargs: {kwargs}")
        
        # Use a dedicated telechat loading function with retry logic
        tokenizer = _ensure_fresh_telechat_load(name_or_path, **kwargs)
    else:
        print(f"> Loading Tokenizer from {name_or_path} with kwargs: {kwargs}")
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        # if "telechat" in name_or_path:
        #     # kwargs["trust_remote_code"] = True
        #     print(f"> Loading processor from {name_or_path}...")
        #     processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True, **kwargs)
        # else:
        #     processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        print(f"> Loading processor from {name_or_path} with kwargs: {kwargs}")
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
