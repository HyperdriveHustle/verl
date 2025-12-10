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
"""
Utility functions for extracting and storing token probabilities from training and inference stages.
"""

import json
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute stable softmax using log-sum-exp trick for numerical stability.
    
    This function ensures consistent softmax computation across training and inference stages.
    
    Args:
        logits: Input logits tensor of shape (..., vocab_size)
        dim: Dimension along which to apply softmax
        
    Returns:
        Probability tensor of the same shape as logits
    """
    # Subtract max for numerical stability
    logits_max = logits.max(dim=dim, keepdim=True)[0]
    logits_shifted = logits - logits_max
    
    # Compute exp and sum
    exp_logits = torch.exp(logits_shifted)
    sum_exp = exp_logits.sum(dim=dim, keepdim=True)
    
    # Compute probabilities
    probs = exp_logits / (sum_exp + 1e-10)  # Add small epsilon to avoid division by zero
    
    return probs


def extract_token_probability(logits: torch.Tensor, token_id: int, dim: int = -1) -> float:
    """
    Extract probability for a specific token from logits.
    
    Args:
        logits: Logits tensor of shape (..., vocab_size) or (vocab_size,)
        token_id: Token ID to extract probability for
        dim: Dimension along which vocab_size is located
        
    Returns:
        Probability value as float, rounded to 8 decimal places
    """
    probs = stable_softmax(logits, dim=dim)
    
    # Extract probability for the specific token
    # Handle different tensor dimensions
    if logits.dim() == 1:
        # 1D case (vocab_size,)
        token_prob = probs[token_id].item()
    elif logits.dim() == 2:
        # 2D case (batch, vocab_size) - take first batch item
        token_prob = probs[0, token_id].item()
    elif logits.dim() == 3:
        # 3D case (batch, seq_len, vocab_size) - take last sequence position
        token_prob = probs[0, -1, token_id].item()
    else:
        # Fallback: use indexing
        indices = [0] * (logits.dim() - 1) + [token_id]
        token_prob = probs[tuple(indices)].item()
    
    return round(token_prob, 8)


class ProbabilityLogger:
    """
    Logger for storing training and inference probabilities to JSONL file.
    """
    
    def __init__(self, output_file: str):
        """
        Initialize the probability logger.
        
        Args:
            output_file: Path to the JSONL output file
        """
        self.output_file = output_file
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """Ensure the directory for the output file exists."""
        directory = os.path.dirname(self.output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def log_probability(
        self,
        training_probability: float,
        inference_probability: float,
        token_index: int,
    ):
        """
        Log a single token's probabilities to the JSONL file.
        
        Args:
            training_probability: Probability from training stage
            inference_probability: Probability from inference stage
            token_index: Index of the token in the sequence
        """
        entry = {
            "Training Possibility": training_probability,
            "Inference Possibility": inference_probability,
            "Token Index": token_index,
        }
        
        try:
            # Append mode to ensure new data doesn't overwrite existing data
            with open(self.output_file, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write probability entry to {self.output_file}: {e}", exc_info=True)
            raise
    
    def log_batch_probabilities(
        self,
        training_probabilities: list[float],
        inference_probabilities: list[float],
        token_indices: list[int],
    ):
        """
        Log a batch of token probabilities to the JSONL file.
        
        Args:
            training_probabilities: List of probabilities from training stage
            inference_probabilities: List of probabilities from inference stage
            token_indices: List of token indices in the sequence
        """
        if len(training_probabilities) != len(inference_probabilities) or len(training_probabilities) != len(token_indices):
            raise ValueError(
                f"Length mismatch: training_probs={len(training_probabilities)}, "
                f"inference_probs={len(inference_probabilities)}, "
                f"token_indices={len(token_indices)}"
            )
        
        try:
            # Append mode to ensure new data doesn't overwrite existing data
            with open(self.output_file, "a", encoding="utf-8") as f:
                for train_prob, inf_prob, token_idx in zip(training_probabilities, inference_probabilities, token_indices):
                    entry = {
                        "Training Possibility": train_prob,
                        "Inference Possibility": inf_prob,
                        "Token Index": token_idx,
                    }
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write probability batch to {self.output_file}: {e}", exc_info=True)
            raise

