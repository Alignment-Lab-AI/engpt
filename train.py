# train.py (Complete Script with Fixes)

import os
import time
import random
import argparse
import functools
import yaml
import math
import numpy as np
from termcolor import colored
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple, Type, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from datasets import load_dataset, load_from_disk, DatasetDict, Features, Sequence, Value
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, AutoTokenizer
from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler
from torch.cuda.amp import autocast # Keep old import, use new syntax in call

import wandb
from engpt import (
    EfficientNGPTModel, EfficientNGPTConfig, register_l2norm_parametrization, Scale
)
# Import necessary optimizers and wrappers
try:
    from smmf import SMMF
except ImportError:
    SMMF = None
    print(colored("Warning: SMMF optimizer not found. Will fail if selected.", "red"))
try:
    from gro import IntegratedOptimizerWrapper
except ImportError:
    IntegratedOptimizerWrapper = None
    print(colored("Warning: IntegratedOptimizerWrapper not found. Will fail if selected.", "red"))
from torch.optim import AdamW
# Optionally import standalone OrthoGrad if used
try:
    from orthograd import OrthoGrad
except ImportError:
    OrthoGrad = None

# --- Utility functions ---
def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def decode_bytes(byte_list: List[int], encoding: str = 'latin1', errors: str = 'ignore') -> str:
    """Decodes a list of byte values (integers) into a string."""
    filtered = [b for b in byte_list if isinstance(b, int) and 0 <= b <= 255]
    return bytes(filtered).decode(encoding, errors=errors)

# --- Dataset Classes ---
class ConcatenatedSequenceDataset(torch.utils.data.Dataset):
    """
    Concatenates multiple smaller chunks from a base dataset into sequences
    of the model's target block size.
    """
    def __init__(self, base_dataset: Union[Dataset, List], concat_factor: int, model_block_size: int, tokenizer: Optional[Any] = None):
        self.base_dataset = base_dataset
        self.concat_factor = concat_factor
        self.model_block_size = model_block_size
        self.tokenizer = tokenizer
        # Handle case where base_dataset might be empty list
        if not base_dataset:
            self.num_sequences = 0
        else:
            self.num_sequences = len(base_dataset) // concat_factor
        if self.num_sequences == 0 and len(base_dataset) > 0:
             print(f"Warning: Base dataset length ({len(base_dataset)}) is less than concat_factor ({concat_factor}). No sequences generated.")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if idx >= self.num_sequences:
             raise IndexError("Index out of range for ConcatenatedSequenceDataset")
        start_idx = idx * self.concat_factor
        end_idx = start_idx + self.concat_factor
        if end_idx > len(self.base_dataset):
             # This should not happen if __len__ is correct, but as safety
             print(f"Warning: Calculated end_idx {end_idx} exceeds base dataset len {len(self.base_dataset)} for index {idx}")
             return None # Or handle differently

        try:
            chunks = [self.base_dataset[i] for i in range(start_idx, end_idx)]
        except IndexError:
             print(f"Error fetching chunks for index {idx} (range {start_idx}-{end_idx}), base len {len(self.base_dataset)}")
             return None # Skip this item if underlying access fails

        if not all(chunks): # Check if any chunk is None or invalid
             print(f"Warning: Found invalid chunk(s) for index {idx}")
             return None

        # --- Byte-Level Concatenation (Primary Logic) ---
        all_input_ids = []
        for i, chunk in enumerate(chunks):
            if 'input_ids' not in chunk or not hasattr(chunk['input_ids'], '__iter__'):
                 print(f"Warning: Skipping chunk {i} for main index {idx} due to missing/invalid 'input_ids'. Chunk: {chunk}")
                 continue
            all_input_ids.extend(list(chunk['input_ids']))

        if len(all_input_ids) != self.model_block_size:
             # Pad or truncate if length mismatch occurs (should be rare if chunking is correct)
             # print(f"Warning: Concatenated length mismatch. Expected {self.model_block_size}, got {len(all_input_ids)}. Adjusting...")
             all_input_ids = all_input_ids[:self.model_block_size]
             padding_length = self.model_block_size - len(all_input_ids)
             if padding_length > 0: all_input_ids.extend([0] * padding_length) # Pad with 0

        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        labels_tensor = input_ids_tensor.clone()

        return {'input_ids': input_ids_tensor, 'labels': labels_tensor}


def prepare_data(data_cfg: Dict[str, Any], model_cfg: EfficientNGPTConfig, train_cfg: Dict[str, Any], tokenizer: Optional[Any] = None) -> Tuple[Union[Dataset, List], Union[Dataset, List]]:
    """
    Prepares the dataset: Loads from cache or processes, chunks, splits, and concatenates.
    Corrected cache checking logic.
    """
    processed_path = data_cfg['local_processed_path']
    base_chunk_size = data_cfg.get('base_chunk_size', 512)
    model_block_size = model_cfg.block_size
    encoding = data_cfg.get('encoding', 'latin1')
    text_column = data_cfg.get('text_column', 'text')
    seed = train_cfg.get('seed', 42)
    test_split_size = data_cfg.get('test_split_size', 0.01)

    # --- Cache Check ---
    splits = None
    # Check if the exact path provided is a directory containing a saved DatasetDict
    expected_meta_file = os.path.join(processed_path, "dataset_dict.json")

    if os.path.exists(expected_meta_file):
        print(colored(f"Found existing dataset metadata at: {expected_meta_file}", "blue"))
        try:
            print(f"Attempting to load pre-processed dataset from cache: {processed_path}")
            splits = load_from_disk(processed_path)
            assert isinstance(splits, DatasetDict), "Loaded object is not a DatasetDict"
            assert 'train' in splits and 'test' in splits, "Cached dataset missing train/test splits."
            # Allow empty train/test splits at this stage, check later
            print(colored(f"Successfully loaded dataset from cache: {processed_path}", "green"))
        except Exception as e:
            print(colored(f"Error loading dataset from cache ({processed_path}): {e}. Reprocessing...", "yellow"))
            splits = None
    else:
        print(colored(f"Dataset cache metadata not found at: {expected_meta_file}", "blue"))
        print(f"Will process raw dataset and save to: {processed_path}")
        parent_dir = os.path.dirname(processed_path)
        if parent_dir and not os.path.isdir(parent_dir):
             print(f"Creating parent directory: {parent_dir}")
             os.makedirs(parent_dir, exist_ok=True)

    # --- Process if not loaded from cache ---
    if splits is None:
        print(f"Processing raw dataset '{data_cfg['dataset_name']}' into chunks of size {base_chunk_size}...")
        try:
            raw = load_dataset(data_cfg['dataset_name'], split=data_cfg.get('split', 'train'))
        except Exception as e:
            raise RuntimeError(f"Failed to load raw dataset '{data_cfg['dataset_name']}': {e}")

        debug_limit = train_cfg.get('debug_limit_dataset_size')
        if debug_limit is not None and debug_limit > 0:
            print(f"Applying debug limit: Processing only {debug_limit} raw documents.")
            raw = raw.select(range(min(debug_limit, len(raw))))

        if len(raw) == 0:
            raise ValueError("Raw dataset is empty after loading (and potential debug limit). Cannot proceed.")

        assert text_column in raw.column_names, f"Text column '{text_column}' not found in raw dataset."

        # Define chunking function
        def chunk_examples_batched(batch: Dict[str, list]) -> Dict[str, list]:
            all_chunk_ids = []
            num_encoding_errors = 0
            processed_docs = 0
            for text in batch[text_column]:
                processed_docs += 1
                if not isinstance(text, str): text = str(text)
                try:
                    # Strict encoding check first
                    _ = text.encode(encoding, errors='strict')
                    # If successful, encode for real
                    bytes_ids = list(text.encode(encoding)) # No errors expected now
                except Exception:
                    num_encoding_errors += 1
                    continue # Skip documents with encoding errors

                num_full_chunks = len(bytes_ids) // base_chunk_size
                for i in range(num_full_chunks):
                    start = i * base_chunk_size
                    end = (i + 1) * base_chunk_size
                    chunk = bytes_ids[start:end]
                    all_chunk_ids.append(chunk)

            # Optional: More verbose warning about encoding errors
            # if num_encoding_errors > 0:
            #    print(f"Warning: Skipped {num_encoding_errors}/{processed_docs} docs in batch due to encoding errors.")

            # Labels are the same as inputs for byte-level LM
            return {'input_ids': all_chunk_ids, 'labels': all_chunk_ids}

        chunked_features = Features({
            'input_ids': Sequence(Value('int64'), length=base_chunk_size),
            'labels': Sequence(Value('int64'), length=base_chunk_size)
        })

        print("Mapping dataset to chunks...")
        num_map_workers = max(1, dcfg.get('num_workers', 0)) # Use at least 1 process
        try:
             chunked_ds = raw.map(
                 chunk_examples_batched,
                 batched=True,
                 batch_size=dcfg.get('map_batch_size', 1000),
                 remove_columns=raw.column_names,
                 features=chunked_features,
                 num_proc=num_map_workers,
                 load_from_cache_file=False # Force re-computation if splits is None
             )
        except Exception as map_err:
             print(f"Error during dataset mapping: {map_err}")
             raise map_err

        print(f"Chunking complete. Total chunks: {len(chunked_ds)}")
        if len(chunked_ds) == 0:
             raise RuntimeError("Dataset became empty after chunking. Check input data or chunking logic.")

        print("Splitting dataset...")
        if len(chunked_ds) < 2: # Need at least 2 samples to split
             print(colored("Warning: Not enough data chunks (<2) to perform train/test split. Using all data for training.", "yellow"))
             splits = DatasetDict({'train': chunked_ds, 'test': chunked_ds.select([])}) # Empty test set
        elif test_split_size <= 0 or test_split_size >= 1:
             print(colored(f"Warning: Invalid test_split_size ({test_split_size}). Using all data for training.", "yellow"))
             splits = DatasetDict({'train': chunked_ds, 'test': chunked_ds.select([])}) # Empty test set
        else:
             splits = chunked_ds.train_test_split(
                 test_size=test_split_size,
                 seed=seed,
                 shuffle=True
             )

        print(f"Saving processed dataset splits to disk: {processed_path}")
        try:
            # Save directly to the target processed_path
            splits.save_to_disk(processed_path)
            print(colored(f"Successfully saved dataset to cache: {processed_path}", "green"))
        except Exception as e:
            raise RuntimeError(f"Failed to save processed dataset to {processed_path}: {e}")

    # --- Concatenate Chunks ---
    train_ds_chunks = splits['train']
    test_ds_chunks = splits['test']

    if len(train_ds_chunks) == 0:
         raise RuntimeError("Train split is empty after loading/processing. Cannot proceed.")
    if len(test_ds_chunks) == 0:
         print(colored("Warning: Test split is empty after loading/processing.", "yellow"))

    assert model_block_size % base_chunk_size == 0, \
        f"Model block size ({model_block_size}) must be a multiple of base_chunk_size ({base_chunk_size})"
    concat_factor = model_block_size // base_chunk_size
    if concat_factor < 1:
         raise ValueError("concat_factor is less than 1. Check model_block_size and base_chunk_size.")

    print(f"Concatenating {concat_factor} chunks of size {base_chunk_size} -> final sequence length {model_block_size}")
    train_ds = ConcatenatedSequenceDataset(train_ds_chunks, concat_factor, model_block_size, tokenizer)
    test_ds = ConcatenatedSequenceDataset(test_ds_chunks, concat_factor, model_block_size, tokenizer) if len(test_ds_chunks) > 0 else [] # Handle empty test set

    print(f"Final Train Sequences: {len(train_ds)}")
    print(f"Final Test Sequences: {len(test_ds)}")
    if len(train_ds) == 0:
         # This can happen if len(train_ds_chunks) < concat_factor
         raise RuntimeError(f"Train dataset is empty after concatenation. Need at least {concat_factor} chunks for one sequence.")

    return train_ds, test_ds


def simple_collate_fn(
    batch: List[Optional[Dict[str, torch.Tensor]]],
    pad_token_id: int = 0,
    ign_token_id: int = -100,
    mask_prob: float = 0.15,
    middle_token_count: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    1) Filters out None
    2) Concatenates & stacks input_ids
    3) Clones labels from input_ids
    4) Applies one of {Teacher-Forced, R-, S-, X-, Causal} per sequence
    5) Builds attention_mask = (input_ids != pad) & (input_ids != ignore)
    """
    # 1) filter
    valid = [b for b in batch if b is not None
                          and isinstance(b.get("input_ids"), torch.Tensor)]
    if not valid:
        return {}

    # 2) stack
    input_ids = torch.stack([b["input_ids"] for b in valid], dim=0)  # (B, L)
    labels    = input_ids.clone()

    B, L = input_ids.shape
    rng  = np.random.RandomState()  # or seed however you like

    # helper denoisers (you can also import your _apply_* here)
    def r_denoise(x, y):
        max_preds = int(mask_prob * L)
        for _ in range(max_preds):
            span = max(1, int(rng.normal(3, 1)))
            start = rng.randint(0, L - span)
            x[:, start : start+span] = ign_token_id
            y[:, start : start+span] = ign_token_id

    def s_denoise(x, y):
        mid_start = (L // 2) - middle_token_count // 2
        mid_end   = mid_start + middle_token_count
        # mask everything except first, last, and middle block
        x[:, 1:mid_start] = ign_token_id
        x[:, mid_end:-1]  = ign_token_id
        y[:, 1:mid_start] = ign_token_id
        y[:, mid_end:-1]  = ign_token_id

    def x_denoise(x, y):
        max_preds = int(mask_prob * L)
        ngram = rng.randint(12, 20)
        for _ in range(max_preds):
            st = rng.randint(0, L - ngram)
            x[:, st:st+ngram] = ign_token_id
            y[:, st:st+ngram] = ign_token_id

    def causal(x, y):
        y[:, :-1] = x[:, 1:].clone()
        y[:, -1]  = ign_token_id

    # 3) choose strategy per batch element
    strategies = rng.choice(
        [0, 1, 2, 3, 4], size=B, p=[0.1, 0.1, 0.1, 0.1, 0.6]
    )
    for i, strat in enumerate(strategies):
        xi = input_ids[i : i+1]   # keep batch dim
        yi = labels[i : i+1]
        if strat == 1:        r_denoise(xi, yi)
        elif strat == 2:      s_denoise(xi, yi)
        elif strat == 3:      x_denoise(xi, yi)
        elif strat == 4:      causal(xi, yi)
        # strat == 0 ⇒ teacher-forced (no change)

    # 4) build attention mask—no fallback, we ALWAYS do this:
    attention_mask = (input_ids != pad_token_id) & (input_ids != ign_token_id)

    return {
        "input_ids":      input_ids,
        "labels":         labels,
        "attention_mask": attention_mask.long(),
    }
# --- Optimizer Building ---
def build_optimizer(model: nn.Module, train_cfg: Dict[str, Any], device: torch.device) -> torch.optim.Optimizer:
    """Builds the optimizer based on the training configuration."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: raise ValueError("Model has no parameters requiring gradients.")

    optimizer_type = train_cfg.get('optimizer_type', 'adamw').lower()
    lr = train_cfg.get('lr') # Get LR, might be None
    print(f"Attempting to build optimizer of type: {optimizer_type}, Config LR: {lr}")

    if optimizer_type == 'integrated_sam_ortho_schedulefree':
        if IntegratedOptimizerWrapper is None: raise ImportError("IntegratedOptimizerWrapper not imported.")
        print(colored("Using IntegratedOptimizerWrapper (SAM + OrthoGrad + ScheduleFree)", "yellow"))
        base_optimizer_name = train_cfg.get('base_optimizer_class_name', 'AdamW').lower()
        base_optimizer_cls: Type[torch.optim.Optimizer]
        base_args: Dict[str, Any] = {}

        print(f"Attempting to use base optimizer: {base_optimizer_name}")
        if base_optimizer_name == 'smmf':
            if SMMF is None: raise ImportError("SMMF optimizer class not imported.")
            base_optimizer_cls = SMMF
            base_args = train_cfg.get('base_optimizer_params', {}).get('smmf_params', {})
            if not isinstance(base_args, dict): base_args = {}
        elif base_optimizer_name == 'adamw':
            base_optimizer_cls = AdamW
            base_args = train_cfg.get('base_optimizer_params', {}).get('adamw_params', {})
            if not isinstance(base_args, dict): base_args = {}
            base_args.setdefault('fused', False) # Default fused to False for wrapper base
        else: raise ValueError(f"Unsupported base_optimizer_class_name for integrated wrapper: {base_optimizer_name}")

        wrapper_args = train_cfg.get('integrated_optimizer_params', {})
        if not isinstance(wrapper_args, dict): wrapper_args = {}

        print(f"Wrapper Args: {wrapper_args}")
        print(f"Base Optimizer Args (for {base_optimizer_name}): {base_args}")
        if lr is None: raise ValueError("lr cannot be None when using IntegratedOptimizerWrapper")

        try:
            optimizer = IntegratedOptimizerWrapper(params=params, base_optimizer_class=base_optimizer_cls, lr=lr, base_optimizer_args=base_args, **wrapper_args)
            print(colored("Successfully built IntegratedOptimizerWrapper.", "green"))
        except Exception as e: print(colored(f"Error building IntegratedOptimizerWrapper: {e}", "red")); raise e

    elif optimizer_type == 'adamw':
        adamw_params = train_cfg.get('adamw_params', {})
        if not isinstance(adamw_params, dict): adamw_params = {}
        weight_decay = train_cfg.get('weight_decay', 0.01) # Standard WD from top level
        fused_adam = train_cfg.get('use_fused_optimizer', True) and device.type == 'cuda' and not train_cfg.get('use_orthograd', False)
        if lr is None: raise ValueError("lr cannot be None when using AdamW")
        print(f"Using standard AdamW. LR={lr}, WD={weight_decay}, Betas={adamw_params.get('betas')}, Eps={adamw_params.get('eps')}, Fused={fused_adam}")
        try:
             optimizer = AdamW(params, lr=lr, weight_decay=weight_decay, betas=adamw_params.get('betas', [0.9, 0.95]), eps=adamw_params.get('eps', 1e-8), fused=fused_adam)
        except RuntimeError as e:
             if "found parameters occupying contiguous memory" in str(e) and fused_adam:
                  print(colored("Fused AdamW failed, falling back to non-fused.", "yellow"))
                  optimizer = AdamW(params, lr=lr, weight_decay=weight_decay, betas=adamw_params.get('betas', [0.9, 0.95]), eps=adamw_params.get('eps', 1e-8), fused=False)
             else: raise e
        if train_cfg.get('use_orthograd', False):
             if OrthoGrad is None: raise ImportError("OrthoGrad requested but not imported.")
             print(colored("Wrapping AdamW with OrthoGrad", "yellow"))
             ortho_base_args = {'lr': lr, 'weight_decay': weight_decay, 'betas': adamw_params.get('betas', [0.9, 0.95]), 'eps': adamw_params.get('eps', 1e-8), 'fused': False}
             optimizer = OrthoGrad(params, base_optimizer_cls=AdamW, **ortho_base_args)

    elif optimizer_type == 'smmf':
        if SMMF is None: raise ImportError("SMMF optimizer class not imported.")
        smmf_params = train_cfg.get('smmf_params', {})
        if not isinstance(smmf_params, dict): smmf_params = {}
        if lr is None: raise ValueError("lr cannot be None when using SMMF")
        print(f"Using standard SMMF. LR={lr}, Params={smmf_params}")
        optimizer = SMMF(params, lr=lr, **smmf_params)
        if train_cfg.get('use_orthograd', False):
             if OrthoGrad is None: raise ImportError("OrthoGrad requested but not imported.")
             print(colored("Wrapping SMMF with OrthoGrad", "yellow"))
             ortho_base_args = {'lr': lr, **smmf_params}
             optimizer = OrthoGrad(params, base_optimizer_cls=SMMF, **ortho_base_args)

    # --- Adafactor handling (commented out unless needed) ---
    # elif optimizer_type == 'adafactor':
    #     from transformers import Adafactor # Ensure import only if needed
    #     adafactor_params = train_cfg.get('adafactor_params', {})
    #     if not isinstance(adafactor_params, dict): adafactor_params = {}
    #     print(f"Using Adafactor. Params={adafactor_params}")
    #     adafactor_args = { 'relative_step': adafactor_params.get('relative_step', True), # ... other adafactor args }
    #     if not adafactor_args['relative_step']:
    #         if lr is None: raise ValueError("lr cannot be None for Adafactor if relative_step=False")
    #         adafactor_args['lr'] = lr
    #     else:
    #          adafactor_args['lr'] = None # Explicitly None if relative
    #     optimizer = Adafactor(params, **adafactor_args)
    #     # OrthoGrad wrapper for Adafactor would go here if needed

    else:
        raise ValueError(f"Unsupported optimizer_type: '{optimizer_type}'")

    return optimizer


# --- Scheduler Building ---
def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: Dict[str, Any], num_training_steps: int, steps_per_epoch: int) -> Optional[_LRScheduler]:
    """Builds the learning rate scheduler based on the training configuration."""
    if not train_cfg.get('use_lr_scheduler', False):
        print("LR Scheduler disabled.")
        return None

    scheduler_type = train_cfg.get('scheduler_type')
    if scheduler_type is None:
         print("use_lr_scheduler is true, but scheduler_type is null. Disabling scheduler.")
         return None
    scheduler_type = scheduler_type.lower()

    # Check if optimizer type is compatible with external schedulers
    optimizer_type = train_cfg.get('optimizer_type', '').lower()
    if 'schedulefree' in optimizer_type:
         print(colored("Warning: External LR scheduler requested but ScheduleFree logic is active in the optimizer. Disabling external scheduler.", "yellow"))
         return None

    warmup_steps = train_cfg.get('warmup_steps', 0)
    # Fetch LR from optimizer groups *after* optimizer is built
    try:
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr is None and scheduler_type == 'onecycle':
             raise ValueError("Cannot use OneCycleLR if optimizer's base LR is None (e.g., Adafactor with relative_step=True)")
    except (IndexError, KeyError):
         print("Warning: Could not retrieve LR from optimizer param_groups. Scheduler might not work correctly.")
         current_lr = train_cfg.get('lr', 0.0) # Fallback, might be incorrect


    print(f"Building LR Scheduler: Type={scheduler_type}, WarmupSteps={warmup_steps}, TotalOptSteps={num_training_steps}")

    if scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    elif scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    elif scheduler_type == 'onecycle':
        onecycle_params = train_cfg.get('onecycle_params', {})
        if current_lr is None: # Should have been caught above, but double check
            raise ValueError("Base LR is None, cannot use OneCycleLR.")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=current_lr, # Use the optimizer's current LR as max_lr
            total_steps=num_training_steps,
            pct_start=onecycle_params.get('pct_start', 0.3),
            anneal_strategy=onecycle_params.get('anneal_strategy', 'cos'),
            div_factor=onecycle_params.get('div_factor', 25.0),
            final_div_factor=onecycle_params.get('final_div_factor', 10000.0)
        )
    else:
        print(f"Unsupported scheduler_type: {scheduler_type}. No scheduler will be used.")
        scheduler = None

    return scheduler

# --- Checkpoint Functions ---
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[_LRScheduler],
                    global_step: int, epoch: int, config: Dict, filename: str):
    """Saves model, optimizer, scheduler state, and config to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Get model state dict, handling potential compile wrappers
    model_state = model.state_dict()
    if hasattr(model, '_orig_mod'): # Check if model was torch.compiled
        print("Saving state_dict from original model (before compile)...")
        model_state = model._orig_mod.state_dict()

    state = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'global_step': global_step,
        'config_snapshot': config # Save the config used for this run
    }
    try:
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename} (Step: {global_step}, Epoch: {epoch})")
    except Exception as e:
        print(f"Error saving checkpoint to {filename}: {e}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[_LRScheduler],
                    device: torch.device, train_cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Loads state from a checkpoint file if specified in train_cfg."""
    resume_path = train_cfg.get('resume_from_checkpoint')
    global_step, start_epoch = 0, 0

    if resume_path and resume_path != 'null' and os.path.isfile(resume_path):
        print(f"Attempting to resume from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)

            # --- Load Model State ---
            model_state = checkpoint['model_state_dict']
            # Basic check for compile wrapper state dict keys
            is_compiled_state = any(k.startswith('_orig_mod.') for k in model_state.keys())
            is_compiled_model = hasattr(model, '_orig_mod')

            if is_compiled_state and not is_compiled_model:
                 print("Warning: Checkpoint seems to be from a compiled model, but current model isn't compiled. Attempting to load _orig_mod state.")
                 model_state = {k.replace('_orig_mod.', '', 1): v for k, v in model_state.items() if k.startswith('_orig_mod.')}
            elif not is_compiled_state and is_compiled_model:
                 print("Warning: Checkpoint seems to be from a non-compiled model, but current model is compiled. Loading into _orig_mod.")
                 model._orig_mod.load_state_dict(model_state, strict=True)
                 model_loaded = True # Flag that we loaded into the original module
            else: # Either both compiled or both not compiled
                 target_model = model._orig_mod if is_compiled_model else model
                 target_model.load_state_dict(model_state, strict=True)
                 model_loaded = True

            if not model_loaded: # If loading into _orig_mod failed or wasn't applicable
                print("Loading model state directly...")
                model.load_state_dict(model_state, strict=True)

            # --- Load Optimizer State ---
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # --- Load Scheduler State ---
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"Warning: Could not load scheduler state dict ({e}). Scheduler will start from scratch.")
            elif scheduler:
                 print("Warning: Checkpoint does not contain scheduler state, starting scheduler from scratch.")

            # --- Load Step and Epoch ---
            global_step = checkpoint.get('global_step', 0)
            # Load epoch + 1 to start *after* the saved epoch
            start_epoch = checkpoint.get('epoch', 0) # If saved at end of epoch N, start epoch N+1 (index N)
            print(colored(f"Checkpoint loaded successfully. Resuming from Step: {global_step}, Start Epoch: {start_epoch}", "green"))

        except Exception as e:
            print(colored(f"Error loading checkpoint: {e}. Starting from scratch.", "red"))
            import traceback
            traceback.print_exc()
            global_step, start_epoch = 0, 0
    else:
        print("No valid checkpoint path provided or file not found. Starting from scratch.")
        global_step, start_epoch = 0, 0

    return global_step, start_epoch


# --- Evaluation Function ---
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, config: Dict[str, Any], device: torch.device, is_final_eval: bool = False, step: Optional[int] = None, use_amp: bool = False, amp_dtype: torch.dtype = torch.float16 ) -> float:
    model.eval(); total_loss, total_samples = 0.0, 0
    eval_desc = "Final Eval" if is_final_eval else f"Step {step} Eval"; data_cfg, model_cfg_dict = config['data_config'], config['model_config']
    eval_encoding = data_cfg.get('encoding', 'latin1'); pad_token_id, ignore_index = data_cfg['pad_token_id'], model_cfg_dict['ignore_index']
    printed_this_eval = False # Flag to print only one example *per call* to evaluate

    pbar_eval = tqdm(loader, desc=eval_desc, leave=False)
    for batch in pbar_eval:
        if not batch: continue
        input_ids, targets, attention_mask = batch['input_ids'].to(device, non_blocking=True), batch['labels'].to(device, non_blocking=True), batch['attention_mask'].to(device, non_blocking=True)
        with autocast(enabled=use_amp, dtype=amp_dtype): logits, loss = model(x=input_ids, targets=targets, attention_mask=attention_mask)
        assert loss is not None and not torch.isnan(loss).any() and not torch.isinf(loss).any(), f"Eval loss invalid: {loss}"
        if loss.numel() > 1: loss = loss.mean()
        loss_item = loss.item(); bs = input_ids.size(0); total_loss += loss_item * bs; total_samples += bs
        pbar_eval.set_postfix_str(f"Loss: {loss_item:.4f}")

        # --- Modified Example Printing ---
        # Try to print one random example per evaluation run
        if not printed_this_eval and bs > 0:
            try: # Add try-except for safety during random selection/decoding
                idx = random.randrange(bs); # Select random index from batch
                inp_ids, tgt_ids = input_ids[idx].cpu().tolist(), targets[idx].cpu().tolist(); pred_ids = torch.argmax(logits[idx].float(), dim=-1).cpu().tolist()
                inp_txt = decode_bytes([t for t in inp_ids if t != pad_token_id], eval_encoding); tgt_txt = decode_bytes([t for t in tgt_ids if t != pad_token_id and t != ignore_index], eval_encoding); pred_txt = decode_bytes([t for t in pred_ids if t != pad_token_id and t != ignore_index], eval_encoding)
                print(f"\n--- Eval Example (Step {step}, Batch Idx {idx}) ---\nInput : {colored(inp_txt, 'green')}\nTarget: {colored(tgt_txt, 'yellow')}\nPred  : {colored(pred_txt, 'blue')}\n" + "-" * 30)
                printed_this_eval = True # Mark as printed for this eval run
            except Exception as e:
                print(colored(f"Error printing eval example: {e}", "yellow"))
                printed_this_eval = True # Still mark as printed to avoid spamming errors

    model.train(); assert total_samples > 0, "No samples processed during evaluation."
    return total_loss / total_samples

# --- Main Training Function ---
def main(config_path: str):
    """Main function to load config, set up, and run training."""
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: print(f"Error reading/parsing config file {config_path}: {e}"); return

    mcfg_dict = config.get('model_config', {}); dcfg = config.get('data_config', {})
    tcfg = config.get('training_config', {}); wcfg = config.get('wandb_config', {})

    seed = tcfg.get('seed', 42); set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    use_amp_config = tcfg.get('use_amp', False); amp_enabled = use_amp_config and device.type == 'cuda'
    amp_dtype_str = tcfg.get('amp_dtype', 'float16').lower(); amp_dtype = torch.float16
    if amp_enabled:
        if amp_dtype_str == 'bfloat16':
            if torch.cuda.is_bf16_supported(): amp_dtype = torch.bfloat16; print(colored("AMP enabled: bfloat16", "green"))
            else: print(colored("Warning: bfloat16 specified but not supported, falling back to float16.", "yellow")); amp_dtype = torch.float16; print(colored("AMP enabled: float16", "green"))
        else: amp_dtype = torch.float16; print(colored("AMP enabled: float16", "green"))
    else: amp_dtype = torch.float32; print("AMP disabled.")

    try: # Encapsulate setup steps
        mcfg = EfficientNGPTConfig(**mcfg_dict); model = EfficientNGPTModel(mcfg).to(device)
        register_l2norm_parametrization(model)
        model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Config Block Size: {mcfg.block_size}"); print(f"Model Trainable Params: {model_param_count:,}")

        tokenizer = None # No tokenizer needed for byte-level
        if tcfg.get('Tokenizer'): print("Warning: 'Tokenizer' field found in config but byte-level model doesn't use it.")

        use_torch_compile = tcfg.get('use_torch_compile', False)
        if use_torch_compile and hasattr(torch, 'compile'):
            compile_mode = tcfg.get('torch_compile_mode', 'default'); print(f"Attempting torch.compile (mode: {compile_mode})...")
            try: model = torch.compile(model, mode=compile_mode); print(colored("Compile OK.", "green"))
            except Exception as e: print(colored(f"torch.compile failed: {e}. Proceeding without compile.", "yellow"))
        elif use_torch_compile: print("torch.compile requested but not available.")

        train_ds, test_ds = prepare_data(dcfg, mcfg, tcfg, tokenizer)

        batch_size = tcfg.get('batch_size', 32)
        collate_fn = functools.partial(simple_collate_fn, pad_token_id=dcfg.get('pad_token_id', 0), ign_token_id=mcfg.ignore_index)
        pin_mem = device.type == 'cuda'; num_work = dcfg.get('num_workers', 0); persist = dcfg.get('dataloader_persistent_workers', True) and num_work > 0
        print(f"DataLoader: Workers={num_work}, PinMem={pin_mem}, Persistent={persist}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_work, pin_memory=pin_mem, drop_last=True, persistent_workers=persist)
        test_loader_full = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_work, pin_memory=pin_mem, persistent_workers=persist) if test_ds else None

        eval_subset_size = tcfg.get('eval_subset_size', 200)
        test_loader_subset = None
        if test_ds:
            eval_indices = range(min(eval_subset_size, len(test_ds)))
            eval_subset_ds = Subset(test_ds, eval_indices)
            test_loader_subset = DataLoader(eval_subset_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_work, pin_memory=pin_mem, persistent_workers=persist)
            print(f"Intermediate eval sequences: {len(eval_subset_ds)}")
        else: print("No test data, skipping subset loader creation.")

        optimizer = build_optimizer(model, tcfg, device)

        n_epochs = tcfg.get('num_epochs', 1); steps_per_ep = len(train_loader); grad_accum = tcfg.get('gradient_accumulation_steps', 1)
        if steps_per_ep == 0: raise RuntimeError("Training DataLoader is empty!")
        if grad_accum < 1: grad_accum = 1
        num_opt_steps = max(1, steps_per_ep // grad_accum) * n_epochs
        print(f"Grad Accum: {grad_accum}, Steps/Epoch: {steps_per_ep}, Total Est. Opt Steps: {num_opt_steps}")

        scheduler = build_scheduler(optimizer, tcfg, num_opt_steps, steps_per_ep)

    except Exception as setup_err:
        print(colored(f"Error during setup: {setup_err}", "red"))
        import traceback
        traceback.print_exc()
        return # Exit if setup fails

    # --- WandB ---
    use_wandb = wcfg.get('project') and not tcfg.get('disable_wandb', False)
    if use_wandb:
        try:
            eff_bs = batch_size * grad_accum; print(f"Effective BS: {eff_bs}")
            if 'training_config' not in config: config['training_config'] = {}
            config['training_config']['effective_batch_size'] = eff_bs
            config['training_config']['model_param_count'] = model_param_count
            wandb.init(project=wcfg['project'], entity=wcfg.get('entity'), name=wcfg.get('run_name'), config=config)
            wandb.watch(model, log='gradients', log_freq=tcfg.get('log_interval', 100) * grad_accum)
        except Exception as e: print(f"Error initializing WandB: {e}. Disabling WandB."); use_wandb = False
    else: print("WandB disabled.")

    # --- Output Directories ---
    output_dir = tcfg.get('output_dir', 'output_ngpt')
    checkpoint_dir = tcfg.get('checkpoint_dir', os.path.join(output_dir, 'checkpoints'))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Load Checkpoint ---
    start_step, start_epoch = load_checkpoint(model, optimizer, scheduler, device, tcfg)
    global_step = start_step

    # --- Training Loop ---
    print(colored(f"--- Starting Training ---", "cyan"))
    latest_eval_loss = float('nan'); train_loss_accum = 0.0; batches_in_step = 0; last_micro_loss = float('nan')
    is_integrated_optimizer = isinstance(optimizer, IntegratedOptimizerWrapper)
    print(f"Using Integrated Optimizer: {is_integrated_optimizer}")

    # Define closure function outside the loop if possible (captures necessary variables)
    # Note: This requires input_ids, targets, attention_mask to be accessible.
    # Defining inside the loop is safer if variables change unexpectedly.
    def training_closure() -> torch.Tensor:
        # Closure needs access to current batch's data
        nonlocal input_ids, targets, attention_mask, model, amp_enabled, amp_dtype, grad_accum
        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
            logits, loss = model(x=input_ids, targets=targets, attention_mask=attention_mask)
            if loss is None: raise ValueError("Model returned None loss")
            loss_val = loss.item() # Get value before potential division
            if math.isnan(loss_val) or math.isinf(loss_val):
                raise ValueError(f"NaN/Inf loss detected in closure: {loss_val}")
            loss_scaled = loss / grad_accum
        loss_scaled.backward()
        return loss # Return original loss

    try: # Wrap training loop in try/except for graceful exit
        for epoch in range(start_epoch, n_epochs):
            model.train()
            pbar_train = tqdm(enumerate(train_loader), total=steps_per_ep, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False)
            epoch_start_time = time.time(); train_loss_accum = 0.0; batches_in_step = 0; total_norm = None

            for micro_batch_idx, batch in pbar_train:
                 # --- Skip logic based on global_step ---
                current_effective_step = global_step + (batches_in_step // grad_accum) # Estimate current opt step
                # This skip logic might be complex, simpler to just run and let state resume?
                # Let load_checkpoint handle starting step.

                if not batch: print(f"Warning: Skipping empty batch at micro_batch_idx {micro_batch_idx}"); continue

                input_ids = batch['input_ids'].to(device, non_blocking=True)
                targets = batch['labels'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)

                # --- Forward/Backward/Step ---
                micro_loss = float('nan') # Initialize micro_loss
                try:
                    if is_integrated_optimizer:
                        original_loss = optimizer.step(training_closure) # Handles internal F/B
                        if original_loss is not None: micro_loss = original_loss.item()
                    else: # Standard optimizer
                        if batches_in_step % grad_accum == 0: # Zero only at start of accum cycle
                             optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
                             logits, loss = model(x=input_ids, targets=targets, attention_mask=attention_mask)
                             if loss is None: raise ValueError("Model returned None loss")
                             loss_val = loss.item()
                             if math.isnan(loss_val) or math.isinf(loss_val): raise ValueError(f"NaN/Inf loss: {loss_val}")
                             loss_for_backward = loss / grad_accum
                        micro_loss = loss.item()
                        loss_for_backward.backward()
                        # Step happens after accumulating
                        if (batches_in_step + 1) % grad_accum == 0:
                             apply_clip = tcfg.get('apply_grad_clipping', True); clip_norm = tcfg.get('grad_clip_norm')
                             if apply_clip and clip_norm is not None and clip_norm > 0:
                                 total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                             else: total_norm = None
                             optimizer.step()

                except Exception as e:
                     print(colored(f"\nError during training step {global_step}, micro-batch {micro_batch_idx}: {e}", "red"))
                     import traceback; traceback.print_exc()
                     raise e # Stop training

                last_micro_loss = micro_loss

                # --- Accumulate Loss & Update TQDM ---
                if not math.isnan(micro_loss) and not math.isinf(micro_loss): train_loss_accum += micro_loss
                batches_in_step += 1
                current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else None
                lr_str = f"{current_lr:.2e}" if current_lr is not None else "N/A"
                pbar_train.set_postfix(opt_step=global_step, micro_loss=f"{last_micro_loss:.4f}" if not math.isnan(last_micro_loss) else "N/A", eval=f"{latest_eval_loss:.4f}" if not math.isnan(latest_eval_loss) else "N/A", lr=lr_str)

                # --- Post-Step Logic ---
                if batches_in_step % grad_accum == 0:
                    # --- Scheduler Step ---
                    if scheduler: scheduler.step()

                    # --- Logging ---
                    avg_loss_in_step = train_loss_accum / batches_in_step if batches_in_step > 0 else float('nan')
                    pbar_train.set_postfix(opt_step=global_step, avg_loss=f"{avg_loss_in_step:.4f}" if not math.isnan(avg_loss_in_step) else "N/A", eval=f"{latest_eval_loss:.4f}" if not math.isnan(latest_eval_loss) else "N/A", lr=lr_str)
                    if use_wandb:
                        log_data = {"train/avg_loss_step": avg_loss_in_step, "train/learning_rate": current_lr if current_lr is not None else 0.0, "epoch": epoch + (micro_batch_idx + 1) / steps_per_ep, "train/last_micro_loss": last_micro_loss}
                        if total_norm is not None: log_data["train/gradient_norm"] = total_norm.item() if isinstance(total_norm, torch.Tensor) and not torch.isnan(total_norm).any() else float('nan')
                        wandb.log(log_data, step=global_step)
                    train_loss_accum = 0.0 # Reset accumulators *after* logging avg

                    # --- Eval & Checkpoint ---
                    eval_interval = tcfg.get('eval_interval')
                    if eval_interval and eval_interval > 0 and global_step > 0 and global_step % eval_interval == 0:
                         eval_loss = evaluate(model, test_loader_subset, config, device, step=global_step, use_amp=amp_enabled, amp_dtype=amp_dtype)
                         latest_eval_loss = eval_loss # Update known eval loss
                         if use_wandb: wandb.log({"eval/loss": eval_loss}, step=global_step)
                         # No need to print here, evaluate function prints

                    save_interval = tcfg.get('save_interval')
                    if save_interval and save_interval > 0 and global_step > 0 and global_step % save_interval == 0:
                         save_checkpoint(model, optimizer, scheduler, global_step, epoch, config, os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pth"))

                    global_step += 1
                    # Reset counters for next accumulation cycle
                    batches_in_step = 0; total_norm = None


            # --- End of Epoch ---
            pbar_train.close()
            epoch_duration_min = (time.time() - epoch_start_time) / 60
            print(colored(f"\nEpoch {epoch + 1} finished ({epoch_duration_min:.2f} min). Global Step: {global_step}", "cyan"))
            if tcfg.get('eval_full_at_epoch_end', True):
                 eval_loss = evaluate(model, test_loader_full, config, device, is_final_eval=True, step=global_step, use_amp=amp_enabled, amp_dtype=amp_dtype)
                 if use_wandb: wandb.log({"eval/epoch_end_loss": eval_loss}, step=global_step)
                 latest_eval_loss = eval_loss
            if tcfg.get('save_at_epoch_end', True):
                 save_checkpoint(model, optimizer, scheduler, global_step, epoch + 1, config, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    except Exception as train_err: # Catch errors during training loop
         print(colored(f"Training interrupted by error: {train_err}", "red"))
         import traceback
         traceback.print_exc()
    finally: # Ensure final save and wandb finish even if loop breaks
        # --- End of Training ---
        print(colored("--- Training Finished or Interrupted ---", "green"))
        final_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_final.pth")
        print(f"Saving final checkpoint to {final_ckpt_path}...")
        save_checkpoint(model, optimizer, scheduler, global_step, epoch if 'epoch' in locals() else n_epochs, config, final_ckpt_path)
        if use_wandb and wandb.run is not None:
             print("Finishing WandB run...")
             wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNGPT Model (V6 Corrected).")
    parser.add_argument("--config_path", type=str, default="config_final_v6.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config_path)