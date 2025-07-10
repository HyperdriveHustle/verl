Using Checkpoints to Support Fault Tolerance Training
=====================================================

There could be training errors or machine failure during the whole RLHF training process, 
so it is recommended to enable checkpoints to minimize your loss.

The API Interface has already been listed in :ref:`config-explain-page`,
and we will not repeat them. But there are still some technique details
we hope to clarify.

.. note:: 

    Notice that the ``checkpoint.contents`` field has no effect to FSDP checkpoint except ``hf_model``, 
    the other 3 fields are binded together to save and load. We recommend to include ``model``, ``optimizer`` and ``extra`` all.

Checkpoint Saving Directory Structure
-------------------------------------

Commonly, we use the ``default_local_dir`` declared in ``ppo_trainer.yaml`` or ``ppo_megatron_trainer.yml``
to work as preffix when saving checkpoints, which is ``checkpoints/${trainer.project_name}/${trainer.experiment_name}``.

So the inner checkpoint structure of **FSDP** is like:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
<<<<<<< HEAD
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   ├── actor_huggingface
    │   ├── critic
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   └── critic_huggingface
    └── latest_checkpointed_iteration.txt

All model shards, optimizers and extra states are stored togather, in a sharded and distributed way.
=======
    │   │   ├── huggingface     # default save config and tokenizer, save huggingface model if include ``hf_model`` in checkpoint.contents
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   ├── critic
    │   │   ├── huggingface
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    └── latest_checkpointed_iteration.txt

All model shards, optimizers and extra states are stored together, in a sharded and distributed way.
>>>>>>> verl_0626

While **Megatron** current checkpoint structure is:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
<<<<<<< HEAD
    │   │   ├── huggingface     # default save tokenizer, save huggingface model if include ``hf_mode`` in checkpoint.contents
    │   │   ├── model           # save sharded model, naming the same as Megatron
    │   │   │   ├── mp_rank_xx_yyy          # xx is tp_rank in 2 digits, yyy is pp_rank in 3 digits
    │   │   │   │   └── model_states.pt
    │   │   │   └── mp_rank_xx_xxx
    │   │   ├── optim
    │   │   │   ├── distrib_optim_pp{x}_tp{y}.pt
    │   │   │   └── distrib_optim_pp{x}_tp{y}.pt
    │   │   └── rng_states
    │   └── critic
    │   │   ├── huggingface
    │   │   ├── model
    │   │   ├── optim
    │   │   └── rng_states
=======
    │   │   ├── huggingface     # default save config and tokenizer, save huggingface model if include ``hf_mode`` in checkpoint.contents
    │   │   └── dist_ckpt       # save sharded model/optimizer/rng_states, naming the same as Megatron
    │   └── critic
    │   │   ├── huggingface
    │   │   └── dist_ckpt
>>>>>>> verl_0626
    └── latest_checkpointed_iteration.txt

Convert FSDP and Megatron Checkpoints to HuggingFace Format Model
-----------------------------------------------------------------

We provide a tool to convert the FSDP and Megatron checkpoints to HuggingFace format model.
<<<<<<< HEAD
The tool is located in ``scripts/model_merger.py``.

The arguments are as follows:

.. code:: bash

    usage: model_merger.py [-h] [--backend {fsdp,megatron}]
                           [--tie-word-embedding whether the model share embedding weights]
                           [--is-value-model whether the model is critic model]
                           [--hf_model_path $original_model_path, like {Qwen/Qwen2-7B}]
                           [--local_dir $local_directory saved fsdp or megatron models]
                           [--target_dir $target_dir to save converted models, default is tmp]
                           [--hf_upload_path $huggingface_repo to upload]

So example use of Megatron model merger is:

.. code:: bash

    python3 scripts/model_merger.py --backend megatron \
        --is-value-model \
        --hf_model_path Qwen/Qwen2-7B \
        --local_dir checkpoints/verl_megatron_gsm8k_examples/deepseek_megatron_checkpoint_saveload/global_step_1/actor/model
=======
The tool is located in ``verl/model_merger``.

The script supports two main sub-commands: `merge` (to convert and save checkpoints) and `test` (to validate merged checkpoints against a reference model).
The arguments for the `merge` sub-command are as follows:

.. code:: bash

    usage: python -m verl.model_merger merge [-h] --backend {fsdp,megatron} --local_dir LOCAL_DIR [--hf_model_path HF_MODEL_PATH]
                                [--tie-word-embedding] [--is-value-model] [--target_dir TARGET_DIR]
                                [--hf_upload_path HF_UPLOAD_PATH] [--private]

    options:
    -h, --help            show this help message and exit
    --backend {fsdp,megatron}
                            The backend of the model
    --local_dir LOCAL_DIR
                            Path to the saved model checkpoints
    --hf_model_path HF_MODEL_PATH
                            (Deprecated) Path to the original Hugging Face model for config.
    --tie-word-embedding  Whether to tie word embedding weights (currently only Megatron supported)
    --is-value-model      Whether the model is a value model (currently only Megatron supported)
    --target_dir TARGET_DIR
                            Directory to save the merged huggingface model
    --hf_upload_path HF_UPLOAD_PATH
                            Hugging Face repository ID to upload the model
    --private             Whether to upload the model to a private Hugging Face repository

Example usage for merging Megatron checkpoints:

.. code:: bash

    python -m verl.model_merger merge \
        --backend megatron \
        --tie-word-embedding \
        --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
        --target_dir /path/to/merged_hf_model

Example usage for merging FSDP checkpoints:

.. code:: bash

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
        --target_dir /path/to/merged_hf_model

>>>>>>> verl_0626

Megatron Merger details
-----------------------

Current implement of decoder layers uses ``nn.ModuleList`` to store the layers, 
and thus the model layers on every PP rank and VPP rank starts their index from 0.

There are 3 ways to correct this behavior:

1. Modify the decoder layer's state_dict, add ``offset`` to each layer's index, thus rewrite ``nn.ModuleList`` implementation.
2. Modify the layer index when saving checkpoint and recover them when loading checkpoint.
3. The Checkpoint merger do this work, calculate the actual ``offset`` from ``state_dict`` only, a little complex.

Current implementation use solution 2.

<<<<<<< HEAD
=======

HuggingFace to Megatron DistCheckpoint details
----------------------------------------------

If your model is quite huge, we recommend you to use Megatron dist-checkpoint to load the model.
Megatron dist-checkpoint supports loading with different kinds of model parallelism,
and it is much faster than the original checkpoint loading.

To convert original HuggingFace model to Megatron dist-checkpoint,
you can use the ``scripts/converter_hf_to_mcore.py`` script. Large MoE models are temporarily supported with CPU initialization,
which is a little slower. While we are working on a better solution to support large models.

Example command to convert the model is as follows:

.. code:: bash

    python scripts/converter_hf_to_mcore.py \
        --hf_model_path Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --output_path /mnt/disk/Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --use_cpu_initialization    # Only work for MoE models


>>>>>>> verl_0626
Original Checkpoint Utils
-------------------------

Original Checkpoint Utils refer to original checkpoint implementation in ``verl/models/[model]/megatron/checkpoint_utils``.

We only need ``[model]_loader.py`` in original checkpoint utils now, since we get rid of storing ``hf_model`` every time (which is not recommended for large model training, try only saving sharded models if you can).

.. note:: 

    Note that ``[model]_loader`` only support environments where **storage clusters are able to connect with every calculation nodes**. 
    Because it utilizes **sharded load way to minimize the loading checkpoint overhead**. 
    Every rank loads its own data from ``state_dict`` which can be accessed by all of them.
    While there is also no need to broadcast among DP ranks, since the saved state_dict is only produced by DP rank 0.

    For users who can **only place the huggingface model on one device**, we keep the original costly implementation in ``[model]_loader_deprecated``. In this implementation, rank 0 broadcast all weights to each tp and pp rank, and then dp rank 0 broadcast to all dp ranks. There may be at risks of OOM.

    To use deprecated loader, change the import package of ``load_state_dict_to_megatron_llama``.
