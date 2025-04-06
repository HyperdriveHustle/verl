import warnings
try:
    import megatron
    print('file:', megatron.__file__)
    from megatron.core import parallel_state as mpu
except:
    raise ImportError('megatron not found')

import os
import sys
from verl.single_controller.base.decorator import register, Dispatch, Execute
from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
from verl.single_controller.base.megatron.worker import MegatronWorker
from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs
from omegaconf import OmegaConf

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
import torch

import pandas as pd
import time


def query_ray_cluster():
    print('*'*100)
    # Connect to the Ray cluster if not already connected
    if not ray.is_initialized():
        # Connect to an existing Ray cluster
        #ray.init(address='auto', runtime_env={"pip": ["megatron-core"]})
        ray.init(address='auto')
        print(f"Connected to Ray cluster at {ray.get_runtime_context().gcs_address}")

    # Get cluster resources
    nodes = ray.nodes()

    # Process node information
    node_data = []
    for node in nodes:
        node_info = {
            'Node ID': node['NodeID'][:8] + '...',  # Truncated node ID for readability
            'IP Address': node['NodeManagerAddress'],
            'Hostname': node.get('Hostname', 'N/A'),
            'Status': 'ALIVE' if node['Alive'] else 'DEAD',
            'CPU': f"{node['Resources'].get('CPU', 0):.1f}",
            'Memory (GB)': f"{node['Resources'].get('memory', 0) / (1024 * 1024 * 1024):.2f}",
            'GPU': f"{node['Resources'].get('GPU', 0):.0f}"
        }

        # Add any custom resources
        for resource, value in node['Resources'].items():
            if resource not in ['CPU', 'memory', 'GPU', 'object_store_memory']:
                node_info[resource] = value

        node_data.append(node_info)

    # Create a DataFrame for better display
    df = pd.DataFrame(node_data)
    print('ray cluster:')
    print(df)

    # list GPUs
    resources = ray.cluster_resources()
    gpu_info = {}
    for key, value in resources.items():
        if key.startswith("GPU") or "gpu" in key.lower():
            gpu_info[key] = value

    print(f"All cluster GPU resources: {gpu_info}")
    print('*'*100)


@ray.remote
class MLPLayerWorker(MegatronWorker):

    def __init__(self):

        # supress ray worker warning
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        super().__init__()

        rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])

        # print
        worker_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "None")
        node_id = ray.get_runtime_context().get_node_id()

        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)
        world_size = torch.distributed.get_world_size()

        device = torch.cuda.current_device()
        print(f"Ray node ID: {node_id}; CUDA_VISIBLE_DEVICES: {worker_gpus}")
        print(f'ray worker: {global_rank=} {rank=} {world_size=} {torch.cuda.current_device()=}; {torch.cuda.get_device_name(device)=}; {torch.cuda.get_device_capability(device)=}')

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            nccl_communicator_config_path=None,
        )
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(10)

        if global_rank == 0:
            # NOTE: multi-controller code:
            print(f'init model parallel: {global_rank=}')

            print('tensor model parallel size: ', mpu.get_tensor_model_parallel_world_size(),
                  mpu.get_tensor_model_parallel_rank())
            print('pipeline model parallel size: ', mpu.get_pipeline_model_parallel_world_size(),
                  mpu.get_pipeline_model_parallel_rank())
            print('data parallel size: ', mpu.get_data_parallel_world_size(), mpu.get_data_parallel_rank())
            print('virtual pipeline parallel: ', mpu.get_virtual_pipeline_model_parallel_world_size(),
                  mpu.get_virtual_pipeline_model_parallel_rank())

            print('all rank: ', mpu.get_all_ranks())

    @register(Dispatch.ONE_TO_ALL)
    def init_model(self, config):
        from omegaconf import OmegaConf
        from verl.utils.megatron_utils import init_model_parallel_config
        from verl.models.llama.megatron.layers import ParallelLlamaMLP
        megatron_config = OmegaConf.create({
            'sequence_parallel': False,
            'param_dtype': 'fp32',
            'tensor_model_parallel_size': mpu.get_tensor_model_parallel_world_size(),
            'pipeline_model_parallel_rank': mpu.get_pipeline_model_parallel_rank(),
            'pipeline_model_parallel_size': mpu.get_pipeline_model_parallel_world_size(),
            'virtual_pipeline_model_parallel_rank': mpu.get_virtual_pipeline_model_parallel_rank(),
            'virtual_pipeline_model_parallel_size': mpu.get_virtual_pipeline_model_parallel_world_size()
        })

        megatron_config = init_model_parallel_config(megatron_config)
        self.parallel_layer = ParallelLlamaMLP(config=config, megatron_config=megatron_config)

    def worker_sync(self):
        print(f'worker sync {self.rank=}')

    @register(Dispatch.ONE_TO_ALL)
    def get_weights(self):
        output = {}
        for key, val in self.parallel_layer.named_parameters():
            output[key] = val
        return output

    @register(Dispatch.MEGATRON_COMPUTE)
    def run_layer(self, x):
        x = x.to('cuda')
        y = self.parallel_layer(x)
        return y


class RayClassWithInitArgsAndSched(RayClassWithInitArgs):
    def __init__(self, cls, *args, **kwargs) -> None:

        self.target_node_id = kwargs.pop('target_node_id', None)
        self.cuda_visible_devices = kwargs.pop('cuda_visible_devices', None)

        # assume visible devices start from 0
        self.cnt = 0

        assert self.target_node_id is not None, f'target node must specify'
        assert self.cuda_visible_devices is not None, f'visible device must specify'

        super().__init__(cls, *args, **kwargs)

    def __call__(self,
                 placement_group,
                 placement_group_bundle_idx,
                 use_gpu: bool,
                 num_gpus=1,
                ):
        # the signature stay the same
        options = {
            "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=self.target_node_id,
            soft=False,
            )
        }
        options.update(self._options)
        if use_gpu:
            options["num_gpus"] = num_gpus

        if len(self._additional_resource) > 1:
            for k, v in self._additional_resource.items():
                options[k] = v

        # cuda_visible_devices = self.cuda_visible_devices[self.cnt]
        # self.cnt+=1

        return self.cls.options(
                **options).remote(
                    *self.args,
                    #cuda_visible_devices=cuda_visible_devices,
                    **self.kwargs,
                )


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    query_ray_cluster()
    print()

    # NOTE: notice the difference between [4] and [1,1,1,1] results in different placement groups
    resource_pool = RayResourcePool([4], use_gpu=True, max_colocate_count=1)
    # resource_pool = RayResourcePool([1,1,1,1], use_gpu=True, max_colocate_count=1)

    # just inspect
    print('*'*100)
    print('resource pool: ')
    print(resource_pool.store)
    print(resource_pool.max_collocate_count, resource_pool.use_gpu, resource_pool.world_size)
    pg_scheme = [[{
        "CPU": resource_pool.max_collocate_count,
        "GPU": 1
    } if resource_pool.use_gpu else {
        "CPU": resource_pool.max_collocate_count
    } for _ in range(process_count)] for process_count in resource_pool.store]
    print('pg scheme: ')
    print(pg_scheme)

    # RayClassWithInitArgs
    #layer_cls = RayClassWithInitArgs(cls=MLPLayerWorker)

    nodes = ray.nodes()
    target_node_id = nodes[-1]["NodeID"]
    print(f'{type(target_node_id)=}, {target_node_id=}')
    kwargs = {
        'target_node_id': target_node_id,
        'cuda_visible_devices': [0,1,2,3],
    }
    layer_cls = RayClassWithInitArgsAndSched(cls=MLPLayerWorker, **kwargs)

    # worker group
    wg = NVMegatronRayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=layer_cls,
    )
    print('placement Group: ', len(resource_pool.pgs))

    # NOTE: sync all worker
    # print('sync all worker')
    # wg.execute_all_sync('worker_sync')

    # barrier = ray.get(ray.remote(ray.util.wait.Barrier).remote(3))

    print('parallel sizes: ')
    print(wg.world_size, 
          wg.tp_size, 
          wg.pp_size,
          wg.dp_size,
    )
    print('*'*100)

    ffn_hidden_size = 11008
    batch_size = 16
    seq_len = 2048
    hidden_size = 4096

    config = OmegaConf.create({
        'hidden_size': hidden_size,
        'intermediate_size': ffn_hidden_size,
        'hidden_act': 'silu',
        'pretraining_tp': 1,
        'tp': wg.tp_size,
    })
    x = torch.rand(size=(seq_len, batch_size, hidden_size), dtype=torch.float32)

    wg.init_model(config)
    output = wg.run_layer(
        [x])  # This must be a list of size 1, ensuring that the input equals the data parallel (dp).

    print('model output: ')
    print(output[0].shape)

    # Shutdown ray cluster
    ray.shutdown()
