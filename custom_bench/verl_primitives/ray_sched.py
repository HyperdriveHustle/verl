import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from time import sleep


@ray.remote
def my_function():
    sleep(2)
    return ray.get_runtime_context().get_node_id()


def main():
    ray.init()

    # Get the node IDs
    nodes = ray.nodes()
    target_node_id = nodes[-1]["NodeID"]
    print(f'{target_node_id=}')

    # Schedule the function to run on a specific node using NodeAffinitySchedulingStrategy
    result = my_function.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=target_node_id,
            soft=False,
        )).remote()
    print(ray.get(result))


if __name__ == "__main__":
    main()
