import sys

sys.path.append('.')

import yaml
import subprocess
import argparse
import json
import sys
from typing import List, Dict, Any
from time import sleep

from get_idle_nodes import get_all_nodes, get_pods_per_node, find_empty_nodes, run_kubectl_command


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Check for empty nodes in a Kubernetes cluster.')
    parser.add_argument('--context',
                        type=str,
                        default=None,
                        help='Kubernetes context to use')
    parser.add_argument('--kubeconfig',
                        type=str,
                        default=None,
                        help='Path to kubeconfig file')
    parser.add_argument('--output',
                        type=str,
                        choices=['json', 'text'],
                        default='text',
                        help='Output format (json or text)')

    #kubectl get nodes --selector=team=rlib
    parser.add_argument('--selector',
                        type=str,
                        default=None,
                        help='Label selector for nodes')
    parser.add_argument('--verbose',
                        '-v',
                        action='store_true',
                        help='Enable verbose output')

    #
    parser.add_argument('-n', type=int, default=None, help='num of node')
    parser.add_argument('-f', type=str, default=None)
    parser.add_argument('--ban', type=str, default='ban_list.txt')
    parser.add_argument("-p",
                        nargs='*',
                        type=int,
                        default=None,
                        help="patterns to match")

    return parser.parse_args()


def get_ban_node(filename):
    if filename is None:
        return []

    node_names = []  # Initialize an empty list to store node names
    with open(filename, 'r') as f:  # Open the file for reading
        for line in f:  # Iterate over each line
            # Strip leading/trailing whitespace from the line first
            cleaned_line = line.strip()

            # Find the position of "Node: "
            node_keyword = "Node: "
            start_index = cleaned_line.find(node_keyword)

            # Check if the keyword was found
            if start_index != -1:
                # Calculate where the node name starts (after "Node: ")
                name_start = start_index + len(node_keyword)

                # Extract the rest of the string from where the name starts
                remaining_string = cleaned_line[name_start:]

                # Split the remaining string by space to isolate the node name
                parts = remaining_string.split()

                # Check if parts is not empty (to avoid index errors)
                if parts:
                    node_name = parts[
                        0]  # The first part should be the node name
                    node_names.append(node_name)  # Append to the list
    print(f'*' * 100)
    print(f'ban list: ', node_names)
    return node_names


def get_empty_nodes(args):
    nodes = get_all_nodes(args)
    pod_counts = get_pods_per_node(args)
    empty_nodes = find_empty_nodes(nodes, pod_counts)

    assert len(empty_nodes) > 0, "no empty nodes found"

    print(f"Found {len(empty_nodes)} empty nodes:")

    # group by region
    empty_nodes_by_region = {}
    for node in empty_nodes:
        tor = node['labels'].get('region', 'unknown')
        if tor not in empty_nodes_by_region:
            empty_nodes_by_region[tor] = []
        empty_nodes_by_region[tor].append(node)

    # sort by TOR
    empty_nodes_by_region = sorted(empty_nodes_by_region.items(),
                                   key=lambda x: x[0])
    for tor, nodes in empty_nodes_by_region:
        print(f"TOR: {tor}, {len(nodes)} empty nodes:")
        for idx, node in enumerate(nodes):
            gpu_type = node['labels'].get('nvidia.com/gpu.product', 'unknown')
            total_gpu = node['labels'].get('nvidia.com/gpu.count', 'unknown')
            allocatable_gpu = node['allocatable_gpus']
            print(
                f"{idx}. Node: {node['name']} GPU Type: {gpu_type}. GPUs: {allocatable_gpu}/{total_gpu}."
            )
    return empty_nodes_by_region


def build_yaml(
    yaml_file: str,
    avail_nodes,
):
    """
    Args:
        yaml_file (str): Path to the input YAML file (should base on volcano)
    """
    nnode = len(avail_nodes)
    assert nnode > 1, f'{avail_nodes=}'
    with open(yaml_file, 'r') as f:
        doc = yaml.safe_load(f)

    # Set minAvailable to n
    try:
        doc['spec']['minAvailable'] = nnode
        print(f"Set minAvailable to {nnode}")
    except KeyError as e:
        raise RuntimeError(f"Could not set minAvailable: {e}")

    # set number
    try:
        tasks = doc['spec']['tasks']
        for task in tasks:
            # NOTE: assume 1 master task, n-1 worker task
            if task['name'] == 'master':
                task['replicas'] = 1
            elif task['name'] == 'worker':
                task['replicas'] = nnode - 1
            else:
                raise RuntimeError(f"Unknown task name: {task['name']}")
    except (KeyError, IndexError, RuntimeError) as e:
        print(f"Error: Could not locate nodeAffinity section in YAML: {e}")
        raise e

    # set affinity
    try:
        for task in doc['spec']['tasks']:
            node_affinity = task['template']['spec']['affinity'][
                'nodeAffinity']
            selector_terms = node_affinity[
                'requiredDuringSchedulingIgnoredDuringExecution'][
                    'nodeSelectorTerms']

            for term in selector_terms:
                for i, expr in enumerate(term.get('matchExpressions', [])):
                    if expr.get('key') == 'kubernetes.io/hostname':
                        # Replace the values with our available nodes
                        term['matchExpressions'][i]['values'] = avail_nodes
    except (KeyError, IndexError) as e:
        print(f"Error: Could not locate nodeAffinity section in YAML: {e}")
        raise e

    name = yaml_file.split('.')[0]
    name += '_output.yaml'
    with open(name, 'w') as f:
        #yaml.dump_all(documents, f, default_flow_style=False)
        yaml.dump(doc, f)
    return name


def resource_scheduling(
        empty_nodes_by_region,
        n,
        ban_nodes,
        patterns: list[int],  # each int specify node number under a Tor
):
    print('*' * 100)
    print(f'[resource scheduling]: ')
    assert n is not None, "n is None"
    ns = []
    for tor, nodes in empty_nodes_by_region:
        tmp = []
        for idx, node in enumerate(nodes):
            gpu_type = node['labels'].get('nvidia.com/gpu.product', 'unknown')
            total_gpu = node['labels'].get('nvidia.com/gpu.count', 'unknown')
            allocatable_gpu = node['allocatable_gpus']
            if gpu_type == 'unknown':
                continue
            if total_gpu == 'unknown':
                continue
            if int(allocatable_gpu) != 8:
                continue
            if node['name'] in ban_nodes:
                continue

            tmp.append(node['name'])
            if len(tmp) >= n:
                break
        if len(tmp) >= n:
            print(f'Allocated: {tor=} {tmp=}')
            ns = tmp
            break
    print(f'nodes: {ns=}')
    return ns


def main():
    args = parse_arguments()

    # get empty nodes
    empty_nodes_by_region = get_empty_nodes(args)

    # get ban nodes
    ban_nodes = get_ban_node(args.ban)

    # schedule strategy
    nodes = resource_scheduling(
        empty_nodes_by_region,
        args.n,
        ban_nodes,
        args.p,
    )
    output_file = build_yaml(args.f, nodes)
    sleep(1)

    # exec k
    try:
        print(f'*' * 100)
        print(f"Applying to Kubernetes cluster...")
        result = subprocess.run(['kubectl', 'apply', '-f', output_file],
                                check=True,
                                capture_output=True,
                                text=True)
        print(f"Successfully applied! Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error applying to cluster: {e}")
        print(f"Error details: {e.stderr}")
        raise e


if __name__ == "__main__":
    main()
