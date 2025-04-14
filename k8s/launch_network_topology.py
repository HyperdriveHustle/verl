import sys

sys.path.append('.')

import yaml
import subprocess
import argparse
import json
import re
import sys

from typing import List, Dict, Any
from time import sleep

from get_idle_nodes import get_all_nodes, get_pods_per_node, find_empty_nodes, run_kubectl_command, get_empty_nodes


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
    #parser.add_argument('-n', type=int, default=None, help='num of node')
    parser.add_argument(
        '-f',
        type=str,
        default=None,
        help='yaml template to dynamically change affinity and node num')

    parser.add_argument('--ban', type=str, default=None)
    parser.add_argument('--only', type=str, default=None)

    parser.add_argument("-p",
                        nargs='*',
                        type=int,
                        default=None,
                        help="patterns to match")

    return parser.parse_args()


def get_predefined_nodes(filename):
    prefix = 'zjzx1h'
    node_names = []
    # Compile a regular expression pattern to find words starting with the prefix.
    # \b        - Matches a word boundary to ensure we match whole words.
    # (         - Starts a capturing group (to extract the full name).
    # {prefix}  - Matches the literal prefix provided.
    # [\w-]* - Matches zero or more word characters (letters, numbers, _)
    #             or hyphens (-). This assumes node names consist of these.
    # )         - Ends the capturing group.
    # \b        - Matches a word boundary at the end.
    # re.escape ensures the prefix is treated literally even if it contains regex special characters.
    pattern = re.compile(r"\b(" + re.escape(prefix) + r"[\w-]*)\b")

    try:
        with open(filename, 'r') as f:  # Open the file for reading
            for line in f:
                # Find all non-overlapping matches of the pattern in the current line
                matches = pattern.findall(line)
                # Add all found matches (node names) from this line to our list
                node_names.extend(matches)

    except FileNotFoundError as e:
        print(f"Error: File '{filename}' not found.")
        raise e
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise e

    # Return a list of unique node names found (preserves order)
    # Using dict.fromkeys is an efficient way to remove duplicates
    print(f'*' * 100)
    print(f'ban list: ', node_names)
    return list(dict.fromkeys(node_names))


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


def filter_node(
    node,
    predefined_strategy: str,
    predefined_nodes: list[str],
):
    gpu_type = node['labels'].get('nvidia.com/gpu.product', 'unknown')
    total_gpu = node['labels'].get('nvidia.com/gpu.count', 'unknown')
    allocatable_gpu = node['allocatable_gpus']
    if gpu_type == 'unknown':
        return False
    if total_gpu == 'unknown':
        return False
    if int(allocatable_gpu) != 8:
        return False

    if predefined_strategy == 'ban':
        if node['name'] in predefined_nodes:
            return False
    elif predefined_strategy == 'only':
        if node['name'] not in predefined_nodes:
            return False
    else:
        raise RuntimeError(f'Unknown strategy: {predefined_strategy}')
    return True


def resource_scheduling(
        empty_nodes_by_region: dict,
        predefined_strategy: str,
        predefined_nodes: list[str],
        patterns: list[int],  # each int specify node number under a Tor
):
    assert patterns is not None, 'patterns is None'

    tor2nodes = sorted(
        empty_nodes_by_region.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )
    patterns = list(sorted(patterns, reverse=True))
    tor_len = [(x[0], len(x[1])) for x in tor2nodes]
    print('*' * 100)
    print(f'[resource scheduling]: {patterns=} {tor_len=}')
    assert sum([i[1] for i in tor_len
                ]) >= sum(patterns), f'{sum(tor_len)=} {sum(patterns)=}'

    # match patterns to tor-topology
    pat_idx = 0
    tor_idx = 0
    ns = []
    while pat_idx < len(patterns) and tor_idx < len(tor2nodes):

        pat = patterns[pat_idx]
        nodes = tor2nodes[tor_idx][1]
        tor = tor2nodes[tor_idx][0]
        if tor == 'unknown':
            # unknown TOR
            tor_idx += 1
            continue

        avail_nodes = []
        for idx, node in enumerate(nodes):
            ok = filter_node(node, predefined_strategy, predefined_nodes)
            if ok:
                avail_nodes.append(node)

        if len(avail_nodes) >= pat:
            selected_nodes = [node['name'] for node in avail_nodes[:pat]]
            print(f'Allocated: tor: {tor2nodes[tor_idx][0]} {selected_nodes=}')
            ns.extend(selected_nodes)
            pat_idx += 1
        tor_idx += 1

    # not enough
    if pat_idx < len(patterns):
        raise RuntimeError(
            f'Not enough nodes to match patterns: {patterns=} {tor_len=}')
    print(f'=' * 50)
    print(f'nodes: {ns=}')
    return ns


def main():
    args = parse_arguments()

    # get empty nodes
    empty_nodes_by_region = get_empty_nodes(args)

    # get ban nodes or only (mutual exclusive)
    if args.ban is None and args.only is None:
        # all allowed
        select = 'ban'
        select_nodes = []
    elif args.ban is not None:
        assert args.only is None
        select = 'ban'
        select_nodes = get_predefined_nodes(args.ban)
    elif args.only is not None:
        assert args.ban is None
        select = 'only'
        select_nodes = get_predefined_nodes(args.only)

    # schedule strategy
    nodes = resource_scheduling(
        empty_nodes_by_region,
        select,
        select_nodes,
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
