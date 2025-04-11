import sys
sys.path.append('.')

import yaml
import subprocess
import argparse
import json
import sys
from typing import List, Dict, Any

from get_idle_nodes import get_all_nodes, get_pods_per_node, find_empty_nodes, run_kubectl_command

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Check for empty nodes in a Kubernetes cluster.')
    parser.add_argument('--context', type=str, default=None, help='Kubernetes context to use')
    parser.add_argument('--kubeconfig', type=str, default=None, help='Path to kubeconfig file')
    parser.add_argument('--output', type=str, choices=['json', 'text'], default='text',
                        help='Output format (json or text)')

    #kubectl get nodes --selector=team=rlib
    parser.add_argument('--selector', type=str, default=None, help='Label selector for nodes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    # 
    parser.add_argument('-n', type=int, default=None, help='num of node')
    parser.add_argument('-f', type=str, default=None)

    return parser.parse_args()

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
    empty_nodes_by_region = sorted(empty_nodes_by_region.items(), key=lambda x: x[0])
    for tor, nodes in empty_nodes_by_region:
        print(f"TOR: {tor}, {len(nodes)} empty nodes:")
        for idx, node in enumerate(nodes):
            gpu_type = node['labels'].get('nvidia.com/gpu.product', 'unknown')
            total_gpu = node['labels'].get('nvidia.com/gpu.count', 'unknown')
            allocatable_gpu = node['allocatable_gpus']
            print(f"{idx}. Node: {node['name']} GPU Type: {gpu_type}. GPUs: {allocatable_gpu}/{total_gpu}.")
    return empty_nodes_by_region


def build_yaml(yaml_file, avail_nodes):
    """
    Args:
        yaml_file (str): Path to the input YAML file (should base on volcano)
        output_file (str): Path to save the modified YAML
        avail_nodes (list): List of available node names
    """
    with open(yaml_file, 'r') as f:
        doc = yaml.safe_load(f)
    
    if doc.get('kind') == 'Job':
        try:
            node_affinity = doc['spec']['tasks'][0]['template']['spec']['affinity']['nodeAffinity']
            selector_terms = node_affinity['requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms']
            
            # Find the hostname match expression
            for term in selector_terms:
                for i, expr in enumerate(term.get('matchExpressions', [])):
                    if expr.get('key') == 'kubernetes.io/hostname':
                        # Replace the values with our available nodes
                        term['matchExpressions'][i]['values'] = avail_nodes
                        break
        except (KeyError, IndexError) as e:
            print(f"Error: Could not locate nodeAffinity section in YAML: {e}")
            raise e
    
    name = yaml_file.split('.')[0]
    name += '_output.yaml'
    with open(name, 'w') as f:
        #yaml.dump_all(documents, f, default_flow_style=False)
        yaml.dump(doc, f)
    return name

def resource_scheduling(empty_nodes_by_region, n):
    print('*'*100)
    print(f'resource_scheduling: ')
    assert n is not None, "n is None"
    ns = []
    for tor, nodes in empty_nodes_by_region:
        tmp = []
        for idx, node in enumerate(nodes):
            gpu_type = node['labels'].get('nvidia.com/gpu.product', 'unknown')
            total_gpu = node['labels'].get('nvidia.com/gpu.count', 'unknown')
            allocatable_gpu = node['allocatable_gpus']
            print(f"{idx}. Node: {node['name']} GPU Type: {gpu_type}. GPUs: {allocatable_gpu}/{total_gpu}.")

            if gpu_type == 'unknown':
                continue
            if total_gpu == 'unknown':
                continue
            if int(allocatable_gpu) != 8:
                continue

            tmp.append(node['name'])
            if len(tmp) >= n:
                break
        if len(tmp) >= n:
            print(f'resource_scheduling: {tor=} {tmp=}')
            ns = tmp
            break
    print(f'nodes: {ns=}')
    return ns


def main():
    args = parse_arguments()

    # get empty nodes
    empty_nodes_by_region = get_empty_nodes(args)

    # schedule strategy
    nodes = resource_scheduling(empty_nodes_by_region, args.n)
    output_file = build_yaml(args.f, nodes)

    # exec k
    try:
        print(f'*'*100)
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