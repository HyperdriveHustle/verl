import subprocess
import argparse
import json
import sys
from typing import List, Dict, Any
from get_idle_nodes import get_all_nodes


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
    return parser.parse_args()


def main():
    args = parse_arguments()
    nodes = get_all_nodes(args)

    labels2values = {}
    for node in nodes:
        for k, v in node['metadata']['labels'].items():
            if k not in labels2values:
                labels2values[k] = set()
            labels2values[k].add(v)

    for k, v in labels2values.items():
        print(f"{k}: {list(v)}")


if __name__ == "__main__":
    main()
