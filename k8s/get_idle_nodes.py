import subprocess
import argparse
import json
import sys
import yaml
from typing import List, Dict, Any


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


def run_kubectl_command(command: List[str], verbose: bool = False) -> str:
    """Execute a kubectl command and return the output."""
    if verbose:
        print(f"Executing command: kubectl {' '.join(command)}")

    try:
        result = subprocess.run(["kubectl"] + command,
                                capture_output=True,
                                text=True,
                                check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing kubectl command: {e}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def get_all_nodes(args) -> List[Dict[str, Any]]:
    """Get all nodes in the cluster."""
    cmd = ["get", "nodes", "-o", "json"]

    if args.context:
        cmd.extend(["--context", args.context])

    if args.kubeconfig:
        cmd.extend(["--kubeconfig", args.kubeconfig])

    # if args.selector:
    #     cmd.extend(["--selector", args.selector])

    output = run_kubectl_command(cmd, args.verbose)
    nodes_json = json.loads(output)
    return nodes_json["items"]


def get_pods_per_node(args) -> Dict[str, int]:
    """Get the number of pods running on each node."""
    cmd = ["get", "pods", "--all-namespaces", "-o", "json"]
    #cmd = ["get", "pods", "-o", "json"]

    if args.context:
        cmd.extend(["--context", args.context])

    if args.kubeconfig:
        cmd.extend(["--kubeconfig", args.kubeconfig])

    output = run_kubectl_command(cmd, args.verbose)
    pods_json = json.loads(output)

    # Count pods per node
    pod_counts = {}
    for pod in pods_json["items"]:
        pod_name = pod['metadata']['name']

        # skip non-gpu pod
        need_gpu = False
        for container in pod["spec"]["containers"]:
            request_gpu = container.get("resources",
                                        {}).get("requests",
                                                {}).get("nvidia.com/gpu", '0')
            if int(request_gpu) > 0:
                need_gpu = True
                break
        if not need_gpu:
            continue

        node_name = pod.get("spec", {}).get("nodeName")
        if node_name:
            pod_counts[node_name] = pod_counts.get(node_name, 0) + 1
    return pod_counts


def find_empty_nodes(
    nodes: List[Dict[str, Any]],
    pod_counts: Dict[str, int],
    label_selector_file: str,
) -> List[Dict[str, Any]]:

    # get selectors
    label_selector = {}
    if label_selector_file is not None:
        with open(label_selector_file, 'r') as f:
            data = yaml.safe_load(f)
        for k, v in data.items():
            assert not isinstance(
                v, dict), f"only support one level of labels: {v}"
            label_selector[k] = v

    print(f'=' * 80)
    print(f'label selector: {len(label_selector)}')
    for k, v in label_selector.items():
        print(f'{k} -> {v}')
    print(f'=' * 80)

    empty_nodes = []
    for node in nodes:
        ## skip not gpu=true
        if 'gpu' not in node['metadata']['labels'] or \
                node['metadata']['labels']['gpu'] != 'true':
            continue

        ## skip if node has pod
        node_name = node["metadata"]["name"]
        if node_name in pod_counts:
            continue

        ## skip non NV GPU
        if 'nvidia.com/gpu' not in node['status']['allocatable']:
            continue

        ## filter labels (ALL)
        # ok = True
        # for k, v in label_selector.items():
        #     if k not in node_labels or node_labels[k] != v:
        #         ok = False
        #         break

        ## filter labels (Any)
        node_labels = node["metadata"]["labels"]
        if len(label_selector) > 0:
            ok = False
        else:
            ok = True
        for k, v in label_selector.items():
            if k in node_labels and node_labels[k] == v:
                ok = True
                break
        if not ok:
            continue

        # collect
        node_conditions = node["status"]["conditions"]
        ready_condition = next(
            (cond for cond in node_conditions if cond["type"] == "Ready"),
            None)

        allocatable_gpus = node['status']['allocatable']['nvidia.com/gpu']

        node_info = {
            "name": node_name,
            "ready": ready_condition and ready_condition["status"] == "True",
            "capacity": node["status"]["capacity"],
            "labels": node["metadata"]["labels"],
            'allocatable_gpus': allocatable_gpus,
        }

        empty_nodes.append(node_info)

    return empty_nodes


def get_empty_nodes(args):
    nodes = get_all_nodes(args)
    pod_counts = get_pods_per_node(args)
    empty_nodes = find_empty_nodes(nodes, pod_counts, args.selector)

    assert len(empty_nodes) > 0, "no empty nodes found"
    print(f"Found {len(empty_nodes)} empty nodes:")

    # group by region
    empty_nodes_by_region = {}
    for node in empty_nodes:

        # XXX I am told unknown == a, but not sure exactly
        #tor = node['labels'].get('region', 'unknown')
        tor = node['labels'].get('region', 'a')

        if tor not in empty_nodes_by_region:
            empty_nodes_by_region[tor] = []
        empty_nodes_by_region[tor].append(node)

    # sort by TOR
    kv = sorted(
        empty_nodes_by_region.items(),
        key=lambda x: x[0],
    )
    for tor, nodes in kv:
        print(f"TOR: {tor}, {len(nodes)} empty nodes:")
        for idx, node in enumerate(nodes):
            gpu_type = node['labels'].get('nvidia.com/gpu.product', 'unknown')
            total_gpu = node['labels'].get('nvidia.com/gpu.count', 'unknown')
            allocatable_gpu = node['allocatable_gpus']
            print(
                f"{idx}. Node: {node['name']} GPU Type: {gpu_type}. GPUs: {allocatable_gpu}/{total_gpu}."
            )
    return empty_nodes_by_region


def main():
    """Main function to check for empty nodes in the cluster."""
    args = parse_arguments()

    if args.verbose:
        print("Gathering information about nodes and pods...")

    get_empty_nodes(args)


if __name__ == "__main__":
    main()
