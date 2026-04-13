#!/usr/bin/env -S uv run --quiet
# /// script
# dependencies = ["kubernetes"]
# ///
"""NCCL health check script for GPU clusters.

Runs all_reduce_perf across all nodes, then tests node groups
(pairs, groups of 4, groups of 8, etc.) to identify bad nodes.

Pair tests use a round-robin tournament schedule for maximum parallelism.
"""

import os
import shlex
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))
HOSTFILE = os.environ.get("HOSTFILE", "/etc/mpi/hostfile")
LOG_DIR = Path(os.environ.get("LOG_DIR", "/tmp/nccl-healthcheck"))
GROUP_SIZES = os.environ.get("GROUP_SIZES", "2")
TEST_MODE = os.environ.get("TEST_MODE", "all")
BW_THRESHOLD_2NODE = float(os.environ.get("BW_THRESHOLD_2NODE", "450"))
BW_THRESHOLD_DEFAULT = float(os.environ.get("BW_THRESHOLD_DEFAULT", "350"))
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "0"))  # 0 = unlimited (N/2)
SAME_LOCAL_BLOCK_ONLY = os.environ.get("SAME_LOCAL_BLOCK_ONLY", "false").lower() == "true"
ORDERED_HOSTFILE = os.environ.get("ORDERED_HOSTFILE", "false").lower() == "true"
BIND_TO = os.environ.get("BIND_TO", "numa")
MAP_BY = os.environ.get("MAP_BY", "")

# Build mpirun base command -- shape-specific args come from MPIRUN_EXTRA_ARGS env var
MPIRUN_BASE = [
    "mpirun", "--allow-run-as-root",
    "-mca", "coll", "^hcoll",
    "-mca", "plm_rsh_args", "-p 2222 -o StrictHostKeyChecking=no -o LogLevel=ERROR",
    "--bind-to", BIND_TO,
]
if MAP_BY:
    MPIRUN_BASE += ["--map-by", MAP_BY]
MPIRUN_EXTRA = os.environ.get("MPIRUN_EXTRA_ARGS", "")
if MPIRUN_EXTRA:
    MPIRUN_BASE += shlex.split(MPIRUN_EXTRA)

# NCCL test binary and args -- configurable per shape via NCCL_TEST_ARGS env var
NCCL_TEST_ARGS = os.environ.get("NCCL_TEST_ARGS", "-b 8 -f 2 -g 1 -e 4G -c 1")
NCCL_TEST = ["/workspace/nccl-tests/build/all_reduce_perf"] + shlex.split(NCCL_TEST_ARGS)


# Maps full hostnames to short names, IPs, and local block (rail group) IDs
HOST_TO_IP = {}
HOST_SHORT = {}
HOST_LOCAL_BLOCK = {}


def resolve_host(hostname):
    """Resolve a hostname to its IP address."""
    import socket
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return hostname


def short_name(hostname):
    """Get short display name (IP or fallback to hostname)."""
    if hostname in HOST_SHORT:
        return HOST_SHORT[hostname]
    return hostname


def load_node_local_blocks():
    """Load local block (rail group) labels for all nodes from the Kubernetes API."""
    try:
        from kubernetes import client, config
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        nodes = v1.list_node().items
        ip_to_block = {}
        for node in nodes:
            labels = node.metadata.labels or {}
            block = labels.get("oci.oraclecloud.com/rdma.local_block_id", "unknown")
            for addr in (node.status.addresses or []):
                if addr.type == "InternalIP":
                    ip_to_block[addr.address] = block
        return ip_to_block
    except Exception as e:
        print(f"{YELLOW}WARN: Could not read node labels from Kubernetes API: {e}{RESET}", flush=True)
        return {}


def get_rails_info(test_hosts):
    """Get local block (rail group) distribution for a group of hosts."""
    blocks = defaultdict(list)
    for h in test_hosts:
        block = HOST_LOCAL_BLOCK.get(h, "unknown")
        blocks[block].append(h)
    block_ids = sorted(blocks.keys(), key=lambda b: -len(blocks[b]))
    if len(blocks) == 1:
        return f"1 block: {block_ids[0]}, {len(test_hosts)} nodes"
    return f"{len(blocks)} blocks: {', '.join(f'{b} {len(blocks[b])} nodes' for b in block_ids)}"


def read_hosts():
    with open(HOSTFILE) as f:
        hosts = [line.split()[0] for line in f if line.strip()]
    # Build hostname -> IP mapping
    for h in hosts:
        ip = resolve_host(h)
        HOST_TO_IP[h] = ip
        HOST_SHORT[h] = ip
    # Load local block (rail group) labels from Kubernetes API
    ip_to_block = load_node_local_blocks()
    for h in hosts:
        ip = HOST_TO_IP.get(h, h)
        HOST_LOCAL_BLOCK[h] = ip_to_block.get(ip, "unknown")
    # Order hosts by local block (largest blocks first) so sequential groups
    # stay within the same block as much as possible
    if ORDERED_HOSTFILE:
        block_sizes = defaultdict(int)
        for h in hosts:
            block_sizes[HOST_LOCAL_BLOCK.get(h, "unknown")] += 1
        hosts.sort(key=lambda h: (-block_sizes[HOST_LOCAL_BLOCK.get(h, "unknown")],
                                  HOST_LOCAL_BLOCK.get(h, "unknown")))
    return hosts


def wait_for_workers(hosts):
    import time
    while True:
        all_ready = True
        for host in hosts:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                 "-o", "LogLevel=ERROR", "-p", "2222", host, "exit"],
                capture_output=True)
            if r.returncode != 0:
                all_ready = False
                break
        if all_ready:
            return
        print("Waiting for workers to be ready...", flush=True)
        time.sleep(5)


def run_nccl_test(hosts, logfile=None):
    """Run NCCL test on the given hosts. Returns (returncode, output)."""
    np = len(hosts) * NUM_GPUS
    host_list = ",".join(f"{h}:{NUM_GPUS}" for h in hosts)
    cmd = MPIRUN_BASE + ["-np", str(np), "-npernode", str(NUM_GPUS),
                         "-H", host_list] + NCCL_TEST
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if logfile:
        logfile.write_text(output)
    return result.returncode, output


def find_inplace_busbw_col(output):
    """Find the column index for in-place busbw by parsing the NCCL header.
    Returns the 0-based index, or 11 as fallback.
    """
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if "out-of-place" in line and "in-place" in line:
            # Next line has column names: size count type redop root time algbw busbw #wrong time algbw busbw #wrong
            if i + 1 < len(lines):
                header = lines[i + 1].lstrip("#").split()
                # Find the second occurrence of "busbw" (in-place)
                busbw_indices = [j for j, col in enumerate(header) if col == "busbw"]
                if len(busbw_indices) >= 2:
                    return busbw_indices[1]
            break
    return 11  # fallback


def extract_4g_busbw(output):
    """Extract in-place busbw at 4GB message size."""
    col = find_inplace_busbw_col(output)
    for line in output.splitlines():
        if "4294967296" in line and "nThread" not in line:
            parts = line.split()
            if len(parts) > col:
                try:
                    return float(parts[col])
                except ValueError:
                    pass
    return None


def generate_pair_rounds(n):
    """Generate round-robin tournament schedule (circle method).
    Returns list of rounds, each round is a list of (a, b) pairs.
    """
    effective_n = n if n % 2 == 0 else n + 1
    rotate = list(range(1, effective_n))
    rounds = []
    for _ in range(effective_n - 1):
        pairs = []
        b = rotate[-1]
        if b < n:
            pairs.append((0, b))
        for j in range((effective_n - 2) // 2):
            a, b = rotate[j], rotate[effective_n - 3 - j]
            if a < n and b < n:
                pairs.append((a, b))
        rounds.append(pairs)
        rotate = [rotate[-1]] + rotate[:-1]
    return rounds


def _run_pair(args):
    """Worker function for parallel pair execution."""
    host_a, host_b, logfile = args
    rc, output = run_nccl_test([host_a, host_b], logfile)
    bw = extract_4g_busbw(output)
    return host_a, host_b, rc, bw


# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def hosts_display(test_hosts):
    """Format host list for display using short names."""
    return ", ".join(short_name(h) for h in test_hosts)


def check_result(label, rc, bw, threshold, test_hosts):
    """Check a test result and return (status, label, bw, hosts)."""
    if rc != 0:
        print(f"{RED}FAIL: {label} (crashed){RESET}", flush=True)
        return ("FAIL", label, None, test_hosts)
    elif bw is None:
        print(f"{YELLOW}WARN: {label} no bandwidth data{RESET}", flush=True)
        return ("WARN", label, None, test_hosts)
    elif bw < threshold:
        print(f"{YELLOW}SUSPECT: {label} busbw={bw:.2f} GB/s (< {threshold}){RESET}", flush=True)
        return ("SUSPECT", label, bw, test_hosts)
    else:
        print(f"{GREEN}PASS: {label} busbw={bw:.2f} GB/s{RESET}", flush=True)
        return ("PASS", label, bw, test_hosts)


def run_all_pairs(hosts):
    """Run all C(N,2) pairs using round-robin tournament with parallel rounds."""
    n = len(hosts)
    rounds = generate_pair_rounds(n)
    total_pairs = n * (n - 1) // 2
    max_per_round = MAX_PARALLEL if MAX_PARALLEL > 0 else n // 2

    print(f"\n{'='*50}")
    print(f"  ALL-PAIRS TEST")
    print(f"  {total_pairs} pairs in {len(rounds)} rounds")
    print(f"  Max parallel: {max_per_round}")
    print(f"  Threshold: {BW_THRESHOLD_2NODE} GB/s")
    print(f"{'='*50}", flush=True)

    results = []
    for round_num, pairs in enumerate(rounds):
        print(f"\n--- Round {round_num + 1}/{len(rounds)} ---", flush=True)
        all_tasks = []
        for a, b in pairs:
            host_a, host_b = hosts[a], hosts[b]
            logfile = LOG_DIR / f"pair_{host_a}_{host_b}.log"
            all_tasks.append((host_a, host_b, logfile))

        # Process tasks in batches of max_per_round
        for batch_start in range(0, len(all_tasks), max_per_round):
            batch = all_tasks[batch_start:batch_start + max_per_round]
            with ProcessPoolExecutor(max_workers=len(batch)) as pool:
                futures = {pool.submit(_run_pair, t): t for t in batch}
                for future in as_completed(futures):
                    host_a, host_b, rc, bw = future.result()
                    label = f"{short_name(host_a)}, {short_name(host_b)}"
                    r = check_result(label, rc, bw, BW_THRESHOLD_2NODE, [host_a, host_b])
                    results.append(r)
    return results


def run_sequential_groups(hosts, group_size):
    """Run non-overlapping groups sequentially with bandwidth checking."""
    num_groups = len(hosts) // group_size
    remainder = len(hosts) % group_size

    print(f"\n{'='*40}")
    mode = "SEQUENTIAL" if TEST_MODE == "sequential" else "NON-OVERLAPPING"
    extra = f" + 1 remainder of {remainder}" if remainder > 0 else ""
    print(f"{mode} GROUP-OF-{group_size} TESTS ({num_groups} groups{extra})")
    print(f"{'='*40}", flush=True)

    results = []
    groups = []
    for i in range(0, len(hosts) - group_size + 1, group_size):
        group_num = i // group_size + 1
        groups.append((f"group-of-{group_size} #{group_num}", hosts[i:i + group_size]))
    if remainder > 0:
        groups.append((f"group-of-{group_size}-remainder", hosts[-remainder:]))

    for label, group_hosts in groups:
        # Pick threshold based on the actual size of this group (remainder may differ)
        threshold = BW_THRESHOLD_2NODE if len(group_hosts) == 2 else BW_THRESHOLD_DEFAULT
        rails = get_rails_info(group_hosts)
        print(f"\n--- {label}: {hosts_display(group_hosts)} ({rails}) ---", flush=True)
        logfile = LOG_DIR / f"{label}.log"
        rc, output = run_nccl_test(group_hosts, logfile)
        bw = extract_4g_busbw(output)
        r = check_result(label, rc, bw, threshold, group_hosts)
        results.append(r)

    return results


def print_summary(hosts, all_results):
    passed_count = sum(1 for r in all_results if r[0] == "PASS")
    total_count = len(all_results)

    W = 48
    L = 26
    V = W - L - 1
    result_str = f"{passed_count}/{total_count} passed"
    print(f"\n+{'-'*W}+")
    print(f"|{'SUMMARY':^{W}}|")
    print(f"+{'-'*W}+")
    print(f"| {'Hosts:':<{L}}{len(hosts):<{V}}|")
    print(f"| {'Test mode:':<{L}}{TEST_MODE:<{V}}|")
    print(f"| {'Group sizes:':<{L}}{GROUP_SIZES:<{V}}|")
    print(f"| {'BW threshold (2 nodes):':<{L}}{str(BW_THRESHOLD_2NODE) + ' GB/s':<{V}}|")
    print(f"| {'BW threshold (>2 nodes):':<{L}}{str(BW_THRESHOLD_DEFAULT) + ' GB/s':<{V}}|")
    print(f"| {'Results:':<{L}}{result_str:<{V}}|")
    print(f"+{'-'*W}+")

    failures = [r for r in all_results if r[0] != "PASS"]
    if not failures:
        print(f"{GREEN}ALL TESTS PASSED{RESET}")
        return True

    print(f"\n{RED}FAILED/SUSPECT TESTS ({len(failures)}):{RESET}")
    for status, label, bw, test_hosts in failures:
        bw_str = f" busbw={bw:.2f} GB/s" if bw else ""
        color = RED if status == "FAIL" else YELLOW
        # Collapse long host lists to a count to keep lines readable
        hosts_str = (f"{len(test_hosts)} nodes"
                     if len(test_hosts) > 8
                     else hosts_display(test_hosts))
        print(f"  {color}{status}: {label}{bw_str} [{hosts_str}]{RESET}")

    node_counts = defaultdict(int)
    for status, label, bw, test_hosts in failures:
        # Skip all-nodes (every node appears) and validation tests
        # (would count the known-good reference node unfairly)
        if label == "all-nodes" or label.startswith("validate-"):
            continue
        for h in test_hosts:
            node_counts[short_name(h)] += 1

    if node_counts:
        print(f"\n{YELLOW}SUSPECT/FAIL NODE FREQUENCY (most frequent first):{RESET}")
        for node, count in sorted(node_counts.items(), key=lambda x: -x[1]):
            print(f"  {YELLOW}{node}: appeared in {count} failed/suspect tests{RESET}")

    return False


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    hosts = read_hosts()

    W = 48
    L = 26
    V = W - L - 1
    print(f"\n+{'-'*W}+")
    print(f"|{'NCCL HEALTH CHECK':^{W}}|")
    print(f"+{'-'*W}+")
    print(f"| {'Hosts:':<{L}}{len(hosts):<{V}}|")
    print(f"| {'Test mode:':<{L}}{TEST_MODE:<{V}}|")
    print(f"| {'Group sizes:':<{L}}{GROUP_SIZES:<{V}}|")
    print(f"| {'BW threshold (2 nodes):':<{L}}{str(BW_THRESHOLD_2NODE) + ' GB/s':<{V}}|")
    print(f"| {'BW threshold (>2 nodes):':<{L}}{str(BW_THRESHOLD_DEFAULT) + ' GB/s':<{V}}|")
    print(f"| {'Max parallel:':<{L}}{str(MAX_PARALLEL) if MAX_PARALLEL > 0 else 'unlimited':<{V}}|")
    print(f"| {'Same block only:':<{L}}{str(SAME_LOCAL_BLOCK_ONLY).lower():<{V}}|")
    print(f"| {'Ordered hostfile:':<{L}}{str(ORDERED_HOSTFILE).lower():<{V}}|")
    print(f"+{'-'*W}+", flush=True)

    wait_for_workers(hosts)
    print(f"\n{GREEN}All {len(hosts)} workers are ready!{RESET}", flush=True)

    # Print local block (rail group) topology summary
    block_groups = defaultdict(list)
    for h in hosts:
        block_groups[HOST_LOCAL_BLOCK.get(h, "unknown")].append(h)
    print(f"\nLocal blocks (rail groups):", flush=True)
    for block_id, members in sorted(block_groups.items(), key=lambda x: -len(x[1])):
        if len(members) > 8:
            print(f"  {block_id}: {len(members)} nodes", flush=True)
        else:
            print(f"  {block_id}: {len(members)} nodes [{hosts_display(members)}]", flush=True)

    all_results = []

    if SAME_LOCAL_BLOCK_ONLY:
        # Run tests only within each local block
        for block_id, block_hosts in sorted(block_groups.items(), key=lambda x: -len(x[1])):
            if len(block_hosts) < 2:
                print(f"\n{YELLOW}Skipping local block (rail group) {block_id} (only {len(block_hosts)} node){RESET}", flush=True)
                continue

            print(f"\n{'='*50}")
            print(f"  LOCAL BLOCK (RAIL GROUP): {block_id} ({len(block_hosts)} nodes)")
            print(f"{'='*50}", flush=True)

            # All nodes in this local block
            if len(block_hosts) > 2:
                label = f"local-block-{block_id}"
                logfile = LOG_DIR / f"local_block_{block_id}.log"
                rc, output = run_nccl_test(block_hosts, logfile)
                bw = extract_4g_busbw(output)
                r = check_result(label, rc, bw, BW_THRESHOLD_DEFAULT, block_hosts)
                all_results.append(r)

            # Group tests within this local block
            for size_str in GROUP_SIZES.split(","):
                group_size = int(size_str.strip())
                if group_size > len(block_hosts):
                    continue
                if TEST_MODE == "all" and group_size == 2:
                    results = run_all_pairs(block_hosts)
                else:
                    results = run_sequential_groups(block_hosts, group_size)
                all_results.extend(results)
    else:
        # Phase 1: All nodes
        rails = get_rails_info(hosts)
        print(f"\n{'='*40}")
        print(f"ALL NODES TEST ({len(hosts)} hosts, {len(hosts) * NUM_GPUS} GPUs, {rails})")
        print(f"{'='*40}", flush=True)
        logfile = LOG_DIR / "all_nodes.log"
        rc, output = run_nccl_test(hosts, logfile)
        bw = extract_4g_busbw(output)
        r = check_result("all-nodes", rc, bw, BW_THRESHOLD_DEFAULT, hosts)
        all_results.append(r)

        # Phase 2: Group tests
        for size_str in GROUP_SIZES.split(","):
            group_size = int(size_str.strip())
            if TEST_MODE == "all" and group_size == 2:
                results = run_all_pairs(hosts)
            else:
                results = run_sequential_groups(hosts, group_size)
            all_results.extend(results)

    # Phase 3: Bisect suspect groups to isolate bad nodes
    suspect_groups = [
        (label, test_hosts) for status, label, bw, test_hosts in all_results
        if status in ("SUSPECT", "FAIL") and len(test_hosts) > 2
        and label != "all-nodes"
    ]

    # Pair failures are already at minimum size — both nodes are suspects,
    # carry them directly into cross-validation without bisecting.
    isolated_suspects = set()
    for status, label, bw, test_hosts in all_results:
        if (status in ("SUSPECT", "FAIL")
                and len(test_hosts) == 2
                and label != "all-nodes"):
            for h in test_hosts:
                isolated_suspects.add(h)

    if suspect_groups:
        print(f"\n{'='*50}")
        print(f"  BISECT PHASE")
        print(f"  Isolating bad nodes from {len(suspect_groups)} suspect group(s)")
        print(f"{'='*50}", flush=True)

        for orig_label, orig_hosts in suspect_groups:
            print(f"\n--- Bisecting {orig_label} ({len(orig_hosts)} nodes) ---", flush=True)
            queue = [orig_hosts[:]]
            round_num = 0

            while queue:
                next_queue = []
                round_num += 1
                print(f"\n  Bisect round {round_num}:", flush=True)

                for group in queue:
                    if len(group) <= 2:
                        # Test the pair directly
                        threshold = BW_THRESHOLD_2NODE if len(group) == 2 else BW_THRESHOLD_DEFAULT
                        label = f"bisect-{hosts_display(group)}"
                        logfile = LOG_DIR / f"bisect_{'_'.join(short_name(h) for h in group)}.log"
                        rc, output = run_nccl_test(group, logfile)
                        bw = extract_4g_busbw(output)
                        r = check_result(label, rc, bw, threshold, group)
                        all_results.append(r)
                        if r[0] in ("SUSPECT", "FAIL"):
                            for h in group:
                                isolated_suspects.add(h)
                    else:
                        # Split in half and test each
                        mid = len(group) // 2
                        halves = [group[:mid], group[mid:]]
                        for half in halves:
                            threshold = BW_THRESHOLD_2NODE if len(half) == 2 else BW_THRESHOLD_DEFAULT
                            label = f"bisect-{hosts_display(half)}"
                            logfile = LOG_DIR / f"bisect_{'_'.join(short_name(h) for h in half)}.log"
                            rc, output = run_nccl_test(half, logfile)
                            bw = extract_4g_busbw(output)
                            r = check_result(label, rc, bw, threshold, half)
                            all_results.append(r)
                            if r[0] in ("SUSPECT", "FAIL"):
                                if len(half) <= 2:
                                    for h in half:
                                        isolated_suspects.add(h)
                                else:
                                    next_queue.append(half)
                queue = next_queue

        if isolated_suspects:
            print(f"\n{YELLOW}Bisect isolated {len(isolated_suspects)} suspect node(s): "
                  f"{', '.join(short_name(h) for h in isolated_suspects)}{RESET}", flush=True)
        else:
            print(f"\n{GREEN}Bisect found no individual suspect nodes "
                  f"(issue may only appear at scale){RESET}", flush=True)

    # Phase 4: Cross-validate suspect nodes against known-good nodes
    if isolated_suspects:
        # Find nodes that only appeared in PASS results (never in any failed/suspect test)
        failed_nodes = set()
        for status, label, bw, test_hosts in all_results:
            if status != "PASS":
                for h in test_hosts:
                    failed_nodes.add(h)
        good_nodes = [h for h in hosts if h not in failed_nodes and h not in isolated_suspects]
        if not good_nodes:
            # Fallback: pick any node not in isolated_suspects
            good_nodes = [h for h in hosts if h not in isolated_suspects]
        if len(good_nodes) >= 1:
            good_node = good_nodes[0]

            print(f"\n{'='*50}")
            print(f"  CROSS-VALIDATION PHASE")
            print(f"  Testing {len(isolated_suspects)} suspect node(s) paired with known-good node {short_name(good_node)}")
            print(f"{'='*50}", flush=True)

            confirmed_bad = []
            confirmed_good = []

            for suspect in sorted(isolated_suspects, key=short_name):
                label = f"validate-{short_name(suspect)}"
                logfile = LOG_DIR / f"validate_{short_name(suspect)}.log"
                rc, output = run_nccl_test([suspect, good_node], logfile)
                bw = extract_4g_busbw(output)
                r = check_result(label, rc, bw, BW_THRESHOLD_2NODE, [suspect, good_node])
                all_results.append(r)
                if r[0] in ("SUSPECT", "FAIL"):
                    confirmed_bad.append(suspect)
                else:
                    confirmed_good.append(suspect)

            if confirmed_bad:
                print(f"\n{RED}CONFIRMED BAD NODE(S): "
                      f"{', '.join(short_name(h) for h in confirmed_bad)}{RESET}", flush=True)
            if confirmed_good:
                print(f"\n{GREEN}CLEARED NODE(S) (passed with known-good): "
                      f"{', '.join(short_name(h) for h in confirmed_good)}{RESET}", flush=True)

    print_summary(hosts, all_results)
    # Always exit 0 -- MPIJob controller restarts the launcher on non-zero
    # exit regardless of restartPolicy. Results are in the logs.

    # Keep the launcher pod alive after completion so `kubectl logs -f`
    # doesn't show "stream closed: EOF" and logs remain easy to retrieve.
    # Set KEEP_ALIVE=false to exit immediately instead.
    if os.environ.get("KEEP_ALIVE", "true").lower() != "false":
        import time
        print(f"\n{GREEN}Health check complete. Sleeping to keep pod alive. "
              f"Delete with: kubectl delete mpijob nccl-healthcheck{RESET}",
              flush=True)
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
