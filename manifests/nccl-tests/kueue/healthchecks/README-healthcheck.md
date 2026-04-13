# NCCL Health Check for BM.GPU.H100.8

Runs NCCL `all_reduce_perf` across GPU nodes to identify bad nodes or degraded RDMA performance.

## How it works

1. Deploys an MPIJob with all GPU nodes as workers
2. Reads node topology (local blocks / rail groups) from the Kubernetes API
3. Runs an all-nodes NCCL test
4. Runs group tests (pairs, groups of 4, groups of 8, etc.) depending on configuration
5. Reports per-test bandwidth and flags results below threshold as SUSPECT
6. Bisects suspect groups to isolate individual bad nodes
7. Cross-validates suspect nodes by pairing them with a known-good node
8. Prints a summary with suspect node frequency and confirmed bad nodes

## Quick start

```bash
# Apply shared resources + MPIJob
kubectl apply -f BM.GPU.H100.8-healthcheck.yaml

# Follow logs
kubectl logs -f $(kubectl get pods -l nccl-test-replica=mpi-launcher -o jsonpath='{.items[0].metadata.name}')

# Clean up
kubectl delete mpijob nccl-healthcheck
kubectl delete configmap nccl-healthcheck-script
kubectl delete clusterrolebinding nccl-healthcheck-node-reader
kubectl delete clusterrole nccl-healthcheck-node-reader
kubectl delete sa nccl-healthcheck
```

## Configuration

All settings are controlled via environment variables on the launcher container.

| Variable | Default | Description |
|---|---|---|
| `GROUP_SIZES` | `"2"` | Comma-separated group sizes to test (e.g. `"2"`, `"2,4,8"`) |
| `TEST_MODE` | `"all"` | `"all"` = all pair combinations with parallel rounds; `"sequential"` = non-overlapping groups, each node tested once |
| `BW_THRESHOLD_2NODE` | `450` | In-place busbw threshold (GB/s) at 4GB for 2-node tests |
| `BW_THRESHOLD_DEFAULT` | `350` | In-place busbw threshold (GB/s) at 4GB for all other group sizes |
| `MAX_PARALLEL` | `0` | Max concurrent tests per round. `0` = unlimited (N/2 for pairs) |
| `NUM_GPUS` | `8` | GPUs per node |
| `SAME_LOCAL_BLOCK_ONLY` | `"false"` | Only test nodes within the same local block (rail group) |
| `ORDERED_HOSTFILE` | `"false"` | Sort hosts by local block (largest first) before testing |

## Test phases

### Phase 1: All-nodes test

Runs NCCL across every worker node. Establishes baseline performance.

### Phase 2: Group tests

Depending on `TEST_MODE`:

- **`all`** (pairs only): Tests every possible pair using a round-robin tournament schedule. N/2 pairs run in parallel per round.
- **`sequential`**: Tests non-overlapping groups in order. Each node tested once per group size.

### Phase 3: Bisect

For each suspect group with more than 2 nodes, recursively splits in half and tests each half until reaching pairs. Collects individual suspect nodes.

### Phase 4: Cross-validation

Each isolated suspect node is paired with a known-good reference node (one that only appeared in PASS results). If the pair still fails, the node is confirmed bad. If it passes, the node is cleared.

## Time estimates

Each individual NCCL test takes approximately 60-70 seconds. Total run time depends on `MAX_PARALLEL` and the number of tests:

| N nodes | Mode | Group size | Tests | MAX_PARALLEL=0 (N/2) | MAX_PARALLEL=1 |
|---|---|---|---|---|---|
| 15 | all | 2 | 106 (all-nodes + 105 pairs) | ~15 min | ~2 hours |
| 26 | all | 2 | 326 (all-nodes + 325 pairs) | ~30 min | ~6 hours |
| 26 | sequential | 2,4,8 | ~24 | ~25 min | ~25 min |

Setting `MAX_PARALLEL` too high can cause RDMA fabric contention and false positives on some shapes (e.g. BM.GPU.B4.8). Start with `MAX_PARALLEL=2` if unsure.

## Examples

### All pair combinations (recommended for finding bad nodes)

```yaml
- name: GROUP_SIZES
  value: "2"
- name: TEST_MODE
  value: "all"
```

For 26 nodes: 325 pairs in 25 rounds, 13 parallel per round. Takes ~30 minutes with default parallelism.

### Quick sequential check

```yaml
- name: GROUP_SIZES
  value: "2,4,8"
- name: TEST_MODE
  value: "sequential"
```

For 26 nodes: 13 pairs + 6 groups of 4 + 3 groups of 8 + remainders. Takes ~25 minutes.

### Same local block (rail group) only

Tests nodes only within their own local block. Useful for isolating intra-block RDMA issues from cross-block issues.

```yaml
- name: SAME_LOCAL_BLOCK_ONLY
  value: "true"
- name: GROUP_SIZES
  value: "2"
- name: TEST_MODE
  value: "all"
```

### Ordered hostfile

Sorts hosts by local block (largest block first) so sequential groups naturally stay within the same block where possible.

```yaml
- name: ORDERED_HOSTFILE
  value: "true"
- name: GROUP_SIZES
  value: "4"
- name: TEST_MODE
  value: "sequential"
```

### Limit parallelism

```yaml
- name: MAX_PARALLEL
  value: "4"
```

## Output

Each test prints a colored result line with local block (rail group) topology info:

- Green `PASS`: busbw above threshold
- Yellow `SUSPECT`: busbw below threshold
- Red `FAIL`: mpirun crashed

Example:

```
--- group-of-4 #1: 10.140.65.237, 10.140.93.155, 10.140.74.14, 10.140.87.30 (1 block: kkq7ubugwsa (4 nodes)) ---
PASS: group-of-4 #1 busbw=365.85 GB/s
```

### Summary

```
+------------------------------------------------+
|                    SUMMARY                     |
+------------------------------------------------+
| Hosts:                 26                       |
| Test mode:             all                      |
| Group sizes:           2                        |
| BW threshold (2N):     450.0 GB/s               |
| BW threshold (>2N):    350.0 GB/s               |
| Results:               298/301 passed           |
+------------------------------------------------+

FAILED/SUSPECT TESTS (3):
  SUSPECT: 10.140.84.146, 10.140.65.231 busbw=412.50 GB/s [10.140.84.146, 10.140.65.231]
  SUSPECT: 10.140.84.146, 10.140.93.155 busbw=398.20 GB/s [10.140.84.146, 10.140.93.155]
  SUSPECT: 10.140.84.146, 10.140.80.58 busbw=405.10 GB/s [10.140.84.146, 10.140.80.58]

SUSPECT/FAIL NODE FREQUENCY (most frequent first):
  10.140.84.146: appeared in 3 failed/suspect tests

CONFIRMED BAD NODE(S): 10.140.84.146
```

## Adjusting worker count

Set `replicas` to match the number of schedulable GPU nodes:

```bash
kubectl get nodes -l node.kubernetes.io/instance-type=BM.GPU.H100.8 --field-selector=spec.unschedulable!=true --no-headers | wc -l
```

## Per-test log files

Detailed NCCL output for each test is saved inside the launcher pod at `/tmp/nccl-healthcheck/`. To inspect:

```bash
LAUNCHER=$(kubectl get pods -l nccl-test-replica=mpi-launcher -o jsonpath='{.items[0].metadata.name}')

# List all logs
kubectl exec $LAUNCHER -- ls /tmp/nccl-healthcheck/

# View a specific pair
kubectl exec $LAUNCHER -- cat /tmp/nccl-healthcheck/pair_10.140.84.146_10.140.65.231.log

# View a bisect result
kubectl exec $LAUNCHER -- cat /tmp/nccl-healthcheck/bisect_10.140.84.146_10.140.65.231.log

# View a cross-validation result
kubectl exec $LAUNCHER -- cat /tmp/nccl-healthcheck/validate_10.140.84.146.log
```
