# SR-IOV VFs for RDMA pods on OCI DOCA-OFED images

This guide gives each pod one SR-IOV virtual function (VF) on every RDMA rail of a GPU node. The NVIDIA Network Operator creates the VFs, gives them IP addresses, and moves them into the pods. The guide covers both RDMA network namespace modes, `exclusive` and `shared`.

## Why a pod needs its own interface on each rail

The tested OCI HPC image uses NVIDIA DOCA-OFED. The DOCA-OFED RDMA core module (`ib_core`) shows a RoCE GID to a process only when the network interface of that GID is in the network namespace of the process. The upstream Linux kernel does not do this check.

A pod that gets an RDMA device but no network interface on that device sees only zero GIDs in `/sys/class/infiniband/<device>/ports/1/gids/`. The driver also rejects address handles and queue pair connections that use those GIDs (`ENODEV`).

In this guide, each pod gets one VF network interface per rail inside its own network namespace. Each interface has its own IP address, so the pod has its own valid GIDs.

## Tested environment

| Item | Value |
|---|---|
| Shape | `BM.GPU4.8`, 2 nodes, 16 RDMA rails per node |
| Image | `Canonical-Ubuntu-24.04-2026.09.18-0-KERNEL-ORACLE-6.14-DOCA-OFED-3.5.0-GPU-595-OPEN-CUDA-13.2-2026.09.23-0` |
| Kernel | `6.14.0-1018-oracle` |
| DOCA-OFED | 3.5.0 (`ofed_info -s`: `OFED-internal-26.07-0.7.7`) |
| Kubernetes | OKE v1.36.4 |
| NVIDIA Network Operator | 26.7.0 |
| Helm | 3.22.1 |
| RDMA rails | IPv4, MTU 4220 |

The guide was not tested on other shapes, other images, or IPv6 RDMA rails. The files in this directory use the `BM.GPU4.8` node label and the `BM.GPU4.8` PCI addresses.

## RDMA network namespace modes

| | Exclusive (`netns_mode=0`) | Shared (`netns_mode=1`) |
|---|---|---|
| RDMA devices a pod sees | Only the RDMA devices of its own VFs | All RDMA devices on the node |
| GIDs a pod can use | GIDs of its own VFs | GIDs of its own VFs. GIDs of other devices read as zero. |
| Network file | `sriov-network-exclusive.yaml` (with the RDMA CNI) | `sriov-network-shared.yaml` (no RDMA CNI) |
| Default on the tested image | Yes | No |

The RDMA CNI moves the VF RDMA device into the pod. It fails when the mode is shared, so the shared network file does not use it.

This guide changes the mode with a reboot of each GPU node.

## Prerequisites

- `kubectl` access to the cluster, and Helm 3.
- `git`.
- Node Feature Discovery (NFD) runs in the cluster. The NVIDIA GPU Operator usually deploys it.
- No other Multus, SR-IOV Network Operator, NVIDIA Network Operator, or VF management tool runs in the cluster.
- The MPI Operator, only for the NCCL test in step 9.

Make sure that NFD labels the GPU nodes as SR-IOV capable:

```bash
kubectl get nodes -l node.kubernetes.io/instance-type=BM.GPU4.8,feature.node.kubernetes.io/network-sriov.capable=true
```

Each `BM.GPU4.8` node must be in the output.

Make sure that no other Multus or SR-IOV component runs:

```bash
kubectl get daemonsets -A | grep -i -E 'multus|sriov|network-operator' || echo "none found"
```

The output must be `none found`.

## Step 1. Get the files

```bash
git clone --branch doca-vf-network-operator https://github.com/OguzPastirmaci/oci-hpc-oke.git
```

```bash
cd oci-hpc-oke/manifests/nvidia-network-operator/standalone
```

Run all later commands from this directory.

## Step 2. Check the current RDMA mode

```bash
for NODE in $(kubectl get nodes -l node.kubernetes.io/instance-type=BM.GPU4.8 -o jsonpath='{.items[*].metadata.name}'); do
  echo "$NODE netns_mode=$(kubectl debug node/"$NODE" --profile=sysadmin --image=busybox:1.37 --attach --quiet -- cat /host/sys/module/ib_core/parameters/netns_mode)"
done
```

`N` is exclusive. `Y` is shared.

The command leaves completed `node-debugger-*` pods. Delete them:

```bash
kubectl get pods -o name | grep node-debugger- | xargs -r kubectl delete
```

## Step 3. Change the RDMA mode (only if needed)

Do this step only when the current mode is not the mode that you want. Do it before step 4 when you can. To change the mode after the VFs exist, read [Change the RDMA mode later](#change-the-rdma-mode-later).

Set `NETNS_MODE` to `1` for shared or `0` for exclusive:

```bash
NETNS_MODE=1
```

Write the setting on every GPU node and rebuild the initramfs:

```bash
for NODE in $(kubectl get nodes -l node.kubernetes.io/instance-type=BM.GPU4.8 -o jsonpath='{.items[*].metadata.name}'); do
  kubectl debug node/"$NODE" --profile=sysadmin --image=busybox:1.37 --attach --quiet -- \
    chroot /host bash -c "echo 'options ib_core netns_mode=$NETNS_MODE' > /etc/modprobe.d/ib_core.conf && update-initramfs -u -k all"
done
```

Reboot the GPU nodes one at a time. The loop drains each node, reboots it, waits until it is `Ready` with a new boot ID, and uncordons it:

```bash
for NODE in $(kubectl get nodes -l node.kubernetes.io/instance-type=BM.GPU4.8 -o jsonpath='{.items[*].metadata.name}'); do
  BOOT_ID=$(kubectl get node "$NODE" -o jsonpath='{.status.nodeInfo.bootID}')
  kubectl drain "$NODE" --ignore-daemonsets --delete-emptydir-data --timeout=15m
  kubectl debug node/"$NODE" --profile=sysadmin --image=busybox:1.37 --attach --quiet -- chroot /host systemctl reboot
  until [ "$(kubectl get node "$NODE" -o jsonpath='{.status.nodeInfo.bootID}')" != "$BOOT_ID" ]; do sleep 30; done
  kubectl wait --for=condition=Ready node/"$NODE" --timeout=30m
  kubectl uncordon "$NODE"
done
```

A `BM.GPU4.8` node took about 8 minutes to reboot in the test. The Oracle Cloud Agent configures the RDMA rails after the node starts, which took about 10 more minutes.

Delete the debug pods, then repeat step 2 to confirm the new mode:

```bash
kubectl get pods -o name | grep node-debugger- | xargs -r kubectl delete
```

## Step 4. Install the NVIDIA Network Operator

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
```

```bash
helm repo update nvidia
```

```bash
helm upgrade --install network-operator nvidia/network-operator \
  --namespace nvidia-network-operator --create-namespace \
  --version 26.7.0 --values values.yaml --wait --timeout 15m
```

`values.yaml` disables the NFD that the chart includes, because the cluster already runs NFD. It also disables the `mellanox` plugin of the SR-IOV Network Operator, so that the operator does not change the NIC firmware or request a reboot.

Make sure that the SR-IOV Network Operator uses these settings:

```bash
kubectl -n nvidia-network-operator get sriovoperatorconfig default -o jsonpath='{.spec.configurationMode} {.spec.disableDrain} {.spec.disablePlugins}{"\n"}'
```

The output must be `daemon false ["mellanox"]`.

## Step 5. Deploy Multus, the CNI plugins, and NV-IPAM

```bash
kubectl apply -f nic-cluster-policy.yaml
```

```bash
kubectl wait --for=jsonpath='{.status.state}'=ready nicclusterpolicy/nic-cluster-policy --timeout=15m
```

## Step 6. Create the IP pool and the network

```bash
kubectl apply -f ip-pool.yaml -f sriov-network-pool-config.yaml
```

For exclusive mode, apply the exclusive network:

```bash
kubectl apply -f sriov-network-exclusive.yaml
```

For shared mode, apply the shared network instead:

```bash
kubectl apply -f sriov-network-shared.yaml
```

Both files create the network `rdma-vf` in the `default` namespace. Use the file that matches the mode from step 2.

## Step 7. Create the VFs

```bash
kubectl apply -f sriov-network-node-policy.yaml
```

The operator creates one VF on each of the 16 RDMA PFs of each node, one node at a time. In the test, this did not drain or reboot the nodes. Wait until each GPU node offers 16 VFs:

```bash
kubectl wait nodes -l node.kubernetes.io/instance-type=BM.GPU4.8 --for=jsonpath='{.status.allocatable.nvidia\.com/rdma-vf}'=16 --timeout=15m
```

## Step 8. Check the GIDs in a pod

```bash
kubectl apply -f vf-check.yaml
```

```bash
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/rdma-vf-check --timeout=5m
```

```bash
kubectl logs rdma-vf-check
```

The last line must start with `PASS`. Example output in exclusive mode:

```text
RDMA netns mode: N (Y = shared, N = exclusive)
RDMA devices visible in the pod: 1
Character devices: rdma_cm umad30 uverbs30
VF interface net1: ip=192.168.1.190 rdma_device=mlx5_30
  gid 0: fe80:0000:0000:0000:2438:8fff:fe64:94cd IB/RoCE v1
  gid 1: fe80:0000:0000:0000:2438:8fff:fe64:94cd RoCE v2
  gid 2: 0000:0000:0000:0000:0000:ffff:c0a8:01be IB/RoCE v1
  gid 3: 0000:0000:0000:0000:0000:ffff:c0a8:01be RoCE v2
PASS: GID index 3 of mlx5_30 is the RoCE v2 GID of 192.168.1.190
```

In shared mode, the pod sees all RDMA devices of the node. The output then also shows that another device has a zero GID:

```text
RDMA netns mode: Y (Y = shared, N = exclusive)
RDMA devices visible in the pod: 34
Character devices: rdma_cm umad30 uverbs30
VF interface net1: ip=192.168.1.45 rdma_device=mlx5_30
...
Other RDMA device mlx5_0 gid 3: 0000:0000:0000:0000:0000:0000:0000:0000
PASS: GID index 3 of mlx5_30 is the RoCE v2 GID of 192.168.1.45
```

Delete the pod:

```bash
kubectl delete -f vf-check.yaml
```

## Step 9. Test the RDMA bandwidth

### All 16 VF pairs with ib_write_bw

The test starts one pod with 16 VFs on each of two GPU nodes. It runs `ib_write_bw` on all 16 VF pairs at the same time. The client also checks that the route to each server address uses the matching VF.

```bash
kubectl apply -f ib-write-bw-all-vfs.yaml
```

```bash
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/rdma-vf-write-bw-client --timeout=10m
```

```bash
kubectl logs rdma-vf-write-bw-client | grep -E '^(Client bandwidth summary|PASS|FAIL|ERROR)'
```

The last line must be `PASS: ib_write_bw completed concurrently across all 16 VF pairs`. The test results were:

| Mode | Per pair | Total |
|---|---|---|
| Exclusive | 41.59 to 56.88 Gbit/s | 768.11 Gbit/s |
| Shared | 43.46 to 49.74 Gbit/s | 763.96 Gbit/s |

The 16 pairs run at the same time. On the host PFs, the same test with the same image gave 50.29 to 54.03 Gbit/s per pair (842.93 Gbit/s total). One VF pair alone reached 97.97 Gbit/s.

Delete the test:

```bash
kubectl delete -f ib-write-bw-all-vfs.yaml
```

### NCCL all_reduce over 16 VFs per node

This test needs the MPI Operator. It runs `all_reduce_perf` with 8 GPUs and 16 VFs on each of two GPU nodes. Each rank sets `NCCL_IB_HCA` to the RDMA devices of the VFs in its own pod. The 40 GB GPUs cannot hold 16 GiB buffers with validation, so the test stops at 8 GiB.

```bash
kubectl apply -f nccl-test.yaml
```

```bash
kubectl wait --for=condition=Succeeded mpijob/nccl-test-vf --timeout=20m
```

```bash
kubectl logs job/nccl-test-vf-launcher | grep -E '^ +[0-9]{9,}|Out of bounds'
```

The test results were:

| Mode | Size | Bus bandwidth, out of place | Bus bandwidth, in place |
|---|---|---|---|
| Exclusive | 4 GiB | 190.07 GB/s | 190.25 GB/s |
| Exclusive | 8 GiB | 190.90 GB/s | 190.83 GB/s |
| Shared | 4 GiB | 189.79 GB/s | 189.93 GB/s |
| Shared | 8 GiB | 190.91 GB/s | 191.08 GB/s |

The same test on the host PFs with `hostNetwork: true` gave 190.99 GB/s out of place and 190.97 GB/s in place at 8 GiB.

The log shows `NCCL WARN NET/MLX5: Call to mlx5dv_devx_general_cmd failed with error Remote I/O error` once for each rank. The same message comes from the same test image on the host PFs with `hostNetwork: true`. It did not change the result.

Delete the test:

```bash
kubectl delete -f nccl-test.yaml
```

## Use the VFs in your own pods

Use `nccl-test.yaml` as the example. A pod that uses the VFs needs these settings:

- The annotation `k8s.v1.cni.cncf.io/networks` with `rdma-vf` once for each VF. For 16 VFs, list `rdma-vf` 16 times, separated by commas.
- The resource limit `nvidia.com/rdma-vf` with the same number.
- The `IPC_LOCK` capability. The test pods do not run privileged.
- The pod must run in the `default` namespace, because the network exists there.
- `NCCL_IB_HCA` set to the RDMA devices of the VFs in the pod, and `NCCL_IB_GID_INDEX=3`. The VF interfaces are `net1` to `net16`. This command prints the device list:

```bash
basename -a /sys/class/net/net*/device/infiniband/* | paste -sd, -
```

## Change the RDMA mode later

1. Delete all pods that use VFs.
2. Do step 3 with the new `NETNS_MODE` value.
3. Wait until the operator creates the VFs again. Use the commands of step 7 to check the node states and the 16 VFs per node.
4. Apply the network file that matches the new mode (step 6). It replaces the network `rdma-vf`.
5. Do step 8 to check the GIDs.

## Remove the configuration

Delete the node policy. The operator then removes the VFs, one node at a time:

```bash
kubectl delete -f sriov-network-node-policy.yaml
```

Wait until no node reports VFs. The loop ends when the list is empty:

```bash
until [ -z "$(kubectl -n nvidia-network-operator get sriovnetworknodestates -o jsonpath='{.items[*].status.interfaces[?(@.numVfs>0)].name}')" ]; do sleep 10; done
```

Delete the network, the pool, and the operator:

```bash
kubectl -n nvidia-network-operator delete sriovnetwork rdma-vf
```

```bash
kubectl delete -f sriov-network-pool-config.yaml -f ip-pool.yaml -f nic-cluster-policy.yaml
```

```bash
helm uninstall network-operator -n nvidia-network-operator
```

Helm does not delete the CRDs. Delete them and the namespace:

```bash
kubectl delete crd --ignore-not-found \
  hostdevicenetworks.mellanox.com ipoibnetworks.mellanox.com macvlannetworks.mellanox.com \
  nicclusterpolicies.mellanox.com nicnodepolicies.mellanox.com \
  ovsnetworks.sriovnetwork.openshift.io sriovibnetworks.sriovnetwork.openshift.io \
  sriovnetworknodepolicies.sriovnetwork.openshift.io sriovnetworknodestates.sriovnetwork.openshift.io \
  sriovnetworkpoolconfigs.sriovnetwork.openshift.io sriovnetworks.sriovnetwork.openshift.io \
  sriovoperatorconfigs.sriovnetwork.openshift.io \
  deviceattributes.sriovnetwork.k8snetworkplumbingwg.io sriovresourcepolicies.sriovnetwork.k8snetworkplumbingwg.io \
  cidrpools.nv-ipam.nvidia.com ippools.nv-ipam.nvidia.com \
  network-attachment-definitions.k8s.cni.cncf.io
```

```bash
kubectl delete namespace nvidia-network-operator
```

Do not delete `clusterpolicies.nvidia.com` or `nvidiadrivers.nvidia.com`. The GPU Operator uses them.

Remove the node labels that the operator added:

```bash
kubectl label nodes -l node.kubernetes.io/instance-type=BM.GPU4.8 network.nvidia.com/operator.mofed.wait- network.nvidia.com/operator.nic-configuration.wait- sriovnetwork.openshift.io/device-plugin-
```

The operator does not change the RDMA mode back. To restore the original mode, do step 3 with the original `NETNS_MODE` value.
