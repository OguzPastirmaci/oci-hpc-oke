# RDMA pods with host device PFs and IPvlan on OCI DOCA-OFED images

This guide is for pods that get the RDMA physical functions (PFs) of a GPU node as `nvidia.com/hostdev` devices and stay on the pod network. It adds one IPvlan interface on each RDMA rail to each pod. The NVIDIA Network Operator deploys the device plugin, Multus, the CNI plugins, and NVIDIA IPAM.

## Why the pod needs its own interface on each rail

The tested OCI HPC image uses NVIDIA DOCA-OFED. The DOCA-OFED RDMA core module (`ib_core`) shows a RoCE GID to a process only when the network interface of that GID is in the network namespace of the process. The upstream Linux kernel does not do this check.

The RDMA network interfaces (`rdma0` to `rdma15`) stay in the host network namespace. A pod that gets only the `nvidia.com/hostdev` devices fails in the tested environment:

- `/sys/class/infiniband/<device>/ports/1/gids/` shows only zero GIDs.
- `ib_write_bw` with GID index 3 fails with `Failed to modify QP ... to RTR`. The kernel returns `ENODEV`.
- `ucx_perftest` fails with `uct_iface_open(rc_mlx5/mlx5_6:1) failed: Address not valid`.

An IPvlan L2 interface on a rail uses the MAC address of the rail and gets its own IP address. The interface is in the pod network namespace. The kernel adds a RoCE GID for its address on the RDMA device of the rail, and the pod can use that GID.

## Files and shapes

`values.yaml` does not depend on the shape. Each shape folder holds the other files:

| Folder | Status | Rail addresses |
|---|---|---|
| `BM.GPU4.8` | Tested. See [Tested environment](#tested-environment). | IPv4. Each IPvlan interface gets an IPv4 address from an NVIDIA IPAM pool inside the rail subnet. |
| `BM.GPU.B300.8` | Not tested. | IPv6. Each node has its own routed `/64` on each rail, assigned by the rail router (SLAAC). See [BM.GPU.B300.8 design](#bmgpub3008-design). |

Each shape folder contains:

- `nic-cluster-policy.yaml`: the device plugin for the 16 RDMA PFs of the shape, Multus, the CNI plugins, and NVIDIA IPAM.
- `ip-pool.yaml`: the NVIDIA IPAM pool for the IPvlan interfaces.
- `ipvlan-networks.yaml`: 16 networks `rdma-ipvlan-0` to `rdma-ipvlan-15`, one IPvlan interface on each rail.
- `rdma-ipvlan-tools.yaml`: scripts that find the RDMA device and GID index of each IPvlan interface.
- `ipvlan-check.yaml`, `ib-write-bw-all-rails.yaml`, `nccl-test.yaml`: the tests.

Use one shape per cluster with this guide. Both shape folders create objects with the same names.

### BM.GPU.B300.8 design

The `BM.GPU.B300.8` files were not tested on B300 hardware. They come from these sources:

- The 16 RDMA PF PCI addresses come from the `bm-gpu-b300-8` policy in `terraform/files/nvidia-network-operator/sriov-network-node-policy.yaml` in this repository.
- The NCCL settings and the 4 worker nodes come from `manifests/nccl-tests/virtual-functions/mpi-operator/BM.GPU.B300.8.yaml` in this repository.
- The address design comes from an earlier IPvlan test on `BM.GPU.GB300.4` nodes with IPv6 rails. That test created the IPvlan interfaces with DRANET, not with Multus.

The IPvlan CNI does not start without an IPAM. On IPv6 rails with one routed `/64` for each node, a shared NVIDIA IPAM pool does not route. So the B300 files work this way:

- `ip-pool.yaml` gives each IPvlan interface a placeholder address from `fd01:0:0:100::/112`. This address only lets the IPvlan CNI start. It does not route.
- The rail router gives each IPvlan interface its routed address and a default route (SLAAC). In the GB300 test, this took 3 to 4 seconds after the pod started.
- `rail-map.sh` waits for the SLAAC address and uses its GID. It ignores the placeholder address.
- The networks have no `tuning` or `sbr` plugin. The ARP settings are for IPv4, and source routing for the placeholder address is not useful.

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
| RDMA rails | IPv4, subnet `10.224.0.0/12`, MTU 4220 |

The guide was not tested on other shapes, other images, or IPv6 RDMA rails.

## RDMA network namespace mode

This setup needs the shared RDMA network namespace mode (`netns_mode=1`). The tested image uses the exclusive mode by default.

In exclusive mode, a pod with the `nvidia.com/hostdev` devices gets the `/dev/infiniband` character devices, but no RDMA device is visible in `/sys/class/infiniband`. `ibv_devices` lists no device.

This guide changes the mode with a reboot of each GPU node.

## Prerequisites

- `kubectl` access to the cluster, and Helm 3.
- `git`.
- Node Feature Discovery (NFD) runs in the cluster.
- No other Multus, SR-IOV Network Operator, NVIDIA Network Operator, or SR-IOV device plugin runs in the cluster.
- The MPI Operator, only for the NCCL test in step 10.

## Step 1. Get the files and set the shape

```bash
git clone --branch doca-ipvlan-hostdev https://github.com/OguzPastirmaci/oci-hpc-oke.git
```

```bash
cd oci-hpc-oke/manifests/nvidia-network-operator/ipvlan-hostdev
```

Set `SHAPE` to the shape of your GPU nodes. Use the name of a shape folder, `BM.GPU4.8` or `BM.GPU.B300.8`. Only `BM.GPU4.8` was tested:

```bash
SHAPE=BM.GPU4.8
```

Run all later commands from this directory and in the same shell, because they use `SHAPE`.

## Step 2. Check the prerequisites

Make sure that the GPU nodes have the shape label:

```bash
kubectl get nodes -l node.kubernetes.io/instance-type="$SHAPE"
```

Make sure that no other Multus or SR-IOV component runs:

```bash
kubectl get daemonsets -A | grep -i -E 'multus|sriov|network-operator' || echo "none found"
```

The output must be `none found`.

## Step 3. Check the current RDMA mode

```bash
for NODE in $(kubectl get nodes -l node.kubernetes.io/instance-type="$SHAPE" -o jsonpath='{.items[*].metadata.name}'); do
  echo "$NODE netns_mode=$(kubectl debug node/"$NODE" --profile=sysadmin --image=busybox:1.37 --attach --quiet -- cat /host/sys/module/ib_core/parameters/netns_mode)"
done
```

`N` is exclusive. `Y` is shared.

The command leaves completed `node-debugger-*` pods. Delete them:

```bash
kubectl get pods -o name | grep node-debugger- | xargs -r kubectl delete
```

## Step 4. Change the RDMA mode to shared (only if needed)

Do this step only when step 3 shows `N`. Do it before step 5 when you can.

```bash
NETNS_MODE=1
```

Write the setting on every GPU node and rebuild the initramfs:

```bash
for NODE in $(kubectl get nodes -l node.kubernetes.io/instance-type="$SHAPE" -o jsonpath='{.items[*].metadata.name}'); do
  kubectl debug node/"$NODE" --profile=sysadmin --image=busybox:1.37 --attach --quiet -- \
    chroot /host bash -c "echo 'options ib_core netns_mode=$NETNS_MODE' > /etc/modprobe.d/ib_core.conf && update-initramfs -u -k all"
done
```

Reboot the GPU nodes one at a time. The loop drains each node, reboots it, waits until it is `Ready` with a new boot ID, and uncordons it:

```bash
for NODE in $(kubectl get nodes -l node.kubernetes.io/instance-type="$SHAPE" -o jsonpath='{.items[*].metadata.name}'); do
  BOOT_ID=$(kubectl get node "$NODE" -o jsonpath='{.status.nodeInfo.bootID}')
  kubectl drain "$NODE" --ignore-daemonsets --delete-emptydir-data --timeout=15m
  kubectl debug node/"$NODE" --profile=sysadmin --image=busybox:1.37 --attach --quiet -- chroot /host systemctl reboot
  until [ "$(kubectl get node "$NODE" -o jsonpath='{.status.nodeInfo.bootID}')" != "$BOOT_ID" ]; do sleep 30; done
  kubectl wait --for=condition=Ready node/"$NODE" --timeout=30m
  kubectl uncordon "$NODE"
done
```

A `BM.GPU4.8` node took about 8 minutes to reboot in the test. The Oracle Cloud Agent configures the RDMA rails after the node starts. Wait until all 16 RDMA rails of each node have an IP address again:

```bash
for NODE in $(kubectl get nodes -l node.kubernetes.io/instance-type="$SHAPE" -o jsonpath='{.items[*].metadata.name}'); do
  until [ "$(kubectl debug node/"$NODE" --profile=sysadmin --image=busybox:1.37 --attach --quiet -- chroot /host ip -o addr show scope global | awk '$2 ~ /^rdma[0-9]+$/ {print $2}' | sort -u | wc -l)" -eq 16 ]; do sleep 60; done
  echo "$NODE: 16 RDMA rails have an IP address"
done
```

Delete the debug pods, then repeat step 3 to confirm the new mode:

```bash
kubectl get pods -o name | grep node-debugger- | xargs -r kubectl delete
```

## Step 5. Install the NVIDIA Network Operator

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

`values.yaml` disables the NFD that the chart includes, because the cluster already runs NFD. It also disables the SR-IOV Network Operator, because this setup uses the PFs.

## Step 6. Deploy the device plugin, Multus, the CNI plugins, and NVIDIA IPAM

```bash
kubectl apply -f "$SHAPE/nic-cluster-policy.yaml"
```

```bash
kubectl wait --for=jsonpath='{.status.state}'=ready nicclusterpolicy/nic-cluster-policy --timeout=15m
```

The device plugin offers the 16 RDMA PFs of each node as `nvidia.com/hostdev`. The file lists the PFs by PCI address, because the front-end NIC is also an RDMA device from the same vendor. On `BM.GPU4.8`, it is the same NIC model. Wait until each GPU node offers 16 devices:

```bash
kubectl wait nodes -l node.kubernetes.io/instance-type="$SHAPE" --for=jsonpath='{.status.allocatable.nvidia\.com/hostdev}'=16 --timeout=5m
```

## Step 7. Check the IP address range

Show the rail addresses of a GPU node:

```bash
NODE=$(kubectl get nodes -l node.kubernetes.io/instance-type="$SHAPE" -o jsonpath='{.items[0].metadata.name}')
```

```bash
kubectl debug node/"$NODE" --profile=sysadmin --image=busybox:1.37 --attach --quiet -- chroot /host ip -o addr show dev rdma0 scope global
```

### BM.GPU4.8

Each IPvlan interface needs an IPv4 address in the subnet of the RDMA rails. No host may use the address. `BM.GPU4.8/ip-pool.yaml` uses `10.239.255.0/24`.

In the tested environment, all 16 rails use addresses in `10.224.0.0/12`, and `10.239.255.0/24` is inside it. If your rails use another subnet, change `subnet` in `BM.GPU4.8/ip-pool.yaml` to a range inside it.

Make sure that no host answers on any address of the range. The command pings every address of `10.239.255.0/24` on each of the 16 rails. Change the address prefix in the command if you changed the pool:

```bash
# shellcheck disable=SC2016  # the loop runs on the node, not in your shell
kubectl debug node/"$NODE" --profile=sysadmin --image=busybox:1.37 --attach --quiet -- chroot /host bash -c 'for rail in $(seq 0 15); do for i in $(seq 1 254); do (ping -c 1 -W 1 -I rdma$rail 10.239.255.$i >/dev/null 2>&1 && echo "in use: rdma$rail 10.239.255.$i") & done; wait; done; echo "check done"'
```

The only output line must be `check done`.

### BM.GPU.B300.8

The rails must show IPv6 addresses. `BM.GPU.B300.8/ip-pool.yaml` uses the placeholder range `fd01:0:0:100::/112`. It must not overlap any rail prefix. On all B300 and GB300 rails seen so far, the rail prefixes started with `fdcd:`. If a rail prefix overlaps the placeholder range, change `subnet` in `BM.GPU.B300.8/ip-pool.yaml`.

### Both shapes

Delete the debug pods:

```bash
kubectl get pods -o name | grep node-debugger- | xargs -r kubectl delete
```

## Step 8. Create the IP pool and the networks

```bash
kubectl apply -f "$SHAPE/ip-pool.yaml" -f "$SHAPE/ipvlan-networks.yaml" -f "$SHAPE/rdma-ipvlan-tools.yaml"
```

These files create:

- The NVIDIA IPAM pool `rdma-ipvlan-pool`. Each GPU node gets a block of 32 addresses.
- 16 networks `rdma-ipvlan-0` to `rdma-ipvlan-15` in the `default` namespace. Network `rdma-ipvlan-N` creates an IPvlan L2 interface on `rdmaN`. On `BM.GPU4.8`, the `tuning` plugin sets the ARP settings of the interface, and the `sbr` plugin adds source routing.
- The ConfigMap `rdma-ipvlan-tools` with two scripts for the pods. `rail-map.sh` prints the RDMA device and GID index of each IPvlan interface. `with-rails.sh` sets `NCCL_IB_HCA` and `NCCL_IB_GID_INDEX` from that list and runs a command.

## Step 9. Check the GIDs in a pod

```bash
kubectl apply -f "$SHAPE/ipvlan-check.yaml"
```

```bash
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/rdma-ipvlan-check --timeout=5m
```

```bash
kubectl logs rdma-ipvlan-check
```

The last line must start with `PASS`. Example output on `BM.GPU4.8`:

```text
RDMA netns mode: Y (Y = shared, N = exclusive)
Allocated RDMA PFs: 16
interface address device gid_index
net1 10.239.255.1 mlx5_6 5
net2 10.239.255.2 mlx5_7 5
...
net16 10.239.255.16 mlx5_13 5
Host GID index 3 of mlx5_6 in this pod: 0000:0000:0000:0000:0000:0000:0000:0000
PASS: 16 IPvlan interfaces have a RoCE v2 GID on an allocated RDMA PF (GID index: 5)
```

Interface `netN` is the IPvlan interface on `rdma(N-1)`. The GIDs of the host rail addresses still read as zero in the pod. On `BM.GPU.B300.8`, the address column shows the SLAAC IPv6 address, and the GID index can differ.

Delete the pod:

```bash
kubectl delete -f "$SHAPE/ipvlan-check.yaml"
```

## Step 10. Test the RDMA bandwidth

### All 16 rails with ib_write_bw

The test starts one pod on each of two GPU nodes. It runs `ib_write_bw` on all 16 rails at the same time. Each rail uses the RDMA device and GID of the IPvlan interface. The test control connection uses the pod network.

```bash
kubectl apply -f "$SHAPE/ib-write-bw-all-rails.yaml"
```

```bash
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/rdma-ipvlan-write-bw-client --timeout=10m
```

```bash
kubectl logs rdma-ipvlan-write-bw-client | grep -E '^(Client bandwidth summary|PASS|FAIL|ERROR)'
```

The last line must be `PASS: ib_write_bw completed concurrently across all 16 IPvlan rails`. On `BM.GPU4.8`, the test ran twice:

| Run | Per rail | Total |
|---|---|---|
| 1 | 41.72 to 56.28 Gbit/s | 804.86 Gbit/s |
| 2, from a fresh clone of this guide | 38.04 to 51.64 Gbit/s | 746.03 Gbit/s |

The 16 rails run at the same time, and the results change from run to run.

Delete the test:

```bash
kubectl delete -f "$SHAPE/ib-write-bw-all-rails.yaml"
```

### NCCL all_reduce over 16 rails per node

This test needs the MPI Operator. It runs `all_reduce_perf` with 8 GPUs on each GPU node. `with-rails.sh` sets `NCCL_IB_HCA` and `NCCL_IB_GID_INDEX` in each rank.

| Shape | Worker nodes | Largest message |
|---|---|---|
| `BM.GPU4.8` | 2 | 8 GiB. The 40 GB GPUs cannot hold 16 GiB buffers with validation. |
| `BM.GPU.B300.8` | 4 | 16 GiB |

```bash
kubectl apply -f "$SHAPE/nccl-test.yaml"
```

```bash
kubectl wait --for=condition=Succeeded mpijob/nccl-test-ipvlan --timeout=20m
```

```bash
kubectl logs job/nccl-test-ipvlan-launcher | grep -E '^ +[0-9]{9,}|Out of bounds'
```

The results on `BM.GPU4.8` were:

| Size | Bus bandwidth, out of place | Bus bandwidth, in place |
|---|---|---|
| 4 GiB | 190.11 GB/s | 190.05 GB/s |
| 8 GiB | 190.69 GB/s | 190.53 GB/s |

These results come from a run that followed this guide from a fresh clone. The test had 0 wrong values.

The log shows `NCCL WARN NET/MLX5: Call to mlx5dv_devx_general_cmd failed with error Remote I/O error` once for each rank. The same message comes from the same test image on the host PFs with `hostNetwork: true`. It did not change the result.

Delete the test:

```bash
kubectl delete -f "$SHAPE/nccl-test.yaml"
```

## Use the rails in your own pods

Use `$SHAPE/nccl-test.yaml` as the example. A pod needs these settings:

- The resource limit `nvidia.com/hostdev: 16`.
- The annotation `k8s.v1.cni.cncf.io/networks` with the 16 networks in order: `rdma-ipvlan-0,rdma-ipvlan-1,...,rdma-ipvlan-15`.
- The `IPC_LOCK` capability. The test pods do not run privileged.
- The pod must run in the `default` namespace, because the networks exist there.
- For NCCL: `NCCL_IB_HCA` set to the RDMA devices of the IPvlan interfaces, and `NCCL_IB_GID_INDEX` set to their GID index. Mount the ConfigMap `rdma-ipvlan-tools` and start NCCL through `with-rails.sh`, or run `rail-map.sh` to get the values.

On `BM.GPU4.8`, UCX 1.22.0 selected GID index 5 by itself. `ucx_perftest` (`ucp_put_bw`, 1 MiB messages, one rail) reached 11630.75 MB/s between two pods. These UCX commands are not part of this guide.

## Remove the configuration

Delete the networks, the pool, the scripts, and the device plugin configuration:

```bash
kubectl delete -f "$SHAPE/ipvlan-networks.yaml" -f "$SHAPE/rdma-ipvlan-tools.yaml" -f "$SHAPE/ip-pool.yaml" -f "$SHAPE/nic-cluster-policy.yaml"
```

```bash
helm uninstall network-operator -n nvidia-network-operator
```

Helm does not delete the CRDs. Delete them and the namespace:

```bash
kubectl delete crd --ignore-not-found \
  hostdevicenetworks.mellanox.com ipoibnetworks.mellanox.com macvlannetworks.mellanox.com \
  nicclusterpolicies.mellanox.com nicnodepolicies.mellanox.com \
  cidrpools.nv-ipam.nvidia.com ippools.nv-ipam.nvidia.com \
  network-attachment-definitions.k8s.cni.cncf.io
```

```bash
kubectl delete namespace nvidia-network-operator
```

Do not delete `clusterpolicies.nvidia.com` or `nvidiadrivers.nvidia.com`. The GPU Operator uses them.

Remove the node labels that the operator added:

```bash
kubectl label nodes -l node.kubernetes.io/instance-type="$SHAPE" network.nvidia.com/operator.mofed.wait- network.nvidia.com/operator.nic-configuration.wait-
```

The operator does not change the RDMA mode back. To restore the exclusive mode, do step 4 with `NETNS_MODE=0`.
