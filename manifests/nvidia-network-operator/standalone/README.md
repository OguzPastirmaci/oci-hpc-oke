# Standalone NVIDIA Network Operator with operator-managed VFs

This deployment installs the standalone NVIDIA Network Operator alongside an
existing GPU Operator. It is scoped to `BM.GPU.B4.8` nodes and makes the SR-IOV
Network Operator, rather than `vf-config`, create one VF on each selected PF.

Do **not** deploy `manifests/vf-config/vf-config.yaml`. This configuration uses
`externallyManaged: false`; running both VF owners would introduce an unsafe
race over `sriov_numvfs`.

## Version choice

The deployment uses the published NVIDIA Network Operator `26.4.0` chart and
images because its `plugins` image contains the SBR `addSourceHints` option.
The `26.1.1` plugins image contains CNI plugins v1.9.1 but its SBR binary does
not contain that option.

As of 2026-06-27, NVIDIA's Helm repository and documentation publish `26.4.0`
and the platform-support documentation calls 26.4.x supported. However, the
public `Mellanox/network-operator` GitHub releases page still exposes only
`v26.4.0-rc.1` as a pre-release. This deployment intentionally accepts that
release-status inconsistency to obtain `addSourceHints`.

## Ownership boundaries

- The OCI image provisioning flow and Oracle Cloud Agent (OCA) own PF firmware
  and host preparation: `SRIOV_EN=True`, `NUM_OF_VFS=127`, Ethernet link type,
  the host driver, PF addresses, QoS/PFC, and `ib_core netns_mode=0`.
- The generic SR-IOV Network Operator plugin owns the runtime VF count. The
  Mellanox firmware plugin is disabled, so the operator writes `sriov_numvfs`
  without changing firmware or asking for a reboot.
- The SR-IOV device plugin advertises `nvidia.com/rdma-vf`, and SR-IOV CNI moves
  the allocated VF into a pod.
- RDMA CNI provides exclusive RDMA-device namespace isolation. Tuning applies
  per-network-namespace sysctls. SBR installs source-policy routes and
  `addSourceHints` keeps the directly connected VF subnet in the main table
  with the VF source address, allowing an unbound `ibv_rc_pingpong` TCP control
  connection to choose the VF instead of `eth0`.

No privileged init container, DOCA/OFED driver container, RDMA shared-device
plugin, NIC Configuration Operator, or GPU Operator component is deployed by
these manifests.

## PF preflight

Before installing, each selected PF must already have:

- vendor/device `15b3:1019` bound to `mlx5_core`;
- `sriov_totalvfs` at least 1 and `sriov_numvfs=0` for a zero-to-one test;
- firmware `SRIOV_EN=True`, sufficient `NUM_OF_VFS`, and Ethernet link type;
- an UP Ethernet and RDMA device with MTU 4220;
- `mlx5_ib` loaded and `ib_core netns_mode=0`;
- OCI HPC QoS/PFC configuration.

Capture node boot IDs before making changes so a no-reboot result can be
proved:

```bash
kubectl get nodes -l node.kubernetes.io/instance-type=BM.GPU.B4.8 -o name \
  | while read -r node; do
      printf '%s ' "$node"
      kubectl get "$node" -o jsonpath='{.status.nodeInfo.bootID}{"\n"}'
    done
```

## Install

The commands below run on the OKE operator host. OCI CLI authentication must
use the instance principal.

```bash
export PATH=/home/ubuntu/bin:/home/ubuntu/lib/oracle-cli/bin:$PATH
export OCI_CLI_AUTH=instance_principal

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm upgrade --install network-operator nvidia/network-operator \
  --namespace nvidia-network-operator \
  --create-namespace \
  --version 26.4.0 \
  --values manifests/nvidia-network-operator/standalone/values.yaml \
  --wait \
  --timeout 15m
```

Verify the safety settings before creating any policy:

```bash
kubectl -n nvidia-network-operator get sriovoperatorconfig default \
  -o jsonpath='{.spec.configurationMode}{" "}{.spec.disableDrain}{" "}{.spec.disablePlugins}{"\n"}'
```

The expected values are `daemon false ["mellanox"]`. Also verify there is one
NFD deployment (the existing GPU Operator instance), not a second Network
Operator-owned NFD deployment.

Deploy the networking components. The intentionally minimal
`NicClusterPolicy` does not include an OFED driver or shared RDMA device plugin:

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/nic-cluster-policy.yaml
kubectl wait --for=jsonpath='{.status.state}'=ready nicclusterpolicy/nic-cluster-policy --timeout=15m
```

Create IPAM and the network before the VF policy:

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/nv-ipam-ip-pool.yaml
kubectl apply -f manifests/nvidia-network-operator/standalone/sriov-network-pool-config.yaml
kubectl apply -f manifests/nvidia-network-operator/standalone/sriov-network.yaml
```

Finally, let the SR-IOV Network Operator create the VFs. This is the only VF
creation step:

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/sriov-network-node-policy.yaml
```

## Verify VF creation without reboot

Watch node state until both B4 nodes report `Succeeded`; fail the test if a
state reports `Reboot_Required`:

```bash
kubectl -n nvidia-network-operator get sriovnetworknodestates -w
```

Re-run the boot-ID command and compare it with the preflight capture. Then
confirm 16 allocatable VFs per B4 node:

```bash
kubectl get nodes -l node.kubernetes.io/instance-type=BM.GPU.B4.8 \
  -o custom-columns='NODE:.metadata.name,RDMA-VFS:.status.allocatable.nvidia\.com/rdma-vf'
```

On each host, every selected PF should now have `sriov_numvfs=1`. Firmware
capacity remains 127; the operator changes only the runtime VF count.

## Functional tests

Apply the existing two-node VF NCCL test:

```bash
kubectl apply -f manifests/nccl-tests/kueue/virtual-functions/BM.GPU.B4.8.yaml
kubectl get mpijob,pods -w
kubectl logs -f job/nccl-test-launcher
```

For a stable point-to-point test independent of MPIJob cleanup, deploy the two
single-VF ping-pong pods. Required pod anti-affinity places them on different
B4 nodes:

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/ibv-rc-pingpong.yaml
kubectl wait --for=condition=Ready pod/rdma-vf-ping-a pod/rdma-vf-ping-b --timeout=10m
```

Use the `k8s.v1.cni.cncf.io/network-status` annotation or `ip -4 addr` to find
the VF IP and `rdma link` to find its associated mlx5 device. Confirm the route
to the peer VF uses the VF directly and carries the VF source address without
adding a manual `/32` route:

```bash
ip route get PEER_VF_IP
```

Expected shape:

```text
PEER_VF_IP dev netN src LOCAL_VF_IP
```

Run `ibv_rc_pingpong` with the VF-associated mlx5 device and GID index 3. The
server listens without a peer argument; the client supplies the server VF IP:

```bash
ibv_rc_pingpong -d VF_MLX5_DEVICE -g 3
ibv_rc_pingpong -d VF_MLX5_DEVICE -g 3 SERVER_VF_IP
```

The client must complete without `ip route replace`. Also verify in each pod:

```bash
rdma link
ip rule
ip route show table all
sysctl net.ipv4.conf.all.arp_announce net.ipv4.conf.all.rp_filter
```

Remove the dedicated test pods when done:

```bash
kubectl delete -f manifests/nvidia-network-operator/standalone/ibv-rc-pingpong.yaml
```

## Validation record: 2026-06-27 test cluster

The manifests in this directory were tested on an OKE Kubernetes 1.35.2
cluster with two `BM.GPU.B4.8` nodes. GPU Operator was already `ready`, its
driver deployment was disabled because the image supplied the host driver,
and its single NFD installation was healthy. The standalone Network Operator
did not deploy a duplicate NFD installation.

### Release and SBR binary verification

The official Helm repository exposed these relevant chart versions:

```text
26.4.0
26.1.1
26.1.0
25.10.0
```

The Linux amd64 `plugins:network-operator-v26.1.1` image had manifest digest
`sha256:8ec35312756be0c6329adf802d07654b953559185f9e9496ea62893761edcdd1`.
Its SBR binary identified itself as CNI plugins v1.9.1 and did not contain an
`AddSourceHints` field. Upstream added that field after v1.9.1 in commit
`025aca14c330f0fd5aa1fa9a7890b03bcd80f6f0` on 2026-04-13.

The published Linux amd64 `plugins:network-operator-v26.4.0` image had
manifest digest
`sha256:9d50e188ecba0071aaecee735770f75d6c9a76c3705f0731da3f7153b20b7e00`.
Its actual SBR binary contains both `AddSourceHints` and
`json:"addSourceHints,omitempty"`, even though its embedded build-version
string still says v1.9.1. This binary-level check, rather than the embedded
version string, is why 26.4.0 was selected.

The release-status caveat remains: NVIDIA's chart repository and product docs
publish 26.4.0 and call 26.4.x supported, while the public source repository
still lists only `v26.4.0-rc.1` as a pre-release on the test date.

### Operator installation and ownership proof

The Helm release installed successfully as revision 1 in
`nvidia-network-operator`. Before any node policy was created,
`SriovOperatorConfig/default` reported:

```text
configurationMode: daemon
disableDrain: false
disablePlugins: [mellanox]
```

The minimal `NicClusterPolicy` reached `ready`. Its applied-state list reported
Multus, container-networking plugins, and NV-IPAM as `ready`, while OFED, the
RDMA shared-device plugin, Network Operator NIC feature discovery, NIC
Configuration Operator, DTS, and Spectrum-X were `ignore`.

The generated `default/rdma-vf` NetworkAttachmentDefinition contained, in
order:

```text
sriov -> nv-ipam -> tuning -> rdma -> sbr(addSourceHints=true)
```

The B4 node policy explicitly reported `externallyManaged: false`. No
`vf-config` DaemonSet or privileged VF-creation init container was applied.
Config-daemon logs showed only the `generic` and `k8s` configuration plugins.
For every zero-to-one PF transition the generic plugin logged `no need drain`,
and the aggregate result was `drain-required=false,
reboot-required=false`.

The generic plugin also ran its normal kernel-argument cleanup path and tried
to remove managed SR-IOV kernel arguments that were not requested by this
policy, including `ib_core.netns_mode=0` and `ib_core.netns_mode=1`. The OCI
image does not rely on those command-line arguments: its persistent setting is
`options ib_core netns_mode=0` in `/etc/modprobe.d/ib_core.conf`. The live
setting remained exclusive, RDMA CNI isolation worked, and no reboot occurred.
The modprobe file and live value should still be included in post-install and
post-reboot checks.

### Managed VF creation without reboot

Both node states moved from `InProgress` to `Succeeded` between 08:12:12 and
08:12:35 UTC. Neither state had a `lastSyncError`, and no failure or
`Reboot_Required` state appeared.

| Node | Boot ID before | Boot ID after | PFs at `numVfs=1` | Capacity/allocatable |
|---|---|---|---:|---:|
| `10.140.69.103` | `7bb1f485-34bd-423e-abc3-40a053c53b50` | unchanged | 16/16 | 16/16 |
| `10.140.74.195` | `b1b3e833-24ff-4b25-8f2a-9ebcd9efb933` | unchanged | 16/16 | 16/16 |

Every selected PF still reported `sriov_totalvfs=127`, Ethernet link type, and
MTU 4220, while `sriov_numvfs` changed from 0 to 1. Both nodes were Ready and
uncordoned after reconciliation.

### CNI, routing, sysctl, and isolation proof

The requested VF NCCL workers each received 16 interfaces at MTU 4220 and 16
active RDMA links. Each VF source had an SBR rule and a dedicated table with a
connected route and default route. With `addSourceHints`, the main table also
retained VF-subnet routes carrying VF source addresses. For example, without
any manual `/32` route:

```text
192.168.0.17 dev net16 src 192.168.0.116
192.168.0.116 dev net16 src 192.168.0.17
```

When 16 interfaces all use the same `/16`, unbound destination routing selects
one of the equal-prefix VF routes; in this run it selected `net16`. Traffic
explicitly bound to a VF source continues to use that VF's SBR table. Therefore
`addSourceHints` fixes the original fallback to `eth0`, but it is not a
per-rail destination map for 16 identical subnets. Use a matching source
address/device for rail-specific diagnostics.

The dedicated ping-pong pods each requested one VF. RDMA CNI isolation was
visible directly: pod A saw only `mlx5_30/net1` and pod B saw only
`mlx5_26/net1`, despite each host having 16 PFs and VFs. The allocator happened
to select different mlx5 indices on the two hosts, and communication still
succeeded. Routes were:

```text
pod A: 192.168.0.117 dev net1 src 192.168.0.18
pod B: 192.168.0.18  dev net1 src 192.168.0.117
```

Both the `all` and `net1` values were verified for `arp_announce=2`,
`arp_filter=1`, `arp_ignore=1`, `rp_filter=0`, and `accept_local=1`. The
lightweight test pods needed an explicit `nvidia.com/gpu:NoSchedule`
toleration because they run on GPU nodes without requesting a GPU; that
toleration is included in `ibv-rc-pingpong.yaml`.

`ibv_rc_pingpong` used GID index 3 and completed without `ip route replace`:

```text
client: 6260.01 Mbit/sec, 10.47 usec/iter
server: 5866.09 Mbit/sec, 11.17 usec/iter
```

### NCCL result

The unmodified
`manifests/nccl-tests/kueue/virtual-functions/BM.GPU.B4.8.yaml` test completed
successfully with 16 ranks across the two B4 nodes. There were zero wrong or
out-of-bounds values. Results were:

| Size | Out-of-place bus bandwidth | In-place bus bandwidth |
|---:|---:|---:|
| 1 GiB | 186.84 GB/s | 186.69 GB/s |
| 2 GiB | 189.05 GB/s | 189.12 GB/s |
| 4 GiB | 190.24 GB/s | 190.29 GB/s |

Average bus bandwidth was 188.705 GB/s. The MPIJob reached `Succeeded`; the
first ping-pong attempt inside its workers raced the MPI controller's expected
worker cleanup after completion. The dedicated long-running pods removed that
lifecycle race and produced the successful result above.

The dedicated ping-pong pods are privileged only as a diagnostic workload so
they can mount `/dev/infiniband`, matching the existing NCCL test pattern. No
privileged pod or init container participates in PF/VF configuration.

After evidence collection, the ping-pong pods and the NCCL MPIJob, LocalQueue,
ClusterQueue, and test ResourceFlavor were deleted. The Network Operator Helm
release and managed VF configuration were intentionally left installed. Both
B4 nodes remained Ready and uncordoned with unchanged boot IDs, node states
`Succeeded`, and 16 capacity/16 allocatable `nvidia.com/rdma-vf` resources.

## Rollback

To return the PFs to zero VFs under the same owner, first change `numVfs` to
`0` in `sriov-network-node-policy.yaml`, apply it, and wait for node states to
return to `Succeeded`. Only then remove the network, pool, policy, and Helm
release. Do not introduce `vf-config` during rollback.
