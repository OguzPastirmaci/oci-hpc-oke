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

For a more targeted sysctl scope, apply
`sriov-network-targeted-sysctls.yaml` instead of `sriov-network.yaml`. Both
files define the same `rdma-vf` resource and are alternatives, not two
resources to install together:

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/sriov-network-targeted-sysctls.yaml
```

The targeted variant applies the ARP and `accept_local` settings only to the
VF represented by `IFNAME`, which Tuning CNI replaces with `net1`, `net2`, and
so on. It retains both `all.rp_filter=0` and `IFNAME.rp_filter=0` because Linux
uses the maximum of the global and interface values when performing reverse
path validation; setting only the interface value cannot override a nonzero
global value. This variant avoids applying the positive ARP and `accept_local`
values to the pod's primary `eth0` interface.

The targeted variant also omits `spoofChk`. Consequently, SR-IOV CNI does not
change the VF's existing spoof-check state during pod attachment. This is
intentional for testing; record the PF-reported VF spoof-check state before and
during allocation. The fully validated `sriov-network.yaml` explicitly used
`spoofChk: "off"`.

The first 2026-06-27 validation record below used the fully namespace-wide
`sriov-network.yaml`. A second 2026-06-27 record documents a separate cluster
validation that used only `sriov-network-targeted-sysctls.yaml`, including
baseline comparison, route, spoof-check, ping-pong, and NCCL checks.

Finally, let the SR-IOV Network Operator create the VFs. This is the only VF
creation step:

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/sriov-network-node-policy.yaml
```

## Dynamic Resource Allocation (DRA) variant

The steps above hand VFs to pods through the SR-IOV device plugin, which advertises
`nvidia.com/rdma-vf` as an extended resource. The SR-IOV Network Operator can instead
publish VFs through DRA, where pods claim devices with a `ResourceClaimTemplate`. The
two are alternatives; pick one before installing.

Install with `values-dra.yaml` instead of `values.yaml`. It is the same file plus the
`dynamicResourceAllocation` feature gate:

```bash
helm upgrade --install network-operator nvidia/network-operator \
  --namespace nvidia-network-operator \
  --create-namespace \
  --version 26.4.0 \
  --values manifests/nvidia-network-operator/standalone/values-dra.yaml \
  --wait \
  --timeout 15m
```

Confirm the gate reached the operator config:

```bash
kubectl -n nvidia-network-operator get sriovoperatorconfig default \
  -o jsonpath='{.spec.featureGates.dynamicResourceAllocation}{"\n"}'
```

Everything else in the install is unchanged: the same `NicClusterPolicy`, IPAM pool,
pool config, and `SriovNetwork`. Apply the node policy for the shape you are running.
`sriov-network-node-policy.yaml` covers `BM.GPU.B4.8`;
`sriov-network-node-policy-mi300x.yaml` covers `BM.GPU.MI300X.8`, which has 8 PFs
rather than 16.

### Verifying VFs under DRA

The allocatable-resource check in the next section does **not** apply. With the
feature gate on, the device plugin no longer advertises `nvidia.com/rdma-vf`, so
`.status.allocatable` has no VF entry and that command prints `<none>`. VFs appear as
ResourceSlices instead:

```bash
kubectl get resourceslices
```

Expect one slice per node, each holding one device per PF. The operator also
generates a `DeviceClass` named after the policy's `resourceName`, selecting on
`nvidia.com/<resourceName>`:

```bash
kubectl get deviceclass rdma-vf -o yaml
```

A `resourceName` mismatch between the node policy and the `SriovNetwork` is silent
until a pod starts, and then surfaces as `SRIOV-CNI failed to load netconf:
LoadConf(): VF pci addr is required`. Multus resolves the VF's PCI address by matching
the NAD's `k8s.v1.cni.cncf.io/resourceName` annotation against the claim's allocated
devices, so the two names must agree.

### Claiming VFs from a workload

Pods do not request `nvidia.com/rdma-vf` under DRA. They reference a
`ResourceClaimTemplate`, and one claim can carry every VF on the node:

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: rccl-sriov-vf-8
spec:
  spec:
    devices:
      requests:
      - name: vf
        exactly:
          deviceClassName: sriovnetwork.k8snetworkplumbingwg.io
          count: 8
          selectors:
          - cel:
              expression: device.attributes["k8s.cni.cncf.io"].resourceName == "nvidia.com/rdma-vf"
```

The pod keeps the usual Multus annotation, repeated once per VF, and adds
`resourceClaims` plus `resources.claims`. Each repeated attachment consumes a distinct
device from the claim, so eight entries yield eight VFs, one per PF, surfacing as
`net1` through `net8`.

`manifests/rccl-tests/dra/BM.GPU.MI300X.8.yaml` is a complete example. It allocates
all 8 VFs through DRA while GPUs still come from the AMD device plugin, and it drops
Kueue.

### AMD GPU nodes

The operator injects only an `nvidia.com/gpu` toleration into the Multus, CNI plugin,
and NV-IPAM DaemonSets. On AMD nodes tainted `amd.com/gpu`, those DaemonSets skip the
very nodes holding the VFs. The failure is quiet: the DaemonSets report healthy
because they run everywhere else, and pods reach `Running` with their network
annotation ignored and no secondary interface. `nic-cluster-policy.yaml` sets
`spec.tolerations` for both taint keys to prevent this. Workload pods need the
`amd.com/gpu` toleration too.

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

For a self-running point-to-point test, use the automated manifest. It places
the client and server on different B4 nodes, discovers their NV-IPAM addresses
and VF-associated mlx5 devices, verifies that each pod sees exactly one RDMA
link, checks the client route, runs `ibv_rc_pingpong`, and includes both client
and server output in the client log:

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/ibv-rc-pingpong-automated.yaml
kubectl wait \
  --for=jsonpath='{.status.phase}'=Succeeded \
  pod/rdma-vf-ping-auto-a \
  --timeout=10m
kubectl logs pod/rdma-vf-ping-auto-a
```

The last line must be:

```text
PASS: ibv_rc_pingpong completed over isolated VFs without a manual route
```

The automated test uses a namespace-scoped ServiceAccount and Role. The Role
can only get and patch the named server pod and read that pod's log. The
server publishes its dynamically allocated VF IP as a temporary annotation on
its own pod; the client reads it directly through the Kubernetes API. It does
not grant `pods/exec`, use a launcher image, or require hard-coded VF
addresses. Delete the bundle before rerunning it because the endpoint pods
have fixed names:

```bash
kubectl delete -f manifests/nvidia-network-operator/standalone/ibv-rc-pingpong-automated.yaml
```

To exercise every VF concurrently, apply the automated `ib_write_bw` bundle.
Each endpoint requests all 16 `nvidia.com/rdma-vf` resources. The test maps
`net1` through `net16` to their RDMA devices, pairs matching interfaces across
the nodes, verifies each source-specific SBR route, and starts 16 concurrent
tests on ports 18515 through 18530:

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/ib-write-bw-all-vfs-automated.yaml
kubectl wait \
  --for=jsonpath='{.status.phase}'=Succeeded \
  pod/rdma-vf-write-bw-client \
  --timeout=10m
kubectl logs pod/rdma-vf-write-bw-client
```

The test uses the repo's established
`oguzpastirmaci/mofed-perftest:5.4-3.6.8.1-ubuntu20.04-amd64` image because the
CUDA 13.1 NCCL image used by the MPIJob contains `ibv_rc_pingpong` but does not
contain an `ib_write_bw` binary or installed `perftest` package. Each rail runs
for ten seconds with GID index 3, RDMA CM, traffic class 41, four QPs, and a
64-KiB message size. A passing log ends with a bandwidth summary followed by:

```text
PASS: ib_write_bw completed concurrently across all 16 isolated VF pairs
```

Remove the fixed-name resources before another run:

```bash
kubectl delete -f manifests/nvidia-network-operator/standalone/ib-write-bw-all-vfs-automated.yaml
```

For manual diagnostics, the original manifest remains available unchanged. It
deploys two long-running single-VF pods, and required pod anti-affinity places
them on different B4 nodes:

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

## Validation record: 2026-06-27 targeted-sysctls test cluster

This was a separate OKE Kubernetes 1.35.2 cluster with two `BM.GPU.B4.8`
nodes. This run applied only
`sriov-network-targeted-sysctls.yaml`; it did not apply or transition through
the namespace-wide `sriov-network.yaml` configuration.

### OCA and PF gate

The Oracle Cloud Agent HPC configuration was checked before installing the
Network Operator. Both nodes had exactly one completion marker and no log
lines matching error, fatal, failed, failure, or traceback:

| Node | OCA completion marker | Error matches |
|---|---|---:|
| `10.140.77.117` | `Fully Configuredoci-hpc-mlx-configure1.0.0` at 19:00:01 UTC | 0 |
| `10.140.79.236` | `Fully Configuredoci-hpc-mlx-configure1.0.0` at 18:59:58 UTC | 0 |

On each node, all 16 selected PFs were `15b3:1019`, bound to `mlx5_core`, UP,
MTU 4220, and paired with an active mlx5 RDMA device. Every PF exposed
`sriov_totalvfs=127` and began with `sriov_numvfs=0`. The live
`ib_core.netns_mode` value was exclusive (`N`, corresponding to
`netns_mode=0`), and `/etc/modprobe.d/ib_core.conf` contained
`options ib_core netns_mode=0`.

### Targeted network and operator ownership

The standalone Network Operator 26.4.0 Helm release was installed with NFD
disabled because the GPU Operator already owned the single cluster NFD
deployment. Before creating a node policy, the safety settings were:

```text
configurationMode: daemon
disableDrain: false
disablePlugins: [mellanox]
node policies: none
```

The minimal `NicClusterPolicy` reached `ready`. Only Multus, container CNI
plugins, and NV-IPAM were enabled; OFED, the shared RDMA device plugin, NIC
Configuration Operator, and other optional components remained ignored. No
`vf-config` resource or privileged VF-creation init container was used.

The generated `default/rdma-vf` NAD contained this chain:

```text
sriov -> nv-ipam -> tuning(targeted sysctls) -> rdma -> sbr(addSourceHints=true)
```

Its tuning configuration contained only:

```text
net.ipv4.conf.all.rp_filter=0
net.ipv4.conf.IFNAME.arp_announce=2
net.ipv4.conf.IFNAME.arp_filter=1
net.ipv4.conf.IFNAME.arp_ignore=1
net.ipv4.conf.IFNAME.rp_filter=0
net.ipv4.conf.IFNAME.accept_local=1
```

The generated NAD had zero `spoofchk` occurrences. Therefore this test did not
ask SR-IOV CNI to change spoof checking.

### VF creation without drain or reboot

The generic SR-IOV plugin changed every selected PF from zero to one runtime
VF. Config-daemon logs reported `no need drain` for all 16 addresses on both
nodes and aggregated `drain-required=false, reboot-required=false`. Both node
states reached `Succeeded` with an empty `lastSyncError`.

| Node | Boot ID before and after | Final PFs at `numVfs=1` | Allocatable VFs |
|---|---|---:|---:|
| `10.140.77.117` | `3d5f3d39-bdc6-4908-a268-29cc0badf265` (unchanged) | 16/16 | 16 |
| `10.140.79.236` | `2c9f3840-8579-44da-9258-a49390f978a2` (unchanged) | 16/16 | 16 |

The nodes remained Ready and schedulable. Firmware capacity stayed 127; only
the runtime VF count changed.

### Targeted sysctl comparison

A primary-network-only pod using the same image established the OCI CNI
baseline. A one-VF pod then showed the targeted values below:

| Scope | Primary-only baseline | With targeted `rdma-vf` |
|---|---|---|
| `all` ARP announce/filter/ignore | `0/0/0` | `0/0/0` |
| `all.rp_filter` | `0` | `0` |
| `all.accept_local` | `1` | `1` |
| `eth0` ARP announce/filter/ignore | `0/0/0` | `0/0/0` |
| `eth0.rp_filter` / `accept_local` | `0/0` | `0/0` |
| `net1` ARP announce/filter/ignore | not present | `2/1/1` |
| `net1.rp_filter` / `accept_local` | not present | `0/1` |

This proves that the positive ARP and `accept_local` values were not copied to
`all` or `eth0`. The observed global `accept_local=1` came from the OCI primary
network baseline, not from the targeted NAD. The global `rp_filter=0` was both
the baseline value and an explicit targeted-manifest requirement.

Each ping-pong pod saw exactly one active RDMA link, proving RDMA CNI namespace
isolation. SBR source hints installed the VF subnet in the main table with the
VF source address. No manual route was added:

```text
pod A: 192.168.0.101 dev net1 src 192.168.0.201
pod B: 192.168.0.201 dev net1 src 192.168.0.101
```

ICMP had zero loss. `ibv_rc_pingpong -g 3` completed 1,000 iterations:

```text
client: 6201.95 Mbit/sec, 10.57 usec/iter
server: 5943.23 Mbit/sec, 11.03 usec/iter
```

The initial server/client automation attempt was invalid because the
background server retained the `kubectl exec` stream until its timeout, so the
client started only after the server exited. Running server and client in
separate concurrent exec sessions produced the successful result above.

### Omitted spoofChk result

A representative VF reported `spoof checking off` immediately after VF
creation and before allocation. The two VFs selected by the ping-pong pods
remained off while attached. After all tests, every one of the 16 VFs on both
nodes reported `spoof checking off`.

Omitting `spoofChk` therefore worked on this cluster and did not toggle the
existing/default state. It does not enforce a portable desired state: a host
whose VF starts with spoof checking on would retain that state. Keep the field
omitted only when preserving host/default behavior is intentional.

### Targeted-manifest NCCL result

The unmodified `BM.GPU.B4.8.yaml` Kueue MPIJob allocated all 16 VFs and all
eight GPUs on each node, ran 16 ranks, and reached `Succeeded`. It reported
zero wrong or out-of-bounds values:

| Size | Out-of-place bus bandwidth | In-place bus bandwidth |
|---:|---:|---:|
| 1 GiB | 187.43 GB/s | 187.15 GB/s |
| 2 GiB | 188.96 GB/s | 189.07 GB/s |
| 4 GiB | 190.21 GB/s | 190.21 GB/s |

Average bus bandwidth was 188.84 GB/s. The launcher initially appeared
Pending only while pulling the 5.2 GB test image; it then ran and exited zero.

### Automated diagnostic manifest results

The separate `ibv-rc-pingpong-automated.yaml` bundle was applied after the
targeted-manifest validation. It discovered dynamic NV-IPAM addresses and
mlx5 devices without manual `kubectl exec`, confirmed one RDMA link per pod,
printed the direct `net1` source route, and included both server and client
results in the client log. Both pods exited zero and the final line was PASS:

```text
client: 6087.88 Mbit/sec, 10.77 usec/iter
server: 5751.29 Mbit/sec, 11.40 usec/iter
```

The original long-running `ibv-rc-pingpong.yaml` was not changed. The new
automated bundle has its own resource names and narrowly scoped RBAC. Its
ServiceAccount cannot list pods, access the client pod through the API, or use
`pods/exec`; it can only get/patch the named server pod and read its log.

The separate `ib-write-bw-all-vfs-automated.yaml` bundle then allocated 16 VFs
to each endpoint and ran all 16 matching VF pairs concurrently. Every
secondary attachment was present, all client source lookups selected the
matching interface and SBR table (`net1`/table 100 through `net16`/table 115),
and all 32 client/server perftest processes exited zero. A repeat run of the
final manifest reported:

```text
Client bandwidth summary: rails=16 min=6.12 max=6.15 aggregate=98.00 Gbit/sec
PASS: ib_write_bw completed concurrently across all 16 isolated VF pairs
```

Both all-VF endpoint pods reached `Succeeded`. The same limited annotation and
pod-log RBAC pattern was used; there was no `pods/exec` permission and no
manual route addition.

The ping-pong pods, primary-network baseline pod, MPIJob, LocalQueue,
ClusterQueue, and test ResourceFlavor were removed. The Network Operator and
managed VF configuration were intentionally left installed. All Network
Operator pods were Running with zero restarts; both node states remained
`Succeeded`, both boot IDs were unchanged, all 32 PFs remained at
`sriov_numvfs=1`, and both nodes retained 16 allocatable VFs.

## Validation record: 2026-07-28 MI300X DRA test cluster

This run validated the DRA variant on an OKE Kubernetes 1.35.2 cluster with two
`BM.GPU.MI300X.8` nodes (`10.140.76.70`, `10.140.78.175`) and three operational
nodes, running Ubuntu 24.04.4, kernel `6.8.0-1057-oracle`, and CRI-O 1.35.2. The
AMD GPU device plugin was already installed and each GPU node advertised
`amd.com/gpu: 8`.

Unlike the two B4.8 records above, this was a teardown-and-reinstall: the cluster
already had a working but hand-patched configuration, which was fully removed
before reinstalling from these manifests only. The point was to prove the files
reproduce a working setup without manual intervention.

### Teardown

Rollback followed the procedure in the next section. The node policy was patched to
`numVfs: 0` first; both node states returned to `Succeeded` and all ResourceSlices
disappeared. All 8 PFs on each node reported `sriov_numvfs=0`. The SriovNetwork,
IPPool, pool config, node policy, and NicClusterPolicy were then deleted and the
Helm release uninstalled. Afterwards the `nvidia-network-operator` namespace had no
pods, no NetworkAttachmentDefinition remained, and both nodes stayed Ready and
uncordoned.

### Install and ownership proof

The 26.4.0 chart was installed with `values-dra.yaml` as revision 1. Before any node
policy existed, `SriovOperatorConfig/default` reported:

```text
configurationMode: daemon
disableDrain: false
disablePlugins: [mellanox]
featureGates.dynamicResourceAllocation: true
```

Only the GPU Operator's NFD deployment was present; the Network Operator did not
install a second one. The minimal `NicClusterPolicy` reached `ready` with Multus,
container-networking plugins, and NV-IPAM `ready`, and OFED, both device plugins,
NIC feature discovery, NIC Configuration Operator, DTS, and Spectrum-X `ignore`.

The `sriov-dra-driver` DaemonSet sits at `Init:0/1` until a node policy exists. Its
`wait-for-config` init container annotates its own pod with
`sriovnetwork.openshift.io/device-plugin-wait-config` and waits for the config daemon
to clear it. This is expected between the NicClusterPolicy and node policy steps, not
a failure. It reached 2/2 immediately after the node policy was applied.

### DaemonSet placement on AMD nodes

This is the regression check for the `amd.com/gpu` toleration in
`nic-cluster-policy.yaml`. On a clean install all three DaemonSets covered every node:

| DaemonSet | Desired | Ready |
|---|---:|---:|
| `kube-multus-ds` | 5 | 5 |
| `cni-plugins-ds` | 5 | 5 |
| `nv-ipam-node` | 5 | 5 |

Without the toleration these report 3/3 and are absent from the two GPU nodes, which
looks healthy but leaves pods on those nodes with no Multus and therefore no
secondary interface.

### VF creation without reboot

Both node states reached `Succeeded` with an empty `lastSyncError`. No
`Reboot_Required` state appeared.

| Node | Boot ID before and after | PFs at `numVfs=1` | ResourceSlice devices |
|---|---|---:|---:|
| `10.140.76.70` | `4fd5f539-1688-4d1a-b277-7bdee144ab70` (unchanged) | 8/8 | 8 |
| `10.140.78.175` | `d81d415d-cb13-4379-8778-624672754d3c` (unchanged) | 8/8 | 8 |

Every selected PF reported `sriov_totalvfs=127` and MTU 4220 while `sriov_numvfs`
changed from 0 to 1. Firmware capacity was unchanged; only the runtime VF count moved.
Both nodes stayed Ready and uncordoned.

### DRA object verification

Each node published one ResourceSlice holding 8 devices, one per PF `rdma0` through
`rdma7`, all carrying `k8s.cni.cncf.io/resourceName: nvidia.com/rdma-vf`. Node
allocatable contained no VF resource, as expected with the feature gate on.

The operator generated `DeviceClass/rdma-vf`:

```text
device.driver == "sriovnetwork.k8snetworkplumbingwg.io" &&
  device.attributes["k8s.cni.cncf.io"].resourceName == "nvidia.com/rdma-vf"
```

The generated `default/rdma-vf` NAD carried
`k8s.v1.cni.cncf.io/resourceName: nvidia.com/rdma-vf` and this chain, with NV-IPAM
configured as the SR-IOV plugin's IPAM against pool `sriov-pool`:

```text
sriov -> nv-ipam -> tuning(targeted sysctls) -> rdma -> sbr(addSourceHints=true)
```

A pod requesting a `count: 8` claim plus eight repeated `rdma-vf` attachments received
eight distinct VFs, one per PF, visible as `mlx5_10` through `mlx5_17` and `net1`
through `net8`. Each repeated attachment consumed a different device from the claim.

### resourceName mismatch

This run caught a defect in the first version of
`manifests/rccl-tests/dra/BM.GPU.MI300X.8.yaml`, which selected
`nvidia.com/sriov-rdma-vf`. That name came from a hand-applied node policy on the
original cluster, not from this repository. A clean deploy produces
`nvidia.com/rdma-vf`.

The mismatch is silent until a pod starts. Scheduling succeeds and the claim is
allocated, then the sandbox fails repeatedly with:

```text
SRIOV-CNI failed to load netconf: LoadConf(): VF pci addr is required
```

Multus resolves the VF PCI address by matching the NAD's
`k8s.v1.cni.cncf.io/resourceName` annotation against the claim's allocated devices, so
`SriovNetwork.spec.resourceName` and `SriovNetworkNodePolicy.spec.resourceName` must
agree. Correcting the selector resolved it.

### RCCL result

`manifests/rccl-tests/dra/BM.GPU.MI300X.8.yaml` allocated all 8 VFs through DRA and
all 8 GPUs through the AMD device plugin on each node, ran 16 ranks, and reached
`Succeeded` with zero wrong and zero out-of-bounds values:

| Size | Out-of-place bus bandwidth | In-place bus bandwidth |
|---:|---:|---:|
| 1 GiB | 351.11 GB/s | 350.92 GB/s |
| 2 GiB | 353.58 GB/s | 353.56 GB/s |
| 4 GiB | 355.30 GB/s | 355.39 GB/s |
| 8 GiB | 356.79 GB/s | 356.67 GB/s |
| 16 GiB | 359.70 GB/s | 359.78 GB/s |

The launcher ran on an operational node and needed no toleration. Worker pods required
the `amd.com/gpu` toleration, which the reference manifests for untainted clusters do
not carry.

Workers logged `NCCL WARN Missing "iommu=pt" from kernel command line`. It did not
prevent the run, but it is a node image concern worth tracking separately.

After the test the Network Operator and managed VF configuration were intentionally
left installed. All Network Operator pods were Running with zero restarts, both node
states remained `Succeeded`, both boot IDs were unchanged, all 16 PFs across the two
nodes stayed at `sriov_numvfs=1`, and both ResourceSlices remained published.

### Not covered by this run

`ibv-rc-pingpong-automated.yaml` and `ib-write-bw-all-vfs-automated.yaml` were not
run. Both request `nvidia.com/rdma-vf` as an extended resource, which does not exist
under DRA, and both assume 16 VFs. They need DRA claims and an 8-VF variant before
they can run on MI300X. Consequently this record contains no per-rail SBR route
check, sysctl comparison, or spoof-check observation; the RCCL result is the only
functional evidence.

## Rollback

To return the PFs to zero VFs under the same owner, first change `numVfs` to
`0` in `sriov-network-node-policy.yaml`, apply it, and wait for node states to
return to `Succeeded`. Only then remove the network, pool, policy, and Helm
release. Do not introduce `vf-config` during rollback.
