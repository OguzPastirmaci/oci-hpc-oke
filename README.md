# Running RDMA GPU Workloads on OKE with NVIDIA GPU Operator and Network Operator

> [!IMPORTANT]
> Using SR-IOV Virtual Functions (VFs) for RDMA is not currently a supported OKE configuration. Use this guide only for experiments and testing.

## Overview

This guide configures RDMA VFs on Oracle Kubernetes Engine (OKE) GPU nodes with an existing NVIDIA GPU Operator and a standalone NVIDIA Network Operator. The Network Operator creates one VF on each selected physical function (PF), advertises the VFs as Kubernetes resources, and provides the SR-IOV, NV-IPAM, Tuning, RDMA, and source-based routing CNI chain used by GPU workloads.

The example is scoped to `BM.GPU.B4.8`. The OCI image provisioning flow and Oracle Cloud Agent own the PF firmware, host driver, QoS, and RDMA host preparation. The SR-IOV Network Operator owns only the runtime VF count.

For the ownership model, preflight checks, test evidence, and rollback procedure, see the [standalone NVIDIA Network Operator reference](./manifests/nvidia-network-operator/standalone/README.md).

## Prerequisites

- An OKE cluster with an operational node pool and at least two `BM.GPU.B4.8` nodes
- `kubectl` configured with cluster administrator permissions
- An existing NVIDIA GPU Operator deployment with Node Feature Discovery (NFD)
- Helm 3 on the system used to manage the cluster
- GPU nodes built from a specialized image listed below

## Node Images

### GPU Node Requirements

The GPU image must provide the NVIDIA GPU and networking drivers, RDMA packages, OCI HPC QoS configuration, and exclusive RDMA namespace mode. Verify that `/etc/modprobe.d/ib_core.conf` contains:

```text
options ib_core netns_mode=0
```

Do not enable the Network Operator OFED driver or Mellanox firmware plugin for this workflow. The host image and Oracle Cloud Agent already own those components.

## Deployment Steps

Run these commands from the repository root on the OKE operator host or another system with cluster access.

### 1. Verify Cluster Nodes

Wait for the `BM.GPU.B4.8` nodes to reach `Ready`:

```bash
kubectl get nodes -l node.kubernetes.io/instance-type=BM.GPU.B4.8
```

Capture each node boot ID before creating VFs. You will compare these values after the Network Operator reconciles the node policy.

```bash
kubectl get nodes -l node.kubernetes.io/instance-type=BM.GPU.B4.8 -o name \
  | while read -r node; do
      printf '%s ' "$node"
      kubectl get "$node" -o jsonpath='{.status.nodeInfo.bootID}{"\n"}'
    done
```

Before continuing, verify the PF and host requirements in the [PF preflight section](./manifests/nvidia-network-operator/standalone/README.md#pf-preflight).

### 2. Verify Helm 3

The OKE operator host includes Helm. Verify it before installing the Network Operator:

```bash
helm version --short
```

### 3. Add the NVIDIA Helm Repository

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
```

### 4. Verify NVIDIA GPU Operator and NFD

The GPU Operator must already be ready, and the cluster must have only one NFD deployment:

```bash
kubectl get pods -n gpu-operator
kubectl get deployments -A | grep node-feature-discovery
```

### 5. Deploy NVIDIA Network Operator

Install the standalone Network Operator with its NFD deployment disabled. Version `26.4.0` is the version tested by this branch.

```bash
helm upgrade --install network-operator nvidia/network-operator \
  --namespace nvidia-network-operator \
  --create-namespace \
  --version 26.4.1 \
  --values manifests/nvidia-network-operator/standalone/values.yaml \
  --wait \
  --timeout 15m
```

Verify the SR-IOV safety settings before creating a node policy:

```bash
kubectl -n nvidia-network-operator get sriovoperatorconfig default \
  -o jsonpath='{.spec.configurationMode}{" "}{.spec.disableDrain}{" "}{.spec.disablePlugins}{"\n"}'
```

The expected output is:

```text
daemon false ["mellanox"]
```

### 6. Configure the NIC Cluster Policy

Apply the minimal policy that deploys Multus, NVIDIA IPAM, and the required CNI plugins. It does not deploy an OFED driver, RDMA shared device plugin, or GPU components.

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/nic-cluster-policy.yaml
kubectl wait --for=jsonpath='{.status.state}'=ready \
  nicclusterpolicy/nic-cluster-policy \
  --timeout=15m
```

### 7. Create the IP Pool and SR-IOV Network

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/nv-ipam-ip-pool.yaml
kubectl apply -f manifests/nvidia-network-operator/standalone/sriov-network-targeted-sysctls.yaml
```

The targeted network applies ARP and `accept_local` settings only to the VF interfaces. It also leaves the existing VF spoof-check state unchanged. Use the namespace-wide [`sriov-network.yaml`](./manifests/nvidia-network-operator/standalone/sriov-network.yaml) only when you specifically need its global sysctls and explicit `spoofChk: "off"` setting. Do not apply both network files.

## SR-IOV Configuration

### 8. Configure Node Drain Behavior

Limit Network Operator reconciliation to one `BM.GPU.B4.8` node at a time:

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/sriov-network-pool-config.yaml
```

### 9. Create Virtual Functions

Apply the node policy. The SR-IOV Network Operator creates one VF on each of the 16 selected PFs per node.

> [!WARNING]
> Do not deploy `manifests/vf-config/vf-config.yaml`. This workflow sets `externallyManaged: false`, so the SR-IOV Network Operator must be the only owner of `sriov_numvfs`.

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/sriov-network-node-policy.yaml
```

## Verification

### 10. Verify VF Creation Without Reboot

Watch the node states until both `BM.GPU.B4.8` nodes report `Succeeded`:

```bash
kubectl -n nvidia-network-operator get sriovnetworknodestates -w
```

Stop and investigate if a state reports `Reboot_Required` or a non-empty `lastSyncError`. Re-run the boot ID command from step 1 and confirm that every boot ID is unchanged.

### 11. Verify VF Allocation

Each `BM.GPU.B4.8` node should expose 16 allocatable VFs:

```bash
kubectl get nodes -l node.kubernetes.io/instance-type=BM.GPU.B4.8 \
  -o custom-columns='NODE:.metadata.name,RDMA-VFS:.status.allocatable.nvidia\.com/rdma-vf'
```

Expected shape:

```text
NODE            RDMA-VFS
10.140.69.103   16
10.140.74.195   16
```

The generated `default/rdma-vf` NetworkAttachmentDefinition should contain this plugin order:

```text
sriov -> nv-ipam -> tuning -> rdma -> sbr
```

### 12. Use RDMA VFs in Pod Manifests

Add the `rdma-vf` network annotation and request the matching extended resource. Each comma-separated `rdma-vf` annotation entry requests one VF.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: rdma-test
  annotations:
    k8s.v1.cni.cncf.io/networks: rdma-vf
spec:
  containers:
    - name: rdma-test
      image: your-image
      resources:
        limits:
          nvidia.com/rdma-vf: 1
```

For a complete two-node example, see [`ibv-rc-pingpong.yaml`](./manifests/nvidia-network-operator/standalone/ibv-rc-pingpong.yaml).

## Running NCCL Tests (Optional)

### 13. Verify Kueue and MPI Operator

Kueue and MPI Operator are deployed by default by stack version 26.3.0 and later. Verify both deployments:

```bash
kubectl get deployments -n kueue-system
kubectl get deployments -n mpi-operator
```

### 14. Run the NCCL Test

> [!IMPORTANT]
> The test manifest is specific to `BM.GPU.B4.8`. Verify that the CUDA major version in the test image matches the CUDA major version installed on the nodes.

Apply the existing two-node VF test. Each worker requests eight GPUs and all 16 `nvidia.com/rdma-vf` resources on its node.

```bash
kubectl apply -f manifests/nccl-tests/kueue/virtual-functions/BM.GPU.B4.8.yaml
kubectl get mpijob,pods -w
```

### 15. Monitor the NCCL Test

The initial image pull can take several minutes. After the launcher starts, follow its logs:

```bash
kubectl logs -f job/nccl-test-launcher
```

A successful run reaches `Succeeded` and reports zero wrong or out-of-bounds values. The detailed validation record is in the [standalone Network Operator reference](./manifests/nvidia-network-operator/standalone/README.md#validation-record-2026-06-27-test-cluster).

## Additional VF Diagnostics

Run the automated one-VF `ibv_rc_pingpong` test:

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

Run the automated test across all 16 VFs on each node:

```bash
kubectl apply -f manifests/nvidia-network-operator/standalone/ib-write-bw-all-vfs-automated.yaml
kubectl wait \
  --for=jsonpath='{.status.phase}'=Succeeded \
  pod/rdma-vf-write-bw-client \
  --timeout=10m
kubectl logs pod/rdma-vf-write-bw-client
```

The last line must be:

```text
PASS: ib_write_bw completed concurrently across all 16 isolated VF pairs
```

## Guides

- [Accessing a Private OKE Cluster via OCI Bastion Service](./docs/accessing-private-oke-cluster-via-oci-bastion-service.md)
- [Adding SSH keys to worker nodes](./docs/adding-ssh-keys-to-worker-nodes.md)
- [Deploying the Monitoring Stack manually](./docs/deploying-monitoring-stack-manually.md)
- [Deploying the Slurm Operator Full Suite](./docs/deploying-slurm-operator-full-suite.md)
- [Importing Container Images from OCI File Storage Service Using Skopeo](./docs/importing-images-from-fss-skopeo.md)
- [OCI HPC OKE Utils (Node Labeler, Image Prepuller, Hostexec)](./docs/oci-hpc-oke-utils.md)
- [Replacing the boot volume of self-managed nodes and managed node pools using the Boot Volume Replacement (BVR) script](./docs/replacing-the-boot-volume-of-self-managed-nodes.md)
- [Running GPU & RDMA active health checks](./docs/running-active-health-checks.md)
- [Running GPU & RDMA passive health checks](./docs/running-gpu-rdma-healthchecks-with-node-problem-detector.md)
- [Running ib_write_bw Tests Between Nodes](./docs/running-ib-write-bw-test.md)
- [Running NCCL and RCCL Tests from Slurm](./docs/running-nccl-rccl-tests-from-slurm-operator.md)
- [Running PyTorch Jobs on OKE Using Host Network with RDMA](./docs/running-pytorch-jobs-on-oke-using-hostnetwork-with-rdma.md)
- [Upgrading OKE clusters](./docs/oke-hpc-upgrade.md)
- [Using Cluster Autoscaler with Cluster Networks](./docs/using-cluster-autoscaler-with-cluster-networks.md)
- [Using Dynamic Resource Allocation (DRA) for Multi-Node NVLink](./docs/using-dynamic-resource-allocation-for-multi-node-nvlink-imex.md)
- [Using RDMA Network Locality When Running Workloads on OKE](./docs/using-rdma-network-locality-when-running-workloads-on-oke.md)
- [CVE-2026-31431 ("Copy Fail")](./docs/copy-fail.md)
- [Dirty Frag: CVE-2026-43284, CVE-2026-43500](./docs/dirty-frag.md)

## Contributing

This project welcomes contributions from the community. Before submitting a pull request, please [review our contribution guide](./CONTRIBUTING.md).

## Security

Please consult the [security guide](./SECURITY.md) for our responsible security vulnerability disclosure process.
