module "oke" {
  source  = "oracle-terraform-modules/oke/oci"
  version = "5.0.3"

  # Provider
  providers           = { oci.home = oci.home }
  config_file_profile = var.config_file_profile
  home_region         = var.home_region
  region              = var.region
  tenancy_id          = var.tenancy_id
  compartment_id      = var.compartment_id
  ssh_public_key_path = var.ssh_public_key_path
  ssh_private_key_path = var.ssh_private_key_path
  
  kubernetes_version = var.kubernetes_version
  cluster_type = var.cluster_type
  cluster_name         = var.cluster_name
  bastion_allowed_cidrs = ["0.0.0.0/0"]
  allow_worker_ssh_access     = true
  control_plane_allowed_cidrs = ["0.0.0.0/0"]

  control_plane_is_public = true

  sriov_cni_plugin_install       = true
  sriov_cni_plugin_namespace     = "kube-system"
  
  rdma_cni_plugin_install       = true
  rdma_cni_plugin_namespace     = "kube-system"
  
  # Resource creation
  assign_dns           = false
  create_vcn           = true
  create_bastion       = true
  create_cluster       = true
  create_operator      = true
  create_iam_resources = false
  use_defined_tags     = false

  worker_pools = {
    ubuntu_a100 = {
      description = "GPU pool", enabled = true,
      disable_default_cloud_init=true,
      mode        = "cluster-network",
      size = 2,
      shape = var.a100_shape,
      boot_volume_size = 250,
      placement_ads = [1],
      image_type = "custom",
      image_id = var.ubuntu_a100_image,
      node_labels = { "oci.oraclecloud.com/disable-gpu-device-plugin" : "true" },
      cloud_init = [{ content = "./cloud-init/ubuntu_gpu.sh" }],
      agent_config = {
       are_all_plugins_disabled = false,
       is_management_disabled   = false,
       is_monitoring_disabled   = false,
       plugins_config = {
         "Compute HPC RDMA Authentication"     = "ENABLED",
         "Compute HPC RDMA Auto-Configuration" = "ENABLED",
         "Compute Instance Monitoring"         = "ENABLED",
         "Compute Instance Run Command"        = "ENABLED",
         "Compute RDMA GPU Monitoring"         = "DISABLED",
         "Custom Logs Monitoring"              = "ENABLED",
         "Management Agent"                    = "ENABLED",
         "Oracle Autonomous Linux"             = "DISABLED",
         "OS Management Service Agent"         = "DISABLED",
       }
     },
    }
    ubuntu-system = {
      description = "CPU pool", enabled = true,
      disable_default_cloud_init=true,
      mode        = "node-pool",
      boot_volume_size = 150,
      shape = "VM.Standard.E4.Flex",
      ocpus = 8,
      memory = 64,
      size = 2,
      image_type = "custom",
      image_id = var.ubuntu_system_pool_image,
      cloud_init = [{ content = "./cloud-init/ubuntu_system.sh" }],
    }
  }
}