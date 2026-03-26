package test

import (
	"strings"
	"testing"

	"github.com/gruntwork-io/terratest/modules/terraform"
	"github.com/stretchr/testify/require"
)

// TestPlanSmoke runs terraform plan without applying to validate configuration.
// This is a fast smoke test for CI pipelines.
func TestPlanSmoke(t *testing.T) {
	options := newTerraformOptions(t, nil)
	terraform.InitAndPlan(t, options)
}

func TestCoreProvisioning(t *testing.T) {
	options := newTerraformOptions(t, nil)

	defer terraform.Destroy(t, options)
	terraform.InitAndApply(t, options)

	// Validate state_id is present
	require.NotEmpty(t, terraform.Output(t, options, "state_id"))

	// Validate OCIDs have correct format
	clusterID := terraform.Output(t, options, "cluster_id")
	require.True(t, isValidOCID(clusterID), "cluster_id should be a valid OCID: %s", clusterID)

	vcnID := terraform.Output(t, options, "vcn_id")
	require.True(t, isValidOCID(vcnID), "vcn_id should be a valid OCID: %s", vcnID)

	controlPlaneSubnetID := terraform.Output(t, options, "control_plane_subnet_id")
	require.True(t, isValidOCID(controlPlaneSubnetID), "control_plane_subnet_id should be a valid OCID: %s", controlPlaneSubnetID)

	controlPlaneNsgID := terraform.Output(t, options, "control_plane_nsg_id")
	require.True(t, isValidOCID(controlPlaneNsgID), "control_plane_nsg_id should be a valid OCID: %s", controlPlaneNsgID)

	// Pod subnet/NSG only exist for VCN-Native Pod Networking (not Flannel)
	cniType := terraform.Output(t, options, "cni_type")
	if cniType == "npn" || cniType == "VCN-Native Pod Networking" {
		podSubnetID := terraform.Output(t, options, "pod_subnet_id")
		require.True(t, isValidOCID(podSubnetID), "pod_subnet_id should be a valid OCID: %s", podSubnetID)

		podNsgID := terraform.Output(t, options, "pod_nsg_id")
		require.True(t, isValidOCID(podNsgID), "pod_nsg_id should be a valid OCID: %s", podNsgID)
	}

	workerSubnetID := terraform.Output(t, options, "worker_subnet_id")
	require.True(t, isValidOCID(workerSubnetID), "worker_subnet_id should be a valid OCID: %s", workerSubnetID)

	workerNsgID := terraform.Output(t, options, "worker_nsg_id")
	require.True(t, isValidOCID(workerNsgID), "worker_nsg_id should be a valid OCID: %s", workerNsgID)

	// Ops worker pool is always created
	workerOpsPoolID := terraform.Output(t, options, "worker_ops_pool_id")
	require.True(t, isValidOCID(workerOpsPoolID), "worker_ops_pool_id should be a valid OCID: %s", workerOpsPoolID)

	// Internal LB subnet/NSG are always created
	intLbSubnetID := terraform.Output(t, options, "int_lb_subnet_id")
	require.True(t, isValidOCID(intLbSubnetID), "int_lb_subnet_id should be a valid OCID: %s", intLbSubnetID)

	intLbNsgID := terraform.Output(t, options, "int_lb_nsg_id")
	require.True(t, isValidOCID(intLbNsgID), "int_lb_nsg_id should be a valid OCID: %s", intLbNsgID)

	// Cluster private endpoint is always present and must be HTTPS
	clusterPrivateEndpoint := terraform.Output(t, options, "cluster_private_endpoint")
	require.NotEmpty(t, clusterPrivateEndpoint)
	require.True(t, strings.HasPrefix(clusterPrivateEndpoint, "https://"), "cluster_private_endpoint should start with https://: %s", clusterPrivateEndpoint)

	// Public endpoint only present when control plane is public
	clusterPublicEndpoint := terraform.Output(t, options, "cluster_public_endpoint")
	if clusterPublicEndpoint != "" {
		require.True(t, strings.HasPrefix(clusterPublicEndpoint, "https://"), "cluster_public_endpoint should start with https://: %s", clusterPublicEndpoint)
	}

	// Public LB subnet/NSG only present when public subnets are enabled
	pubLbSubnetID := terraform.Output(t, options, "pub_lb_subnet_id")
	if pubLbSubnetID != "" {
		require.True(t, isValidOCID(pubLbSubnetID), "pub_lb_subnet_id should be a valid OCID: %s", pubLbSubnetID)

		pubLbNsgID := terraform.Output(t, options, "pub_lb_nsg_id")
		require.True(t, isValidOCID(pubLbNsgID), "pub_lb_nsg_id should be a valid OCID: %s", pubLbNsgID)
	}

	// Bastion only present when create_bastion = true
	bastionID := terraform.Output(t, options, "bastion_id")
	if bastionID != "" {
		require.True(t, isValidOCID(bastionID), "bastion_id should be a valid OCID: %s", bastionID)
		require.NotEmpty(t, terraform.Output(t, options, "bastion_public_ip"), "bastion_public_ip should not be empty when bastion is created")
	}

	// Operator only present when create_operator = true
	operatorID := terraform.Output(t, options, "operator_id")
	if operatorID != "" {
		require.True(t, isValidOCID(operatorID), "operator_id should be a valid OCID: %s", operatorID)
	}
}
