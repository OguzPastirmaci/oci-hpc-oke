package test

import (
	"os"
	"regexp"
	"testing"

	"github.com/gruntwork-io/terratest/modules/k8s"
	"github.com/gruntwork-io/terratest/modules/terraform"
	"github.com/stretchr/testify/require"
)

func TestStorageLustre(t *testing.T) {
	skipUnlessEnv(t, "RUN_LUSTRE_TESTS")

	vars := map[string]interface{}{
		"create_lustre": true,
	}
	// Optional: override AD if LUSTRE_AD is set, otherwise falls back to worker_ops_ad
	if lustreAD := envOrDefault([]string{"LUSTRE_AD", "OCI_LUSTRE_AD", "TF_VAR_lustre_ad"}, ""); lustreAD != "" {
		vars["lustre_ad"] = lustreAD
	}
	options := newTerraformOptions(t, vars)

	defer terraform.Destroy(t, options)
	terraform.InitAndApply(t, options)

	// State assertions
	resources := terraformStateList(t, options)
	requireStateHasPrefix(t, resources, "oci_lustre_file_storage_lustre_file_system.lustre")
	requireStateHasPrefix(t, resources, "oci_core_network_security_group.lustre")
	requireStateHasPrefix(t, resources, "oci_core_subnet.lustre_subnet")
	requireStateHasPrefix(t, resources, "kubectl_manifest.lustre_pv")

	// Attribute validation
	fsID := terraform.Output(t, options, "lustre_file_system_id")
	require.True(t, isValidOCID(fsID), "lustre_file_system_id should be a valid OCID: %s", fsID)

	mgsAddr := terraform.Output(t, options, "lustre_management_service_address")
	require.Regexp(t, regexp.MustCompile(`^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$`), mgsAddr,
		"lustre_management_service_address should be a valid IPv4: %s", mgsAddr)

	nsgID := terraform.Output(t, options, "lustre_nsg_id")
	require.True(t, isValidOCID(nsgID), "lustre_nsg_id should be a valid OCID: %s", nsgID)

	subnetID := terraform.Output(t, options, "lustre_subnet_id")
	require.True(t, isValidOCID(subnetID), "lustre_subnet_id should be a valid OCID: %s", subnetID)

	// Kubernetes tests — gated on public endpoint (same pattern as storage_fss_test.go)
	publicEndpoint := optionalOutput(t, options, "cluster_public_endpoint")
	if publicEndpoint == "" {
		t.Log("Skipping Kubernetes Lustre tests: no public endpoint")
		return
	}
	clusterID := terraform.Output(t, options, "cluster_id")
	region := os.Getenv("OCI_REGION")
	kubeconfigPath := generateKubeconfig(t, clusterID, region)
	testLustreKubernetes(t, kubeconfigPath)
}

// testLustreKubernetes verifies PVC binding and shared filesystem write/read.
// Both tests share one PVC because lustre-pv uses the Retain reclaim policy —
// after a PVC is deleted the PV enters Released state and cannot be rebound.
func testLustreKubernetes(t *testing.T, kubeconfigPath string) {
	t.Helper()
	opts := k8s.NewKubectlOptions("", kubeconfigPath, "default")

	pvcYAML := `
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: lustre-test-pvc
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ""
  resources:
    requests:
      storage: 1Ti
  volumeName: lustre-pv
`
	k8s.KubectlApplyFromString(t, opts, pvcYAML)
	defer k8s.RunKubectl(t, opts, "delete", "pvc", "lustre-test-pvc", "--ignore-not-found=true")

	t.Log("Waiting for Lustre PVC to bind")
	k8s.RunKubectl(t, opts, "wait",
		"--for=jsonpath={.status.phase}=Bound",
		"pvc/lustre-test-pvc",
		"--timeout=120s",
	)

	writerYAML := `
apiVersion: v1
kind: Pod
metadata:
  name: lustre-writer
spec:
  restartPolicy: Never
  containers:
  - name: writer
    image: busybox
    command: ["sh", "-c", "echo 'lustre-test-content' > /mnt/lustre/testfile.txt"]
    volumeMounts:
    - name: lustre
      mountPath: /mnt/lustre
  volumes:
  - name: lustre
    persistentVolumeClaim:
      claimName: lustre-test-pvc
`
	k8s.KubectlApplyFromString(t, opts, writerYAML)
	defer k8s.RunKubectl(t, opts, "delete", "pod", "lustre-writer", "--ignore-not-found=true")

	t.Log("Waiting for Lustre writer pod to complete")
	k8s.RunKubectl(t, opts, "wait",
		"--for=jsonpath={.status.phase}=Succeeded",
		"pod/lustre-writer",
		"--timeout=120s",
	)

	readerYAML := `
apiVersion: v1
kind: Pod
metadata:
  name: lustre-reader
spec:
  restartPolicy: Never
  containers:
  - name: reader
    image: busybox
    command: ["sh", "-c", "cat /mnt/lustre/testfile.txt"]
    volumeMounts:
    - name: lustre
      mountPath: /mnt/lustre
  volumes:
  - name: lustre
    persistentVolumeClaim:
      claimName: lustre-test-pvc
`
	k8s.KubectlApplyFromString(t, opts, readerYAML)
	defer k8s.RunKubectl(t, opts, "delete", "pod", "lustre-reader", "--ignore-not-found=true")

	t.Log("Waiting for Lustre reader pod to complete")
	k8s.RunKubectl(t, opts, "wait",
		"--for=jsonpath={.status.phase}=Succeeded",
		"pod/lustre-reader",
		"--timeout=120s",
	)

	output, err := k8s.RunKubectlAndGetOutputE(t, opts, "logs", "lustre-reader")
	require.NoError(t, err)
	require.Contains(t, output, "lustre-test-content",
		"reader pod output should contain written content")
	t.Log("Lustre write/read test passed")
}
