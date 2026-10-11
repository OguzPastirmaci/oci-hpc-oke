# Copyright (c) 2025 Oracle Corporation and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl

# Drain all Kueue CRs before the chart is uninstalled so the CRD cascade does not
# hang on resource-in-use finalizers. Operator path only; ORM/local use wait = false
# on helm_release.kueue. Input updates in place, so destroy uses the current operator.
resource "terraform_data" "kueue_predestroy_drain_via_operator" {
  count = alltrue([var.install_kueue, local.deploy_from_operator]) ? 1 : 0

  input = {
    bastion_host    = module.oke.bastion_public_ip
    bastion_user    = local.bastion_user
    operator_host   = module.oke.operator_private_ip
    operator_user   = local.operator_user
    ssh_private_key = tls_private_key.stack_key.private_key_openssh
    drain_script    = file("${path.module}/files/kueue/predestroy-drain.sh")
  }

  connection {
    type                = "ssh"
    bastion_host        = self.output.bastion_host
    bastion_user        = self.output.bastion_user
    bastion_private_key = self.output.ssh_private_key
    host                = self.output.operator_host
    user                = self.output.operator_user
    private_key         = self.output.ssh_private_key
    timeout             = "10m"
  }

  provisioner "remote-exec" {
    when       = destroy
    on_failure = continue
    inline = [
      "export PATH=$PATH:/home/${self.output.operator_user}/bin",
      "export OCI_CLI_AUTH=instance_principal",
      "export PYTHONWARNINGS=\"ignore:the 'strict' parameter::urllib3.poolmanager\"",
      "cat > /tmp/kueue-predestroy-drain.sh <<'KUEUE_DRAIN_EOF'\n${self.output.drain_script}\nKUEUE_DRAIN_EOF",
      "bash /tmp/kueue-predestroy-drain.sh",
    ]
  }

  depends_on = [module.kueue]
}
