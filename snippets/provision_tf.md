
::: {.cell .markdown}

## Provision infrastructure with Terraform

Now that everything is set up, we are ready to provision our VM resources with Terraform! We will use Terraform to provision 3 VM instances and associated network resources on the OpenStack cloud.

![Using Terraform to provision resources.](images/step1-tf.svg)



:::

::: {.cell .markdown}

### Preliminaries

:::

::: {.cell .markdown}

Let's navigate to the directory with the Terraform configuration for our KVM deployment:

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/tf/kvm
```
:::


::: {.cell .markdown}

and make sure we'll be able to run the `terraform` executable by adding the directory in which it is located to our `PATH`:

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
```
:::

::: {.cell .markdown}

We also need to un-set some OpenStack-related environment variables that are set automatically in the Chameleon Jupyter environment, since these will override some Terraform settings that we *don't* want to override:

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
unset $(set | grep -o "^OS_[A-Za-z0-9_]*")
```
:::


::: {.cell .markdown}

We should also check that our `clouds.yaml` is set up:

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cat  clouds.yaml
```
:::


::: {.cell .markdown}

### Understanding our Terraform configuration

:::

::: {.cell .markdown}

The `tf/kvm` directory in our IaC repository includes the following files, which we'll briefly discuss now.

```
├── data.tf
├── main.tf
├── outputs.tf
├── provider.tf
├── variables.tf
└── versions.tf
```

A Terraform configuration defines infrastructure elements using stanzas, which include different components such as 

* data sources (see `data.tf`)
* resources, (ours are in `main.tf`)
* outputs, (see `outputs.tf`)
* one or more providers (see `providers.tf`) with reference to providers listed in our `clouds.yaml`,
* variables, (see `variables.tf`)
* and by convention there is a `versions.tf` which describes what version of Terraform and what version of the OpenStack plugin for Terraform our configuration is defined for. 

We'll focus especially on data sources, resources, outputs, and variables. Here's an example of a Terraform configuration that includes all four:

```
resource "openstack_compute_instance_v2" "my_vm" {
  name            = "${var.instance_hostname}"
  flavor_name     = "m1.small"
  image_id        = data.openstack_images_image_v2.ubuntu.id
  key_pair        = "my-keypair"
  network {
    name = "private-network"
  }
}

data "openstack_images_image_v2" "ubuntu" {
  name = "CC-Ubuntu24.04"
}

variable "instance_hostname" {
  description = "Hostname to use for the image"
  type        = string
  default     = "example-vm"
}

output "instance_ip" {
  value = openstack_compute_instance_v2.my_vm.access_ip_v4
}
```

Each item is in a **stanza** which has a block type, an identifier, and a body enclosed in curly braces {}. For example, the resource stanza for the OpenStack instance above has the block type `resource`, the resource type `openstack_compute_instance_v2`, and the name `my_vm`. (This name can be anything you want - it is used to refer to the resource elsewhere in the configuration.) Inside the body, we would specify attributes such as `flavor_name`, `image_id`, and `network` (you can see a complete list in the [documentation](https://registry.terraform.io/providers/terraform-provider-openstack/openstack/latest/docs/resources/compute_instance_v2)).

The data sources, variables, and resources are used to define and manage infrastructure. 

* **data** sources get existing infrastructure details from OpenStack about resources *not* managed by Terraform, e.g. available images or flavors. For example, here we had a data stanza of type "openstack_images_image_v2" with name "ubuntu". Terraform will get the details of this image from the OpenStack provider; then, when we use `data.openstack_images_image_v2.ubuntu.id` in defining the resource, it knows the ID of the image without us having to look it up. (Note that we can refer to another part of the Terraform file using `block_type.resource_type.name`, e.g. `data.openstack_images_image_v2.ubuntu` here.) You can look at our `data.tf` and see that we are asking Terraform to find out about the existing `sharednet1` network, its associated subnet, and several security groups.
* **variables** let us define inputs and reuse the configuration across different environments. The value of variables can be passed in the command line arguments when we run a `terraform` command, or by defining environment variables that start with `TF_VAR`. In this example, there's a variable `instance_hostname` so that we can re-use this configuration to create a VM with any hostname - the variable is used inside the resource block with `name = "${var.instance_hostname}"`. If you look at our `variables.tf`, you can see that we'll use variables to define a suffix to include in all our resource names (e.g. your net ID), and the name of your key pair.
* **resources** represent actual OpenStack components such as compute instances, networks, ports, floating IPs, and security groups. You can see the types of resources available in the [documentation](https://registry.terraform.io/providers/terraform-provider-openstack/openstack/latest/docs). Our resoures are defined in `main.tf`.


You may notice the use of `for_each` in `main.tf`. This is used to iterate over a collection, such as a map variable, to create multiple instances of a resource. Since `for_each` assigns unique keys to each element, that makes it easier to reference specific resources. For example, we provision a port on `sharednet1` for each instance, but when we assign a floating IP, we can specifically refer to the port for "node1" with `openstack_networking_port_v2.sharednet1_ports["node1"].id`.

Terraform also supports outputs, which provide information about the infrastructure after deployment. For example, if we want to print a dynamically assigned floating IP after the infrastructure is deployed, we might put it in an output. This will save us from having to look it up in the Horizon GUI. You can see in `outputs.tf` that we do exactly this.

Terraform is *declarative*, not imperative, so we don't need to write the exact steps needed to provision this infrastructure - Terraform will examine our configuration and figure out a plan to realize it.


:::


::: {.cell .markdown}

### Applying our Terraform configuration

:::


::: {.cell .markdown}

First, we need Terraform to set up our working directory, make sure it has "provider" plugins to interact with our infrastructure provider (it will read in `provider.tf` to check), and set up storage for keeping track of the infrastructure state:

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
terraform init
```
:::

::: {.cell .markdown}

We need to set some [variables](https://developer.hashicorp.com/terraform/language/values/variables). In our Terraform configuration, we define a variable named `suffix` that we will substitute with our own net ID, and then we use that variable inside the hostname of instances and the names of networks and other resources in `main.tf`, e.g. we name our network <pre>private-subnet-mlops-<b>${var.suffix}</b></pre>. We'll also use a variable to specify a key pair to install.

In the following cell, **replace `netID` with your actual net ID, and replace `id_rsa_chameleon` with the name of *your* personal key that you use to access Chameleon resources**.

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
export TF_VAR_suffix=netID
export TF_VAR_key=id_rsa_chameleon
```
:::

::: {.cell .markdown}

We should confirm that our planned configuration is valid:

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
terraform validate
```
:::


::: {.cell .markdown}

Then, let's preview the changes that Terraform will make to our infrastructure. In this stage, Terraform communicates with the cloud infrastructure provider to see what have *already* deployed and to 

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
terraform plan
```
:::

::: {.cell .markdown}

Finally, we will apply those changes. (We need to add an `-auto-approve` argument because ordinarily, Terraform prompts the user to type "yes" to approve the changes it will make.)

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
terraform apply -auto-approve
```
:::

::: {.cell .markdown}

Make a note of the floating IP assigned to your instance, from the Terraform output.

:::

::: {.cell .markdown}

From the KVM@TACC Horizon GUI, check the list of compute instances and find yours. Take a screenshot for later reference.

:::


::: {.cell .markdown}

### Changing our infrastructure

:::

::: {.cell .markdown}

One especially nice thing about Terraform is that if we change our infrastructure definition, it can apply those changes without having to re-provision everything from scratch. 

:::


::: {.cell .markdown}

For example, suppose the physical node on which our "node3" VM becomes non-functional. To replace our "node3", we can simply run

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
terraform apply -replace='openstack_compute_instance_v2.nodes["node3"]' -auto-approve
```
:::

::: {.cell .markdown}

Similarly, we could make changes to the infrastructure description in the `main.tf` file and then use `terraform apply` to update our cloud infrastructure. Terraform would determine which resources can be updated in place, which should be destroyed and recreated, and which should be left alone.

This declarative approach - where we define the desired end state and let the tool get there - is much more robust than imperative-style tools for deploying infrastructure (`openstack` CLI, `python-chi` Python API) (and certainly more robust than ClickOps!).

:::

