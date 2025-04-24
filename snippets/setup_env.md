
::: {.cell .markdown}

## Prepare the environment

In keeping with good DevOps practices, we will deploy our infrastructure - starting with the Kubernetes cluster - using infrastructure-as-code and configuration-as-code principles:

* The process of provisioning and deploying the infrastructure is scalable, because it is heavily automated. It is easy to rebuild the system, without requiring effort or expertise.
* Everything needed to deploy the infrastructure is in version control.
* The infrastructure is immutable - no manual updates or changes.

We will use two IaC/CaC tools to prepare our Kubernetes cluster: 

* [Terraform](https://www.terraform.io/), which we'll use to provision the resources on our cloud infrastructure provider. (A popular alternative is [OpenTofu](https://opentofu.org/).)
* [Ansible](https://github.com/ansible/ansible), which we'll use to configure and deploy Kubernetes, and then to set up the Kubernetes cluster and the services running on it. (A popular alternative is [Salt](https://github.com/saltstack/salt).)

both of which are aligned with the principles above.

In this notebook, which will run in the Chameleon Jupyter environment, we will install and configure these tools in that environment. This is a *one-time* step that an engineer would ordinarily do just once, on their own computer.

> **Note**: This is a Bash notebook, so you will run it with a Bash kernel. You can change the kernel (if needed) by clicking the kernel name in the top right of the Jupyter interface.

:::

::: {.cell .markdown}

### Get infrastructure configuration

:::

::: {.cell .markdown}

Following IaC principles, our infrastructure configuration is all in version control! We have organized all of the materials that "describe" the deployment in our "IaC repository": [https://github.com/teaching-on-testbeds/gourmetgram-iac.git](https://github.com/teaching-on-testbeds/gourmetgram-iac.git). 

This repository has the following structure:


```
├── tf
│   └── kvm
├── ansible
│   ├── general
│   ├── pre_k8s
│   ├── k8s
│   ├── post_k8s
│   └── argocd
├── k8s
│   ├── platform
│   ├── staging
│   ├── canary
│   └── production
└── workflows
```

* The `tf` directory includes materials needed for Terraform to provision resources from the cloud provider. This is a "Day 0" setup task.
* The "Day 1" setup task is to install and configure Kubernetes on the resources. We use Ansible, and the materials are in the `ansible` directory in the `pre_k8s`, `k8s` and `post_k8s` subdirectories. (The `general` directory is just for learning.)
* The applications that we will be deployed in Kubernetes are defined in the `k8s` directory:
  * `platform` has all the "accessory" services we need to support our machine learning application. In this example, it has a model registry and the associated database and object store services used by the model registry; more generally "platform" may include experiment tracking, evaluation and monitoring, and other related services.
  * `staging`, `canary`, and `production` are deployments of our GourmetGram application. A new model or application version starts off in `staging`; after some internal tests it may be promoted to `canary` where it is served to some live users; and after further evaluation and monitoring, it may be promoted to `production`. 
* We use Ansible to "register" these applications in ArgoCD, using the playbooks in the `ansible/argocd` directory. ArgoCD is a continuous delivery tool for Kubernetes that automatically deploys and updates applications based on the latest version of its manifests.
* From "Day 2" and on, during the lifecycle of the application, we use ArgoCD and Argo Workflows to handle model and application versions, using the pipelines in `workflows`.


In the next cell, we get a copy of the [GourmetGram infrastructure repository](https://github.com/teaching-on-testbeds/gourmetgram-iac.git):

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
git clone --recurse-submodules https://github.com/teaching-on-testbeds/gourmetgram-iac.git /work/gourmetgram-iac
```
:::

::: {.cell .markdown}

Note that we use the `--recurse-submodules` argument to `git clone` - we are including Kubespray, an Ansible-based project for deploying Kubernetes, inside our IaC repository as a submodule.

:::

::: {.cell .markdown}


Among the automation and CI/CD tools mentioned above:

* Terraform and Ansible run on the engineer's own computer, and communicate with the cloud provider/cloud resources over a network. 
* ArgoCD and Argo Workflows run on the cloud resources themselves.

So, a necessary prerequisite for this workflow is to download, install, and configure Terraform and Ansible on "our own computer" - except in this case, we will use the Chameleon Jupyter environment as "our computer".

:::

::: {.cell .markdown}

### Install and configure Terraform

:::

::: {.cell .markdown}

Before we can use Terraform, we'll need to download a Terraform client. The following cell will download the Terraform client and "install" it in this environment:

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
mkdir -p /work/.local/bin
wget https://releases.hashicorp.com/terraform/1.10.5/terraform_1.10.5_linux_amd64.zip
unzip -o -q terraform_1.10.5_linux_amd64.zip
mv terraform /work/.local/bin
rm terraform_1.10.5_linux_amd64.zip
```
:::


::: {.cell .markdown}

The Terraform client has been installed to: `/work/.local/bin`. In order to run `terraform` commands, we will have to add this directory to our `PATH`, which tells the system where to look for executable files.

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
```
:::


::: {.cell .markdown}

Let's make sure we can now run `terraform` commands. The following cell should print usage information for the `terraform` command, since we run it without any subcommands:
:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
terraform
```
:::



::: {.cell .markdown}

Terraform works by communicating with a cloud provider (either a commercial cloud, like AWS or GCP, or a private cloud, like an on-premises OpenStack cloud, or a hybrid cloud with both types of resources). We will need to prepare credentials with which it can act on our behalf on the Chameleon OpenStack cloud. This is a one-time procedure.

To get credentials, open the Horizon GUI:

* from the Chameleon website
* click "Experiment" > "KVM@TACC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.

On the left side, expand the "Identity" section and click on "Application Credentials". Then, click "Create Application Credential".

* In the "Name", field, use "mlops-lab".
* Set the "Expiration" date and time to the due date of this lab. (Note that this will be in UTC time, not your local time zone.) This ensures that if your credential is leaked (e.g. you accidentially push it to a public Github repository), the damage is mitigated.
* Click "Create Application Credential".
* Choose "Download clouds.yaml".

:::

::: {.cell .markdown}


The `clouds.yaml` file will look something like this (expect with an alphanumeric string in place of `REDACTED_UNIQUE_ID` and `REDACTED_SECRET`):

```
clouds:
  openstack:
    auth:
      auth_url: https://kvm.tacc.chameleoncloud.org:5000
      application_credential_id: "REDACTED_UNIQUE_ID"
      application_credential_secret: "REDACTED_SECRET"
    region_name: "KVM@TACC"
    interface: "public"
    identity_api_version: 3
    auth_type: "v3applicationcredential"
```

It lists one or more clouds - in this case, a single cloud named "openstack", and then for each cloud, specifies how to connect and authenticate to that cloud. In particular, the `application_credential_id` and `application_credential_secret` allow an application like Terraform to interact with the Chameleon cloud on your behalf, without having to use your personal Chameleon login.

Then, in our Terraform configuration, we will have a block like

```
provider "openstack" {
  cloud = "openstack"
}
```

where the value assigned to `cloud` tells Terraform which cloud in the `clouds.yaml` file to authenticate to.

:::

::: {.cell .markdown}

One nice feature of Terraform is that we can use it to provision resource on multiple clouds. For example, if we wanted to provision resources on both KVM@TACC and CHI@UC (e.g. the training resources on CHI@UC and everything else on KVM@TACC), we might generate application credentials on both sites, and combine them into a `clouds.yaml` like this:

```
clouds:
  kvm:
    auth:
      auth_url: https://kvm.tacc.chameleoncloud.org:5000
      application_credential_id: "REDACTED_UNIQUE_ID_KVM"
      application_credential_secret: "REDACTED_SECRET_KVM"
    region_name: "KVM@TACC"
    interface: "public"
    identity_api_version: 3
    auth_type: "v3applicationcredential"
  uc:
    auth:
      auth_url: https://chi.uc.chameleoncloud.org:5000
      application_credential_id: "REDACTED_UNIQUE_ID_UC"
      application_credential_secret: "REDACTED_SECRET_UC"
    region_name: "CHI@UC"
    interface: "public"
    identity_api_version: 3
    auth_type: "v3applicationcredential"

```

and then in our Terraform configuration, we could specify which OpenStack cloud to use, e.g.

```
provider "openstack" {
  cloud = "kvm"
}
```

or 


```
provider "openstack" {
  cloud = "uc"
}
```

For now, since we are just using one cloud, we will leave our `clouds.yaml` as is.

:::

::: {.cell .markdown}

In the file browser in the Chameleon Jupyter environment, you will see a template `clouds.yaml`.  Use the file browser to open it, and paste in the 

```
      application_credential_id: "REDACTED_UNIQUE_ID"
      application_credential_secret: "REDACTED_SECRET"
```

lines from the `clouds.yaml` that you just downloaded from the KVM@TACC GUI (so that it has the "real" credentials in it). Save the file.

:::

::: {.cell .markdown}

Terraform will look for the `clouds.yaml` in either ` ~/.config/openstack` or the directory from which we run `terraform` - we will move it to the latter directory:

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cp clouds.yaml /work/gourmetgram-iac/tf/kvm/clouds.yaml
```
:::

::: {.cell .markdown}

### Install and configure Ansible

:::


::: {.cell .markdown}

Next, we'll set up Ansible! We will similarly need to get the Ansible client, which we install in the following cell:

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
PYTHONUSERBASE=/work/.local pip install --user ansible-core==2.16.9 ansible==9.8.0
```
:::


::: {.cell .markdown}

The Ansible client has been installed to: `/work/.local/bin`. In order to run `ansible-playbook` commands, we will have to add this directory to our `PATH`, which tells the system where to look for executable files. We also need to let it know where to find the corresponding Python packages.

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
export PYTHONPATH=/work/.local/lib/python3.10/site-packages:$PYTHONPATH
```
:::


::: {.cell .markdown}

Let's make sure we can now run `ansible-playbook` commands. The following cell should print usage information for the `ansible-playbook` command, since we run it with `--help`:
:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
ansible-playbook --help
```
:::



::: {.cell .markdown}

Now, we'll configure Ansible. The `ansible.cfg` configuration file modifies the default behavior of the Ansible commands we're going to run. Open this file using the file browser on the left side.

:::

::: {.cell .markdown}


Our configuration will include:

```
[defaults]
stdout_callback = yaml
inventory = /work/gourmetgram-iac/ansible/inventory.yaml

```

The first line is just a matter of preference, and directs the Ansible client to display output from commands in a more structured, readable way. The second line specifies the location of a default *inventory* file - the list of hosts that Ansible will configure.

It will also include:

```
[ssh_connection]
ssh_args = -o StrictHostKeyChecking=off -o UserKnownHostsFile=/dev/null -o ForwardAgent=yes -o ProxyCommand="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -W %h:%p cc@A.B.C.D"
```

which says that when Ansible uses SSH to connect to the resources it is managing, it should "jump" through `A.B.C.D` and forward the keys from this environment, through `A.B.C.D`, to the final destination. (Also, we disable host key checking when using SSH.)

You will need to edit `A.B.C.D.` *after* you provision your resources, and replace it with the floating IP assigned to your experiment.

*After* you have edited the floating IP and saved the `ansible.cfg` file, you can move it - Ansible will look in either `~/.ansible.cfg` or the directory that we run Ansible commands from, we will use the latter:

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
# ONLY AFTER YOU HAVE PROVISIONED RESOURCES AND UPDATED THE CFG
cp ansible.cfg /work/gourmetgram-iac/ansible/ansible.cfg
```
:::

::: {.cell .markdown}

### Configure the PATH

:::

::: {.cell .markdown}

Both Terraform and Ansible executables have been installed to a location that is not the system-wide location for executable files: `/work/.local/bin`. In order to run `terraform` or `ansible-playbook` commands, we will have to add this directory to our `PATH`, which tells the system where to look for executable files.

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
```
:::

::: {.cell .markdown}

and, we'll have to do that in *each new Bash session*.

:::

::: {.cell .markdown}

### Prepare Kubespray

To install Kubernetes, we'll use Kubespray, which is a set of Ansible playbooks for deploying Kubernetes. We'll also make sure we have its dependencies now:

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
PYTHONUSERBASE=/work/.local pip install --user -r /work/gourmetgram-iac/ansible/k8s/kubespray/requirements.txt
```
:::



