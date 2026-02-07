

# Build an MLOps Pipeline

In [Cloud Computing on Chameleon](https://teaching-on-testbeds.github.io/cloud-chi/), following the premise:

> You are working at a machine learning engineer at a small startup company called GourmetGram. They are developing an online photo sharing community focused on food. You are testing a new model you have developed that automatically classifies photos of food into one of a set of categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. You have built a simple web application with which to test your model and get feedback from others.

we deployed a basic machine learning service to an OpenStack cloud. However, that deployment involved a lot of manual steps ("ClickOps"), and any updates to it would similarly involve lots of manual effort, be difficult to track, etc.

In this tutorial, we will learn how to automate both the initial deployment, and updates during the lifecycle of the application. We will:

* practice deploying systems following infrastructure-as-code and configuration-as-code principles using automated deployment tools
* and create an automated pipeline to manage a machine learning model through its lifecycle 

Our experiment will use the following automated deployment and lifecycle management tools:

* Terraform: A declarative Infrastructure as Code (IaC) tool used to provision and manage cloud infrastructure (servers, networks, etc.) by defining the desired end state in configuration files. Here, we use it to provision our infrastructure.
* Ansible: An imperative Configuration as Code (CaC) tool that automates system configuration, software installation, and application deployment through task-based YAML playbooks describing the steps to achieve a desired setup. Here, we use it to install Kubernetes and the Argo tools on our infrastructure after it is provisioned
* Argo CD: A declarative GitOps continuous delivery tool for Kubernetes that automatically syncs and deploys applications based on the desired state stored in Git repositories.
* Argo Workflows: A Kubernetes-native workflow engine where you define workflows, which execute tasks inside containers to run pipelines, jobs, or automation processes.

**Note**: that we use Argo CD and Argo Workflows, which are tightly integrated with Kubernetes, because we are working in the context of a Kubernetes deployment. If our service was not deployed in Kubernetes (for example: it was deployed using "plain" Docker containers without a container orchestration framework), we would use other tools for managing the application and model lifecycle.

The expected *hands-on* duration of this experiment is 5-6 hours. However, there is an unattended installation step in the middle (Kubernetes setup) that you may need to leave running for several hours. You should plan accordingly, to e.g. leave that stage running while you do something else, then return to finish.

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You should also have added your SSH key to the KVM@TACC site.



## Experiment topology 

In this experiment, we will deploy a 3-node Kubernetes cluster on Chameleon instances. The Kubernetes cluster will be self-managed, which means that the infrastructure provider is not responsbile for setting up and maintaining our cluster; *we* are.  

However, the cloud infrastructure provider will provide the compute resources and network resources that we need. We will provision the following resources for this experiment:

![Experiment topology.](images/lab-topology.svg)




## Provision a key

Before you begin, open this experiment on Trovi:

* Use this link: [MLOps Pipeline](https://chameleoncloud.org/experiment/share/1eb302de-4707-4ae9-ae2d-391b9b8e5261) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it.

You will see several notebooks inside the `mlops-chi` directory - look for the one titled `0_intro.ipynb`. Open this notebook and execute the following cell (and make sure the correct project is selected):


```python
# runs in Chameleon Jupyter environment
from chi import server, context

context.version = "1.0" 
context.choose_project()
context.choose_site(default="KVM@TACC")
```

```python
# runs in Chameleon Jupyter environment
server.update_keypair()
```


One more note: the rest of these materials assume that the required security groups are already set up within the project! If you are a student working on these materials as part of an assignment, your instructor will have done this already on behalf of the entire class (it only needs to be run once within a project!), so you can move on to the next step. Otherwise, use `x_setup_sg.ipynb` to make sure you have all the security groups you will need.




Then, you may continue following along at [Build an MLOps Pipeline](https://teaching-on-testbeds.github.io/mlops-chi/).



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



### Get infrastructure configuration



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



```bash
# runs in Chameleon Jupyter environment
git clone --recurse-submodules https://github.com/teaching-on-testbeds/gourmetgram-iac.git /work/gourmetgram-iac
```


Note that we use the `--recurse-submodules` argument to `git clone` - we are including Kubespray, an Ansible-based project for deploying Kubernetes, inside our IaC repository as a submodule.




Among the automation and CI/CD tools mentioned above:

* Terraform and Ansible run on the engineer's own computer, and communicate with the cloud provider/cloud resources over a network. 
* ArgoCD and Argo Workflows run on the cloud resources themselves.

So, a necessary prerequisite for this workflow is to download, install, and configure Terraform and Ansible on "our own computer" - except in this case, we will use the Chameleon Jupyter environment as "our computer".



### Install and configure Terraform



Before we can use Terraform, we'll need to download a Terraform client. The following cell will download the Terraform client and "install" it in this environment:


```bash
# runs in Chameleon Jupyter environment
mkdir -p /work/.local/bin
wget https://releases.hashicorp.com/terraform/1.10.5/terraform_1.10.5_linux_amd64.zip
unzip -o -q terraform_1.10.5_linux_amd64.zip
mv terraform /work/.local/bin
rm terraform_1.10.5_linux_amd64.zip
```



The Terraform client has been installed to: `/work/.local/bin`. In order to run `terraform` commands, we will have to add this directory to our `PATH`, which tells the system where to look for executable files.



```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
```



Let's make sure we can now run `terraform` commands. The following cell should print usage information for the `terraform` command, since we run it without any subcommands:


```bash
# runs in Chameleon Jupyter environment
terraform
```




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



In the file browser in the Chameleon Jupyter environment, you will see a template `clouds.yaml`.  Use the file browser to open it, and paste in the 

```
      application_credential_id: "REDACTED_UNIQUE_ID"
      application_credential_secret: "REDACTED_SECRET"
```

lines from the `clouds.yaml` that you just downloaded from the KVM@TACC GUI (so that it has the "real" credentials in it). Save the file.



Terraform will look for the `clouds.yaml` in either ` ~/.config/openstack` or the directory from which we run `terraform` - we will move it to the latter directory:


```bash
# runs in Chameleon Jupyter environment
cp clouds.yaml /work/gourmetgram-iac/tf/kvm/clouds.yaml
```


### Install and configure Ansible




Next, we'll set up Ansible! We will similarly need to get the Ansible client, which we install in the following cell:


```bash
# runs in Chameleon Jupyter environment
PYTHONUSERBASE=/work/.local pip install --user ansible-core==2.16.9 ansible==9.8.0
```



The Ansible client has been installed to: `/work/.local/bin`. In order to run `ansible-playbook` commands, we will have to add this directory to our `PATH`, which tells the system where to look for executable files. We also need to let it know where to find the corresponding Python packages.



```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
```



Let's make sure we can now run `ansible-playbook` commands. The following cell should print usage information for the `ansible-playbook` command, since we run it with `--help`:


```bash
# runs in Chameleon Jupyter environment
ansible-playbook --help
```




Now, we'll configure Ansible. The `ansible.cfg` configuration file modifies the default behavior of the Ansible commands we're going to run. Open this file using the file browser on the left side.




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
ssh_args = -o ControlMaster=auto -o ControlPersist=60s \
           -o StrictHostKeyChecking=off -o UserKnownHostsFile=/dev/null \
           -o ForwardAgent=yes \
           -o ProxyCommand="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -W %h:%p cc@A.B.C.D"
pipelining = True
```

which says that when Ansible uses SSH to connect to the resources it is managing, it should "jump" through `A.B.C.D` and forward the keys from this environment, through `A.B.C.D`, to the final destination. (Also, we disable host key checking when using SSH, and configure it to minimize the number of SSH sessions and the number of network operations wherever possible.)

You will need to edit `A.B.C.D.` *after* you provision your resources, and replace it with the floating IP assigned to your experiment.

*After* you have edited the floating IP and saved the `ansible.cfg` file, you can move it - Ansible will look in either `~/.ansible.cfg` or the directory that we run Ansible commands from, we will use the latter:


```bash
# runs in Chameleon Jupyter environment
# ONLY AFTER YOU HAVE PROVISIONED RESOURCES AND UPDATED THE CFG
cp ansible.cfg /work/gourmetgram-iac/ansible/ansible.cfg
```


### Configure the PATH



Both Terraform and Ansible executables have been installed to a location that is not the system-wide location for executable files: `/work/.local/bin`. In order to run `terraform` or `ansible-playbook` commands, we will have to add this directory to our `PATH`, which tells the system where to look for executable files.



```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
```


and, we'll have to do that in *each new Bash session*.



### Prepare Kubespray

To install Kubernetes, we'll use Kubespray, which is a set of Ansible playbooks for deploying Kubernetes. We'll also make sure we have its dependencies now:



```bash
# runs in Chameleon Jupyter environment
PYTHONUSERBASE=/work/.local pip install --user -r /work/gourmetgram-iac/ansible/k8s/kubespray/requirements.txt
```





## Provision infrastructure with Terraform

Now that everything is set up, we are ready to provision our VM resources with Terraform! We will use Terraform to provision 3 VM instances and associated network resources on the OpenStack cloud.

![Using Terraform to provision resources.](images/step1-tf.svg)





### Create a server lease



While Terraform is able to provision most kinds of resources, it cannot create or manage a reservation - this feature of OpenStack is not used very widely, so the Terraform provider for OpenStack does not support it. We will separately create a lease for three server instances outside of Terraform.



### Authentication

In the cell below, replace `CHI-XXXXXX` with the name of *your* Chameleon project, then run the cell.


```bash
# runs in Chameleon Jupyter environment
export OS_AUTH_URL=https://kvm.tacc.chameleoncloud.org:5000/v3
export OS_PROJECT_NAME="CHI-XXXXXX"
export OS_REGION_NAME="KVM@TACC"
```


and in *BOTH* cells below, replace **netID** with your own net ID, then run to request a lease:


```bash
# runs in Chameleon Jupyter environment
# replace netID in this line
openstack reservation lease create lease_mlops_netID \
  --start-date "$(date -u '+%Y-%m-%d %H:%M')" \
  --end-date "$(date -u -d '+1 days' '+%Y-%m-%d %H:%M')" \
  --reservation "resource_type=flavor:instance,flavor_id=$(openstack flavor show m1.medium -f value -c id),amount=3"
```



and print the UUID of the reserved "flavor":



```bash
# runs in Chameleon Jupyter environment
# also replace netID in this line
flavor_id=$(openstack reservation lease show lease_mlops_netID -f json -c reservations \
      | jq -r '.reservations[0].flavor_id')
echo $flavor_id
```



Make a note of this ID - you will need it later, to provision resources.



### Preliminaries



Let's navigate to the directory with the Terraform configuration for our KVM deployment:


```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/tf/kvm
```



and make sure we'll be able to run the `terraform` executable by adding the directory in which it is located to our `PATH`:



```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
```


We also need to un-set some OpenStack-related environment variables that are set automatically in the Chameleon Jupyter environment, since these will override some Terraform settings that we *don't* want to override:



```bash
# runs in Chameleon Jupyter environment
unset $(set | grep -o "^OS_[A-Za-z0-9_]*")
```



We should also check that our `clouds.yaml` is set up:



```bash
# runs in Chameleon Jupyter environment
cat  clouds.yaml
```



### Understanding our Terraform configuration



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
* **variables** let us define inputs and reuse the configuration across different environments. The value of variables can be passed in the command line arguments when we run a `terraform` command, or by defining environment variables that start with `TF_VAR`. In this example, there's a variable `instance_hostname` so that we can re-use this configuration to create a VM with any hostname - the variable is used inside the resource block with `name = "${var.instance_hostname}"`. If you look at our `variables.tf`, you can see that we'll use variables to define a suffix to include in all our resource names (e.g. your net ID), the name of your key pair, and the reservation ID.
* **resources** represent actual OpenStack components such as compute instances, networks, ports, floating IPs, and security groups. You can see the types of resources available in the [documentation](https://registry.terraform.io/providers/terraform-provider-openstack/openstack/latest/docs). Our resoures are defined in `main.tf`.


You may notice the use of `for_each` in `main.tf`. This is used to iterate over a collection, such as a map variable, to create multiple instances of a resource. Since `for_each` assigns unique keys to each element, that makes it easier to reference specific resources. For example, we provision a port on `sharednet1` for each instance, but when we assign a floating IP, we can specifically refer to the port for "node1" with `openstack_networking_port_v2.sharednet1_ports["node1"].id`.

Terraform also supports outputs, which provide information about the infrastructure after deployment. For example, if we want to print a dynamically assigned floating IP after the infrastructure is deployed, we might put it in an output. This will save us from having to look it up in the Horizon GUI. You can see in `outputs.tf` that we do exactly this.

Terraform is *declarative*, not imperative, so we don't need to write the exact steps needed to provision this infrastructure - Terraform will examine our configuration and figure out a plan to realize it.





### Applying our Terraform configuration




First, we need Terraform to set up our working directory, make sure it has "provider" plugins to interact with our infrastructure provider (it will read in `provider.tf` to check), and set up storage for keeping track of the infrastructure state:


```bash
# runs in Chameleon Jupyter environment
terraform init
```


We need to set some [variables](https://developer.hashicorp.com/terraform/language/values/variables). In our Terraform configuration, we define a variable named `suffix` that we will substitute with our own net ID, and then we use that variable inside the hostname of instances and the names of networks and other resources in `main.tf`, e.g. we name our network <pre>private-subnet-mlops-<b>${var.suffix}</b></pre>. We'll also use a variable to specify a key pair to install.

In the following cell, **replace `netID` with your actual net ID, replace `id_rsa_chameleon` with the name of *your* personal key that you use to access Chameleon resources, and replace the all-zero ID with the reservation ID you printed above.**.


```bash
# runs in Chameleon Jupyter environment
export TF_VAR_suffix=netID
export TF_VAR_key=id_rsa_chameleon
export TF_VAR_reservation=00000000-0000-0000-0000-000000000000
```


We should confirm that our planned configuration is valid:


```bash
# runs in Chameleon Jupyter environment
terraform validate
```



Then, let's preview the changes that Terraform will make to our infrastructure. In this stage, Terraform communicates with the cloud infrastructure provider to see what we have *already* deployed, and to determine what it needs to do to realize the requested configuration:


```bash
# runs in Chameleon Jupyter environment
terraform plan
```


Finally, we will apply those changes. (We need to add an `-auto-approve` argument because ordinarily, Terraform prompts the user to type "yes" to approve the changes it will make.)


```bash
# runs in Chameleon Jupyter environment
terraform apply -auto-approve
```


Make a note of the floating IP assigned to your instance, from the Terraform output.



From the KVM@TACC Horizon GUI, check the list of compute instances and find yours. Take a screenshot for later reference.




### Changing our infrastructure



One especially nice thing about Terraform is that if we change our infrastructure definition, it can apply those changes without having to re-provision everything from scratch. 




For example, suppose the physical node on which our "node3" VM becomes non-functional. To replace our "node3", we can simply run


```bash
# runs in Chameleon Jupyter environment
terraform apply -replace='openstack_compute_instance_v2.nodes["node3"]' -auto-approve
```


Similarly, we could make changes to the infrastructure description in the `main.tf` file and then use `terraform apply` to update our cloud infrastructure. Terraform would determine which resources can be updated in place, which should be destroyed and recreated, and which should be left alone.

This declarative approach - where we define the desired end state and let the tool get there - is much more robust than imperative-style tools for deploying infrastructure (`openstack` CLI, `python-chi` Python API) (and certainly more robust than ClickOps!).




## Practice using Ansible

Now that we have provisioned some infrastructure, we can configure and install software on it using Ansible! 

Ansible is a tool for configuring systems by accessing them over SSH and running commands on them. The commands to run will be defined in advance in a series of *playbooks*, so that instead of using SSH directly and then running commands ourselves interactively, we can just execute a playbook to set up our systems.

First, let's just practice using Ansible.




### Preliminaries



As before, let's make sure we'll be able to use the Ansible executables. We need to put the install directory in the `PATH` inside each new Bash session.



```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
```



If you haven't already, make sure to put your floating IP (which you can see in the output of the Terraform command!) in the `ansible.cfg` configuration file, and move it to the specified location.

The following cell will show the contents of this file, so you can double check - make sure your real floating IP is visible in this output!


```bash
# runs in Chameleon Jupyter environment
cat /work/gourmetgram-iac/ansible/ansible.cfg
```



Finally, we'll `cd` to that directory - 


```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
```




### Verify connectivity

First, we'll run a simple task to check connectivity with all hosts listed in the [`inventory.yml` file](https://github.com/teaching-on-testbeds/gourmetgram-iac/blob/main/ansible/inventory.yml):

```
all:
  vars:
    ansible_python_interpreter: /usr/bin/python3
  hosts:
    node1:
      ansible_host: 192.168.1.11
      ansible_user: cc
    node2:
      ansible_host: 192.168.1.12
      ansible_user: cc
    node3:
      ansible_host: 192.168.1.13
      ansible_user: cc
```

It uses the `ping` module, which checks if Ansible can connect to each host via SSH and run Python code there.




```bash
# runs in Chameleon Jupyter environment
ansible -i inventory.yml all -m ping
```



### Run a "Hello, World" playbook

Once we have verified connectivity to the nodes in our "inventory", we can run a *playbook*, which is a sequence of tasks organized in plays, and defined in a YAML file. Here we will run the following playbook with one "Hello world" play:

```
---
- name: Hello, world - use Ansible to run a command on each host
  hosts: all
  gather_facts: no

  tasks:
    - name: Run hostname command
      command: hostname
      register: hostname_output

    - name: Show hostname output
      debug:
        msg: "The hostname of {{ inventory_hostname }} is {{ hostname_output.stdout }}"
```

The playbook connects to `all` hosts listed in the inventory, and performs two tasks: first, it runs the `hostname` command on each host and saves the result in `hostname_output`, then it prints a message showing the value of `hostname_output` (using the *debug* module).



```bash
# runs in Chameleon Jupyter environment
ansible-playbook -i inventory.yml general/hello_host.yml
```




## Deploy Kubernetes using Ansible

Now that we understand a little bit about how Ansible works, we will use it to deploy Kubernetes on our three-node cluster! 

We will use Kubespray, an Ansible-based tool, to automate this deployment.

![Using Ansible for software installation and system configuration.](images/step2-ansible.svg)





### Preliminaries



As before, let's make sure we'll be able to use the Ansible executables. We need to put the install directory in the `PATH` inside each new Bash session.



```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
```




### Run a preliminary playbook

Before we set up Kubernetes, we will run a preliminary playbook to:

* disable the host firewall on the nodes in the cluster. (The cloud infrastructure provider will anyway block all traffic except for SSH traffic on port 22, as we specified in the security group configuration.) We will also configure each node to permit the local container registry.
* and, configure Docker to use the local registry. (We prefer to do this before deploying Kubernetes, to avoid restarting Docker when there is a live Kubernetes deployment using it already...)



```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml pre_k8s/pre_k8s_configure.yml
```



### Run the Kubespray play

Then, we can run the Kubespray playbook! Inside the `ansible/k8s` subdirectory:

* we have a "copy" of Kubespray as a submodule
* and we have a minimal `inventory` directory, which describes the specific Kubespray configuration for our cluster

The following cell will run for a long time - potentially for hours! - and install Kubernetes on the three-node cluster.

When it is finished the "PLAY RECAP" should indicate that none of the tasks failed.


```bash
# runs in Chameleon Jupyter environment
export ANSIBLE_CONFIG=/work/gourmetgram-iac/ansible/ansible.cfg
export ANSIBLE_ROLES_PATH=roles
```


```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible/k8s/kubespray
ansible-playbook -i ../inventory/mycluster --become --become-user=root ./cluster.yml
```



### Run a post-install playbook

After our Kubernetes install is complete, we run some additional tasks to further configure and customize our Kubernetes deployment. Our post-install playbook will:

* Configure the `kubectl` command so that we can run it directly on "node1" as the `cc` user, and allow the `cc` user to run Docker commands.
* Configure the Kubernetes dashboard, which we can use to monitor our cluster.
* Install [ArgoCD](https://argo-cd.readthedocs.io/en/stable/), [Argo Workflows](https://argoproj.github.io/workflows/), and [Argo Events](https://argoproj.github.io/events/). We will use Argo CD for application and service bootstrapping, and Argo Events/Workflows for application lifecycle management on our Kubernetes cluster.

In the output below, make a note of the Kubernetes dashboard token and the Argo admin password, both of which we will need in the next steps.




```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml post_k8s/post_k8s_configure.yml
```



### Access the Kubernetes dashboard (optional)

To check on our Kubernetes deployment, let's keep an eye on the dashboard. 

First, since we did not configure security group rules to permit any ports besides SSH, we need to use SSH port forwarding to open a tunnel between our *local* device and the remote cluster. Then, since the service is configured only for internal access within the cluster, we need to use port forwarding to also make it available on the host. 

Run the command below in your *local* terminal (not the terminal in the Chameleon Jupyter environment!) and substitute:

* in place of `~/.ssh/id_rsa_chameleon`, the path to your own key
* in place of `A.B.C.D`, the floating IP assigned to your "node1" instance

```bash
# runs in your **local** terminal
ssh -L 8443:127.0.0.1:8443 -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```


then, inside that terminal, run

```bash
# runs on node1 
kubectl port-forward -n kube-system svc/kubernetes-dashboard 8443:443
```

and leave it running.

Now, in a browser, you may open

```
https://127.0.0.1:8443/
```

You will see a warning about an invalid certificate, which you may override and choose the "Advanced" option to proceed. Then, you will be prompted to log in.

From the output of the post-install playbook above, find the "Dashboard token" and paste it into the token space, then log in. You will see the Kubernetes dashboard.

(Note: if your token expires, you can generate a new one with `kubectl -n kube-system create token admin-user`.)

For now, there is not much of interest in the dashboard. You can see some Kubernetes system services in the "kube-system" namespace, and Argo-related services in the "argo", "argocd", and "argo-events" namespaces. We have not yet deployed our GourmetGram services, but we'll do that in the next step!






### Access the ArgoCD dashboard (optional)

Similarly, we may access the Argo CD dashboard. In the following command, substitute

* in place of `~/.ssh/id_rsa_chameleon`, the path to your own key
* in place of `A.B.C.D`, the floating IP assigned to your "node1" instance

```bash
# runs in your **local** terminal
ssh -L 8888:127.0.0.1:8888 -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

then, inside that terminal, run

```bash
# runs on node1 
kubectl port-forward svc/argocd-server -n argocd 8888:443
```

and leave it running.

Now, in a browser, you may open

```
https://127.0.0.1:8888/
```

You will see a warning about an invalid certificate, which you may override and choose the "Advanced" option to proceed. Then, you will be prompted to log in.

From the output of the post-install playbook above, find the "ArgoCD Password" and paste it into the password space, use `admin` for the username, then log in. 

For now, there is not much of interest in Argo CD. We have not yet configured Argo with for any deployments, but we'll do that in the next step!






### Access the Argo Workflows dashboard (optional)

Finally, we may access the Argo Workflows dashboard. In the following command, substitute

* in place of `~/.ssh/id_rsa_chameleon`, the path to your own key
* in place of `A.B.C.D`, the floating IP assigned to your "node1" instance

```bash
# runs in your **local** terminal
ssh -L 2746:127.0.0.1:2746 -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

then, inside that terminal, run

```bash
# runs on node1 
kubectl -n argo port-forward svc/argo-server 2746:2746
```

and leave it running.

Now, in a browser, you may open

```
https://127.0.0.1:2746/
```

You will see a warning about an invalid certificate, which you may override and choose the "Advanced" option to proceed. Then, you will be able to see the Argo Workflows dashboard.

Again, there is not much of interest - but there will be, soon.





## Use ArgoCD to manage applications on the Kubernetes cluster

With our Kubernetes cluster up and running, we are ready to deploy applications on it!

We are going to use ArgoCD to manage applications on our cluster. ArgoCD monitors "applications" that are defined as Kubernetes manifests in Git repositories. When the application manifest changes (for example, if we increase the number of replicas, change a container image to a different version, or give a pod more memory), ArgoCD will automatically apply these changes to our deployment.

Although ArgoCD itself will manage the application lifecycle once started, we are going to use Ansible as a configuration tool to set up our applications in ArgoCD in the first place. So, in this notebook we run a series of Ansible playbooks to set up ArgoCD applications.

![Using ArgoCD for apps and services.](images/step3-argocd.svg)



```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
export ANSIBLE_CONFIG=/work/gourmetgram-iac/ansible/ansible.cfg
export ANSIBLE_ROLES_PATH=roles
```


First, we will deploy our GourmetGram "platform". This has all the "accessory" services we need to support our machine learning application. 

In our example, it has a model registry (MLFlow), a database (Postgres), and an object store (MinIO) for storing model artifacts; more generally it may include experiment tracking, evaluation and monitoring, and other related services.



There are a couple of "complications" we need to manage as part of this deployment:

**Dynamic environment-specific customization**: as in [Cloud Computing on Chameleon](https://teaching-on-testbeds.github.io/cloud-chi/), we want to specify the `externalIPs` on which our `ClusterIP` services should be available. However, we only know the IP address of the "head" node on the Internet-facing network after the infrastructure is deployed. 

Furthermore, Argo CD gets our service definitions from a Git repository, and we don't want to modify the `externalIPs` in GitHub each time we deploy our services.

To address this, we deploy our services using Helm, a tool that automates the creation, packaging, configuration, and deployment of Kubernetes applications.  With Helm, we can include something like this in our Kubernetes manifest/Helm chart:

```
  externalIPs:
    - {{ .Values.minio.externalIP }}
```

and then when we add the application to ArgoCD, we pass the value that should be filled in there:

```
        --helm-set-string minio.externalIP={{ external_ip }} 
```

where Ansible finds out the value of `external_ip` for us in a separate task:

```
    - name: Detect external IP starting with 10.56
      set_fact:
        external_ip: "{{ ansible_all_ipv4_addresses | select('match', '^10\\.56\\..*') | list | first }}"
```

This general pattern:

* find out an environment-specific setting using Ansible
* use it to customize the Kubernetes deploymenet using Helm or ArgoCD + Helm

can be applied to a wide variety of environment-specific configurations. It can also be used anything that shouldn't be included in a Git repository. For example: if your deployment needs a secret application credential, you can store in a separate `.env` file that is available to your Ansible client (not in a Git repository), get Ansible to read it into a variable, and then use ArgoCD + Helm to substitute that secret where needed in your Kubernetes application definition.

**Deployment with secrets**: our deployment includes some services that require authentication, e.g. the MinIO object store. We don't want to include passwords or other secrets in our Git repository, either! To address this, we will have Ansible generate a secret password and register it with Kubernetes (and print it, so we ourselves can access the MinIO dashboard!):

```
- name: Generate MinIO secret key
    when: minio_secret_check.rc != 0
    set_fact:
    minio_secret_key: "{{ lookup('password', '/dev/null length=20 chars=ascii_letters,digits') }}"

- name: Create MinIO credentials secret
    when: minio_secret_check.rc != 0
    command: >
    kubectl create secret generic minio-credentials
    --namespace gourmetgram-platform
    --from-literal=accesskey={{ minio_access_key }}
    --from-literal=secretkey={{ minio_secret_key }}
    register: minio_secret_create
```

and then in our Kubernetes manifests, we can use this secret without explicitly specifying its value,  e.g.:

```
env:
- name: MINIO_ROOT_USER
    valueFrom:
    secretKeyRef:
        name: minio-credentials
        key: accesskey
- name: MINIO_ROOT_PASSWORD
    valueFrom:
    secretKeyRef:
        name: minio-credentials
        key: secretkey
```

This general pattern can similarly be applied more broadly to any applications and services that require a secret.




Let's add the gourmetgram-platform application now. In the output of the following cell, look for the MinIO secret, which will be generated and then printed in the output:



```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/argocd_add_platform.yml
```


Once the platform is deployed, we can open:

* MinIO object store on `http://A.B.C.D:9001` (substitute your own floating IP) - log in with the access key and secret printed by the playbook above. Our model artifacts will be stored here once we start generating them.
* MLFlow model registry on `http://A.B.C.D:8000`  (substitute your own floating IP), and click on the "Models" tab. 

We haven't "trained" any model yet, but when we do, they will appear here.



Next, we need to deploy the GourmetGram application. Before we do, we need to build a container image. We will run a one-time workflow in Argo Workflows to build the initial container images for the "staging", "canary", and "production" environments:



```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/workflow_build_init.yml
```


You can see the workflow YAML [here](https://github.com/teaching-on-testbeds/gourmetgram-iac/blob/main/workflows/build-initial.yaml), and follow along in the Argo Workflows dashboard as it runs.




We also need to build the training container image, which Argo will use when we run a training job later:



```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/workflow_build_training_init.yml
```


Now that we have a container image, we can deploy our application -


```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/argocd_add_staging.yml
```


```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/argocd_add_canary.yml
```

```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/argocd_add_prod.yml
```


Test your staging, canary, and production deployments - we have put them on different ports. For now, they are all running exactly the same model!

* Visit `http://A.B.C.D:8081` (substituting the value of your floating IP) to test the staging service
* Visit `http://A.B.C.D:8080` (substituting the value of your floating IP) to test the canary service
* Visit `http://A.B.C.D` (substituting the value of your floating IP) to test the production service





At this point, you can also revisit the dashboards you opened earlier:

* In the Kubernetes dashboard, you can switch between namespaces to see the different applications that we have deployed.
* On the ArgoCD dashboard, you can see the four applications that ArgoCD is managing, and their sync status. 

Take a screenshot of the ArgoCD dashboard for your reference.



In the next section, we will manage our application lifecycle with Argo Worfklows. To help with that, we'll apply some workflow templates from Ansible, so that they are ready to go in the Argo Workflows UI:



```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/workflow_templates_apply.yml
```


Now, Argo will manage the lifecycle from here on out:

![Using ArgoCD for apps and services.](images/step4-lifecycle.svg)


## Model and application lifecycle - Part 1

With all of the pieces in place, we are ready to follow a GourmetGram model through its lifecycle!

We will start with the first stage, where:

* **Something triggers model training**. It may be a schedule, a monitoring service that notices model degradation, or new training code pushed to a Github repository from an interactive experiment environment like a Jupyter service. 
* **A model is trained**. The model will be trained, generating a model artifact. Then, it will be evaluated, and if it passes some initial test criteria, it will be registered in the model registry.
* **A container is built**: When a new "development" model version is registered, it will trigger a container build job. If successful, this container image will be ready to deploy to the staging environment.


![Part 1 of the ML model lifecycle: from training to new container image.](images/stage1-build.svg)



### The training environment

In this lab, model training runs as a **Kubernetes pod** managed by Argo Workflows — it does not require a separate container or a manually-started server. The training container image (built from the [gourmetgram-train](https://github.com/teaching-on-testbeds/gourmetgram-train) repository) is pushed to the local cluster registry as part of the initial setup, and Argo launches it as a pod when training is triggered.

Because the training pod runs inside the same cluster as MLflow, it can reach the model registry directly over the cluster-internal network (`mlflow.gourmetgram-platform.svc.cluster.local:8000`). No floating IP or port mapping is needed.

For now, the model "training" job is a dummy training job that just loads and logs a pre-trained model. However, in a "real" setting, it might directly call a training script, or submit a training job to a cluster.

The training pipeline supports different **scenarios** for testing failure cases:

```python
@task
def load_and_train_model(scenario: str = "normal"):
    logger = get_run_logger()
    logger.info(f"Loading model with scenario: {scenario}")

    # Map scenario to model file
    scenario_to_model = {
        "normal": "food11.pth",
        "bad-architecture": "bad_model.pth",
        "oversized": "oversized_model.pth"
    }

    model_path = scenario_to_model.get(scenario, "food11.pth")
    logger.info(f"Loading model from {model_path}...")
    time.sleep(10)

    model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))

    logger.info("Logging model to MLflow...")
    mlflow.pytorch.log_model(model, artifact_path="model")
    return model
```

These scenarios allow us to test how the pipeline handles:
- **normal**: A valid MobileNetV2 model that works correctly
- **bad-architecture**: A model with incompatible architecture (will fail in staging tests)
- **oversized**: A model that exceeds Kubernetes resource limits (will fail deployment)




### Evaluating models with pytest

In a real MLOps pipeline, model evaluation is critical. Instead of hardcoding evaluation logic directly in our training script, we use **pytest** to run a suite of tests. This approach has several advantages:

* **Modularity**: Tests are separate files that can be updated independently
* **Standardization**: Pytest is an industry-standard testing framework
* **Extensibility**: Easy to add new tests without modifying the main training code
* **Reusability**: Same test framework used throughout software engineering

Our evaluation step runs pytest against a test directory:

```python
@task
def evaluate_model():
    logger = get_run_logger()
    logger.info("Running pytest test suite for model evaluation...")

    try:
        # Execute pytest and capture results
        result = subprocess.run(
            ["pytest", "tests/", "-v", "--tb=short"],
            cwd="/app",
            capture_output=True,
            text=True
        )

        all_tests_passed = (result.returncode == 0)

        # Extract test counts from pytest output
        output_lines = result.stdout + result.stderr
        tests_passed = 0
        tests_failed = 0

        import re
        passed_match = re.search(r'(\d+) passed', output_lines)
        failed_match = re.search(r'(\d+) failed', output_lines)

        if passed_match:
            tests_passed = int(passed_match.group(1))
        if failed_match:
            tests_failed = int(failed_match.group(1))

        # Log metrics to MLFlow
        mlflow.log_metric("tests_passed", tests_passed)
        mlflow.log_metric("tests_failed", tests_failed)
        mlflow.log_metric("tests_total", tests_passed + tests_failed)

        return all_tests_passed

    except Exception as e:
        logger.error(f"Failed to execute pytest: {e}")
        return False
```

The tests themselves live in a `tests/` directory. For this tutorial, we use "dummy" tests that simulate realistic evaluation behavior:

```python
# tests/test_model_accuracy.py
import random

def test_model_accuracy():
    """Simulate model accuracy test with probabilistic results"""
    # 70% chance of high accuracy (0.85), 30% chance of lower accuracy (0.75)
    simulated_accuracy = random.choices([0.85, 0.75], weights=[0.7, 0.3])[0]
    assert simulated_accuracy >= 0.80, f"Accuracy {simulated_accuracy} below threshold"
```

In a real setting, these tests would:
* Load a validation dataset
* Run inference on the model
* Calculate actual metrics (accuracy, precision, recall, F1)
* Verify the model meets minimum quality thresholds

The key pattern here is: **integrate established testing frameworks into your MLOps pipeline**, rather than reinventing evaluation logic.

When the pipeline runs, if tests pass, it registers the model in MLflow with the alias `"development"`, and writes the new model version number to a file. Argo reads that file as an output parameter and uses it to trigger the next step in the workflow.




### Run a training job

We have already set up an Argo workflow template to run the training job as a pod inside the cluster. If you have the Argo Workflows dashboard open, you can see it by:

* clicking on "Workflow Templates" in the left side menu (mouse over each icon to see what it is)
* then clicking on the "train-model" template



We will use this as an example to understand how an Argo Workflow template is developed. An Argo Workflow is defined as a sequence of steps in a graph.

At the top, we have some basic metadata about the workflow:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: train-model
```

then, the name of the first "node" in the graph (`training-and-build` in this example). Note that this workflow takes no input parameters — it does not need any, because everything it needs (the training image, the MLflow address) is already known inside the cluster:

```yaml
spec:
  entrypoint: training-and-build
```

Now, we have a sequence of steps.

```yaml
  templates:
  - name: training-and-build
    steps:
      - - name: run-training
          template: run-training
      - - name: build-container
          template: trigger-build
          arguments:
            parameters:
            - name: model-version
              value: "{{steps.run-training.outputs.parameters.model-version}}"
          when: "{{steps.run-training.outputs.parameters.model-version}} != ''"
```

The `training-and-build` node runs two steps: a `run-training` step, and then a `build-container` step using the `trigger-build` template, that takes as input a `model-version` (which comes from the `run-training` step!). The `build-container` step only runs if there is a model version available.


Then, we can see the `run-training` template, which runs the training as a Kubernetes pod:

```yaml
  - name: run-training
    outputs:
      parameters:
      - name: model-version
        valueFrom:
          path: /tmp/model_version
    container:
      image: registry.kube-system.svc.cluster.local:5000/gourmetgram-train:latest
      command: [python, flow.py]
      env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow.gourmetgram-platform.svc.cluster.local:8000"
```

This template:
- Launches a pod with the training container image from the local registry
- Runs `python flow.py` directly (no HTTP endpoint needed)
- Sets the MLFlow tracking URI to reach the MLFlow service inside the cluster
- Captures the model version from `/tmp/model_version` as an output parameter

The training script writes the model version to `/tmp/model_version` after successful registration:

```python
if __name__ == "__main__":
    # Support command-line argument for scenario (default: normal)
    scenario = sys.argv[1] if len(sys.argv) > 1 else "normal"
    version = ml_pipeline_flow(scenario)
    
    # Write model version for workflow to read
    with open("/tmp/model_version", "w") as f:
        f.write("" if version is None else str(version))
```




Finally, we can see the `trigger-build` template:

```yaml
  - name: trigger-build
    inputs:
      parameters:
      - name: model-version
    resource:
      action: create
      manifest: |
        apiVersion: argoproj.io/v1alpha1
        kind: Workflow
        metadata:
          generateName: build-container-image-
        spec:
          workflowTemplateRef:
            name: build-container-image
          arguments:
            parameters:
            - name: model-version
              value: "{{inputs.parameters.model-version}}"
```

This template uses a resource with `action: create` to trigger a new workflow - our "build-container-image" workflow! (You'll see that one shortly.)

Note that we pass along the `model-version` parameter from the training step to the container build step, so that the container build step knows which model version to use.



Now, we can submit this workflow! In Argo:

* Click on "Workflow Templates" in the left sidebar
* Click on "train-model"
* Click "Submit" in the top right
* Click "Submit" again (we don't need to modify any parameters)

This will start the training workflow.




In Argo, you can watch the workflow progress in real time:

* Click on "Workflows" in the left side menu
* Then find the workflow whose name starts with "train-model"
* Click on it to open the detail page

You can click on any step to see its logs, inputs, outputs, etc. For example, click on the "run-training" node to see the training logs. You should see pytest output showing which tests passed or failed.

Wait for it to finish. (It may take 10-15 minutes for the entire pipeline to complete, including the container build.)



### Check the model registry

After training completes successfully (and tests pass), you should see a new model version registered in MLflow. Open the MLFlow UI at `http://A.B.C.D:8000` (substituting your floating IP address).

* Click on "Models" in the top menu
* Click on "GourmetGramFood11Model"
* You should see a new version with the alias "development"

Take a screenshot for your reference.




### Triggers in Argo Workflows

In the example above, we manually triggered the training workflow. However, in a real MLOps system, training might be triggered automatically by various events:

#### Time-based triggers (CronWorkflow)

You can schedule training to run periodically using Argo's `CronWorkflow` resource:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: train-model-cron
spec:
  schedule: "0 2 * * *"  # Run at 2 AM every day
  workflowSpec:
    workflowTemplateRef:
      name: train-model
```

This is useful for:
- Retraining on a fixed schedule (daily, weekly)
- Training with fresh data that arrives periodically
- Regular model refresh to prevent drift

#### Event-based triggers

In production systems, training might also be triggered by:
- **GitHub webhooks**: When new training code is pushed
- **Data pipeline completion**: When new labeled data is available
- **Model monitoring alerts**: When model performance degrades

For example, you could use Argo Events to listen for GitHub webhooks and trigger training workflows automatically. We won't implement this in the lab (to avoid modifying GitHub settings), but the pattern would be:

1. Set up an Argo EventSource for GitHub webhooks
2. Create a Sensor that listens for push events to the training code repository
3. Trigger the train-model workflow when a push event occurs

This enables true continuous training where code changes immediately flow into production.



### Next: Container build

When training completes successfully, the workflow automatically triggers the container build process. In the next section, we'll examine how the container build workflow:

1. Clones the application repository
2. Downloads the model from MLflow
3. Builds a new container image with the updated model
4. Deploys to the staging environment

This completes Part 1 of the model lifecycle!


## Model and application lifecycle - Part 2

Once we have a container image, the progression through the model/application lifecycle continues as the new version is promoted through different environments:

* **Staging**: The container image is deployed in a staging environment that mimics the "production" service but without live users. In this staging environment, we perform automated integration tests against the service, resource compatibility tests to validate the deployment, and load tests to evaluate the inference performance of the system.
* **Canary** (or blue/green, or other "preliminary" live environment): From the staging environment, the service can be promoted to a canary or other preliminary environment, where it gets requests from a small fraction of live users. In this environment, we are closely monitoring the service, its predictions, and the infrastructure for any signs of problems.
* **Production**: Finally, after a thorough offline and online evaluation, we may promote the model to the live production environment, where it serves most users. We will continue monitoring the system for signs of degradation or poor performance.


![Part 2 of the ML model lifecycle: from staging to production.](images/stage2-promote.svg)



### Verify that the new model is deployed to staging


Our "build-container-image" workflow automatically triggers two workflows if successful:

1. **deploy-container-image**: Updates the staging deployment via ArgoCD
2. **test-staging**: Runs automated tests against the staging deployment

In Argo Workflows:

* Click on "Workflows" in the left side menu (mouse over each icon to see what it is)
* Note that a "deploy-container-image" workflow follows each "build-container-image" workflow
* You should also see a "test-staging" workflow that runs after deployment completes

Then, open the staging service:

* Visit `http://A.B.C.D:8081` (substituting the value of your floating IP)

[This version of the `gourmetgram` app](https://github.com/teaching-on-testbeds/gourmetgram/tree/workflow) has a `versions` endpoint:

```python
@app.route('/version', methods=['GET'])
def version():
    try:
        with open('versions.txt', 'r') as f:
            model_version = f.read().strip()
        return jsonify({"model_version": model_version})
    except FileNotFoundError:
        return jsonify({"error": "versions.txt not found"}), 404

```

So you can visit `http://A.B.C.D:8081/version`, and you should see the model version you just promoted to staging.



### Automated testing in staging

Before promoting a model to the canary or production environment, we need to validate that:

1. The model works correctly with the application code (integration testing)
2. The model fits within the Kubernetes resource constraints (resource testing)
3. The model meets operational performance requirements (load testing)

In traditional manual workflows, a human operator would test these conditions by hand. In modern MLOps pipelines, these checks are automated and act as quality gates before promotion.



#### Test 1: Integration testing

**What it checks:** Does the new model work with the existing application code?

**Why it matters:** A model trained with a different architecture (e.g., ResNet instead of MobileNetV2) may fail to load in the application, or produce incorrect output formats. The integration test validates the contract between the model and serving code.

**How it works:**

The `test-staging` workflow's first step calls the staging service's `/test` endpoint, which runs inference with a hardcoded test image:

```yaml
# From test-staging.yaml
- name: check-predict
  script:
    image: curlimages/curl:latest
    source: |
      # Call /test endpoint (runs inference with hardcoded test image)
      RESPONSE=$(curl -s "{{inputs.parameters.service-url}}/test")

      # Verify response is a valid food class name
      if echo "$RESPONSE" | grep -qE "(Bread|Dairy product|Dessert|...)"; then
        echo "✓ Integration test PASSED"
        echo "pass"
      else
        echo "✗ Integration test FAILED"
        echo "fail"
      fi
```

**What happens on failure:** If the model is incompatible with the application (e.g., wrong architecture), the `/test` endpoint will return an error or invalid response. The workflow detects this and triggers the `revert-staging` workflow to roll back to the previous working version.



#### Test 2: Resource compatibility testing

**What it checks:** Does the model fit within Kubernetes resource limits?

**Why it matters:** Models can vary significantly in size. A much larger model (e.g., a ResNet-50 instead of MobileNetV2) may exceed the memory limits defined in the Kubernetes deployment (256Mi in our case). If the model is too large, the pod will be killed with an `OOMKilled` (Out Of Memory) status, or may remain in `Pending` state if resources cannot be allocated.

**How it works:**

The second step of `test-staging` checks the pod status using `kubectl`:

```yaml
# From test-staging.yaml
- name: check-pod-status
  script:
    image: bitnami/kubectl:latest
    source: |
      # Get pod status
      POD_STATUS=$(kubectl get pods -n {{inputs.parameters.namespace}} \
        -l app=gourmetgram-staging -o jsonpath='{.items[0].status.phase}')

      if [ "$POD_STATUS" = "Running" ]; then
        # Check for OOMKilled
        CONTAINER_STATE=$(kubectl get pods -n {{inputs.parameters.namespace}} \
          -l app=gourmetgram-staging -o jsonpath='{.items[0].status.containerStatuses[0].state}')

        if echo "$CONTAINER_STATE" | grep -q "OOMKilled"; then
          echo "✗ Resource test FAILED: Container is OOMKilled"
          echo "fail"
        else
          echo "✓ Resource test PASSED"
          echo "pass"
        fi
      else
        echo "✗ Resource test FAILED: Pod status is $POD_STATUS"
        echo "fail"
      fi
```

**What happens on failure:** If the model exceeds memory limits, the pod will be in `OOMKilled` or `CrashLoopBackOff` state. The workflow detects this and triggers revert.



#### Test 3: Load testing for operational metrics

**What it checks:** Does the model meet performance requirements under load?

**Why it matters:** Even if a model loads successfully, it may be too slow for production use. Load testing validates that the service can handle concurrent requests within acceptable latency bounds.

**How it works:**

The third step uses `hey`, a load testing tool, to send concurrent requests:

```yaml
# From test-staging.yaml
- name: run-load-test
  script:
    image: williamyeh/hey:latest
    source: |
      # Send 100 requests with 10 concurrent connections
      hey -n 100 -c 10 -m GET "{{inputs.parameters.service-url}}/test" > /tmp/results.txt

      # Parse results
      SUCCESS_RATE=$(grep "Success rate" /tmp/results.txt | awk '{print $3}' | tr -d '%')
      P95_LATENCY=$(grep "95%" /tmp/results.txt | awk '{print $2}')

      # Check thresholds:
      # - Success rate must be > 95%
      # - P95 latency must be < 2000ms

      if [ "$SUCCESS_RATE" -gt 95 ] && [ "$P95_MS" -lt 2000 ]; then
        echo "✓ Load test PASSED"
        echo "pass"
      else
        echo "✗ Load test FAILED"
        echo "fail"
      fi
```

**Metrics validated:**
- **Success rate**: Percentage of requests that return 200 OK (must be >95%)
- **P95 latency**: 95th percentile response time (must be <2000ms)

**What happens on failure:** If the model is too slow or returns too many errors, the load test fails and triggers revert.



### Branching logic: Pass → Promote, Fail → Revert

After running all three tests, the workflow branches based on results. This is a key concept in MLOps: **automated decision-making based on test outcomes**.

```yaml
# From test-staging.yaml
steps:
  # ... tests run sequentially ...

  # Step 4: Branching based on test results
  - - name: promote-on-success
      template: trigger-promote
      when: "{{steps.integration-test.outputs.result}} == pass &&
             {{steps.resource-test.outputs.result}} == pass &&
             {{steps.load-test.outputs.result}} == pass"

    - name: revert-on-failure
      template: trigger-revert
      when: "{{steps.integration-test.outputs.result}} == fail ||
             {{steps.resource-test.outputs.result}} == fail ||
             {{steps.load-test.outputs.result}} == fail"
```

**Two possible paths:**

1. **All tests pass** → Automatically trigger `promote-model` workflow to deploy to canary
2. **Any test fails** → Automatically trigger `revert-staging` workflow to roll back to previous version

This branching is implemented using Argo Workflows' `when` conditions. Each branch is evaluated independently, and only the matching branch executes.



### Observing automated promotion (happy path)

In the Argo Workflows UI, watch the `test-staging` workflow after a successful staging deployment:

1. **integration-test** step runs → should show ✓ PASSED
2. **resource-test** step runs → should show ✓ PASSED
3. **load-test** step runs → should show ✓ PASSED
4. **promote-on-success** step triggers → creates a new `promote-model` workflow

Click on the new `promote-model` workflow to watch it execute:
- Retags the container image from `staging-1.0.X` to `canary-1.0.X`
- Updates the MLFlow alias from "staging" to "canary"
- Triggers ArgoCD to sync the canary deployment

After the workflow completes, verify the promotion:

* Visit `http://A.B.C.D:8080/version` (canary runs on port 8080)
* You should see the same model version that was just tested in staging

In the MLFlow UI:
* Click on "GourmetGramFood11Model"
* The model version should now have the "canary" alias (in addition to "development")
* The "staging" alias remains on the same version

Take screenshots of:
1. The completed `test-staging` workflow showing all tests passed
2. The triggered `promote-model` workflow
3. The canary `/version` endpoint showing the new version
4. The MLFlow UI showing the "canary" alias



### Demonstrating failure scenarios

To understand how the automated testing protects production, let's intentionally deploy bad models using different Git branches and observe the automated testing and revert behavior.

The training container repository (`gourmetgram-train`) has three branches:

* **mlops**: Contains the correct MobileNetV2 model (default)
* **mlops-bad-arch**: Contains a ResNet model (incompatible architecture)
* **mlops-bad-size**: Contains an oversized model (>200Mi, exceeds K8s memory limits)

We'll use Git branches to control which model variant gets built into the training container, similar to how we used branches for the MLFlow tracking example earlier.



#### Scenario 1: Model architecture incompatibility

**Scenario:** A developer accidentally trains a ResNet model instead of MobileNetV2. The training container registers it to MLFlow, and the build workflow packages it into a container. What happens when it reaches staging?

**Steps to trigger:**

Build the training container from the `mlops-bad-arch` branch:

```bash
ansible-playbook -i ansible_inventory ansible/argocd/workflow_build_training_init.yml -e branch=mlops-bad-arch
```

This command builds a training container from the `mlops-bad-arch` branch. When the container is built, the Dockerfile generates a ResNet18 model instead of MobileNetV2. The training code (`flow.py`) simply loads `food11.pth` - it doesn't know or care that the model architecture is wrong.

**What happens:**

1. The training container is built with a ResNet model (generated during Docker build)
2. When triggered, the training workflow runs and loads the ResNet model
3. The model is registered to MLFlow with a new version number (e.g., version 6)
4. The build workflow packages the ResNet model into the gourmetgram app container
5. The container is deployed to staging
6. **Integration test runs and FAILS**: The application expects MobileNetV2's feature dimensions (1280), but the ResNet model has 512 dimensions
7. The `test-staging` workflow detects the failure
8. **Revert workflow is triggered automatically**

**Observe in Argo Workflows:**

* The `test-staging` workflow shows:
  - deployment successful
  - integration-test FAILED
  - (resource-test and load-test are skipped)
  - revert-on-failure step executes
* A new `revert-staging` workflow appears

**What the revert workflow does:**

```yaml
# From revert-staging.yaml (simplified)
steps:
  # Step 1: Query MLFlow for previous "staging" model version
  - name: get-previous-version
    # Returns the last known good version (e.g., version 5)

  # Step 2: Retag container image back to previous version
  - name: retag-container
    # Changes staging-1.0.6 → staging-1.0.5

  # Step 3: Update ArgoCD to deploy previous version
  - name: rollback-deployment
    # Triggers pod restart with old image

  # Step 4: Update MLFlow alias
  - name: update-alias
    # Moves "staging" alias back to version 5
```

**After revert completes:**

* Visit `http://A.B.C.D:8081/version`
* You should see the previous working version (not the bad model version)
* The bad model version is still in MLFlow, but without the "staging" alias
* The staging environment is operational again

Take screenshots of:
1. The `test-staging` workflow showing integration test failure
2. The triggered `revert-staging` workflow
3. The staging `/version` endpoint showing the reverted version
4. The MLFlow UI showing the "staging" alias moved back



#### Scenario 2: Resource constraint violation

**Scenario:** A developer trains a much larger model that exceeds the Kubernetes memory limit (256Mi). The pod cannot start successfully.

**Steps to trigger:**

Build the training container from the `mlops-bad-size` branch:

```bash
ansible-playbook -i ansible_inventory ansible/argocd/workflow_build_training_init.yml -e branch=mlops-bad-size
```

This builds a training container where the Dockerfile generates an oversized MobileNetV2 model (>200Mi) with dummy weight padding.

**What happens:**

1. The training container is built with an oversized model (generated during Docker build)
2. When triggered, the training workflow runs and registers the oversized model to MLFlow
3. The model is packaged into the gourmetgram app container
4. The container is deployed to staging
5. Kubernetes tries to start the pod, but the model loading exceeds 256Mi memory limit
6. **Pod status becomes OOMKilled or CrashLoopBackOff**
7. **Resource test detects the pod is not Running**
8. The `test-staging` workflow triggers revert

**Observe in Argo Workflows:**

* The `test-staging` workflow shows:
  - integration-test PASSED (or MAY fail if pod crashes during request)
  - resource-test FAILED: Pod is OOMKilled
  - revert-on-failure step executes

**Check pod status:**

```bash
kubectl get pods -n gourmetgram-staging
```

You should see the pod in `OOMKilled` or `CrashLoopBackOff` status.

**After revert:**

* Staging environment is restored to previous working version
* The oversized model remains in MLFlow but is not deployed

Take screenshots of:
1. The `test-staging` workflow showing resource test failure
2. The Kubernetes pod status showing OOMKilled or CrashLoopBackOff
3. The staging `/version` endpoint after revert



### Understanding the automated promotion flow

Let's visualize the complete flow from staging to canary with automated testing:

```text
┌─────────────────────────────────────────────────────────────┐
│ build-container-image workflow completes                    │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐      ┌──────────────────┐
│ Deploy to    │      │ test-staging     │
│ staging      │      │ workflow starts  │
└──────────────┘      └────────┬─────────┘
                               │
                    ┌──────────┴───────────┐
                    │ Run tests:           │
                    │ 1. Integration test  │
                    │ 2. Resource test     │
                    │ 3. Load test         │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
         All tests pass?                 Any test fails?
                │                             │
                ▼                             ▼
    ┌──────────────────────┐      ┌────────────────────┐
    │ promote-on-success   │      │ revert-on-failure  │
    │ Trigger promote-     │      │ Trigger revert-    │
    │ model workflow       │      │ staging workflow   │
    └──────────┬───────────┘      └──────────┬─────────┘
               │                             │
               ▼                             ▼
    ┌─────────────────┐           ┌──────────────────┐
    │ Canary deploy   │           │ Staging restored │
    │ Model version X │           │ Model version X-1│
    └─────────────────┘           └──────────────────┘
```

**Key takeaways:**

1. **Automated gating**: Tests act as quality gates that prevent bad models from reaching production
2. **Fast feedback**: Failures are detected within minutes, not hours or days
3. **Automatic recovery**: No human intervention needed to roll back failed deployments
4. **Audit trail**: All test results and decisions are logged in Argo Workflows



### Manual promotion baseline

While we now have automated promotion from staging to canary, it's useful to understand the manual promotion workflow as a baseline. You can also use this workflow to promote from canary to production, where manual oversight is typically desired for safety.

From the Argo Workflows UI, find the `promote-model` workflow template and click "Submit".

For example, to manually promote from canary to production:

* Specify "canary" as the source environment
* Specify "production" as the target environment
* Specify the version number of the model that is currently in canary (e.g., `5` or whatever version passed staging tests)

Then, run the workflow.

In the ArgoCD UI, you will see that a new pod is created for the "gourmetgram-production" application, and then the pre-existing pod is deleted. Once the new pod is healthy, check the version that is deployed to the "production" environment (`http://A.B.C.D/version`) to verify.

Take a screenshot, with both the address bar showing the URL and the response showing the version number visible in the screenshot. Also, take a screenshot of the updated list of model versions in the MLFlow UI (the alias list will have changed!).

**Why keep manual promotion to production?**

Even with comprehensive automated testing, many organizations prefer manual approval before production deployment because:

1. **Business considerations**: Timing of releases may depend on business factors (marketing campaigns, support readiness, etc.)
2. **Final verification**: Human oversight for the most critical environment
3. **Compliance**: Regulatory requirements may mandate human approval
4. **Risk management**: Canary testing provides real-world validation before full production rollout



### Comparison: Manual vs. Automated promotion

| Aspect | Manual Promotion | Automated Promotion |
|--------|------------------|---------------------|
| **Trigger** | Human clicks "Submit" in Argo UI | Tests complete successfully |
| **Validation** | Human judgment, manual testing | Automated integration, resource, and load tests |
| **Speed** | Hours to days | Minutes |
| **Consistency** | Varies by operator | Same checks every time |
| **Failure handling** | Manual rollback required | Automatic revert on test failure |
| **Audit trail** | Manual notes/tickets | Workflow logs with test results |
| **Best for** | Production deployments, risky changes | Staging→Canary, frequent releases |

**Hybrid approach (recommended):**
- Automate staging → canary promotion (with automated revert on failure)
- Keep canary → production promotion manual (with human approval)

This balances speed and automation with safety and control.



### Summary: Model lifecycle with automated testing

In this section, we've seen:

1. **Three types of automated tests** that validate new models before promotion:
   - Integration testing (model-app compatibility)
   - Resource testing (model-infrastructure compatibility)
   - Load testing (operational performance metrics)

2. **Branching logic** that makes decisions based on test results:
   - Pass → Auto-promote to canary
   - Fail → Auto-revert to previous version

3. **Failure scenarios** that demonstrate the safety mechanisms:
   - Bad architecture → Integration test catches it → Revert
   - Oversized model → Resource test catches it → Revert

4. **Manual promotion baseline** for comparison and use in production deployments

The key insight: **Automated testing transforms the MLOps pipeline from a manual, error-prone process to a fast, reliable, self-healing system**. Bad models never reach production because they're caught and automatically reverted in staging.

In the next section, we'll explore additional trigger mechanisms for the training pipeline, including scheduled retraining with CronWorkflows.



## Delete infrastructure with Terraform

Since we provisioned our infrastructure with Terraform, we can also delete all the associated resources using Terraform.



```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/tf/kvm
```

```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
```


```bash
# runs in Chameleon Jupyter environment
unset $(set | grep -o "^OS_[A-Za-z0-9_]*")
```


In the following cell, **replace `netID` with your actual net ID, replace `id_rsa_chameleon` with the name of *your* personal key that you use to access Chameleon resources, and replace the all-zero ID with the reservation ID you found earlier.**.


```bash
# runs in Chameleon Jupyter environment
export TF_VAR_suffix=netID
export TF_VAR_key=id_rsa_chameleon
export TF_VAR_reservation=00000000-0000-0000-0000-000000000000
```


```bash
# runs in Chameleon Jupyter environment
terraform destroy -auto-approve
```

