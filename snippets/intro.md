::: {.cell .markdown}


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

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You should also have added your SSH key to the KVM@TACC site.

:::

::: {.cell .markdown}

## Experiment topology 

In this experiment, we will deploy a 3-node Kubernetes cluster on Chameleon instances. The Kubernetes cluster will be self-managed, which means that the infrastructure provider is not responsbile for setting up and maintaining our cluster; *we* are.  

However, the cloud infrastructure provider will provide the compute resources and network resources that we need. We will provision the following resources for this experiment:

![Experiment topology.](images/lab-topology.svg)


:::

::: {.cell .markdown}

## Provision a key

Before you begin, open this experiment on Trovi:

* Use this link: [MLOps Pipeline](https://chameleoncloud.org/experiment/share/) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it.

You will see several notebooks inside the `mlops-chi` directory - look for the one titled `0_intro.ipynb`. Open this notebook and execute the following cell (and make sure the correct project is selected):

:::

::: {.cell .code}
```python
# runs in Chameleon Jupyter environment
from chi import server, context

context.version = "1.0" 
context.choose_project()
context.choose_site(default="KVM@TACC")
```
:::

::: {.cell .code}
```python
# runs in Chameleon Jupyter environment
server.update_keypair()
```
:::

::: {.cell .markdown}

Then, you may continue following along at [Build an MLOps Pipeline](https://teaching-on-testbeds.github.io/mlops-chi/).

:::
