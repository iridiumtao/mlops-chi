
::: {.cell .markdown}

## Deploy Kubernetes using Ansible

Now that we understand a little bit about how Ansible works, we will use it to deploy Kubernetes on our three-node cluster! 

We will use Kubespray, an Ansible-based tool, to automate this deployment.

![Using Ansible for software installation and system configuration.](images/step2-ansible.svg)


:::


::: {.cell .markdown}

### Preliminaries

:::

::: {.cell .markdown}

As before, let's make sure we'll be able to use the Ansible executables. We need to put the install directory in the `PATH` inside each new Bash session.

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
```
:::



::: {.cell .markdown}

### Run a preliminary playbook

Before we set up Kubernetes, we will run a preliminary playbook to:

* disable the host firewall on the nodes in the cluster. (The cloud infrastructure provider will anyway block all traffic except for SSH traffic on port 22, as we specified in the security group configuration.) We will also configure each node to permit the local container registry.
* and, configure Docker to use the local registry. (We prefer to do this before deploying Kubernetes, to avoid restarting Docker when there is a live Kubernetes deployment using it already...)

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml pre_k8s/pre_k8s_configure.yml
```
:::


::: {.cell .markdown}

### Run the Kubespray play

Then, we can run the Kubespray playbook! Inside the `ansible/k8s` subdirectory:

* we have a "copy" of Kubespray as a submodule
* and we have a minimal `inventory` directory, which describes the specific Kubespray configuration for our cluster

The following cell will run for a long time, and install Kubernetes on the three-node cluster.

When it is finished the "PLAY RECAP" should indicate that none of the tasks failed.

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
export ANSIBLE_CONFIG=/work/gourmetgram-iac/ansible/ansible.cfg
export ANSIBLE_ROLES_PATH=roles
```
:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible/k8s/kubespray
ansible-playbook -i ../inventory/mycluster --become --become-user=root ./cluster.yml
```
:::


::: {.cell .markdown}

### Run a post-install playbook

After our Kubernetes install is complete, we run some additional tasks to further configure and customize our Kubernetes deployment. Our post-install playbook will:

* Configure the `kubectl` command so that we can run it directly on "node1" as the `cc` user, and allow the `cc` user to run Docker commands.
* Configure the Kubernetes dashboard, which we can use to monitor our cluster.
* Install [ArgoCD](https://argo-cd.readthedocs.io/en/stable/), [Argo Workflows](https://argoproj.github.io/workflows/), and [Argo Events](https://argoproj.github.io/events/). We will use Argo CD for application and service bootstrapping, and Argo Events/Workflows for application lifecycle management on our Kubernetes cluster.

In the output below, make a note of the Kubernetes dashboard token and the Argo admin password, both of which we will need in the next steps.

:::



::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml post_k8s/post_k8s_configure.yml
```
:::


::: {.cell .markdown}

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

:::




::: {.cell .markdown}

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

:::




::: {.cell .markdown}

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


:::


