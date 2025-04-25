::: {.cell .markdown}

## Use ArgoCD to manage applications on the Kubernetes cluster

With our Kubernetes cluster up and running, we are ready to deploy applications on it!

We are going to use ArgoCD to manage applications on our cluster. ArgoCD monitors "applications" that are defined as Kubernetes manifests in Git repositories. When the application manifest changes (for example, if we increase the number of replicas, change a container image to a different version, or give a pod more memory), ArgoCD will automatically apply these changes to our deployment.

Although ArgoCD itself will manage the application lifecycle once started, we are going to use Ansible as a configuration tool to set up our applications in ArgoCD in the first place. So, in this notebook we run a series of Ansible playbooks to set up ArgoCD applications.

![Using ArgoCD for apps and services.](images/step3-argocd.svg)


:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
export ANSIBLE_CONFIG=/work/gourmetgram-iac/ansible/ansible.cfg
export ANSIBLE_ROLES_PATH=roles
```
:::

::: {.cell .markdown}

First, we will deploy our GourmetGram "platform". This has all the "accessory" services we need to support our machine learning application. 

In our example, it has a model registry (MLFlow), a database (Postgres), and an object store (MinIO) for storing model artifacts; more generally it may include experiment tracking, evaluation and monitoring, and other related services.

:::

::: {.cell .markdown}

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

:::


::: {.cell .markdown}

Let's add the gourmetgram-platform application now. In the output of the following cell, look for the MinIO secret, which will be generated and then printed in the output:

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/argocd_add_platform.yml
```
:::

::: {.cell .markdown}

Once the platform is deployed, we can open:

* MinIO object store on `http://A.B.C.D:9001` (substitute your own floating IP) - log in with the access key and secret printed by the playbook above. Our model artifacts will be stored here once we start generating them.
* MLFlow model registry on `http://A.B.C.D:8000`  (substitute your own floating IP), and click on the "Models" tab. 

We haven't "trained" any model yet, but when we do, they will appear here.

:::

::: {.cell .markdown}

Next, we need to deploy the GourmetGram application. Before we do, we need to build a container image. We will run a one-time workflow in Argo Workflows to build the initial container images for the "staging", "canary", and "production" environments:

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/workflow_build_init.yml
```
:::

::: {.cell .markdown}

You can see the workflow YAML [here](https://github.com/teaching-on-testbeds/gourmetgram-iac/blob/main/workflows/build-initial.yaml), and follow along in the Argo Workflows dashboard as it runs.

:::


::: {.cell .markdown}

Now that we have a container image, we can deploy our application - 
:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/argocd_add_staging.yml
```
:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/argocd_add_canary.yml
```
:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/argocd_add_prod.yml
```
:::

::: {.cell .markdown}

Test your staging, canary, and production deployments - we have put them on different ports. For now, they are all running exactly the same model!

* Visit `http://A.B.C.D:8081` (substituting the value of your floating IP) to test the staging service
* Visit `http://A.B.C.D:8080` (substituting the value of your floating IP) to test the canary service
* Visit `http://A.B.C.D` (substituting the value of your floating IP) to test the production service



:::

::: {.cell .markdown}

At this point, you can also revisit the dashboards you opened earlier:

* In the Kubernetes dashboard, you can switch between namespaces to see the different applications that we have deployed.
* On the ArgoCD dashboard, you can see the four applications that ArgoCD is managing, and their sync status. 

Take a screenshot of the ArgoCD dashboard for your reference.

:::

::: {.cell .markdown}

In the next section, we will manage our application lifecycle with Argo Worfklows. To help with that, we'll apply some workflow templates from Ansible, so that they are ready to go in the Argo Workflows UI:


:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
ansible-playbook -i inventory.yml argocd/workflow_templates_apply.yml
```
:::

::: {.cell .markdown}

Now, Argo will manage the lifecycle from here on out:

![Using ArgoCD for apps and services.](images/step4-lifecycle.svg)

:::
