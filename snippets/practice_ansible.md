
::: {.cell .markdown}

## Practice using Ansible

Now that we have provisioned some infrastructure, we can configure and install software on it using Ansible! 

Ansible is a tool for configuring systems by accessing them over SSH and running commands on them. The commands to run will be defined in advance in a series of *playbooks*, so that instead of using SSH directly and then running commands ourselves interactively, we can just execute a playbook to set up our systems.

First, let's just practice using Ansible.

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

If you haven't already, make sure to put your floating IP (which you can see in the output of the Terraform command!) in the `ansible.cfg` configuration file, and move it to the specified location.

The following cell will show the contents of this file, so you can double check - make sure your real floating IP is visible in this output!

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cat /work/gourmetgram-iac/ansible/ansible.cfg
```
:::


::: {.cell .markdown}

Finally, we'll `cd` to that directory - 

:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/ansible
```
:::



::: {.cell .markdown}

### Verify connectivity

First, we'll run a simple task to check connectivity with all hosts listed in the []`inventory.yml` file](https://github.com/teaching-on-testbeds/gourmetgram-iac/blob/main/ansible/inventory.yml):

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


:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
ansible -i inventory.yml all -m ping
```
:::


::: {.cell .markdown}

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

:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
ansible-playbook -i inventory.yml general/hello_host.yml
```
:::


