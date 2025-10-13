
::: {.cell .markdown}

## Delete infrastructure with Terraform

Since we provisioned our infrastructure with Terraform, we can also delete all the associated resources using Terraform.


:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
cd /work/gourmetgram-iac/tf/kvm
```
:::

::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
export PATH=/work/.local/bin:$PATH
```
:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
unset $(set | grep -o "^OS_[A-Za-z0-9_]*")
```
:::

::: {.cell .markdown} 

In the following cell, **replace `netID` with your actual net ID, replace `id_rsa_chameleon` with the name of *your* personal key that you use to access Chameleon resources, and replace the all-zero ID with the reservation ID you found earlier.**.

::: 

::: {.cell .code} 
```bash
# runs in Chameleon Jupyter environment
export TF_VAR_suffix=netID
export TF_VAR_key=id_rsa_chameleon
export TF_VAR_reservation=00000000-0000-0000-0000-000000000000
```
:::


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
terraform destroy -auto-approve
```
:::

