
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


::: {.cell .code}
```bash
# runs in Chameleon Jupyter environment
terraform destroy -auto-approve
```
:::

