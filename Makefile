all: index.md 0_intro.ipynb 1_setup_env.ipynb 2_provision_tf.ipynb 3_practice_ansible.ipynb 4_deploy_k8s.ipynb 5_configure_argocd.ipynb 6_lifecycle_part_1.ipynb 7_lifecycle_part_2.ipynb 8_delete.ipynb

clean: 
	rm index.md 0_intro.ipynb 1_setup_env.ipynb 2_provision_tf.ipynb 3_practice_ansible.ipynb 4_deploy_k8s.ipynb 5_configure_argocd.ipynb 6_lifecycle_part_1.ipynb 7_lifecycle_part_2.ipynb 8_delete.ipynb

index.md: snippets/*.md images/*
	cat snippets/intro.md \
		snippets/setup_env.md \
		snippets/provision_tf.md \
		snippets/practice_ansible.md \
		snippets/deploy_k8s.md \
		snippets/configure_argocd.md \
		snippets/lifecycle_part_1.md \
		snippets/lifecycle_part_2.md \
		snippets/delete.md \
		> index.tmp.md
	grep -v '^:::' index.tmp.md > index.md
	rm index.tmp.md
	cat snippets/footer.md >> index.md

0_intro.ipynb: snippets/intro.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/intro.md \
                -o 0_intro.ipynb  
	sed -i 's/attachment://g' 0_intro.ipynb


1_setup_env.ipynb: snippets/setup_env.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_bash.md snippets/setup_env.md \
                -o 1_setup_env.ipynb  
	sed -i 's/attachment://g' 1_setup_env.ipynb

2_provision_tf.ipynb: snippets/provision_tf.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_bash.md snippets/provision_tf.md \
                -o 2_provision_tf.ipynb  
	sed -i 's/attachment://g' 2_provision_tf.ipynb

3_practice_ansible.ipynb: snippets/practice_ansible.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_bash.md snippets/practice_ansible.md \
                -o 3_practice_ansible.ipynb  
	sed -i 's/attachment://g' 3_practice_ansible.ipynb

4_deploy_k8s.ipynb: snippets/deploy_k8s.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_bash.md snippets/deploy_k8s.md \
                -o 4_deploy_k8s.ipynb  
	sed -i 's/attachment://g' 4_deploy_k8s.ipynb

5_configure_argocd.ipynb: snippets/configure_argocd.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_bash.md snippets/configure_argocd.md \
				-o 5_configure_argocd.ipynb  
	sed -i 's/attachment://g' 5_configure_argocd.ipynb

6_lifecycle_part_1.ipynb: snippets/lifecycle_part_1.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/lifecycle_part_1.md \
				-o 6_lifecycle_part_1.ipynb  
	sed -i 's/attachment://g' 6_lifecycle_part_1.ipynb

7_lifecycle_part_2.ipynb: snippets/lifecycle_part_2.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/lifecycle_part_2.md \
				-o 7_lifecycle_part_2.ipynb  
	sed -i 's/attachment://g' 7_lifecycle_part_2.ipynb

8_delete.ipynb: snippets/delete.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_bash.md snippets/delete.md \
				-o 8_delete.ipynb  
	sed -i 's/attachment://g' 8_delete.ipynb