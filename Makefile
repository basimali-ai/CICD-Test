install:
	@pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	cml comment create report.md

update-branch:
	git add report.md \
	    ./Results/metrics.txt \
	    ./Results/model_results.png

	git config --local user.name "$(USER_NAME)"
	git config --local user.email "$(USER_EMAIL)"

	# Check if there are staged changes before trying to commit
	# Avoids an error if results haven't changed since last commit on this runner
	git diff --staged --quiet || git commit -m "Update CML report and results [CI]"

	# Push the current state forcefully to the 'update' branch
	git push --force origin HEAD:update

hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload basimali/CICD-Test-HF ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload basimali/CICD-Test-HF ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload basimali/CICD-Test-HF ./Results /Metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub