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