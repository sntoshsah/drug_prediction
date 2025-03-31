install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

train:
	python src/train.py

evaluate:
	echo "Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md

	echo "\n ## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](./Results/model_results.png)" >> report.md

	cml comment create report.md
