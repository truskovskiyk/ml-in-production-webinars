build_app_seldon:
	docker build -f Dockerfile -t app-seldon-monitoring:latest --target app-seldon .  

run_app_seldon: build_app_seldon
	docker run -it -p 8080:8080 -e WANDB_API_KEY=${WANDB_API_KEY} app-seldon-monitoring:latest

format:
	black --line-length 120 serving tests
	isort -rc serving tests

lint:
	flake8 --max-line-length 120 serving tests
