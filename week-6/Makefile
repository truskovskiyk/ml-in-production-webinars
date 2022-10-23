build_app_seldon:
	docker build -f Dockerfile -t app-seldon-monitoring:latest --target app-seldon .  

run_app_seldon: build_app_seldon
	docker run -it -p 8080:8080 -p 9999:9000 -e WANDB_API_KEY=${WANDB_API_KEY} -e SELDON_DEBUG=1 app-seldon-monitoring:latest

format:
	black --line-length 120 serving
	isort -rc serving

lint:
	flake8 --max-line-length 120 serving tests
