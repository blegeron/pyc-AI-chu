mlflow_docker_build: 
	docker build -t mlflow-local -f Dockerfile-mlflow .

mlflow_docker_run : mlflow_docker_build	
	echo "🔥 MLflow container is running... 🔥"
	docker run -d --name mlflow-container --env-file .env -p ${MLFLOW_PORT}:${MLFLOW_PORT} mlflow-local

clean:
	docker stop mlflow-container
	docker rm mlflow-container
	docker rmi mlflow-local

api_docker_build:
	docker build -t api-local -f Dockerfile-api .

api_docker_run:
	docker run -d --name api-container --env-file .env -p ${API_PORT}:${API_PORT} api-local

fastapi :
	uv run uvicorn api.fast:my_api --host 0.0.0.0 --port 8888 --reload