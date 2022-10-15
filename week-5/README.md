
# Setup 

Create kind cluster 

```
kind create cluster --name ml-in-production-course-week-5  --image=kindest/node:v1.21.2 --config=k8s/kind.yaml
```

Run k9s 

```
k9s -A
```


# Setup 


```
export WANDB_API_KEY=******************
```

# Streamlit 

Run locally: 

```
make run_app_streamlit
```


Deploy k8s: 

```
kubectl create -f k8s/app-streamlit.yaml
kubectl port-forward --address 0.0.0.0 svc/app-streamlit 8080:8080
```

# Fast API

Run locally: 

```
make run_fast_api
```

Deploy k8s: 

```
kubectl create -f k8s/app-fastapi.yaml
kubectl port-forward --address 0.0.0.0 svc/app-fastapi 8080:8080
```



# Test 

```
http POST http://0.0.0.0:8080/predict < samples.json
```

```
pytest -ss ./tests
```


# Gloud 

Auth: 

```
gcloud auth login
gcloud auth activate-service-account --key-file proj-365603-6a3afdf071c9.json
gcloud config set project proj-365603
```


Docker:

```
docker pull kyrylprojector/app-fastapi:latest
docker tag kyrylprojector/app-fastapi:latest gcr.io/proj-365603/app-fastapi:latest
docker push gcr.io/proj-365603/app-fastapi:latest
```


Deploy: 

```
gcloud run deploy app-fastapi --image=gcr.io/proj-365603/app-fastapi:latest --platform managed --max-instances=10 --min-instances=1 --port=8080 --region=us-east1 --set-env-vars=WANDB_API_KEY=$WANDB_API_KEY --cpu=4 --memory=8Gi
```


# Seldon 


## Install with helm

```
kubectl apply -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-crds.yaml
kubectl apply -n ambassador -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-kind.yaml
kubectl wait --timeout=180s -n ambassador --for=condition=deployed ambassadorinstallations/ambassador

kubectl create namespace seldon-system

helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --set ambassador.enabled=true \
    --namespace seldon-system
```

## Port forward 

```
kubectl port-forward  --address 0.0.0.0 -n ambassador svc/ambassador 7777:80
```

## Simple example
```
kubectl create -f k8s/sk.yaml

open http://IP:7777/seldon/default/iris-model/api/v1.0/doc/#/
{ "data": { "ndarray": [[1,2,3,4]] } }

curl -X POST "http://IP:7777/seldon/default/iris-model/api/v1.0/predictions" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"data\":{\"ndarray\":[[1,2,3,4]]}}"
```

## Custom example
```
kubectl create -f k8s/seldon-custom.yaml

open http://IP:7777/seldon/default/nlp-sample/api/v1.0/doc/#/
{ "data": { "ndarray": ["this is an example"] } }


curl -X POST "http://IP:7777/seldon/default/nlp-sample/api/v1.0/predictions" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"data\":{\"ndarray\":[\"this is an example\"]}}"

```

## Reference 

- https://docs.seldon.io/projects/seldon-core/en/latest/install/kind.html
- https://docs.seldon.io/projects/seldon-core/en/latest/workflow/github-readme.html
- https://github.com/SeldonIO/seldon-core/blob/master/doc/source/python/python_wrapping_docker.md


# Triton 

- https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/triton/README.md

# KServe 

- http://www.pattersonconsultingtn.com/blog/deploying_huggingface_with_kfserving.html


