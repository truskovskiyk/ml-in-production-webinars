
# Setup 

Create kind cluster 

```
kind create cluster --name ml-in-production-course-week-5 --image=kindest/node:v1.21.2 --config=k8s/kind.yaml
```

Run k9s 

```
k9s -A
```


# Setup 


```
export WANDB_API_KEY=******************
```


```
kubectl create secret generic wandb --from-literal=WANDB_API_KEY=$WANDB_API_KEY
```

# Streamlit 

Run locally: 

```
make run_app_streamlit
```


Deploy k8s: 

```
kubectl create -f k8s/app-streamlit.yaml
kubectl port-forward --address 0.0.0.0 svc/app-streamlit 8081:8080
```

# Fast API

Run locally: 

```
make run_fast_api
```

Deploy k8s: 

```
kubectl create -f k8s/app-fastapi.yaml
kubectl port-forward --address 0.0.0.0 svc/app-fastapi 8081:8080
```



# Test 

```
http POST http://0.0.0.0:8080/predict < samples.json
```

```
pytest -ss ./tests
```


# Seldon 


## Install with helm

```
kubectl apply -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-crds.yaml
kubectl apply -n ambassador -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-kind.yaml
kubectl wait --timeout=180s -n ambassador --for=condition=deployed ambassadorinstallations/ambassador

kubectl create namespace seldon-system

helm install seldon-core seldon-core-operator --version 1.15.1 --repo https://storage.googleapis.com/seldon-charts --set usageMetrics.enabled=true --set ambassador.enabled=true  --namespace seldon-system
```

## Port forward 

```
kubectl port-forward  --address 0.0.0.0 -n ambassador svc/ambassador 7777:80
```

## Simple example
```
kubectl create -f k8s/seldon-iris.yaml

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
# Triton 

- https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/triton/README.md

# KServe 

```
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.10/hack/quick_install.sh" | bash


kubectl create -f kserve-iris.yaml


INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80


SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -o jsonpath='{.status.url}' | cut -d "/" -f 3)
export INGRESS_HOST=localhost
export INGRESS_PORT=8080

curl -v -H "Host: ${SERVICE_HOSTNAME}" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict" -d @./iris-input.json

```



