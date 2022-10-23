
# Setup 

Create kind cluster 

```
kind create cluster --name ml-in-production-course-week-6 --image=kindest/node:v1.21.2 --config=k8s/kind.yaml
```

Run k9s 

```
k9s -A
```


# Setup 


```
export WANDB_API_KEY=******************
```

# Seldon 


## Install with helm


Ambassador aka proxy

```
kubectl apply -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-crds.yaml
kubectl apply -n ambassador -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-kind.yaml
kubectl wait --timeout=180s -n ambassador --for=condition=deployed ambassadorinstallations/ambassador
```

Namespace 

```
kubectl create namespace seldon-system
kubectl create namespace seldon
```

Seldon


```
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --set ambassador.enabled=true \
    --namespace seldon-system
```

Seldon analytics

```
helm install seldon-core-analytics seldon-core-analytics --repo https://storage.googleapis.com/seldon-charts --namespace seldon-system
```

## Port forward 

```
kubectl port-forward --address 0.0.0.0 -n ambassador svc/ambassador 7777:80
kubectl port-forward --address 0.0.0.0 svc/seldon-core-analytics-grafana -n seldon-system 3000:80    
kubectl port-forward --address 0.0.0.0 svc/seldon-core-analytics-prometheus-seldon -n seldon-system 5000:80
```

## Seldon examples

```
kubectl create -f k8s/seldon-custom.yaml
kubectl create -f k8s/seldon-ab-test.yaml
kubectl create -f k8s/seldon-shadow-test.yaml
kubectl create -f k8s/seldon-autoscaling.yaml
```

## Load testing 


```
export HOST_NAME=http://54.221.129.217:7777
export NUM_CLIENTS=10
locust -f serving/locust_config.py  --host $HOST_NAME -r 1 -u $NUM_CLIENTS
```

## Reference 

- https://github.com/data-max-hq/ab-testing-in-ml
- https://github.com/SeldonIO/seldon-core/tree/master/examples/feedback/reward-accuracy

