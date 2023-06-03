
# Setup 

Create kind cluster 

```
kind create cluster --name ml-in-production-course-week-7 --image=kindest/node:v1.21.2 --config=k8s/kind.yaml
kubectl create secret generic wandb --from-literal=WANDB_API_KEY=************
```

Run k9s 

```
k9s -A
```


# Grafana 



```
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install monitoring prometheus-community/kube-prometheus-stack

kubectl port-forward --address 0.0.0.0 svc/monitoring-grafana 3000:80
admin/prom-operator

helm uninstall monitoring 
```

- Reference: https://github.com/prometheus-community/helm-charts/blob/main/charts/kube-prometheus-stack/README.md


# Data monitoring 

- https://github.com/evidentlyai/evidently
- https://github.com/SeldonIO/alibi-detect
- https://github.com/whylabs/whylogs 


## Seldon 

```
kubectl apply -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-crds.yaml
kubectl apply -n ambassador -f https://github.com/datawire/ambassador-operator/releases/latest/download/ambassador-operator-kind.yaml
kubectl wait --timeout=180s -n ambassador --for=condition=deployed ambassadorinstallations/ambassador

kubectl create namespace seldon-system

helm install seldon-core seldon-core-operator --version 1.15.1 --repo https://storage.googleapis.com/seldon-charts --set usageMetrics.enabled=true --set ambassador.enabled=true  --namespace seldon-system
helm install seldon-core-analytics seldon-core-analytics --repo https://storage.googleapis.com/seldon-charts --namespace seldon-system

kubectl port-forward --address 0.0.0.0 -n ambassador svc/ambassador 7777:80
kubectl port-forward --address 0.0.0.0 svc/seldon-core-analytics-grafana -n seldon-system 3000:80    
admin/password

kubectl port-forward --address 0.0.0.0 svc/seldon-core-analytics-prometheus-seldon -n seldon-system 5000:80

```


## Seldon & Kserve

- https://docs.seldon.io/projects/seldon-core/en/latest/analytics/outlier_detection.html
- https://docs.seldon.io/projects/seldon-core/en/latest/analytics/drift_detection.html


## Platforms 

- https://docs.arize.com/arize/quickstart
- https://aws.amazon.com/sagemaker/model-monitor/
- https://cloud.google.com/vertex-ai/docs/model-monitoring



