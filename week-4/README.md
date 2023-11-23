
# Setup 

Create kind cluster 

```
kind create cluster --name ml-in-production-course-week-4
```

Run k9s 

```
k9s -A
```

# Kubeflow pipelines 

## Deploy kubeflow pipelines 

Create directly

```
export PIPELINE_VERSION=2.0.3
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```

Create yaml and applay with kubectl (better option)

```
export PIPELINE_VERSION=2.0.3
kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION" > kfp-yml/res.yaml
kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION" > kfp-yml/pipelines.yaml



kubectl create -f kfp-yml/res.yaml
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl create -f kfp-yml/pipelines.yaml
```


Access UI and minio


```
kubectl port-forward --address=0.0.0.0 svc/minio-service 9000:9000 -n kubeflow
kubectl port-forward --address=0.0.0.0 svc/ml-pipeline-ui 8888:80 -n kubeflow
```


## Create pipelines

Setup env variables 

```
export WANDB_PROJECT=nlp-sample
export WANDB_API_KEY=****************
```


Training 

```
python kfp-training-pipeline.py http://0.0.0.0:8080
```

Inference 

```
python kfp-training-pipeline.py http://0.0.0.0:8080
```


# Airflow
## Deploy airflow locally


1. Run standalone airflow

```
export AIRFLOW_HOME=$PWD/airflow-home
airflow standalone
```

2. Configure airlfow <> k8s connection
3. Create storage 

```
kubectl create -f airflow-volumes.yaml
```

4. Read to run pipelines

- https://madewithml.com/courses/mlops/orchestration/


References:

- https://www.astronomer.io/guides/kubepod-operator/
- https://www.astronomer.io/guides/airflow-passing-data-between-tasks/


