
# Setup 

Create kind cluster 

```
kind create cluster --name ml-in-production-course-week-6
kubectl create secret generic wandb --from-literal=WANDB_API_KEY=$WANDB_API_KEY
```

Run k9s 

```
k9s -A
```


# Load test 

Deploy API 

```
kubectl create -f ./k8s/fastapi-app.yaml
kubectl port-forward --address 0.0.0.0 svc/app-fastapi 8080:8080
```

Run test 

```
locust -f load-testing/locustfile.py --host=http://app-fastapi.default.svc.cluster.local:8080 --users 50 --spawn-rate 10 --autostart --run-time 600s
```

Run on k8s 


```
kubectl create -f ./k8s/fastapi-locust.yaml
kubectl port-forward --address 0.0.0.0 pod/load-fastapi-naive 8089:8089
```


# HPA



Install metric server 

```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
```

Create from cli

```
kubectl autoscale deployment app-fastapi --cpu-percent=50 --min=1 --max=10
```

Create from yaml

```
kubectl create -f ./k8s/fastapi-hpa.yaml
```


- https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/


# Async inferece 

## Install KServe

Install kserve

```
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.10/hack/quick_install.sh" | bash
```

Test single model 

```
kubectl create namespace kserve-test
kubectl create -n kserve-test -f ./k8s/kserve-iris.yaml
kubectl get inferenceservices sklearn-iris -n kserve-test
kubectl get svc istio-ingressgateway -n istio-system
kubectl port-forward --namespace istio-system svc/istio-ingressgateway 8080:80
```

```
curl -v -H "Host: sklearn-iris.kserve-test.example.com" "http://0.0.0.0:8080/v1/models/sklearn-iris:predict" -d @data/iris-input.json
```


```
kubectl create -f load-testing/perf.yaml -n kserve-test
```


Install kafka 

```
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install zookeeper bitnami/zookeeper --set replicaCount=1 --set auth.enabled=false --set allowAnonymousLogin=true --set persistance.enabled=false --version 11.0.0
helm install kafka bitnami/kafka --set zookeeper.enabled=false --set replicaCount=1 --set persistance.enabled=false --set logPersistance.enabled=false --set externalZookeeper.servers=zookeeper-headless.default.svc.cluster.local --version 21.0.0
```

Install eventing

```
kubectl apply -f https://github.com/knative/eventing/releases/download/knative-v1.9.7/eventing-crds.yaml
kubectl apply -f https://github.com/knative/eventing/releases/download/knative-v1.9.7/eventing-core.yaml
kubectl apply -f https://github.com/knative-sandbox/eventing-kafka/releases/download/knative-v1.9.1/source.yaml
```


Install minio & creds

```
kubectl apply -f k8s/kafka-infra.yaml
```


Configure minio

```
kubectl port-forward $(kubectl get pod --selector="app=minio" --output jsonpath='{.items[0].metadata.name}') 9000:9000

mc mb myminio/mnist
mc mb myminio/digits

mc admin config set myminio notify_kafka:1 tls_skip_verify="off"  queue_dir="" queue_limit="0" sasl="off" sasl_password="" sasl_username="" tls_client_auth="0" tls="off" client_tls_cert="" client_tls_key="" brokers="kafka-headless.default.svc.cluster.local:9092" topic="mnist" version=""
mc admin service restart myminio
mc event add myminio/mnist arn:minio:sqs::1:kafka -p --event put --suffix .png

gsutil cp -r gs://kfserving-examples/models/tensorflow/mnist .
mc cp -r mnist myminio/
```

Deploy model 

```
kubectl create -f k8s/kafka-model.yaml
```

Trigger the model 

```
mc cp data/0.png myminio/mnist
```


docker build -t kyrylprojector/mnist-transformer:latest -f ./transformer.Dockerfile . && docker push kyrylprojector/mnist-transformer:latest
kubectl delete -f mnist_kafka_new.yaml
kubectl create -f mnist_kafka_new.yaml


docker build -t kyrylprojector/kserve-custom:latest -f Dockerfile . && docker push kyrylprojector/kserve-custom:latest

docker build -t kyrylprojector/kserve-custom:latest -f Dockerfile --target app-kserve .
docker run -e PORT=8080 e WANDB_API_KEY=cb86168a2e8db7edb905da69307450f5e7867d66 -p 8080:8080 kyrylprojector/kserve-custom:latest 





