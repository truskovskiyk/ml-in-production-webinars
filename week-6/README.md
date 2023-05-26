
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
kubectl create -f ./k8s/app-fastapi.yaml
kubectl port-forward --address 0.0.0.0 svc/app-fastapi 8080:8080
```

Run test 

```
locust -f load-testing/locustfile.py --host=http://app-fastapi.default.svc.cluster.local:8080 --users 50 --spawn-rate 10 --autostart --run-time 600s
```

Run on k8s 


```

```

# use kubectl create instead of apply because the job template is using generateName which doesn't work with kubectl apply
kubectl create -f https://raw.githubusercontent.com/kserve/kserve/release-0.8/docs/samples/v1beta1/sklearn/v1/perf.yaml -n kserve-test
