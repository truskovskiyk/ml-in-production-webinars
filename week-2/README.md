
# Setup 

Create kind cluster 

```
kind create cluster --name ml-in-production-course-week-2
```

Run k9s 

```
k9s -A
```


# MINIO 

Based on https://github.com/kubernetes/examples/tree/master/staging/storage/minio

Deploy 

```
kubectl create -f ./minio/minio-standalone.yaml
```


Access UI and API 

```
kubectl port-forward --address=0.0.0.0 pod/minio 9000 9090

```

# S3 access 


```
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
export AWS_ENDPOINT="http://0.0.0.0:9000"

aws s3 ls --endpoint-url $AWS_ENDPOINT
aws s3api create-bucket --bucket test --endpoint-url $AWS_ENDPOINT 
```


# MINIO Client 


```
pytest test_minio_client.py
```

- https://docs.min.io/docs/python-client-api-reference.html
- https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
- https://s3fs.readthedocs.io/en/latest/


# CVS inference 

```
python3 inference_example.py run-single-worker --inference-size 10000000
python3 inference_example.py run-pool --inference-size 10000000
python3 inference_example.py run-ray --inference-size 10000000
python3 inference_example.py run-dask --inference-size 10000000
```

# Pandas profiling 

https://aaltoscicomp.github.io/python-for-scicomp/data-formats/


# Streaming dataset

- https://www.tensorflow.org/tutorials/load_data/tfrecord
- https://github.com/aws/amazon-s3-plugin-for-pytorch
- https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/
- https://github.com/webdataset/webdataset



Create
```
python tutorial.py create-data --path-to-save random-data
```

Upload

```
aws s3api create-bucket --bucket datasets --endpoint-url $AWS_ENDPOINT
aws s3 cp --recursive random-data s3://datasets/random-data --endpoint-url $AWS_ENDPOINT
```

Read

```
python tutorial.py get-dataloader --path-to-save random-data
```


# Vector search 

Deploy with Helm 


```
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.5.3/cert-manager.yaml
helm repo add milvus https://milvus-io.github.io/milvus-helm/
helm repo update
helm upgrade --install vector-search --set cluster.enabled=false --set etcd.replicaCount=1 --set pulsar.enabled=false --set minio.mode=standalone milvus/milvus
```

Deploy with kubeclt


```
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.5.3/cert-manager.yaml
helm repo add milvus https://milvus-io.github.io/milvus-helm/
helm repo update
helm template --set cluster.enabled=false --set etcd.replicaCount=1 --set pulsar.enabled=false --set minio.mode=standalone milvus/milvus > milvus.yaml
kubeclt create -f milvus.yaml
```

Run UI 


```
kubectl port-forward svc/my-vector-db-milvus --address=0.0.0.0 19530:19530
docker run -p 8000:3000 -e MILVUS_URL=0.0.0.0:19530 zilliz/attu:v2.2.3
```


# DVC 



Init DVC

```
dvc init --subdir
git status
git commit -m "Initialize DVC"
```

Add data 

```
mkdir data
touch ./data/big-data.csv
```

Add to dvc

```
dvc add ./data/big-data.csv
git add data/.gitignore data/big-data.csv.dvc
git commit -m "Add raw data"
```

Add remote 

```
aws s3api create-bucket --bucket ml-data --endpoint-url $AWS_ENDPOINT

dvc remote add -d minio s3://ml-data
dvc remote modify minio endpointurl $AWS_ENDPOINT
```

Save code to git 

```
git add .dvc/config
git commit -m "Configure remote storage"
git push 
```

Save data to storage

```
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
dvc push
```

- https://dvc.org/doc/start/data-management
- https://github.com/iterative/dataset-registry


# Labeling 

Install label-studio in docker

```
docker run -it -p 8080:8080 -v `pwd`/mydata:/label-studio/data heartexlabs/label-studio:latest
```


# LakeFS 

Generate template 

```
helm template my-lakefs lakefs/lakefs > lakefs-deploy.yaml
```

Deploy lakefs on k8s 

```
kubectl create -f lakefs-deploy.yaml
```
** Note: it might take ~10 min to deploy it 


Access lakefs 

```
kubectl port-forward svc/my-lakefs 5000:80
```


- https://lakefs.io/
- https://docs.lakefs.io/integrations/python.html
- https://docs.lakefs.io/integrations/kubeflow.html



