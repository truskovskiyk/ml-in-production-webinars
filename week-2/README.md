
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
kubectl port-forward svc/minio-ui 9001:9001
kubectl port-forward svc/minio-ui 9000:9000
```

# MINIO Client 


```
pytest test_minio_client.py
```

- https://docs.min.io/docs/python-client-api-reference.html
- https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
- https://s3fs.readthedocs.io/en/latest/


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
dvc remote add -d minio s3://ml-data
dvc remote modify minio endpointurl http://0.0.0.0:9000
```

Save code to git 

```
git add .dvc/config
git commit -m "Configure remote storage"
git push 
```

Save data to storage

```
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
dvc push
```

- https://dvc.org/doc/start/data-management
- https://github.com/iterative/dataset-registry

# Labeling 

Install label-studio in docker

```
docker run -it -p 8080:8080 -v `pwd`/mydata:/label-studio/data heartexlabs/label-studio:latest
```


# Pandas profiling 

https://aaltoscicomp.github.io/python-for-scicomp/data-formats/


# CVS inference 

```
python inference_example.py
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



