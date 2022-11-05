# Feast implementation 


- https://docs.feast.dev/
- https://docs.feast.dev/getting-started/quickstart
- https://github.com/feast-dev/feast-gcp-driver-ranking-tutorial

## Setup 


```
docker build -t fs:latest .
docker run -it -v $PWD:/app -p 8080:8080 fs:latest /bin/bash
```

```
feast apply
feast ui -p 8080 -h 0.0.0.0
```

## Run training 

```
python traininig.py --model-resutl-path result.pkl
```






## Run inference

```
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME
```

```
uvicorn --host 0.0.0.0 --port 8080 --reload inference:app
```

