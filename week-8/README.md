# Vertex AI 

```
export PROJECT=<>
export BUCKET=<>
```


```
gcloud auth login
gcloud auth application-default login
gcloud auth configure-docker
gcloud config set project $PROJECT
```


```
python cli.py upload-dataset ./data/ nlp-dataset
python cli.py list-of-dataset-version nlp-dataset
python cli.py download-dataset nlp-dataset data
python cli.py run-traininig-job test-job
```

## Reference 

- https://github.com/GoogleCloudPlatform/vertex-ai-samples
- https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai
- https://github.com/statmike/vertex-ai-mlops
- https://github.com/googleapis/python-aiplatform
- https://huggingface.co/blog/deploy-vertex-ai

# SageMaker 


## References 

- https://github.com/aws-samples/mlops-amazon-sagemaker
- https://github.com/aws-samples/amazon-sagemaker-mlops-workshop
- https://github.com/aws-samples/mlops-e2e