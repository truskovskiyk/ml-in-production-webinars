# Docker 

## Build and run 

Build ml sample docker image 

```
docker build --tag app-ml:latest ./app-ml
```

Run ml sample docker container 

```
docker run -it  --rm --name app-ml-test-run app-ml:latest
docker run -it  --rm --name app-ml-test-run app-ml:latest python -c "import time; time.sleep(5); print(f'AUC = {0.0001}')"
```


Build web sample docker image 

```
docker build --tag app-web:latest ./app-web
```

Build web sample docker image 

```
docker run -it --rm -p 8080:8080 --name app-web-test-run app-web:latest
```


Build multi build 

```
docker build --tag app-web:latest --target app-web ./app-multi-build
docker build --tag app-ml:latest --target app-ml ./app-multi-build
```

## Share


Login to docker registry 

```
export DOCKER_HUB_USER=kyrylprojector
export DOCKER_HUB_PASS=**************
docker login -u $DOCKER_HUB_USER -p $DOCKER_HUB_PASS
```

Tag images


```
docker tag app-ml:latest kyrylprojector/app-ml:latest
docker tag app-web:latest kyrylprojector/app-web:latest
```


Push image 


```
docker push kyrylprojector/app-ml:latest
docker push kyrylprojector/app-web:latest
```

## Registry

- [dockerhub](https://hub.docker.com/)
- [github](https://github.com/features/packages)
- [aws](https://aws.amazon.com/ecr/)
- [gcp](https://cloud.google.com/container-registry)


# K8S

## Setup 

Install kind 
https://kind.sigs.k8s.io/docs/user/quick-start/

```
brew install kind
```

Create cluster

```
kind create cluster --name ml-in-production-course-week-1
```

Check current context

```
kubectl config get-contexts
```


Run "htop" for k8s 

```
k9s -A
```

## Use

Create pod for app-web

```
kubectl create -f k8s-resources/pod-app-web.yaml
```

Create pod for app-ml

```
kubectl create -f k8s-resources/pod-app-ml.yaml
```

Create job for app-ml

```
kubectl create -f k8s-resources/job-app-ml.yaml
```

Create deployment for app-web

```
kubectl create -f k8s-resources/deployment-app-web.yaml
```

To access use port-forwarding 

```
kubectl port-forward svc/deployments-app-web 8080:8080
```

## Provides 

- [EKS](https://aws.amazon.com/eks/)
- [GKE](https://cloud.google.com/kubernetes-engine)
- [CloudRun](https://cloud.google.com/run)
- [AWS Fargate/ECS](https://aws.amazon.com/fargate/)

# CI/CD 


## Provides 

- [circleci](https://circleci.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Jenkins](https://www.jenkins.io/)
- [Travis CI](https://www.travis-ci.com/)
- [List of Continuous Integration services](https://github.com/ligurio/awesome-ci)