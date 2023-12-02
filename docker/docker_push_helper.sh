# build base image and push to acr 
az acr build --file docker/BaseDockerFile --registry tezzmlops23 --image base:v1 .
# build final image and push to acr
az acr build --file docker/Dockerfile --registry tezzmlops23 --image digits:v1 .
