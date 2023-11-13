# Projector ML-Ops/ML in prod repo
This will be a repo for various experiments using Docker, k8, FastAPI and a few NN for integration

# MinIO Client Project on Kubernetes

This project includes a Python client for interacting with MinIO, a high-performance object storage service, which is deployed on a Kubernetes cluster. It includes functionality for creating buckets, uploading, downloading, and deleting files from a MinIO server.

## Prerequisites

- Kubernetes cluster or Minikube
- `kubectl` configured to interact with your cluster
- Python 3.6+
- `pip` for installing Python packages

## Getting Started

These instructions will help you to deploy MinIO on Kubernetes and run your client tests against it.

### Deploying MinIO on Kubernetes

1. Navigate to the `k8s/minio` directory where the MinIO Kubernetes manifests are located.

    ```bash
    cd k8s/minio
    ```

2. Create a PersistentVolumeClaim (PVC) for MinIO storage:

    ```bash
    kubectl apply -f minio-pvc.yaml
    ```

3. Deploy MinIO using the provided deployment file:

    ```bash
    kubectl apply -f minio-deployment.yaml
    ```

4. Expose MinIO service to be able to interact with it:

    ```bash
    kubectl apply -f minio-service.yaml
    ```

5. Check the MinIO pod and service to ensure they are correctly deployed and running:

    ```bash
    kubectl get pods
    kubectl get svc
    ```

    Note down the MinIO service's cluster IP or External IP to use as the endpoint.

### Setting Up the Python Environment

1. Clone the repository (if you have not already done so):

    ```bash
    git clone [URL_TO_YOUR_REPO]
    cd ml-ops-proj
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Tests

Before running the tests, make sure to set the environment variables to point to the Kubernetes MinIO service:

```bash
export MINIO_ENDPOINT="[MINIO_CLUSTER_IP]:9000"
export MINIO_ACCESS_KEY="minio"
export MINIO_SECRET_KEY="minio123"
```

Before running the tests, ensure MinIO is accessible. If you are testing locally and MinIO is deployed within Kubernetes, perform port-forwarding:

```bash
kubectl port-forward svc/minio-service 9000:9000
```

Run test suite with following command:

```bash
python -m unittest minio_client.unittests.test_minio_crud_client
```
