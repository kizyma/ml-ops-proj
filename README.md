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

# Setting Up DVC with Google Drive

We use Data Version Control (DVC) to manage and version large files, datasets, and machine learning models, storing them in Google Drive.

## Prerequisites

- DVC
- Google Drive account

## Installation

**Install DVC:**
   Ensure DVC is installed:
   ```bash
   pip install dvc
   ```

   For Google Drive support, install dvc-gdrive:
  ```bash
   pip install dvc-gdrive
   ```

## DVC Initialization

**Initialize DVC:**
   In your project directory, if DVC is not already initialized:
   ```bash
   dvc init
   ```
  
**Google Drive Setup**
1. Create a Google Drive Folder:
2. Create a new folder in Google Drive for your DVC files and note down the folder ID from the URL.

# Add Google Drive as a DVC Remote:
  Configure the remote storage to Google Drive:
  ```bash
  dvc remote add -d mygdrive gdrive://<folder-id>
  ```
# Authenticate with Google Drive:
Follow on-screen instructions to authenticate the first time you push or pull data.

**Usage**
# Add Data to DVC:
  Track your files with DVC:

  ```bash 
  dvc add data/dataset
  ```
# Push Data to Google Drive:
Upload data to Google Drive:
  ```bash
  dvc push
  ```
# Commit DVC Files to Git:
Commit .dvc files and .dvc/config to your repository.
  ```bash
  git add .dvc/config data/dataset.dvc
  git commit -m "Add dataset with DVC"
  git push
  ```
# Pulling Data:
To download data on a different machine:
  ```bash
  dvc pull
  ```

# Setting Up and Using Label Studio for Data Labeling

Label Studio is an open-source data labeling tool. This section provides instructions on deploying Label Studio and labeling around 50 data samples.

## Prerequisites

- Python 3.6 or higher
- pip for installing Python packages

## Installation

**Install Label Studio:**
   Label Studio can be installed using pip. Run the following command:
   ```bash
   pip install label-studio
   ```
## Deployment
**Start Label Studio:**
  Once installed, start Label Studio using the command line:

   ```bash
   label-studio start
   ```
