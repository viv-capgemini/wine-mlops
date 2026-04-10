# Wine MLOps - Local Deployment with DVC and AWS S3

A machine learning project for wine quality prediction using RandomForest with MLflow experiment tracking and DVC for data versioning.

## Project Overview

This project demonstrates a complete MLOps workflow:
- **Model**: RandomForest regression for wine quality prediction
- **Experiment Tracking**: MLflow for logging metrics and models
- **Data Versioning**: DVC for managing training datasets
- **Data Storage**: AWS S3 for remote data storage
- **Local Development**: Easy setup for local experimentation and testing

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed locally
- **Git** installed
- **AWS Account** with S3 bucket access (for remote data storage)
- **AWS CLI** installed and configured
- **DVC** for data version control
- **MLflow** for experiment tracking

### AWS Setup Requirements

1. **AWS S3 Bucket**: Create an S3 bucket to store your data
   ```
   aws s3 mb s3://your-wine-mlops-bucket
   ```

2. **AWS Credentials**: Configure your AWS credentials locally
   ```
   aws configure
   ```
   You'll need:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region (e.g., us-east-1)

3. **IAM Permissions**: Ensure your AWS user has S3 access permissions:
   - `s3:GetObject`
   - `s3:PutObject`
   - `s3:ListBucket`

## Quick Start with Docker (Recommended)

For the quickest setup with all services (MLflow, PostgreSQL) running in containers:

```bash
# Make the startup script executable
chmod +x start-docker.sh

# Copy and configure environment file
cp .env.template .env
# Edit .env with your AWS credentials

# Start all services
./start-docker.sh
```

Then visit `http://localhost:5000` to access MLflow.

To run training:
```bash
docker-compose run train python train.py --experiment wine-prediction --run docker-run-1
```

To stop services:
```bash
docker-compose down
```

**Docker Setup Details**: See the [Docker Deployment](#docker-deployment) section below.

---

## Local Setup Instructions

### 1. Clone Repository

```bash
git clone <repository-url>
cd wine-mlops
```

### 2. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Initialize DVC

Initialize DVC in your local repository:

```bash
dvc init
```

This creates a `.dvc` directory and `dvc.yaml` for tracking.

### 5. Configure DVC with AWS S3

Set up DVC to use AWS S3 as the remote storage backend:

```bash
# Add S3 bucket as DVC remote
dvc remote add -d myremote s3://your-wine-mlops-bucket/dvc-storage

# Verify the configuration
dvc remote list
```

#### DVC Configuration Details

DVC stores its configuration in `.dvc/config`. You can also configure it manually:

```bash
dvc remote modify myremote access_key_id <YOUR_ACCESS_KEY>
dvc remote modify myremote secret_access_key <YOUR_SECRET_KEY>
```

Or use environment variables (recommended for security):

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 6. Set Up Data Directory

Create a data directory structure:

```bash
mkdir -p data/raw
mkdir -p data/processed
```

### 7. Download Dataset from scikit-learn

The project includes a `download_data.py` script that fetches the wine dataset from scikit-learn and saves it as a CSV:

```bash
# Save to default location (data/wine_sample.csv)
python download_data.py

# Or specify a custom output path
python download_data.py --output data/raw/wine_sample.csv
```

Expected output:
```
Downloading wine dataset from scikit-learn...
Saved 178 rows to data/wine_sample.csv
Columns: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
          'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
          'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline', 'quality']
Target distribution:
0    59
1    71
2    48
```

> The scikit-learn wine dataset's `target` column is renamed to `quality` to match what `train.py` expects.

Alternatively, generate the CSV in a single Python command without the script:

```bash
python - <<'EOF'
import pandas as pd
from sklearn.datasets import load_wine
import os

os.makedirs("data", exist_ok=True)
df = load_wine(as_frame=True).frame.rename(columns={"target": "quality"})
df.to_csv("data/wine_sample.csv", index=False)
print(f"Saved {len(df)} rows to data/wine_sample.csv")
EOF
```

### 8. Add Data Files to DVC

Once the data is downloaded locally, add it to DVC tracking:

```bash
# Add to DVC
dvc add data/wine_sample.csv

# Commit the .dvc pointer file to git
git add data/wine_sample.csv.dvc .gitignore
git commit -m "Add wine dataset via DVC"

# Push data to S3
dvc push
```

This creates `data/wine_sample.csv.dvc` which tracks the data version in git while the actual CSV is stored in S3.

You can combine the download and DVC steps into a single workflow:

```bash
# Download data, then immediately track and push to S3
python download_data.py --output data/wine_sample.csv && \
  dvc add data/wine_sample.csv && \
  dvc push
```

### 8. Pull Data from S3

To sync data from S3 to your local machine:

```bash
dvc pull
```

## Running the Model Training

### 1. Start MLflow Server (Optional but Recommended)

In a separate terminal:

```bash
mlflow ui --host 127.0.0.1 --port 7006
```

Then visit `http://localhost:7006` to view experiment runs.

### 2. Train the Model

```bash
# Train with default parameters (uses data/wine_sample.csv)
python train.py

# Train with custom parameters
python train.py \
  --csv data/wine_sample.csv \
  --target quality \
  --experiment wine-prediction \
  --run run-1 \
  --n-estimators 100 \
  --max-depth 10 \
  --test-size 0.2
```

### Available Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `data/wine_sample.csv` | Path to training CSV file |
| `--target` | `quality` | Target column name |
| `--experiment` | `wine-prediction` | MLflow experiment name |
| `--run` | `run-2` | MLflow run name |
| `--n-estimators` | `50` | RandomForest number of estimators |
| `--max-depth` | `5` | RandomForest maximum depth |
| `--test-size` | `0.2` | Test set fraction (0.0-1.0) |
| `--random-state` | `42` | Random seed for reproducibility |

## Docker Deployment

### Overview

The project includes Docker and Docker Compose configuration for running MLflow and model training in containers.

### Files

- **`docker-compose.yml`**: Orchestrates PostgreSQL, MLflow, and training services
- **`Dockerfile.mlflow`**: Image for MLflow tracking server
- **`Dockerfile.train`**: Image for model training
- **`.env.template`**: Environment variables template
- **`start-docker.sh`**: Convenient startup script

### Services

1. **PostgreSQL** (postgres:15-alpine)
   - Backend store for MLflow metadata
   - Port: 5432
   - Credentials configured via `.env`

2. **MLflow** (Custom Python image)
   - Experiment tracking server
   - Port: 5000 (http://localhost:5000)
   - Artifact storage: `/mlflow/artifacts` (persisted volume)

3. **Training** (Custom Python image)
   - Runs model training with MLflow logging
   - Mounts local `data/` and `mlruns/` directories
   - AWS credentials passed via `.env`

### Setup

#### 1. Configure Environment

```bash
cp .env.template .env
```

Edit `.env` with your settings:
```
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

#### 2. Build Images

```bash
docker-compose build
```

#### 3. Start Services

```bash
docker-compose up -d
```

Check status:
```bash
docker-compose ps
```

View logs:
```bash
docker-compose logs -f mlflow
```

### Usage

#### Access MLflow UI

Visit: `http://localhost:5000`

#### Run Training in Container

```bash
# Basic training
docker-compose run train python train.py

# With custom parameters
docker-compose run train python train.py \
  --experiment wine-prediction \
  --run docker-run-1 \
  --n-estimators 100 \
  --max-depth 10
```

#### Access Container Shell

```bash
docker-compose run train bash
```

#### Stop Services

```bash
docker-compose down
```

Remove volumes (warning: deletes PostgreSQL data and artifacts):
```bash
docker-compose down -v
```

### Networking

All services communicate via the `wine-network` Docker network. Training service connects to MLflow via `http://mlflow:5000` (hostname resolution).

### Data and Artifact Persistence

- **PostgreSQL Data**: Stored in `postgres_data` volume
- **MLflow Artifacts**: Stored in `mlflow_artifacts` volume
- **Training Data**: Mounted from local `./data/` directory
- **MLflow Runs**: Mounted from local `./mlruns/` directory

This ensures data persists across container restarts.

### Troubleshooting Docker

**Problem**: Services won't start

Check logs:
```bash
docker-compose logs
```

**Problem**: Port already in use

Change port in `docker-compose.yml`:
```yaml
mlflow:
  ports:
    - "5001:5000"  # Use 5001 instead of 5000
```

**Problem**: PostgreSQL won't connect

```bash
# Reset PostgreSQL
docker-compose down -v
docker-compose up -d postgres
# Wait 30 seconds then start mlflow
```

---

## Project Structure

```
wine-mlops/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── train.py                       # Model training script
├── utils.py                       # Utility functions
├── .gitignore                     # Git ignore patterns
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile.mlflow              # MLflow server image
├── Dockerfile.train               # Training image
├── .env.template                  # Environment variables template
├── start-docker.sh                # Startup script
├── download_data.py               # Download wine dataset from scikit-learn
├── .dvc/                          # DVC configuration directory
│   ├── config                     # DVC remote configuration
│   └── .gitignore                 # DVC ignore patterns
├── data/
│   ├── raw/                       # Raw data files (tracked with DVC)
│   │   └── wine_sample.csv.dvc
│   └── processed/                 # Processed data (optional)
└── mlruns/                        # MLflow runs directory (auto-created)
```

## Data Management with DVC

### Checking Data Status

```bash
# See which files are tracked by DVC
git status

# Check DVC-specific status
dvc status
```

### Update Data Files

When you modify a data file:

```bash
# Re-add the modified file
dvc add data/raw/wine_sample.csv

# Commit changes
git add data/raw/wine_sample.csv.dvc
git commit -m "Update wine dataset"

# Push updated data to S3
dvc push
```

### Retrieve Specific Data Versions

DVC tracks all versions in git. To get a previous data version:

```bash
# Checkout previous git commit
git checkout <commit-hash>

# Pull the corresponding data version
dvc pull
```

## Troubleshooting

### DVC and S3 Issues

**Problem**: `dvc pull` fails with authentication error

**Solution**:
```bash
# Verify AWS credentials are set
aws s3 ls s3://your-wine-mlops-bucket/

# Reconfigure DVC remote
dvc remote remove myremote
dvc remote add -d myremote s3://your-wine-mlops-bucket/dvc-storage
```

**Problem**: SSL certificate verification failed

**Solution**:
```bash
# Disable SSL verification (not recommended for production)
dvc remote modify myremote ssl_verify false
```

### Training Issues

**Problem**: CSV file not found

**Ensure the file exists**:
```bash
ls -la data/wine_sample.csv
```

**Pull data from S3 if using DVC**:
```bash
dvc pull data/raw/wine_sample.csv.dvc
```

**Problem**: MLflow connection refused

**Solution**: Start MLflow server:
```bash
mlflow ui --host 127.0.0.1 --port 7006
```

Or use default local storage (no server required).

### Python Environment Issues

**Problem**: `ModuleNotFoundError` when running train.py

**Solution**:
```bash
# Verify virtual environment is activated
which python  # Should show path to venv/bin/python
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## Best Practices

1. **Always activate virtual environment** before running scripts
2. **Push data to S3** after making changes: `dvc push`
3. **Commit `.dvc` files to git**, but never commit actual data files
4. **Use meaningful experiment names** for MLflow runs
5. **Document hyperparameters** in your MLflow runs for reproducibility
6. **Set random seed** for reproducible experiments
7. **Pin dependency versions** in `requirements.txt`

## Next Steps

- Extend the model with feature engineering in `utils.py`
- Add data validation and testing
- Set up CI/CD pipeline with DVC and MLflow
- Deploy model as a REST API service
- Configure automated data pipelines with `dvc.yaml`

## Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC with AWS S3](https://dvc.org/doc/user-guide/setup-google-drive-remote)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [AWS S3 CLI Reference](https://docs.aws.amazon.com/cli/latest/reference/s3/)
