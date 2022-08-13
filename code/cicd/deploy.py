import mlflow.sagemaker as mfs
import yaml
import boto3
import subprocess

# reading config file
def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

try:
    config = read_yaml("config.yaml")
except Exception as e:
    raise

# variables from config
REGION = config['AWS_BASIC_CONFIG']['REGION']
AWS_ID = config['AWS_BASIC_CONFIG']['AWS_ID']

ACCESS_KEY = config['AWS_BASIC_CONFIG']['AWS_ACCESS_KEY_ID']
SECRET_ACCESS_KEY = config['AWS_BASIC_CONFIG']['AWS_SECRET_ACCESS_KEY']

BUCKET = config['S3']['BUCKET']
ECR_REPO = config['ECR']['REPO']

MODEL_URI = config['MLFLOW']['MODEL_URI']
MLFLOW_MODE = config['MLFLOW']['MODE']

ROLE = config['SAGEMAKER_CONFIG']['ROLE']
APP_NAME = config['SAGEMAKER_CONFIG']['APP_NAME']
TAG_ID= config['SAGEMAKER_CONFIG']['TAG_ID']
INSTANCE_MODEL = config['SAGEMAKER_CONFIG']['INSTANCE_MODEL']

MODEL_URL = f's3://mlflow-sagemaker-{REGION}-{AWS_ID}/{APP_NAME}/model.tar.gz'
IMAGE_URL = f'{AWS_ID}.dkr.ecr.{REGION}.amazonaws.com/mlflow-pyfunc:{TAG_ID}'

# creating s3 bucket for the model
s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                         aws_secret_access_key=SECRET_ACCESS_KEY,
                         region_name=REGION)

try:
    s3_bucket_response = s3_client.create_bucket(Bucket=BUCKET,
                                                 CreateBucketConfiguration={'LocationConstraint': REGION})
except:
    print('Bucket already exists, skipping its creation ...')

# creating ecr repo for the image
ecr_client = boto3.client('ecr', aws_access_key_id=ACCESS_KEY,
                          aws_secret_access_key=SECRET_ACCESS_KEY,
                          region_name=REGION)
try:
    ecr_repo_response = ecr_client.create_repository(repositoryName=ECR_REPO)
except:
    print('ECR repo already exists, skipping its creation ...')


# deploy model
try:
    mfs.deploy(APP_NAME,
            model_uri=MODEL_URI,
            region_name=REGION,
            mode=MLFLOW_MODE,
            execution_role_arn=ROLE,
            image_url=IMAGE_URL,
            instance_type=INSTANCE_MODEL)
except Exception as e:
    print(e)
    print('Model was not deployed on AWS')