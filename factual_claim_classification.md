# TPU environment setup only
**Note**: TPU not recommended for this use case (time spent loading and unloading data
between memory and the TPU exceeds time saved in computation per epoch, 
because of too little training data).


```python
import os
assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
```

TPU compatible pytorch version installation


```python
VERSION = "20200325"  #@param ["1.5" , "20200325", "nightly"]
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  5116  100  5116    0     0  21957      0 --:--:-- --:--:-- --:--:-- 21957
    Updating... This may take around 2 minutes.
    Uninstalling torch-1.6.0+cu101:
      Successfully uninstalled torch-1.6.0+cu101
    Uninstalling torchvision-0.7.0+cu101:
      Successfully uninstalled torchvision-0.7.0+cu101
    Copying gs://tpu-pytorch/wheels/torch-nightly+20200325-cp36-cp36m-linux_x86_64.whl...
    | [1 files][ 83.4 MiB/ 83.4 MiB]                                                
    Operation completed over 1 objects/83.4 MiB.                                     
    Copying gs://tpu-pytorch/wheels/torch_xla-nightly+20200325-cp36-cp36m-linux_x86_64.whl...
    | [1 files][114.5 MiB/114.5 MiB]                                                
    Operation completed over 1 objects/114.5 MiB.                                    
    Copying gs://tpu-pytorch/wheels/torchvision-nightly+20200325-cp36-cp36m-linux_x86_64.whl...
    - [1 files][  2.5 MiB/  2.5 MiB]                                                
    Operation completed over 1 objects/2.5 MiB.                                      
    Processing ./torch-nightly+20200325-cp36-cp36m-linux_x86_64.whl
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torch==nightly+20200325) (1.18.5)
    Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch==nightly+20200325) (0.16.0)
    [31mERROR: fastai 1.0.61 requires torchvision, which is not installed.[0m
    Installing collected packages: torch
    Successfully installed torch-1.5.0a0+d6149a7
    Processing ./torch_xla-nightly+20200325-cp36-cp36m-linux_x86_64.whl
    Installing collected packages: torch-xla
    Successfully installed torch-xla-1.6+e788e5b
    Processing ./torchvision-nightly+20200325-cp36-cp36m-linux_x86_64.whl
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from torchvision==nightly+20200325) (1.15.0)
    Requirement already satisfied: torch in /usr/local/lib/python3.6/dist-packages (from torchvision==nightly+20200325) (1.5.0a0+d6149a7)
    Requirement already satisfied: pillow>=4.1.1 in /usr/local/lib/python3.6/dist-packages (from torchvision==nightly+20200325) (7.0.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torchvision==nightly+20200325) (1.18.5)
    Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch->torchvision==nightly+20200325) (0.16.0)
    Installing collected packages: torchvision
    Successfully installed torchvision-0.6.0a0+3c254fb
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following NEW packages will be installed:
      libomp5
    0 upgraded, 1 newly installed, 0 to remove and 21 not upgraded.
    Need to get 234 kB of archives.
    After this operation, 774 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu bionic/universe amd64 libomp5 amd64 5.0.1-1 [234 kB]
    Fetched 234 kB in 0s (2,728 kB/s)
    Selecting previously unselected package libomp5:amd64.
    (Reading database ... 144611 files and directories currently installed.)
    Preparing to unpack .../libomp5_5.0.1-1_amd64.deb ...
    Unpacking libomp5:amd64 (5.0.1-1) ...
    Setting up libomp5:amd64 (5.0.1-1) ...
    Processing triggers for libc-bin (2.27-3ubuntu1.2) ...
    /sbin/ldconfig.real: /usr/local/lib/python3.6/dist-packages/ideep4py/lib/libmkldnn.so.0 is not a symbolic link
    
    

# Environment config


```python
from pathlib import Path

# WORKING_DIR = Path("./fetched_repo") # Use in Google Colab
WORKING_DIR = Path("./")          # Use in local environment

MODEL_DIR = WORKING_DIR / 'bert-base-multilingual-cased'
FINE_TUNED_DIR = WORKING_DIR / 'models'
```

# Download model and fine-tuned classifiers from GCS bucket


```python
import requests as req
from tqdm import tqdm

for folder, blob in [(MODEL_DIR, 'config.json'), (MODEL_DIR, 'pytorch_model.bin'),
                     (MODEL_DIR, 'vocab.txt'), (FINE_TUNED_DIR, 'bert-base-multilingual-cased_1603012337.pt'),
                     (FINE_TUNED_DIR, 'bert-base-multilingual-cased_1603075225.pt')]:
    
    # Maybe create folder
    if not folder.is_dir():
        folder.mkdir(parents=True, exist_ok=True)
    
    filename = folder / blob
    
    if filename.exists():
        continue
    
    response = req.get(f"https://storage.googleapis.com/bert-base-multilingual-cased/{blob}", stream = True)
    
    print("Downloading", blob, "...")
    with open(folder / blob, "wb") as fp:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            fp.write(chunk)

```

    1it [00:00, 282.50it/s]

    Downloading config.json ...
    

    
    7313it [00:00, 73117.87it/s]

    Downloading pytorch_model.bin ...
    

    694818it [00:09, 71649.48it/s]
    1089it [00:00, 46660.03it/s]
    

    Downloading vocab.txt ...
    

    0it [00:00, ?it/s]

    Downloading bert-base-multilingual-cased_1603012337.pt ...
    

    694833it [00:10, 68425.09it/s]
    

# Fetch repo code and install requirements (if working in Google Colab)
**Attention**: A form field will appear in first two cells for user authentication (username in 1st cell, password in 2nd cell). Also, remember to **reset the kernel after requirements installation**.


```python
import subprocess
import sys
from getpass import getpass
from shutil import rmtree
from pathlib import Path
from urllib.parse import quote as url_quote

# This function persists the pass in the git config file, but this is a Docker env so no one can have access to this workspace
def fetch_repo(repo_url, repo_output_path, branch="master"):
    rmtree(path=repo_output_path, ignore_errors=True)
    # User should be provided in repo_url
    print("Enter repo password:")
    clone_command = "git clone -v -b {branch} --single-branch {repo} {folder}".format(repo=repo_url.format(password=url_quote(getpass())), folder=repo_output_path, branch=branch).split()
    return subprocess.run(clone_command, shell=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

# To pull from repo and avoid re-clone it
def pull_repo(repo_path):
    return subprocess.run("cd {path} && git pull".format(path=repo_path), shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

def link_repo(repo_path):
    abs_path = str(Path(repo_path).resolve())
    sys.path.append(abs_path)
    
def setup_workspace(repo_url, repo_output_path, branch):
    process = fetch_repo(repo_url=repo_url, repo_output_path=repo_output_path, branch=branch)
    if process.returncode == 0:
        print("Git clone successful")
        link_repo(repo_path=repo_output_path)
        print("Repo linked to sys.path") 
    else:
        raise Exception("Git clone failed: {}".format((process.stdout, process.stderr)))


print("Enter repo user:")
REPO_URL = "https://{user}:{{password}}@github.com/danielperezr88/factual_claim_classification.git".format(user=input())
BRANCH = "master"
```

    Enter repo user:
    danielperezr88
    


```python
setup_workspace(REPO_URL, WORKING_DIR, BRANCH)
```

    Enter repo password:
    ··········
    Git clone successful
    Repo linked to sys.path
    


```python
pull_repo(WORKING_DIR)
```




    CompletedProcess(args='cd fetched_repo && git pull', returncode=0, stdout=b'Already up to date.\n', stderr=b'')




```python
!cd {WORKING_DIR} && pip install -r requirements.txt
```

    Obtaining file:///content/fetched_repo (from -r requirements.txt (line 2))
    Requirement already satisfied: gsutil==4.53 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 4)) (4.53)
    Requirement already satisfied: torchtext==0.6.0 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 6)) (0.6.0)
    Requirement already satisfied: transformers==3.3.1 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 7)) (3.3.1)
    Requirement already satisfied: scikit-learn==0.23.2 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 8)) (0.23.2)
    Requirement already satisfied: numpy==1.19.1 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 10)) (1.19.1)
    Requirement already satisfied: pandas==1.1.0 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 11)) (1.1.0)
    Requirement already satisfied: tqdm==4.48.2 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 12)) (4.48.2)
    Requirement already satisfied: matplotlib==3.3.1 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 13)) (3.3.1)
    Requirement already satisfied: seaborn==0.10.1 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 14)) (0.10.1)
    Requirement already satisfied: beautifulsoup4==4.9.1 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 16)) (4.9.1)
    Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (1.15.0)
    Requirement already satisfied: mock==2.0.0 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (2.0.0)
    Requirement already satisfied: google-reauth>=0.1.0 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (0.1.0)
    Requirement already satisfied: argcomplete>=1.9.4 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (1.12.1)
    Requirement already satisfied: monotonic>=1.4 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (1.5)
    Requirement already satisfied: crcmod>=1.7 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (1.7)
    Requirement already satisfied: httplib2>=0.18 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (0.18.1)
    Requirement already satisfied: retry-decorator>=1.0.0 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (1.1.1)
    Requirement already satisfied: google-apitools>=0.5.30 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (0.5.31)
    Requirement already satisfied: gcs-oauth2-boto-plugin>=2.7 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (2.7)
    Requirement already satisfied: fasteners>=0.14.1 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (0.15)
    Requirement already satisfied: pyOpenSSL>=0.13 in /usr/local/lib/python3.6/dist-packages (from gsutil==4.53->-r requirements.txt (line 4)) (19.1.0)
    Requirement already satisfied: torch in /usr/local/lib/python3.6/dist-packages (from torchtext==0.6.0->-r requirements.txt (line 6)) (1.6.0+cu101)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from torchtext==0.6.0->-r requirements.txt (line 6)) (2.23.0)
    Requirement already satisfied: sentencepiece in /usr/local/lib/python3.6/dist-packages (from torchtext==0.6.0->-r requirements.txt (line 6)) (0.1.91)
    Requirement already satisfied: tokenizers==0.8.1.rc2 in /usr/local/lib/python3.6/dist-packages (from transformers==3.3.1->-r requirements.txt (line 7)) (0.8.1rc2)
    Requirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers==3.3.1->-r requirements.txt (line 7)) (3.0.12)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from transformers==3.3.1->-r requirements.txt (line 7)) (20.4)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers==3.3.1->-r requirements.txt (line 7)) (2019.12.20)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from transformers==3.3.1->-r requirements.txt (line 7)) (0.7)
    Requirement already satisfied: sacremoses in /usr/local/lib/python3.6/dist-packages (from transformers==3.3.1->-r requirements.txt (line 7)) (0.0.43)
    Requirement already satisfied: scipy>=0.19.1 in /usr/local/lib/python3.6/dist-packages (from scikit-learn==0.23.2->-r requirements.txt (line 8)) (1.4.1)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn==0.23.2->-r requirements.txt (line 8)) (0.16.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn==0.23.2->-r requirements.txt (line 8)) (2.1.0)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas==1.1.0->-r requirements.txt (line 11)) (2018.9)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.6/dist-packages (from pandas==1.1.0->-r requirements.txt (line 11)) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in /usr/local/lib/python3.6/dist-packages (from matplotlib==3.3.1->-r requirements.txt (line 13)) (2.4.7)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.6/dist-packages (from matplotlib==3.3.1->-r requirements.txt (line 13)) (7.0.0)
    Requirement already satisfied: certifi>=2020.06.20 in /usr/local/lib/python3.6/dist-packages (from matplotlib==3.3.1->-r requirements.txt (line 13)) (2020.6.20)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib==3.3.1->-r requirements.txt (line 13)) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib==3.3.1->-r requirements.txt (line 13)) (1.2.0)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.6/dist-packages (from beautifulsoup4==4.9.1->-r requirements.txt (line 16)) (2.0.1)
    Requirement already satisfied: pbr>=0.11 in /usr/local/lib/python3.6/dist-packages (from mock==2.0.0->gsutil==4.53->-r requirements.txt (line 4)) (5.5.0)
    Requirement already satisfied: pyu2f in /usr/local/lib/python3.6/dist-packages (from google-reauth>=0.1.0->gsutil==4.53->-r requirements.txt (line 4)) (0.1.4)
    Requirement already satisfied: importlib-metadata<3,>=0.23; python_version == "3.6" in /usr/local/lib/python3.6/dist-packages (from argcomplete>=1.9.4->gsutil==4.53->-r requirements.txt (line 4)) (2.0.0)
    Requirement already satisfied: oauth2client>=1.4.12 in /usr/local/lib/python3.6/dist-packages (from google-apitools>=0.5.30->gsutil==4.53->-r requirements.txt (line 4)) (4.1.3)
    Requirement already satisfied: boto>=2.29.1 in /usr/local/lib/python3.6/dist-packages (from gcs-oauth2-boto-plugin>=2.7->gsutil==4.53->-r requirements.txt (line 4)) (2.49.0)
    Requirement already satisfied: cryptography>=2.8 in /usr/local/lib/python3.6/dist-packages (from pyOpenSSL>=0.13->gsutil==4.53->-r requirements.txt (line 4)) (3.1.1)
    Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch->torchtext==0.6.0->-r requirements.txt (line 6)) (0.16.0)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->torchtext==0.6.0->-r requirements.txt (line 6)) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->torchtext==0.6.0->-r requirements.txt (line 6)) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->torchtext==0.6.0->-r requirements.txt (line 6)) (1.24.3)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers==3.3.1->-r requirements.txt (line 7)) (7.1.2)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata<3,>=0.23; python_version == "3.6"->argcomplete>=1.9.4->gsutil==4.53->-r requirements.txt (line 4)) (3.2.0)
    Requirement already satisfied: pyasn1>=0.1.7 in /usr/local/lib/python3.6/dist-packages (from oauth2client>=1.4.12->google-apitools>=0.5.30->gsutil==4.53->-r requirements.txt (line 4)) (0.4.8)
    Requirement already satisfied: pyasn1-modules>=0.0.5 in /usr/local/lib/python3.6/dist-packages (from oauth2client>=1.4.12->google-apitools>=0.5.30->gsutil==4.53->-r requirements.txt (line 4)) (0.2.8)
    Requirement already satisfied: rsa>=3.1.4 in /usr/local/lib/python3.6/dist-packages (from oauth2client>=1.4.12->google-apitools>=0.5.30->gsutil==4.53->-r requirements.txt (line 4)) (4.6)
    Requirement already satisfied: cffi!=1.11.3,>=1.8 in /usr/local/lib/python3.6/dist-packages (from cryptography>=2.8->pyOpenSSL>=0.13->gsutil==4.53->-r requirements.txt (line 4)) (1.14.3)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.6/dist-packages (from cffi!=1.11.3,>=1.8->cryptography>=2.8->pyOpenSSL>=0.13->gsutil==4.53->-r requirements.txt (line 4)) (2.20)
    Installing collected packages: factual-claim-classification
      Found existing installation: factual-claim-classification 0.1.0
        Can't uninstall 'factual-claim-classification'. No files were found to uninstall.
      Running setup.py develop for factual-claim-classification
    Successfully installed factual-claim-classification
    

# Create YAML config file


```python
%%writefile config.yml

# TPU not recommended for this use case (time spent loading and unloading data
# between memory and the TPU exceeds time saved in computation per epoch, 
# because of too little training data).
TPUs: False

datasets:
    path: "./data"
 
train:
    model:
        name: "bert"
        subtype: "bert-base-multilingual-cased"
        output_path: "./models"
        from_folder: "./bert-base-multilingual-cased"
   
    batch_size: 32
    max_length: 32
    lr: 0.0005 # 0.00003
    epochs: 50
    train_pct: .7
 
predict:
    model:
        name: "bert"
        subtype: "bert-base-multilingual-cased"
        from_file: "./models/bert-base-multilingual-cased_1603075225.pt"
```

    Writing config.yml
    


```python
from pathlib import Path
import yaml

def read_yaml(config_path):
    """
    Handles pyyaml's safe_load method call over the given file.
    
    Args:
      config_path (str/pathlib.Path): Path of the YAML file to
          be loaded.
    
    Returns:
      parsed_yaml (dict): Dictionary holding the contents of
          the given file.
    """
    with open(str(Path(config_path).resolve()), 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise Exception(exc)
 
    return parsed_yaml

CONFIG = read_yaml("./config.yml")
```

# Dependencies


```python
import bson
import time
import sys
import torch
import transformers
import pandas as pd
import torch.nn as nn
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from sklearn.utils import resample
from transformers import BertTokenizer, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (get_cosine_with_hard_restarts_schedule_with_warmup,
  get_cosine_schedule_with_warmup)
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

# TPUs for PyTorch
if CONFIG['TPUs']:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
```

# Helper functions


```python
def get_dataset_path(config):
    """
    Retrieve dataset path from `config` in the proper format
    
    Args:
      config (dict): Dictionary holding experiment-specific
          configuration parameters.
    
    Returns:
      (pathlib.Path): Path object, OS-agnostic.
    """
    return Path(WORKING_DIR / config["path"]).resolve()


def upsample_class_imbalance(df, col, pct=1):
    """
    Rebalance an imbalaced dataset by upsampling the minority class
    
    Args:
      df (pandas DataFrame): Original dataframe
      col (str): Column of the imbalanced class
      pct (float): Percentage of balance to reach (default = 1)
      
    Returns:
      df_upsampled (pandas DataFrame): Upsampled version of `df`.
    """
    err = 'Percentage should be in range [0,1]'
    assert pct > 0 and pct <= 1, err
 
    # Split by class
    df_majority = df[df[col] == 0]
    df_minority = df[df[col] == 1]
 
    # Upsample minority class
    df_minority_upsampled = resample(
        df_minority, 
        replace=True,                               # sample with replacement
        n_samples=int(df_majority.shape[0]*pct)     # to match majority class
    )
 
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    return df_upsampled
```

# Load config & define constants


```python
# Features of interest ('id_' field is unnecessary)
FEATURES_TO_READ = ['text_es', 'text_en', 'text_fr', 'claim']
TARGET_FEATURE = 'claim'

CPU_WORKERS = torch.multiprocessing.cpu_count()
GPU_WORKERS = 1

CUDA = torch.cuda.is_available()
N_GPU = torch.cuda.device_count()

n_TPUs = 0
if CONFIG['TPUs']:
    n_TPUs = xm.xrt_world_size()
    DEVICE = xm.xla_device()
else:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("CUDA available:", torch.cuda.is_available())
print("TPUs available:", 0 if not CONFIG['TPUs'] else n_TPUs)
print("CPU count:", CPU_WORKERS)
```

    CUDA available: True
    TPUs available: 0
    CPU count: 6
    

# Load and prepare data
**Note**: Remember to copy/upload `claims.bson` into `{WORKING_DIR}/data` first.


```python
# Read folders from config
dataset_path = get_dataset_path(CONFIG["datasets"])
model_folder = Path(WORKING_DIR / CONFIG["train"]["model"]["output_path"]).resolve()

# Maybe create model folder
if not model_folder.is_dir():
    model_folder.mkdir(parents=True, exist_ok=True)

# Read training data
with open(dataset_path / "claims.bson", 'rb') as fp:
    data = list(bson.decode_file_iter(fp))

# Create DataFrame from original data structure.
df = pd.DataFrame(data, columns=FEATURES_TO_READ)

# Make sure the target feature has the name 'target'
df.rename(columns={TARGET_FEATURE: 'target'}, inplace=True)

# Target class imbalance?
print(f"Class support: {len(df[df.target == 0]) * 3} Non-factual claims vs. {len(df[df.target == 1]) * 3} Factual Claims")

df.head()
```

    Class support: 39864 Non-factual claims vs. 3195 Factual Claims
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text_es</th>
      <th>text_en</th>
      <th>text_fr</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gracias.</td>
      <td>Thank you.</td>
      <td>Merci.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Por ejemplo, cuando estamos hablando de un pa...</td>
      <td>For example, when we are talking about a coun...</td>
      <td>Par exemple, quand on parle d’un pays qui doi...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Entonces como solo creo que es como la políti...</td>
      <td>So as I just think it's like the politics of ...</td>
      <td>Donc, comme je pense que c’est comme la polit...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Y pienso que el Partido Popular no ha estado ...</td>
      <td>And I think the People's Party has not risen ...</td>
      <td>Et je pense que le Parti populaire n’a pas ét...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lo siguiente Nos vamos ya volando y les dejo ...</td>
      <td>The next thing We fly and I leave you with th...</td>
      <td>La prochaine chose que nous volons et je vous...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



There are some considerations to be made regarding this training data:
- There is an important target **class imbalance**, which needs to be handled in different ways depending on the phase of the design process (upsampling for training, Precision/Recall AUC score for evaluation).
- **Multilingual BERT is language-agnostic** by definition, so here we have 3 training samples per row.
- Unless we want to try an ensemble approach with some additional model, **at this point we don't need to take care of text tokenization/vectorization**: tokenization will be performed by BERT itself (although we need to handle some details of that) and vectorization doesn't apply to this kind of models either.


```python
# Shuffle Rows
df = df.sample(frac=1).reset_index(drop=True)

# Calculate quantities of rows for Train/Evaluation/Test splits
total_qty = df.shape[0]

train_pct = CONFIG["train"]['train_pct']

train_qty = int(total_qty * train_pct)
valid_qty = int(total_qty * (1 - train_pct)/2)
test_qty = total_qty - valid_qty - train_qty

# Prepare Train/Validation/Test splits
df_train = df.iloc[:train_qty, :].reset_index(drop=True)
df_valid = df.iloc[train_qty:-test_qty, :].reset_index(drop=True)
df_test = df.iloc[-test_qty:, :].reset_index(drop=True)

# Fix class inbalance in training data by upsampling
df_train_balanced = upsample_class_imbalance(df_train, 'target').reset_index(drop=True)

print('Train Data: Original ratio {} to {}'.format(*df_train.target.value_counts()))
print('Train Data: Rebalanced ratio {} to {}'.format(*df_train_balanced.target.value_counts()))

# Create one row per text. Save language on an additional column
df_train_balanced = df_train_balanced.melt(id_vars=['target'], var_name='language', value_name='text')
df_valid = df_valid.melt(id_vars=['target'], var_name='language', value_name='text')
df_test = df_test.melt(id_vars=['target'], var_name='language', value_name='text')

# Make sure target feature 'claim' is always binary
df_train_balanced = df_train_balanced[df_train_balanced["target"].isin([0,1])]
df_valid = df_valid[df_valid["target"].isin([0,1])]
df_test = df_test[df_test["target"].isin([0,1])]

print('Train df shape: ', df_train_balanced.shape)
print('Valid df shape: ', df_valid.shape)
print('Test df shape: ', df_test.shape)

df_train_balanced.head()
```

    Train Data: Original ratio 9305 to 742
    Train Data: Rebalanced ratio 9305 to 9305
    Train df shape:  (55830, 3)
    Valid df shape:  (6456, 3)
    Test df shape:  (6462, 3)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>language</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>text_es</td>
      <td>Ganamos las elecciones.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>text_es</td>
      <td>Creo que exceden incluso la crítica desagrada...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>text_es</td>
      <td>Bueno, pues son pioneros en la tecnología de ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>text_es</td>
      <td>Y si hay un error, los errores comunes son un...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>text_es</td>
      <td>Le hablo de cosas tan tan importantes como po...</td>
    </tr>
  </tbody>
</table>
</div>



# Define model scaffolding

Here I define a couple of scaffolding classes for the model to be fine-tuned and to handle the dataset. Both follow PyTorch software api, so they can be used seamlessly within a standard PyTorch pipeline. Additionally, I define a helper function to instantiate a BERT-based model together with a BERT-specific tokenizer.


```python
class BERTBaseCased(nn.Module):
    '''
    PyTorch Custom BERT-based model.

    Args:
        subtype (str): Allows different BERT subtype model selection.
    '''
 
    def __init__(self, subtype, from_folder=None):
        
        super(BERTBaseCased, self).__init__()
        
        # TODO: Try to Freeze Bert Layers and Test Performance of just the output Dense Layer
        self.bert = transformers.BertModel.from_pretrained(
            subtype if from_folder is None else (from_folder / "pytorch_model.bin"),
            #from_pt=from_folder is not None,
            config=str(from_folder / "config.json") if from_folder is not None else None
        )
        
        # Freeze all but the classifier layer
        for name, param in self.bert.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False
 
        # Dropout as means of regularization
        self.bert_drop = nn.Dropout(0.3)
        
        # Sigmoid function will be applied by BCELogitsLoss during training and at hand otherwise
        self.out = nn.Linear(2*768, 1)
    
    
    def forward(self, ids, mask, token_type_ids):
        """
        Forward pass function of BERTBaseCased
        
        Args:
          ids (PyTorch Tensor): PyTorch tensor holding the token ids.
          mask (PyTorch Tensor): PyTorch tensor holding the attention
              masks.
          token_type_ids (PyTorch Tensor): PyTorch tensor holding the
              token type ids.
              
        Returns:
          output (PyTorch Tensor): 1-by-1 PyTorch tensor holding the
              output value of the forward pass.
        """
        
        # Forward pass through base BERT
        outputs = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
 
        # Manual pooling + forward through additional layers
        o1, _ = outputs # o1 -> only last hidden state
        mean_pool = torch.mean(o1,1)
        max_pool,_ = torch.max(o1,1)
        cat = torch.cat((mean_pool, max_pool), dim=1)
        bo = self.bert_drop(cat)
        output = self.out(bo)
        
        return output


class BERTDataset:
    '''
    PyTorch Custom Dataset. Allows for better BERT tokenization control.

    Args:
        df (pandas' DataFrame): DataFrame holding the text data into a column called
            `text` and the target value into `target`.
        tokenizer (transformers' BertTokenizer): Tokenizer to be used on text data.
            Must inherit from transformers' Tokenizer class.
        config (dict): Dictionary holding experiment-specific configuration parameters.
    '''
    def __init__(self, df, tokenizer, config):
        self.comment = df.text.values
        self.target = df.target.values if "target" in df.columns else None
        self.labelled_dataset = (self.target is not None)
        self.sample_id = df.sample_id.values if "sample_id" in df.columns else None
        self.tokenizer = tokenizer
        self.max_len = config.get("max_length", 128)

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, item):
        # Minimal checkups and fixes
        comment = str(self.comment[item])
        comment = " ".join(comment.split())

        # Tokenization parameters
        inputs = self.tokenizer.encode_plus(
            comment,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length'
        )

        # [self.tokenizer.ids_to_tokens[i] for i in [101, 32821, 119, 119]] -> Translate back from genereated 'inputs' into tokens

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        target_item =  self.target[item] if self.labelled_dataset else []
        sample_id_item = self.sample_id[item] if self.sample_id is not None else []
        token_type_ids = inputs.get("token_type_ids", torch.zeros_like(torch.tensor(ids, dtype=torch.long)))

        return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(target_item, dtype=torch.float),
                'sample_ids': torch.tensor(sample_id_item, dtype=torch.float)
        }


def get_model_artifacts(model_config, from_folder=None):
    """
    Instantiates both the desired BERT-based model and its corresponding
    tokenizer. Prepared for supporting different BERT-based models.
    
    Args:
      model_config (dict): Dictionary holding experiment-specific
          parameters.
    
    Returns:
      model (PyTorch model): BERT-based model to be fine-tuned.
      tokenizer (transformers Tokenizer): Pytorch BERT tokenizer object.
    """
    
    # Read config
    chosen_model = model_config.get("name", 'bert')
    subtype = model_config.get("subtype", 'bert-base-multilingual-cased')
    
    # Instantiate model and tokenizer objects
    if chosen_model == "bert":
        model = BERTBaseCased(subtype, from_folder=from_folder)
        tokenizer = BertTokenizer.from_pretrained(subtype, do_lower_case=False)
    
    else:
        raise ValueError(f"{chosen_model} not recognized. Available models: {['bert']}")
    
    # Just in case I want to try different BERT versions
    """
    elif chosen_model == 'distil-bert':
      model = DistilBERTBaseCased()
    """
        
    return model, tokenizer
```

# Train + Validation: Helper functions


```python
def get_data_loader(config, df, tokenizer, sample_pct=1., n_TPUs=0):
    """
    Creates a PyTorch DataLoader object with a custom PyTorch Dataset object
    (specific for BERT tokenization handling) and the configuration given.
    
    Args:
      config (dict): Dictionary containing some experiment-specific
          configuration parameters.
      df (pandas DataFrame): DataFrame to be consumed by the DataLoader, once
          embedded in the custom Dataset object `BERTDataset`.
      tokenizer (transformers Tokenizer): Pytorch BERT tokenizer object.
      sample_pct (float): Sampled percentage of the DataFrame to be made
          available in current DataLoader.
    
    Returns:
      dataloader (pytorch DataLoader): Resulting DataLoader object.
    """
    
    dataset = BERTDataset(
        df=df.sample(max(config["batch_size"], int(len(df)*sample_pct))),
        tokenizer=tokenizer,
        config=config)
    
    sampler = None
    if n_TPUs > 0:
        sampler = DistributedSampler(
            dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
    
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=0, drop_last=True, sampler=sampler)
    
    return dataloader


def get_input_tensors(data, device):
    """
    Prepares tensor types for BERT input and makes them available in
    the selected device.
    
    Args:
      data (dict): Dictionary holding all input data to be prepared.
      device (PyTorch device): Selected device on which to load the
          resulting tensors.
    
    Returns:
      ids (PyTorch Tensor): PyTorch tensor holding the token ids.
      token_type_ids (PyTorch Tensor): PyTorch tensor holding the
          token type ids.
      mask (PyTorch Tensor): PyTorch tensor holding the attention
          masks.
      targets (PyTorch Tensor): PyTorch tensor holding target output
          values.
    """
    ids = data["ids"]
    ids = ids.to(device, dtype=torch.long)
    
    mask = data["mask"]
    mask = mask.to(device, dtype=torch.long)
    
    targets = data["targets"]
    targets = targets.to(device, dtype=torch.float).view(-1, 1)     # (batch_size, 1)
    
    token_type_ids = data["token_type_ids"]
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    
    return ids, token_type_ids, mask, targets


def get_optimizer_conf(model, config, n_samples_train, n_TPUs=0):
    """
    Creates PyTorch's criterion, optimizer and scheduler objects for BERT fine-tunning
    training process, with given experiment-specific configuration.
    
    Args:
      model (PyTorch model): BERT-based model to be fine-tuned.
      config (dict): Dictionary containing some experiment-specific configuration
          parameters.
      n_samples_train (int): Size of training data to be used.
    
    Returns:
      result (dict): Dictionary holding PyTorch's criterion, optimizer and scheduler
          objects.
    """
 
    # Reading config parameters
    lr = config.get("lr", 0.1)
    n_epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 32)
    num_train_steps = int(config.get("train_pct", 1.) * n_samples_train / batch_size * n_epochs)

    if n_TPUs > 0:
        num_train_steps /= n_TPUs
        lr *= n_TPUs

    # Takes care of both sigmoid activation layer and Binary Cross Entropy calculation
    # (better numerical stability)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer Configuration
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=lr)

    # Scheduler added to help with convergence (half-wave cosine with warmup)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps/20),
        num_training_steps=int(num_train_steps),
        num_cycles=.5
    )

    result = {}
    result["criterion"] = criterion
    result["optimizer"] = optimizer
    result["scheduler"] = scheduler

    return result


def train_loop_fn(epoch, model, data_loader, optimizer, criterion, device, scheduler=None, n_TPUs=0):
    """
    Function to be called at each epoch for training. Performs mini-batch training
    (iterative forward step + backpropagation) over the whole data available in
    `data_loader` for current epoch. Also features monitoring and returns model
    results obtained during training.
    
    Args:
      epoch (int): Current epoch number.
      model (PyTorch model): BERT-based model to be fine-tuned.
      data_loader (PyTorch DataLoader): DataLoader to be used during this epoch.
      optimizer (PyTorch optimizer): Optimizer to be used during this epoch.
      criterion (PyTorch criterion): Criterion to be used during this epoch.
      device (int): Device to be used during this epoch.
      scheduler (optional, PyTorch scheduler): Scheduler to be used during this epoch.
    
    Returns:
      fin_outputs (list): List holding model output values obtained during this epoch.
      fin_targets (list): List holding target output values for this epoch.
    """
        
    # Training mode
    model.train()
    fin_targets = []
    fin_outputs = []

    # Iterate through batches
    for bi, d in enumerate(data_loader):

        # BERT-friendly input data
        ids, token_type_ids, mask, targets = get_input_tensors(d, device)

        # Forward step
        # xm.optimizer.zero_grad(optimizer)
        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        # Loss function evaluation
        loss = criterion(outputs, targets)

        # Retrieve batch-level results back from GPU/TPU
        targets = targets.cpu().detach().numpy()
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() # Sigmoid needed (defined output activation function was linear to take advantage of BCELogitsLoss)

        # Show some metrics for every 100th batch as means of training process monitoring
        if bi % 100 == 0:
            try:
                roc_auc_sc = roc_auc_score(targets >= 0.5, outputs)
                precision, recall, _ = precision_recall_curve(targets >= 0.5, outputs)
                pr_auc_sc = auc(recall, precision)
                if n_TPUs > 0:
                    xm.master_print(f'Epoch: {epoch:>4}, Batch: {bi:>5}, loss: {loss:>8.5f}, roc_auc: {roc_auc_sc:>8.5f}, pr_auc: {pr_auc_sc:>8.5f}')
                else:
                    print(f'Epoch: {epoch:>4}, Batch: {bi:>5}, loss: {loss:>8.5f}, roc_auc: {roc_auc_sc:>8.5f}, pr_auc: {pr_auc_sc:>8.5f}')
            except ValueError as e:
                print(targets)
                print(outputs)
                print(loss)
                raise e

        # Accumalate all epoch tensors to compute additional metrics
        fin_targets.extend(targets.tolist())
        fin_outputs.extend(outputs.tolist())

        # Backpropagation
        loss.backward()
        
        if n_TPUs > 0:
            xm.optimizer.step()
        else:
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

    return fin_outputs, fin_targets


def eval_loop_fn(epoch, model, data_loader, criterion, device):
    """
    Function to be called at each epoch for validation. Performs mini-batch model
    evaluation (iterative forward step) over the whole data available in
    `data_loader` for current epoch. Also features monitoring and returns model
    results obtained during validation.
    
    Args:
      epoch (int): Current epoch number.
      model (PyTorch model): BERT-based model to be evaluated.
      data_loader (PyTorch DataLoader): DataLoader to be used during this epoch.
      criterion (PyTorch criterion): Criterion to be used during this epoch.
      device (int): Device to be used during this epoch.
    
    Returns:
      fin_outputs (list): List holding model output values obtained during this epoch.
      fin_targets (list): List holding target output values for this epoch.
    """

    # Evaluation mode
    model.eval()
    fin_targets = []
    fin_outputs = []

    # Iterate through batches
    for bi, d in enumerate(data_loader):

        # BERT-friendly input data
        ids, token_type_ids, mask, targets = get_input_tensors(d, device)

        # Forward step
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        # Just in case I want to monitor validation on a batch level
        """
        loss = criterion(outputs, targets)
        roc_auc_sc = roc_auc_score(targets >= 0.5, outputs)
        precision, recall, _ = precision_recall_curve(targets >= 0.5, outputs)
        pr_auc_sc = auc(recall, precision)
        if n_TPUs > 0:
            xm.master_print(f'Epoch: {epoch:>4}, Batch: {bi:>5}, loss: {loss:>8.5f}, roc_auc: {roc_auc_sc:>8.5f}, pr_auc: {pr_auc_sc:>8.5f}')
        else:
            print(f'Epoch: {epoch:>4}, Batch: {bi:>5}, loss: {loss:>8.5f}, roc_auc: {roc_auc_sc:>8.5f}, pr_auc: {pr_auc_sc:>8.5f}')
        """

        # Retrieve and store batch-level results back from GPU/TPU
        targets_np = targets.cpu().detach().numpy().tolist()
        outputs_np = outputs.cpu().detach().numpy().tolist()
        fin_targets.extend(targets_np)
        fin_outputs.extend(outputs_np)    

    return fin_outputs, fin_targets


def _run(model, tokenizer, config, df_train, df_valid):
    """
    Performs all necessary steps for training and validation.
    
    Args:
      model (PyTorch model): BERT-based model to be fine-tuned.
      tokenizer (transformers Tokenizer): Pytorch BERT tokenizer object.
      config (dict): Dictionary containing some experiment-specific
          configuration parameters.
      df_train (pandas DataFrame): DataFrame holding all training data.
      df_valid (pandas DataFrame): DataFrame holding all validation data.
    """
 
    EPOCHS = config["train"]["epochs"]
 
    device = torch.device(DEVICE)
    
    print(f"Loading model to '{device}' device...")
    model = model.to(device)
    print("Model loaded successfully!")

    n_samples_train = len(df_train)

    if n_TPUs > 0:
        n_epochs = config["train"].get("epochs", 50)
        batch_size = config["train"].get("batch_size", 32)
        num_train_steps = int(config["train"].get("train_pct", 1.) * n_samples_train / batch_size * n_epochs)
        num_train_steps /= n_TPUs
        xm.master_print(f'num_train_steps = {num_train_steps}, world_size={n_TPUs}')
        
    # Create optimizer, scheduler and criterion objects    
    optimizer_dict = get_optimizer_conf(model=model, config=config.get("train", dict()), n_samples_train=n_samples_train)
    
    train_loss = []
    valid_loss = []
    best_score = 0
    
    train_begin = time.time()
    for epoch in range(EPOCHS):
 
        start = time.time()
    
        # Epoch monitoring
        print('*'*15)
        print(f'EPOCH: {epoch+1}')
        print('*'*15)
    
        # New DataLoader per epoch to ensure shuffling the data and allow more advanced data loading strategies in the future
        train_data_loader = get_data_loader(df=df_train, tokenizer=tokenizer, config=config["train"], sample_pct=config["train"]["train_pct"], n_TPUs=n_TPUs)
        validation_data_loader = get_data_loader(df=df_valid, tokenizer=tokenizer, config=config["train"], n_TPUs=n_TPUs)
 
        if n_TPUs > 0:
            train_para_loader = pl.ParallelLoader(train_data_loader, [device])
            valid_para_loader = pl.ParallelLoader(validation_data_loader, [device])
 
        print('Training.....')
            
        # Batch by batch training (evaluation + backpropagation + weight optimization step)
        tr_outputs, tr_targets = train_loop_fn(
            epoch=epoch+1,
            model=model,
            data_loader=train_para_loader.per_device_loader(device) if n_TPUs > 0 else train_data_loader,
            optimizer=optimizer_dict['optimizer'], 
            criterion=optimizer_dict['criterion'],
            scheduler=optimizer_dict['scheduler'],
            device=device,
            n_TPUs=n_TPUs
        )
        
        # No gradient evaluation needed for validation
        with torch.no_grad():
            
            print('Validating....')
            
            # Batch by batch validation evaluation
            ev_outputs, ev_targets = eval_loop_fn(
                epoch=epoch+1,
                model=model,
                data_loader=valid_para_loader.per_device_loader(device) if n_TPUs > 0 else validation_data_loader,
                criterion=optimizer_dict['criterion'], 
                device=device)
            
            # Training aggregated metrics calculation
            try:
                tr_targets = np.array(tr_targets) >= 0.5
                tr_roc_auc_sc = roc_auc_score(tr_targets, tr_outputs)
                tr_precision, tr_recall, _ = precision_recall_curve(tr_targets, tr_outputs)
                tr_pr_auc_sc = auc(tr_recall, tr_precision)
            except ValueError as e:
                print(tr_targets.shape)
                print(tr_outputs.shape)
                raise e
            
            # Validation aggregated metrics calculation
            try:
                ev_targets = np.array(ev_targets) >= 0.5
                ev_roc_auc_sc = roc_auc_score(ev_targets, ev_outputs)
                ev_precision, ev_recall, _ = precision_recall_curve(ev_targets, ev_outputs)
                ev_pr_auc_sc = auc(ev_recall, ev_precision)
            except ValueError as e:
                print(ev_targets.shape)
                print(ev_outputs.shape)
                raise e
            
            # Aggregated metrics monitoring (format: <training_metric> / <validation_metric>)
            print(f"ROC AUC Score = {tr_roc_auc_sc} / {ev_roc_auc_sc}")
            print(f"Precision/Recall AUC Score = {tr_pr_auc_sc} / {ev_pr_auc_sc}")
            
            # Save model state if best performance so far, based on Precision / Recall AUC in validation
            if ev_pr_auc_sc > best_score:
                
                if n_TPUs > 0:
                    xm.save(
                        model.state_dict(),
                        str(Path(WORKING_DIR / config["train"]["model"]["output_path"]).resolve() / "{subtype}_{timestamp}.pt".format(
                            subtype=config["train"]["model"]["subtype"],
                            timestamp=int(time.time())
                        ))
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        str(Path(WORKING_DIR / config["train"]["model"]["output_path"]).resolve() / "{subtype}_{timestamp}.pt".format(
                            subtype=config["train"]["model"]["subtype"],
                            timestamp=int(time.time())
                        ))
                    )
                
                best_score = ev_pr_auc_sc
        
        time_spent = time.time() - start
    
        print(f'Epoch completed in {int(time_spent/60)} minutes, {int(time_spent % 60)} seconds')
        
    time_spent = time.time() - train_begin
        
    print(f'Training completed in {int(time_spent/60)} minutes, {int(time_spent % 60)} seconds')

```

## Train + Validation: Job launch


```python
from_folder = Path(WORKING_DIR / CONFIG["train"]["model"]["from_folder"]).resolve()
model, tokenizer = get_model_artifacts(CONFIG["train"]["model"], from_folder=from_folder)

# Start training processes
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = _run(model, tokenizer, CONFIG, df_train_balanced, df_valid)

if n_TPUs > 0:
    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=n_TPUs, start_method='fork')
else:
    _run(model, tokenizer, CONFIG, df_train_balanced, df_valid)
```

    Loading model to 'cuda' device...
    Model loaded successfully!
    ***************
    EPOCH: 1
    ***************
    Training.....
    Epoch:    1, Batch:     0, loss:  0.70951, roc_auc:  0.48438, pr_auc:  0.51920
    Epoch:    1, Batch:   100, loss:  0.69787, roc_auc:  0.52381, pr_auc:  0.60843
    Epoch:    1, Batch:   200, loss:  0.65886, roc_auc:  0.68421, pr_auc:  0.66110
    Epoch:    1, Batch:   300, loss:  0.66732, roc_auc:  0.63968, pr_auc:  0.74774
    Epoch:    1, Batch:   400, loss:  0.61759, roc_auc:  0.78039, pr_auc:  0.80291
    Epoch:    1, Batch:   500, loss:  0.51146, roc_auc:  0.94922, pr_auc:  0.95471
    Epoch:    1, Batch:   600, loss:  0.56295, roc_auc:  0.80162, pr_auc:  0.79291
    Epoch:    1, Batch:   700, loss:  0.54145, roc_auc:  0.86458, pr_auc:  0.94864
    Epoch:    1, Batch:   800, loss:  0.48175, roc_auc:  0.84848, pr_auc:  0.87850
    Epoch:    1, Batch:   900, loss:  0.54687, roc_auc:  0.81349, pr_auc:  0.86126
    Epoch:    1, Batch:  1000, loss:  0.62745, roc_auc:  0.70196, pr_auc:  0.62113
    Epoch:    1, Batch:  1100, loss:  0.38236, roc_auc:  0.93359, pr_auc:  0.93981
    Epoch:    1, Batch:  1200, loss:  0.39617, roc_auc:  0.92969, pr_auc:  0.93215
    Validating....
    ROC AUC Score = 0.7794284825005133 / 0.9038799094095593
    Precision/Recall AUC Score = 0.7733681601152231 / 0.4736801555675949
    Epoch completed in 1 minutes, 26 seconds
    ***************
    EPOCH: 2
    ***************
    Training.....
    Epoch:    2, Batch:     0, loss:  0.45330, roc_auc:  0.87698, pr_auc:  0.90862
    Epoch:    2, Batch:   100, loss:  0.47475, roc_auc:  0.85938, pr_auc:  0.86380
    Epoch:    2, Batch:   200, loss:  0.46956, roc_auc:  0.84706, pr_auc:  0.87871
    Epoch:    2, Batch:   300, loss:  0.50690, roc_auc:  0.84127, pr_auc:  0.90059
    Epoch:    2, Batch:   400, loss:  0.49238, roc_auc:  0.85507, pr_auc:  0.94897
    Epoch:    2, Batch:   500, loss:  0.32759, roc_auc:  0.94332, pr_auc:  0.95845
    Epoch:    2, Batch:   600, loss:  0.35461, roc_auc:  0.94510, pr_auc:  0.93551
    Epoch:    2, Batch:   700, loss:  0.42334, roc_auc:  0.92083, pr_auc:  0.83020
    Epoch:    2, Batch:   800, loss:  0.40917, roc_auc:  0.90476, pr_auc:  0.84826
    Epoch:    2, Batch:   900, loss:  0.42545, roc_auc:  0.87059, pr_auc:  0.85145
    Epoch:    2, Batch:  1000, loss:  0.39068, roc_auc:  0.90980, pr_auc:  0.88516
    Epoch:    2, Batch:  1100, loss:  0.43703, roc_auc:  0.88492, pr_auc:  0.90175
    Epoch:    2, Batch:  1200, loss:  0.35188, roc_auc:  0.96761, pr_auc:  0.98270
    Validating....
    ROC AUC Score = 0.8929377083628128 / 0.9167116864657484
    Precision/Recall AUC Score = 0.8844897124518124 / 0.5172693831104419
    Epoch completed in 1 minutes, 30 seconds
    ***************
    EPOCH: 3
    ***************
    Training.....
    Epoch:    3, Batch:     0, loss:  0.42773, roc_auc:  0.92063, pr_auc:  0.77626
    Epoch:    3, Batch:   100, loss:  0.41383, roc_auc:  0.87109, pr_auc:  0.85493
    Epoch:    3, Batch:   200, loss:  0.47877, roc_auc:  0.88235, pr_auc:  0.89714
    Epoch:    3, Batch:   300, loss:  0.24868, roc_auc:  0.97403, pr_auc:  0.96197
    Epoch:    3, Batch:   400, loss:  0.55305, roc_auc:  0.82251, pr_auc:  0.70995
    Epoch:    3, Batch:   500, loss:  0.26643, roc_auc:  0.95686, pr_auc:  0.94716
    Epoch:    3, Batch:   600, loss:  0.26505, roc_auc:  0.97266, pr_auc:  0.97482
    Epoch:    3, Batch:   700, loss:  0.37818, roc_auc:  0.91667, pr_auc:  0.89506
    Epoch:    3, Batch:   800, loss:  0.52874, roc_auc:  0.83529, pr_auc:  0.77113
    Epoch:    3, Batch:   900, loss:  0.49107, roc_auc:  0.80087, pr_auc:  0.88630
    Epoch:    3, Batch:  1000, loss:  0.41305, roc_auc:  0.87698, pr_auc:  0.83036
    Epoch:    3, Batch:  1100, loss:  0.46909, roc_auc:  0.89177, pr_auc:  0.80400
    Epoch:    3, Batch:  1200, loss:  0.47602, roc_auc:  0.86111, pr_auc:  0.82750
    Validating....
    ROC AUC Score = 0.9040179791677846 / 0.9196146678558739
    Precision/Recall AUC Score = 0.8955914399476339 / 0.531856379983491
    Epoch completed in 1 minutes, 31 seconds
    ***************
    EPOCH: 4
    ***************
    Training.....
    Epoch:    4, Batch:     0, loss:  0.22817, roc_auc:  0.96875, pr_auc:  0.96832
    Epoch:    4, Batch:   100, loss:  0.35790, roc_auc:  0.92208, pr_auc:  0.96134
    Epoch:    4, Batch:   200, loss:  0.49683, roc_auc:  0.82812, pr_auc:  0.79081
    Epoch:    4, Batch:   300, loss:  0.37986, roc_auc:  0.92857, pr_auc:  0.95090
    Epoch:    4, Batch:   400, loss:  0.17337, roc_auc:  1.00000, pr_auc:  1.00000
    Epoch:    4, Batch:   500, loss:  0.36101, roc_auc:  0.90873, pr_auc:  0.90772
    Epoch:    4, Batch:   600, loss:  0.35795, roc_auc:  0.92157, pr_auc:  0.90768
    Epoch:    4, Batch:   700, loss:  0.26141, roc_auc:  0.97255, pr_auc:  0.96875
    Epoch:    4, Batch:   800, loss:  0.33313, roc_auc:  0.93750, pr_auc:  0.96057
    Epoch:    4, Batch:   900, loss:  0.38761, roc_auc:  0.88889, pr_auc:  0.87057
    Epoch:    4, Batch:  1000, loss:  0.49789, roc_auc:  0.84583, pr_auc:  0.90937
    Epoch:    4, Batch:  1100, loss:  0.30702, roc_auc:  0.95142, pr_auc:  0.91461
    Epoch:    4, Batch:  1200, loss:  0.29256, roc_auc:  0.95671, pr_auc:  0.97542
    Validating....
    ROC AUC Score = 0.9059482751313523 / 0.9199911340328796
    Precision/Recall AUC Score = 0.8973972824984288 / 0.5353818920997232
    Epoch completed in 1 minutes, 31 seconds
    ***************
    EPOCH: 5
    ***************
    Training.....
    Epoch:    5, Batch:     0, loss:  0.33262, roc_auc:  0.92208, pr_auc:  0.88890
    Epoch:    5, Batch:   100, loss:  0.37168, roc_auc:  0.91765, pr_auc:  0.91006
    Epoch:    5, Batch:   200, loss:  0.39289, roc_auc:  0.87923, pr_auc:  0.72295
    Epoch:    5, Batch:   300, loss:  0.35811, roc_auc:  0.91667, pr_auc:  0.94189
    Epoch:    5, Batch:   400, loss:  0.42898, roc_auc:  0.85000, pr_auc:  0.89872
    Epoch:    5, Batch:   500, loss:  0.26538, roc_auc:  0.98016, pr_auc:  0.98492
    Epoch:    5, Batch:   600, loss:  0.72815, roc_auc:  0.83333, pr_auc:  0.85093
    Epoch:    5, Batch:   700, loss:  0.28782, roc_auc:  0.96825, pr_auc:  0.94547
    Epoch:    5, Batch:   800, loss:  0.33493, roc_auc:  0.93522, pr_auc:  0.90926
    Epoch:    5, Batch:   900, loss:  0.28594, roc_auc:  0.94902, pr_auc:  0.94515
    Epoch:    5, Batch:  1000, loss:  0.40806, roc_auc:  0.87451, pr_auc:  0.84797
    Epoch:    5, Batch:  1100, loss:  0.33979, roc_auc:  0.91903, pr_auc:  0.90457
    Epoch:    5, Batch:  1200, loss:  0.31093, roc_auc:  0.96471, pr_auc:  0.95446
    Validating....
    ROC AUC Score = 0.9089567409224062 / 0.9213675213675214
    Precision/Recall AUC Score = 0.9001958481933183 / 0.5421484772903109
    Epoch completed in 1 minutes, 31 seconds
    ***************
    EPOCH: 6
    ***************
    Training.....
    Epoch:    6, Batch:     0, loss:  0.38344, roc_auc:  0.90196, pr_auc:  0.84439
    Epoch:    6, Batch:   100, loss:  0.37790, roc_auc:  0.90079, pr_auc:  0.89382
    Epoch:    6, Batch:   200, loss:  0.46259, roc_auc:  0.85547, pr_auc:  0.81095
    Epoch:    6, Batch:   300, loss:  0.61789, roc_auc:  0.78516, pr_auc:  0.77943
    Epoch:    6, Batch:   400, loss:  0.35171, roc_auc:  0.90588, pr_auc:  0.90817
    Epoch:    6, Batch:   500, loss:  0.46206, roc_auc:  0.88492, pr_auc:  0.91295
    Epoch:    6, Batch:   600, loss:  0.47129, roc_auc:  0.88664, pr_auc:  0.81145
    Epoch:    6, Batch:   700, loss:  0.24773, roc_auc:  0.96429, pr_auc:  0.96112
    Epoch:    6, Batch:   800, loss:  0.30463, roc_auc:  0.96471, pr_auc:  0.96188
    Epoch:    6, Batch:   900, loss:  0.38054, roc_auc:  0.90196, pr_auc:  0.88947
    Epoch:    6, Batch:  1000, loss:  0.48238, roc_auc:  0.85882, pr_auc:  0.80268
    Epoch:    6, Batch:  1100, loss:  0.41562, roc_auc:  0.95312, pr_auc:  0.95694
    Epoch:    6, Batch:  1200, loss:  0.42243, roc_auc:  0.86508, pr_auc:  0.87911
    Validating....
    ROC AUC Score = 0.9089452100056629 / 0.9212585292987303
    Precision/Recall AUC Score = 0.9003405244148437 / 0.5360289296024917
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 7
    ***************
    Training.....
    Epoch:    7, Batch:     0, loss:  0.53355, roc_auc:  0.82591, pr_auc:  0.75940
    Epoch:    7, Batch:   100, loss:  0.28606, roc_auc:  0.95238, pr_auc:  0.92070
    Epoch:    7, Batch:   200, loss:  0.29236, roc_auc:  0.94444, pr_auc:  0.95744
    Epoch:    7, Batch:   300, loss:  0.28768, roc_auc:  0.96429, pr_auc:  0.97371
    Epoch:    7, Batch:   400, loss:  0.30370, roc_auc:  0.92857, pr_auc:  0.92954
    Epoch:    7, Batch:   500, loss:  0.33422, roc_auc:  0.91903, pr_auc:  0.87141
    Epoch:    7, Batch:   600, loss:  0.31432, roc_auc:  0.93651, pr_auc:  0.93971
    Epoch:    7, Batch:   700, loss:  0.44228, roc_auc:  0.87109, pr_auc:  0.85842
    Epoch:    7, Batch:   800, loss:  0.44356, roc_auc:  0.86667, pr_auc:  0.89122
    Epoch:    7, Batch:   900, loss:  0.24065, roc_auc:  0.97166, pr_auc:  0.96659
    Epoch:    7, Batch:  1000, loss:  0.35013, roc_auc:  0.92969, pr_auc:  0.91288
    Epoch:    7, Batch:  1100, loss:  0.37519, roc_auc:  0.91797, pr_auc:  0.90695
    Epoch:    7, Batch:  1200, loss:  0.37988, roc_auc:  0.90476, pr_auc:  0.93946
    Validating....
    ROC AUC Score = 0.9113503374670486 / 0.9218119009390355
    Precision/Recall AUC Score = 0.9031135627281439 / 0.5412389217226116
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 8
    ***************
    Training.....
    Epoch:    8, Batch:     0, loss:  0.36031, roc_auc:  0.92549, pr_auc:  0.93886
    Epoch:    8, Batch:   100, loss:  0.33877, roc_auc:  0.91250, pr_auc:  0.84265
    Epoch:    8, Batch:   200, loss:  0.45304, roc_auc:  0.91667, pr_auc:  0.93570
    Epoch:    8, Batch:   300, loss:  0.51183, roc_auc:  0.84416, pr_auc:  0.74804
    Epoch:    8, Batch:   400, loss:  0.30700, roc_auc:  0.96761, pr_auc:  0.98170
    Epoch:    8, Batch:   500, loss:  0.35762, roc_auc:  0.94048, pr_auc:  0.95682
    Epoch:    8, Batch:   600, loss:  0.36340, roc_auc:  0.90873, pr_auc:  0.91584
    Epoch:    8, Batch:   700, loss:  0.39019, roc_auc:  0.91406, pr_auc:  0.92229
    Epoch:    8, Batch:   800, loss:  0.36370, roc_auc:  0.90476, pr_auc:  0.91617
    Epoch:    8, Batch:   900, loss:  0.30474, roc_auc:  0.94444, pr_auc:  0.95875
    Epoch:    8, Batch:  1000, loss:  0.32503, roc_auc:  0.93117, pr_auc:  0.94964
    Epoch:    8, Batch:  1100, loss:  0.34376, roc_auc:  0.94444, pr_auc:  0.94930
    Epoch:    8, Batch:  1200, loss:  0.65964, roc_auc:  0.78516, pr_auc:  0.74034
    Validating....
    ROC AUC Score = 0.9105948858888707 / 0.9226391793163164
    Precision/Recall AUC Score = 0.9017103491914169 / 0.5404895730185617
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 9
    ***************
    Training.....
    Epoch:    9, Batch:     0, loss:  0.38082, roc_auc:  0.90196, pr_auc:  0.90262
    Epoch:    9, Batch:   100, loss:  0.29008, roc_auc:  0.94841, pr_auc:  0.94009
    Epoch:    9, Batch:   200, loss:  0.45054, roc_auc:  0.86667, pr_auc:  0.86353
    Epoch:    9, Batch:   300, loss:  0.22277, roc_auc:  0.98047, pr_auc:  0.98095
    Epoch:    9, Batch:   400, loss:  0.35120, roc_auc:  0.91093, pr_auc:  0.91657
    Epoch:    9, Batch:   500, loss:  0.36088, roc_auc:  0.91373, pr_auc:  0.93137
    Epoch:    9, Batch:   600, loss:  0.37844, roc_auc:  0.90873, pr_auc:  0.91990
    Epoch:    9, Batch:   700, loss:  0.59814, roc_auc:  0.76471, pr_auc:  0.68797
    Epoch:    9, Batch:   800, loss:  0.33642, roc_auc:  0.95294, pr_auc:  0.96615
    Epoch:    9, Batch:   900, loss:  0.41697, roc_auc:  0.88664, pr_auc:  0.92123
    Epoch:    9, Batch:  1000, loss:  0.30475, roc_auc:  0.95294, pr_auc:  0.95847
    Epoch:    9, Batch:  1100, loss:  0.67226, roc_auc:  0.76518, pr_auc:  0.57329
    Epoch:    9, Batch:  1200, loss:  0.31365, roc_auc:  0.93359, pr_auc:  0.92566
    Validating....
    ROC AUC Score = 0.9122178013921158 / 0.9234895632578433
    Precision/Recall AUC Score = 0.9041737491671925 / 0.5447542947718694
    Epoch completed in 1 minutes, 30 seconds
    ***************
    EPOCH: 10
    ***************
    Training.....
    Epoch:   10, Batch:     0, loss:  0.36688, roc_auc:  0.93333, pr_auc:  0.94500
    Epoch:   10, Batch:   100, loss:  0.40431, roc_auc:  0.90980, pr_auc:  0.91489
    Epoch:   10, Batch:   200, loss:  0.36186, roc_auc:  0.90980, pr_auc:  0.92527
    Epoch:   10, Batch:   300, loss:  0.31152, roc_auc:  0.94167, pr_auc:  0.90403
    Epoch:   10, Batch:   400, loss:  0.62894, roc_auc:  0.80087, pr_auc:  0.72988
    Epoch:   10, Batch:   500, loss:  0.51307, roc_auc:  0.85830, pr_auc:  0.78020
    Epoch:   10, Batch:   600, loss:  0.47449, roc_auc:  0.87451, pr_auc:  0.87909
    Epoch:   10, Batch:   700, loss:  0.34334, roc_auc:  0.92549, pr_auc:  0.93743
    Epoch:   10, Batch:   800, loss:  0.41870, roc_auc:  0.90476, pr_auc:  0.92691
    Epoch:   10, Batch:   900, loss:  0.41972, roc_auc:  0.88664, pr_auc:  0.91045
    Epoch:   10, Batch:  1000, loss:  0.48148, roc_auc:  0.87843, pr_auc:  0.85859
    Epoch:   10, Batch:  1100, loss:  0.36446, roc_auc:  0.93750, pr_auc:  0.95061
    Epoch:   10, Batch:  1200, loss:  0.21058, roc_auc:  0.97656, pr_auc:  0.97687
    Validating....
    ROC AUC Score = 0.9105448247192018 / 0.9219176123231949
    Precision/Recall AUC Score = 0.901523167167274 / 0.5407384469223888
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 11
    ***************
    Training.....
    Epoch:   11, Batch:     0, loss:  0.30062, roc_auc:  0.94167, pr_auc:  0.93242
    Epoch:   11, Batch:   100, loss:  0.42282, roc_auc:  0.88259, pr_auc:  0.85834
    Epoch:   11, Batch:   200, loss:  0.30324, roc_auc:  0.95417, pr_auc:  0.96946
    Epoch:   11, Batch:   300, loss:  0.24654, roc_auc:  0.96250, pr_auc:  0.97801
    Epoch:   11, Batch:   400, loss:  0.40219, roc_auc:  0.88235, pr_auc:  0.88939
    Epoch:   11, Batch:   500, loss:  0.28743, roc_auc:  0.95294, pr_auc:  0.94941
    Epoch:   11, Batch:   600, loss:  0.50897, roc_auc:  0.87013, pr_auc:  0.90218
    Epoch:   11, Batch:   700, loss:  0.30883, roc_auc:  0.92578, pr_auc:  0.91132
    Epoch:   11, Batch:   800, loss:  0.24768, roc_auc:  0.96471, pr_auc:  0.96508
    Epoch:   11, Batch:   900, loss:  0.28983, roc_auc:  0.95294, pr_auc:  0.94638
    Epoch:   11, Batch:  1000, loss:  0.27972, roc_auc:  0.95635, pr_auc:  0.96480
    Epoch:   11, Batch:  1100, loss:  0.24644, roc_auc:  0.97500, pr_auc:  0.98533
    Epoch:   11, Batch:  1200, loss:  0.54188, roc_auc:  0.86275, pr_auc:  0.85159
    Validating....
    ROC AUC Score = 0.9115277727248616 / 0.9233166377912623
    Precision/Recall AUC Score = 0.9031330599872509 / 0.5407106633248504
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 12
    ***************
    Training.....
    Epoch:   12, Batch:     0, loss:  0.45247, roc_auc:  0.92460, pr_auc:  0.89174
    Epoch:   12, Batch:   100, loss:  0.29123, roc_auc:  0.95000, pr_auc:  0.90216
    Epoch:   12, Batch:   200, loss:  0.34279, roc_auc:  0.93333, pr_auc:  0.92360
    Epoch:   12, Batch:   300, loss:  0.30423, roc_auc:  0.93725, pr_auc:  0.92284
    Epoch:   12, Batch:   400, loss:  0.39744, roc_auc:  0.89804, pr_auc:  0.88995
    Epoch:   12, Batch:   500, loss:  0.32557, roc_auc:  0.95294, pr_auc:  0.95817
    Epoch:   12, Batch:   600, loss:  0.20095, roc_auc:  0.98824, pr_auc:  0.98615
    Epoch:   12, Batch:   700, loss:  0.26761, roc_auc:  0.95703, pr_auc:  0.95911
    Epoch:   12, Batch:   800, loss:  0.29706, roc_auc:  0.94805, pr_auc:  0.97165
    Epoch:   12, Batch:   900, loss:  0.27077, roc_auc:  0.95703, pr_auc:  0.95742
    Epoch:   12, Batch:  1000, loss:  0.22014, roc_auc:  0.97222, pr_auc:  0.97749
    Epoch:   12, Batch:  1100, loss:  0.43206, roc_auc:  0.86328, pr_auc:  0.84760
    Epoch:   12, Batch:  1200, loss:  0.37566, roc_auc:  0.90909, pr_auc:  0.81642
    Validating....
    ROC AUC Score = 0.9108428490023653 / 0.922609837979857
    Precision/Recall AUC Score = 0.9029202695456656 / 0.5510494871911722
    Epoch completed in 1 minutes, 42 seconds
    ***************
    EPOCH: 13
    ***************
    Training.....
    Epoch:   13, Batch:     0, loss:  0.32079, roc_auc:  0.94902, pr_auc:  0.94913
    Epoch:   13, Batch:   100, loss:  0.46120, roc_auc:  0.85830, pr_auc:  0.83325
    Epoch:   13, Batch:   200, loss:  0.45079, roc_auc:  0.88627, pr_auc:  0.87495
    Epoch:   13, Batch:   300, loss:  0.27727, roc_auc:  0.96356, pr_auc:  0.97354
    Epoch:   13, Batch:   400, loss:  0.31133, roc_auc:  0.94510, pr_auc:  0.95204
    Epoch:   13, Batch:   500, loss:  0.24084, roc_auc:  0.99567, pr_auc:  0.99778
    Epoch:   13, Batch:   600, loss:  0.49924, roc_auc:  0.78333, pr_auc:  0.86543
    Epoch:   13, Batch:   700, loss:  0.33127, roc_auc:  0.92857, pr_auc:  0.86792
    Epoch:   13, Batch:   800, loss:  0.40667, roc_auc:  0.90283, pr_auc:  0.79162
    Epoch:   13, Batch:   900, loss:  0.52492, roc_auc:  0.82251, pr_auc:  0.70793
    Epoch:   13, Batch:  1000, loss:  0.43987, roc_auc:  0.86667, pr_auc:  0.89531
    Epoch:   13, Batch:  1100, loss:  0.32955, roc_auc:  0.95312, pr_auc:  0.94788
    Epoch:   13, Batch:  1200, loss:  0.28180, roc_auc:  0.97917, pr_auc:  0.98845
    Validating....
    ROC AUC Score = 0.9116502910000662 / 0.9224607887924471
    Precision/Recall AUC Score = 0.9028491756101922 / 0.5478398599489253
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 14
    ***************
    Training.....
    Epoch:   14, Batch:     0, loss:  0.51931, roc_auc:  0.83984, pr_auc:  0.77274
    Epoch:   14, Batch:   100, loss:  0.46043, roc_auc:  0.88627, pr_auc:  0.87515
    Epoch:   14, Batch:   200, loss:  0.42334, roc_auc:  0.88627, pr_auc:  0.90782
    Epoch:   14, Batch:   300, loss:  0.33209, roc_auc:  0.93725, pr_auc:  0.94371
    Epoch:   14, Batch:   400, loss:  0.46839, roc_auc:  0.86508, pr_auc:  0.85175
    Epoch:   14, Batch:   500, loss:  0.45288, roc_auc:  0.88281, pr_auc:  0.86092
    Epoch:   14, Batch:   600, loss:  0.36262, roc_auc:  0.91270, pr_auc:  0.86741
    Epoch:   14, Batch:   700, loss:  0.33501, roc_auc:  0.93117, pr_auc:  0.91239
    Epoch:   14, Batch:   800, loss:  0.34623, roc_auc:  0.93333, pr_auc:  0.92034
    Epoch:   14, Batch:   900, loss:  0.60612, roc_auc:  0.82917, pr_auc:  0.68859
    Epoch:   14, Batch:  1000, loss:  0.30681, roc_auc:  0.95703, pr_auc:  0.96799
    Epoch:   14, Batch:  1100, loss:  0.31703, roc_auc:  0.95238, pr_auc:  0.96824
    Epoch:   14, Batch:  1200, loss:  0.53171, roc_auc:  0.86719, pr_auc:  0.85683
    Validating....
    ROC AUC Score = 0.9125648893153923 / 0.9234114294415801
    Precision/Recall AUC Score = 0.9036242235329476 / 0.545454819422086
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 15
    ***************
    Training.....
    Epoch:   15, Batch:     0, loss:  0.49170, roc_auc:  0.89683, pr_auc:  0.87641
    Epoch:   15, Batch:   100, loss:  0.21063, roc_auc:  1.00000, pr_auc:  1.00000
    Epoch:   15, Batch:   200, loss:  0.45007, roc_auc:  0.85024, pr_auc:  0.67985
    Epoch:   15, Batch:   300, loss:  0.39167, roc_auc:  0.91304, pr_auc:  0.82085
    Epoch:   15, Batch:   400, loss:  0.20567, roc_auc:  0.99219, pr_auc:  0.99217
    Epoch:   15, Batch:   500, loss:  0.37418, roc_auc:  0.90079, pr_auc:  0.81329
    Epoch:   15, Batch:   600, loss:  0.28316, roc_auc:  0.96863, pr_auc:  0.97032
    Epoch:   15, Batch:   700, loss:  0.42878, roc_auc:  0.87045, pr_auc:  0.85371
    Epoch:   15, Batch:   800, loss:  0.22265, roc_auc:  0.98047, pr_auc:  0.98475
    Epoch:   15, Batch:   900, loss:  0.27359, roc_auc:  0.96356, pr_auc:  0.95692
    Epoch:   15, Batch:  1000, loss:  0.29487, roc_auc:  0.95686, pr_auc:  0.96112
    Epoch:   15, Batch:  1100, loss:  0.38378, roc_auc:  0.94805, pr_auc:  0.90279
    Epoch:   15, Batch:  1200, loss:  0.34076, roc_auc:  0.91146, pr_auc:  0.74571
    Validating....
    ROC AUC Score = 0.9130014069467006 / 0.9229745408137368
    Precision/Recall AUC Score = 0.9032959550771721 / 0.5386605402914014
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 16
    ***************
    Training.....
    Epoch:   16, Batch:     0, loss:  0.62371, roc_auc:  0.85425, pr_auc:  0.77104
    Epoch:   16, Batch:   100, loss:  0.34546, roc_auc:  0.94510, pr_auc:  0.93598
    Epoch:   16, Batch:   200, loss:  0.21197, roc_auc:  0.97917, pr_auc:  0.95947
    Epoch:   16, Batch:   300, loss:  0.27774, roc_auc:  0.95951, pr_auc:  0.97200
    Epoch:   16, Batch:   400, loss:  0.27262, roc_auc:  0.97266, pr_auc:  0.97640
    Epoch:   16, Batch:   500, loss:  0.49345, roc_auc:  0.83922, pr_auc:  0.80920
    Epoch:   16, Batch:   600, loss:  0.31278, roc_auc:  0.96094, pr_auc:  0.96982
    Epoch:   16, Batch:   700, loss:  0.20837, roc_auc:  0.97656, pr_auc:  0.98255
    Epoch:   16, Batch:   800, loss:  0.41234, roc_auc:  0.90196, pr_auc:  0.89306
    Epoch:   16, Batch:   900, loss:  0.59393, roc_auc:  0.77647, pr_auc:  0.72707
    Epoch:   16, Batch:  1000, loss:  0.48418, roc_auc:  0.85833, pr_auc:  0.89155
    Epoch:   16, Batch:  1100, loss:  0.34512, roc_auc:  0.93074, pr_auc:  0.87852
    Epoch:   16, Batch:  1200, loss:  0.43602, roc_auc:  0.87451, pr_auc:  0.83710
    Validating....
    ROC AUC Score = 0.9115164567444114 / 0.9232537018115765
    Precision/Recall AUC Score = 0.90333142328741 / 0.5460893276549447
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 17
    ***************
    Training.....
    Epoch:   17, Batch:     0, loss:  0.65692, roc_auc:  0.73810, pr_auc:  0.56035
    Epoch:   17, Batch:   100, loss:  0.25550, roc_auc:  0.97222, pr_auc:  0.97621
    Epoch:   17, Batch:   200, loss:  0.35005, roc_auc:  0.92500, pr_auc:  0.95918
    Epoch:   17, Batch:   300, loss:  0.36908, roc_auc:  0.91765, pr_auc:  0.93371
    Epoch:   17, Batch:   400, loss:  0.52007, roc_auc:  0.80162, pr_auc:  0.86504
    Epoch:   17, Batch:   500, loss:  0.33748, roc_auc:  0.93651, pr_auc:  0.95432
    Epoch:   17, Batch:   600, loss:  0.45066, roc_auc:  0.86667, pr_auc:  0.88614
    Epoch:   17, Batch:   700, loss:  0.47955, roc_auc:  0.87059, pr_auc:  0.85992
    Epoch:   17, Batch:   800, loss:  0.43169, roc_auc:  0.85909, pr_auc:  0.77694
    Epoch:   17, Batch:   900, loss:  0.41628, roc_auc:  0.87879, pr_auc:  0.85652
    Epoch:   17, Batch:  1000, loss:  0.72648, roc_auc:  0.70588, pr_auc:  0.74011
    Epoch:   17, Batch:  1100, loss:  0.20063, roc_auc:  0.99206, pr_auc:  0.99430
    Epoch:   17, Batch:  1200, loss:  0.43376, roc_auc:  0.87451, pr_auc:  0.86134
    Validating....
    ROC AUC Score = 0.9122978414485978 / 0.9223004009891835
    Precision/Recall AUC Score = 0.9037917879440301 / 0.5438277699208718
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 18
    ***************
    Training.....
    Epoch:   18, Batch:     0, loss:  0.40507, roc_auc:  0.91373, pr_auc:  0.93493
    Epoch:   18, Batch:   100, loss:  0.57296, roc_auc:  0.83529, pr_auc:  0.82590
    Epoch:   18, Batch:   200, loss:  0.43414, roc_auc:  0.92857, pr_auc:  0.94124
    Epoch:   18, Batch:   300, loss:  0.34225, roc_auc:  0.92857, pr_auc:  0.93390
    Epoch:   18, Batch:   400, loss:  0.41073, roc_auc:  0.89683, pr_auc:  0.88216
    Epoch:   18, Batch:   500, loss:  0.37851, roc_auc:  0.90625, pr_auc:  0.89215
    Epoch:   18, Batch:   600, loss:  0.35232, roc_auc:  0.90625, pr_auc:  0.92246
    Epoch:   18, Batch:   700, loss:  0.48246, roc_auc:  0.82996, pr_auc:  0.80178
    Epoch:   18, Batch:   800, loss:  0.26831, roc_auc:  0.98431, pr_auc:  0.98738
    Epoch:   18, Batch:   900, loss:  0.44557, roc_auc:  0.87059, pr_auc:  0.84453
    Epoch:   18, Batch:  1000, loss:  0.36650, roc_auc:  0.92083, pr_auc:  0.88189
    Epoch:   18, Batch:  1100, loss:  0.45553, roc_auc:  0.90283, pr_auc:  0.83804
    Epoch:   18, Batch:  1200, loss:  0.28899, roc_auc:  0.94048, pr_auc:  0.91316
    Validating....
    ROC AUC Score = 0.9130883105514456 / 0.9229630267541405
    Precision/Recall AUC Score = 0.9046692698704091 / 0.5425819121345495
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 19
    ***************
    Training.....
    Epoch:   19, Batch:     0, loss:  0.30057, roc_auc:  0.95417, pr_auc:  0.97362
    Epoch:   19, Batch:   100, loss:  0.50963, roc_auc:  0.88750, pr_auc:  0.70885
    Epoch:   19, Batch:   200, loss:  0.35363, roc_auc:  0.90873, pr_auc:  0.88240
    Epoch:   19, Batch:   300, loss:  0.45075, roc_auc:  0.89844, pr_auc:  0.92049
    Epoch:   19, Batch:   400, loss:  0.33702, roc_auc:  0.94583, pr_auc:  0.93091
    Epoch:   19, Batch:   500, loss:  0.31966, roc_auc:  0.93359, pr_auc:  0.93738
    Epoch:   19, Batch:   600, loss:  0.12835, roc_auc:  0.99608, pr_auc:  0.99664
    Epoch:   19, Batch:   700, loss:  0.30791, roc_auc:  0.94444, pr_auc:  0.92853
    Epoch:   19, Batch:   800, loss:  0.31952, roc_auc:  0.92917, pr_auc:  0.88401
    Epoch:   19, Batch:   900, loss:  0.49860, roc_auc:  0.87500, pr_auc:  0.92187
    Epoch:   19, Batch:  1000, loss:  0.45398, roc_auc:  0.88095, pr_auc:  0.89645
    Epoch:   19, Batch:  1100, loss:  0.27231, roc_auc:  0.96032, pr_auc:  0.94596
    Epoch:   19, Batch:  1200, loss:  0.38370, roc_auc:  0.91016, pr_auc:  0.92489
    Validating....
    ROC AUC Score = 0.9133802237438908 / 0.9241754525034412
    Precision/Recall AUC Score = 0.9052632775861684 / 0.5464982536211096
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 20
    ***************
    Training.....
    Epoch:   20, Batch:     0, loss:  0.44091, roc_auc:  0.87854, pr_auc:  0.82934
    Epoch:   20, Batch:   100, loss:  0.40136, roc_auc:  0.89804, pr_auc:  0.91358
    Epoch:   20, Batch:   200, loss:  0.46504, roc_auc:  0.87698, pr_auc:  0.86299
    Epoch:   20, Batch:   300, loss:  0.32015, roc_auc:  0.95635, pr_auc:  0.94065
    Epoch:   20, Batch:   400, loss:  0.50320, roc_auc:  0.81781, pr_auc:  0.75170
    Epoch:   20, Batch:   500, loss:  0.42623, roc_auc:  0.93720, pr_auc:  0.86550
    Epoch:   20, Batch:   600, loss:  0.30580, roc_auc:  0.96078, pr_auc:  0.95819
    Epoch:   20, Batch:   700, loss:  0.49182, roc_auc:  0.80567, pr_auc:  0.72616
    Epoch:   20, Batch:   800, loss:  0.33015, roc_auc:  0.94545, pr_auc:  0.80743
    Epoch:   20, Batch:   900, loss:  0.45557, roc_auc:  0.86328, pr_auc:  0.89185
    Epoch:   20, Batch:  1000, loss:  0.34142, roc_auc:  0.96667, pr_auc:  0.98335
    Epoch:   20, Batch:  1100, loss:  0.30751, roc_auc:  0.92273, pr_auc:  0.94817
    Epoch:   20, Batch:  1200, loss:  0.36506, roc_auc:  0.89020, pr_auc:  0.86354
    Validating....
    ROC AUC Score = 0.9135529810989721 / 0.9226282055240967
    Precision/Recall AUC Score = 0.9044825338245803 / 0.539285632002827
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 21
    ***************
    Training.....
    Epoch:   21, Batch:     0, loss:  0.49246, roc_auc:  0.85938, pr_auc:  0.86461
    Epoch:   21, Batch:   100, loss:  0.30873, roc_auc:  0.93750, pr_auc:  0.96319
    Epoch:   21, Batch:   200, loss:  0.33354, roc_auc:  0.93333, pr_auc:  0.96315
    Epoch:   21, Batch:   300, loss:  0.30424, roc_auc:  0.94118, pr_auc:  0.93537
    Epoch:   21, Batch:   400, loss:  0.35559, roc_auc:  0.91270, pr_auc:  0.89306
    Epoch:   21, Batch:   500, loss:  0.54426, roc_auc:  0.82745, pr_auc:  0.86291
    Epoch:   21, Batch:   600, loss:  0.33934, roc_auc:  0.92713, pr_auc:  0.89875
    Epoch:   21, Batch:   700, loss:  0.38282, roc_auc:  0.90476, pr_auc:  0.85401
    Epoch:   21, Batch:   800, loss:  0.39737, roc_auc:  0.90476, pr_auc:  0.86966
    Epoch:   21, Batch:   900, loss:  0.36042, roc_auc:  0.90980, pr_auc:  0.91897
    Epoch:   21, Batch:  1000, loss:  0.19084, roc_auc:  1.00000, pr_auc:  1.00000
    Epoch:   21, Batch:  1100, loss:  0.46415, roc_auc:  0.86111, pr_auc:  0.83722
    Epoch:   21, Batch:  1200, loss:  0.35427, roc_auc:  0.94444, pr_auc:  0.93044
    Validating....
    ROC AUC Score = 0.9133233853979108 / 0.9230556579765795
    Precision/Recall AUC Score = 0.9046657540656717 / 0.5443032939286259
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 22
    ***************
    Training.....
    Epoch:   22, Batch:     0, loss:  0.36332, roc_auc:  0.91093, pr_auc:  0.92919
    Epoch:   22, Batch:   100, loss:  0.31643, roc_auc:  0.94902, pr_auc:  0.95552
    Epoch:   22, Batch:   200, loss:  0.28795, roc_auc:  0.95635, pr_auc:  0.94618
    Epoch:   22, Batch:   300, loss:  0.40725, roc_auc:  0.90980, pr_auc:  0.92355
    Epoch:   22, Batch:   400, loss:  0.34554, roc_auc:  0.91270, pr_auc:  0.88784
    Epoch:   22, Batch:   500, loss:  0.49576, roc_auc:  0.86667, pr_auc:  0.89068
    Epoch:   22, Batch:   600, loss:  0.20740, roc_auc:  1.00000, pr_auc:  1.00000
    Epoch:   22, Batch:   700, loss:  0.32776, roc_auc:  0.92500, pr_auc:  0.94484
    Epoch:   22, Batch:   800, loss:  0.53533, roc_auc:  0.86818, pr_auc:  0.74546
    Epoch:   22, Batch:   900, loss:  0.42698, roc_auc:  0.88259, pr_auc:  0.90947
    Epoch:   22, Batch:  1000, loss:  0.30565, roc_auc:  0.94510, pr_auc:  0.94870
    Epoch:   22, Batch:  1100, loss:  0.33846, roc_auc:  0.93651, pr_auc:  0.93506
    Epoch:   22, Batch:  1200, loss:  0.26826, roc_auc:  0.97255, pr_auc:  0.97760
    Validating....
    ROC AUC Score = 0.9129664421496668 / 0.9227515644601072
    Precision/Recall AUC Score = 0.9052890859911005 / 0.5399067690815867
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 23
    ***************
    Training.....
    Epoch:   23, Batch:     0, loss:  0.44297, roc_auc:  0.89286, pr_auc:  0.87961
    Epoch:   23, Batch:   100, loss:  0.29674, roc_auc:  0.96825, pr_auc:  0.97926
    Epoch:   23, Batch:   200, loss:  0.52469, roc_auc:  0.86111, pr_auc:  0.84283
    Epoch:   23, Batch:   300, loss:  0.25867, roc_auc:  0.97266, pr_auc:  0.97640
    Epoch:   23, Batch:   400, loss:  0.30064, roc_auc:  0.93939, pr_auc:  0.88666
    Epoch:   23, Batch:   500, loss:  0.44788, roc_auc:  0.87891, pr_auc:  0.90677
    Epoch:   23, Batch:   600, loss:  0.35289, roc_auc:  0.91797, pr_auc:  0.91847
    Epoch:   23, Batch:   700, loss:  0.35711, roc_auc:  0.92641, pr_auc:  0.96316
    Epoch:   23, Batch:   800, loss:  0.27335, roc_auc:  0.99216, pr_auc:  0.99191
    Epoch:   23, Batch:   900, loss:  0.40789, roc_auc:  0.92857, pr_auc:  0.90688
    Epoch:   23, Batch:  1000, loss:  0.44317, roc_auc:  0.87446, pr_auc:  0.78221
    Epoch:   23, Batch:  1100, loss:  0.33301, roc_auc:  0.89177, pr_auc:  0.90311
    Epoch:   23, Batch:  1200, loss:  0.45984, roc_auc:  0.85882, pr_auc:  0.85041
    Validating....
    ROC AUC Score = 0.9147508804007543 / 0.9231168964361653
    Precision/Recall AUC Score = 0.9060981985078214 / 0.5408870175677782
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 24
    ***************
    Training.....
    Epoch:   24, Batch:     0, loss:  0.48390, roc_auc:  0.86235, pr_auc:  0.81687
    Epoch:   24, Batch:   100, loss:  0.50457, roc_auc:  0.80952, pr_auc:  0.85052
    Epoch:   24, Batch:   200, loss:  0.33523, roc_auc:  0.92549, pr_auc:  0.90898
    Epoch:   24, Batch:   300, loss:  0.37981, roc_auc:  0.91903, pr_auc:  0.94357
    Epoch:   24, Batch:   400, loss:  0.30080, roc_auc:  0.95238, pr_auc:  0.93964
    Epoch:   24, Batch:   500, loss:  0.42745, roc_auc:  0.89020, pr_auc:  0.90647
    Epoch:   24, Batch:   600, loss:  0.30065, roc_auc:  0.94737, pr_auc:  0.92785
    Epoch:   24, Batch:   700, loss:  0.42921, roc_auc:  0.88889, pr_auc:  0.88900
    Epoch:   24, Batch:   800, loss:  0.28439, roc_auc:  0.94510, pr_auc:  0.95137
    Epoch:   24, Batch:   900, loss:  0.30422, roc_auc:  0.93333, pr_auc:  0.93477
    Epoch:   24, Batch:  1000, loss:  0.22485, roc_auc:  0.98828, pr_auc:  0.98786
    Epoch:   24, Batch:  1100, loss:  0.42774, roc_auc:  0.88636, pr_auc:  0.72630
    Epoch:   24, Batch:  1200, loss:  0.34868, roc_auc:  0.93333, pr_auc:  0.95368
    Validating....
    ROC AUC Score = 0.9117996012346946 / 0.9236125308333518
    Precision/Recall AUC Score = 0.9033234522097057 / 0.5418518953372694
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 25
    ***************
    Training.....
    Epoch:   25, Batch:     0, loss:  0.26075, roc_auc:  0.97255, pr_auc:  0.97760
    Epoch:   25, Batch:   100, loss:  0.37161, roc_auc:  0.92157, pr_auc:  0.90833
    Epoch:   25, Batch:   200, loss:  0.34948, roc_auc:  0.91667, pr_auc:  0.92539
    Epoch:   25, Batch:   300, loss:  0.35486, roc_auc:  0.91250, pr_auc:  0.89486
    Epoch:   25, Batch:   400, loss:  0.31050, roc_auc:  0.93651, pr_auc:  0.92207
    Epoch:   25, Batch:   500, loss:  0.54266, roc_auc:  0.84615, pr_auc:  0.78703
    Epoch:   25, Batch:   600, loss:  0.29923, roc_auc:  0.94737, pr_auc:  0.96066
    Epoch:   25, Batch:   700, loss:  0.43808, roc_auc:  0.88492, pr_auc:  0.85588
    Epoch:   25, Batch:   800, loss:  0.49321, roc_auc:  0.85714, pr_auc:  0.82192
    Epoch:   25, Batch:   900, loss:  0.17368, roc_auc:  0.99609, pr_auc:  0.99621
    Epoch:   25, Batch:  1000, loss:  0.65495, roc_auc:  0.77500, pr_auc:  0.79433
    Epoch:   25, Batch:  1100, loss:  0.37364, roc_auc:  0.91270, pr_auc:  0.93448
    Epoch:   25, Batch:  1200, loss:  0.40990, roc_auc:  0.89412, pr_auc:  0.83794
    Validating....
    ROC AUC Score = 0.9130853355532222 / 0.9223324414279188
    Precision/Recall AUC Score = 0.9042857785253009 / 0.5409440320450873
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 26
    ***************
    Training.....
    Epoch:   26, Batch:     0, loss:  0.18931, roc_auc:  0.98810, pr_auc:  0.99112
    Epoch:   26, Batch:   100, loss:  0.45177, roc_auc:  0.87083, pr_auc:  0.72687
    Epoch:   26, Batch:   200, loss:  0.37365, roc_auc:  0.90588, pr_auc:  0.86852
    Epoch:   26, Batch:   300, loss:  0.45717, roc_auc:  0.87843, pr_auc:  0.91034
    Epoch:   26, Batch:   400, loss:  0.33898, roc_auc:  0.95686, pr_auc:  0.95577
    Epoch:   26, Batch:   500, loss:  0.31926, roc_auc:  0.95686, pr_auc:  0.94361
    Epoch:   26, Batch:   600, loss:  0.36524, roc_auc:  0.88750, pr_auc:  0.85429
    Epoch:   26, Batch:   700, loss:  0.35699, roc_auc:  0.93333, pr_auc:  0.93726
    Epoch:   26, Batch:   800, loss:  0.36766, roc_auc:  0.91016, pr_auc:  0.87141
    Epoch:   26, Batch:   900, loss:  0.25736, roc_auc:  1.00000, pr_auc:  1.00000
    Epoch:   26, Batch:  1000, loss:  0.36102, roc_auc:  0.91270, pr_auc:  0.92476
    Epoch:   26, Batch:  1100, loss:  0.43767, roc_auc:  0.91667, pr_auc:  0.87170
    Epoch:   26, Batch:  1200, loss:  0.33101, roc_auc:  0.93750, pr_auc:  0.91660
    Validating....
    ROC AUC Score = 0.913457422555642 / 0.9221754515724365
    Precision/Recall AUC Score = 0.9036187413830998 / 0.5417269964389501
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 27
    ***************
    Training.....
    Epoch:   27, Batch:     0, loss:  0.30449, roc_auc:  0.93651, pr_auc:  0.89749
    Epoch:   27, Batch:   100, loss:  0.37206, roc_auc:  0.92917, pr_auc:  0.93303
    Epoch:   27, Batch:   200, loss:  0.35314, roc_auc:  0.94902, pr_auc:  0.96338
    Epoch:   27, Batch:   300, loss:  0.38886, roc_auc:  0.90283, pr_auc:  0.92757
    Epoch:   27, Batch:   400, loss:  0.45684, roc_auc:  0.90196, pr_auc:  0.89169
    Epoch:   27, Batch:   500, loss:  0.29469, roc_auc:  0.95312, pr_auc:  0.95644
    Epoch:   27, Batch:   600, loss:  0.39501, roc_auc:  0.91364, pr_auc:  0.82303
    Epoch:   27, Batch:   700, loss:  0.21466, roc_auc:  0.98381, pr_auc:  0.98825
    Epoch:   27, Batch:   800, loss:  0.58700, roc_auc:  0.79688, pr_auc:  0.73366
    Epoch:   27, Batch:   900, loss:  0.34434, roc_auc:  0.93750, pr_auc:  0.96410
    Epoch:   27, Batch:  1000, loss:  0.38684, roc_auc:  0.87451, pr_auc:  0.86267
    Epoch:   27, Batch:  1100, loss:  0.46222, roc_auc:  0.94545, pr_auc:  0.90394
    Epoch:   27, Batch:  1200, loss:  0.51184, roc_auc:  0.87500, pr_auc:  0.85688
    Validating....
    ROC AUC Score = 0.913756051335535 / 0.9221550066584854
    Precision/Recall AUC Score = 0.9056091832348404 / 0.5449904873483654
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 28
    ***************
    Training.....
    Epoch:   28, Batch:     0, loss:  0.31186, roc_auc:  0.94444, pr_auc:  0.94423
    Epoch:   28, Batch:   100, loss:  0.31200, roc_auc:  0.97571, pr_auc:  0.98377
    Epoch:   28, Batch:   200, loss:  0.25021, roc_auc:  0.96429, pr_auc:  0.97252
    Epoch:   28, Batch:   300, loss:  0.29441, roc_auc:  0.99219, pr_auc:  0.99217
    Epoch:   28, Batch:   400, loss:  0.28369, roc_auc:  0.94902, pr_auc:  0.96214
    Epoch:   28, Batch:   500, loss:  0.27850, roc_auc:  0.96875, pr_auc:  0.97018
    Epoch:   28, Batch:   600, loss:  0.43561, roc_auc:  0.88021, pr_auc:  0.66917
    Epoch:   28, Batch:   700, loss:  0.28943, roc_auc:  0.95238, pr_auc:  0.91246
    Epoch:   28, Batch:   800, loss:  0.40855, roc_auc:  0.88627, pr_auc:  0.90177
    Epoch:   28, Batch:   900, loss:  0.52321, roc_auc:  0.83529, pr_auc:  0.81930
    Epoch:   28, Batch:  1000, loss:  0.27030, roc_auc:  0.96032, pr_auc:  0.97389
    Epoch:   28, Batch:  1100, loss:  0.37651, roc_auc:  0.89167, pr_auc:  0.83490
    Epoch:   28, Batch:  1200, loss:  0.48373, roc_auc:  0.85417, pr_auc:  0.73346
    Validating....
    ROC AUC Score = 0.9148880131227244 / 0.9230675525425683
    Precision/Recall AUC Score = 0.9063101722659042 / 0.5444143113200468
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 29
    ***************
    Training.....
    Epoch:   29, Batch:     0, loss:  0.24860, roc_auc:  0.97083, pr_auc:  0.95993
    Epoch:   29, Batch:   100, loss:  0.46208, roc_auc:  0.85000, pr_auc:  0.73391
    Epoch:   29, Batch:   200, loss:  0.36186, roc_auc:  0.92549, pr_auc:  0.93706
    Epoch:   29, Batch:   300, loss:  0.38540, roc_auc:  0.90196, pr_auc:  0.91420
    Epoch:   29, Batch:   400, loss:  0.25046, roc_auc:  0.98413, pr_auc:  0.98366
    Epoch:   29, Batch:   500, loss:  0.35044, roc_auc:  0.86147, pr_auc:  0.91121
    Epoch:   29, Batch:   600, loss:  0.28513, roc_auc:  0.96250, pr_auc:  0.97200
    Epoch:   29, Batch:   700, loss:  0.28025, roc_auc:  0.95000, pr_auc:  0.92663
    Epoch:   29, Batch:   800, loss:  0.21443, roc_auc:  0.98785, pr_auc:  0.99202
    Epoch:   29, Batch:   900, loss:  0.44409, roc_auc:  0.86250, pr_auc:  0.78480
    Epoch:   29, Batch:  1000, loss:  0.35556, roc_auc:  0.92578, pr_auc:  0.94058
    Epoch:   29, Batch:  1100, loss:  0.49632, roc_auc:  0.85714, pr_auc:  0.76998
    Epoch:   29, Batch:  1200, loss:  0.42264, roc_auc:  0.89286, pr_auc:  0.86956
    Validating....
    ROC AUC Score = 0.9144136976380527 / 0.9228837459071487
    Precision/Recall AUC Score = 0.9061055041153065 / 0.5466324406201092
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 30
    ***************
    Training.....
    Epoch:   30, Batch:     0, loss:  0.28836, roc_auc:  0.97083, pr_auc:  0.98332
    Epoch:   30, Batch:   100, loss:  0.30981, roc_auc:  0.94510, pr_auc:  0.95634
    Epoch:   30, Batch:   200, loss:  0.31009, roc_auc:  0.93651, pr_auc:  0.94180
    Epoch:   30, Batch:   300, loss:  0.53576, roc_auc:  0.83529, pr_auc:  0.81097
    Epoch:   30, Batch:   400, loss:  0.61999, roc_auc:  0.75833, pr_auc:  0.59078
    Epoch:   30, Batch:   500, loss:  0.41557, roc_auc:  0.88889, pr_auc:  0.90668
    Epoch:   30, Batch:   600, loss:  0.50302, roc_auc:  0.78543, pr_auc:  0.83344
    Epoch:   30, Batch:   700, loss:  0.26598, roc_auc:  0.96863, pr_auc:  0.97302
    Epoch:   30, Batch:   800, loss:  0.31413, roc_auc:  0.94737, pr_auc:  0.95862
    Epoch:   30, Batch:   900, loss:  0.30896, roc_auc:  0.94372, pr_auc:  0.89679
    Epoch:   30, Batch:  1000, loss:  0.35161, roc_auc:  0.91667, pr_auc:  0.78112
    Epoch:   30, Batch:  1100, loss:  0.25626, roc_auc:  0.97266, pr_auc:  0.98055
    Epoch:   30, Batch:  1200, loss:  0.22759, roc_auc:  0.98785, pr_auc:  0.98510
    Validating....
    ROC AUC Score = 0.9141824089702233 / 0.923102451580845
    Precision/Recall AUC Score = 0.9055584166810335 / 0.5407904711979127
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 31
    ***************
    Training.....
    Epoch:   31, Batch:     0, loss:  0.22340, roc_auc:  0.97619, pr_auc:  0.96896
    Epoch:   31, Batch:   100, loss:  0.41318, roc_auc:  0.88235, pr_auc:  0.82255
    Epoch:   31, Batch:   200, loss:  0.41376, roc_auc:  0.88664, pr_auc:  0.90524
    Epoch:   31, Batch:   300, loss:  0.23905, roc_auc:  0.96863, pr_auc:  0.96614
    Epoch:   31, Batch:   400, loss:  0.32379, roc_auc:  0.92549, pr_auc:  0.94005
    Epoch:   31, Batch:   500, loss:  0.31537, roc_auc:  0.92460, pr_auc:  0.93842
    Epoch:   31, Batch:   600, loss:  0.41410, roc_auc:  0.91667, pr_auc:  0.90735
    Epoch:   31, Batch:   700, loss:  0.34381, roc_auc:  0.92549, pr_auc:  0.92912
    Epoch:   31, Batch:   800, loss:  0.25428, roc_auc:  0.96364, pr_auc:  0.98459
    Epoch:   31, Batch:   900, loss:  0.38222, roc_auc:  0.90980, pr_auc:  0.89986
    Epoch:   31, Batch:  1000, loss:  0.33185, roc_auc:  0.94510, pr_auc:  0.94905
    Epoch:   31, Batch:  1100, loss:  0.28891, roc_auc:  0.95703, pr_auc:  0.96006
    Epoch:   31, Batch:  1200, loss:  0.28595, roc_auc:  0.95635, pr_auc:  0.96558
    Validating....
    ROC AUC Score = 0.9131219976514842 / 0.9241570573638884
    Precision/Recall AUC Score = 0.904227017617999 / 0.5469200288497041
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 32
    ***************
    Training.....
    Epoch:   32, Batch:     0, loss:  0.27562, roc_auc:  0.95686, pr_auc:  0.96454
    Epoch:   32, Batch:   100, loss:  0.54330, roc_auc:  0.80859, pr_auc:  0.76567
    Epoch:   32, Batch:   200, loss:  0.27311, roc_auc:  0.96537, pr_auc:  0.95493
    Epoch:   32, Batch:   300, loss:  0.45997, roc_auc:  0.87500, pr_auc:  0.88719
    Epoch:   32, Batch:   400, loss:  0.33433, roc_auc:  0.92549, pr_auc:  0.92209
    Epoch:   32, Batch:   500, loss:  0.41217, roc_auc:  0.88889, pr_auc:  0.91840
    Epoch:   32, Batch:   600, loss:  0.21970, roc_auc:  0.98333, pr_auc:  0.98943
    Epoch:   32, Batch:   700, loss:  0.25616, roc_auc:  0.95686, pr_auc:  0.96380
    Epoch:   32, Batch:   800, loss:  0.29226, roc_auc:  0.96364, pr_auc:  0.98575
    Epoch:   32, Batch:   900, loss:  0.29212, roc_auc:  0.94118, pr_auc:  0.93762
    Epoch:   32, Batch:  1000, loss:  0.40403, roc_auc:  0.92500, pr_auc:  0.91428
    Epoch:   32, Batch:  1100, loss:  0.41790, roc_auc:  0.87449, pr_auc:  0.84031
    Epoch:   32, Batch:  1200, loss:  0.25454, roc_auc:  0.96032, pr_auc:  0.95175
    Validating....
    ROC AUC Score = 0.913786827144465 / 0.9234265447070484
    Precision/Recall AUC Score = 0.9053927913202465 / 0.5425243756717628
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 33
    ***************
    Training.....
    Epoch:   33, Batch:     0, loss:  0.28309, roc_auc:  0.95294, pr_auc:  0.96232
    Epoch:   33, Batch:   100, loss:  0.39480, roc_auc:  0.89804, pr_auc:  0.90935
    Epoch:   33, Batch:   200, loss:  0.41570, roc_auc:  0.88095, pr_auc:  0.84369
    Epoch:   33, Batch:   300, loss:  0.35392, roc_auc:  0.90455, pr_auc:  0.93178
    Epoch:   33, Batch:   400, loss:  0.39446, roc_auc:  0.90476, pr_auc:  0.84940
    Epoch:   33, Batch:   500, loss:  0.26929, roc_auc:  0.97656, pr_auc:  0.97913
    Epoch:   33, Batch:   600, loss:  0.49942, roc_auc:  0.84127, pr_auc:  0.87606
    Epoch:   33, Batch:   700, loss:  0.33945, roc_auc:  0.93254, pr_auc:  0.90087
    Epoch:   33, Batch:   800, loss:  0.46158, roc_auc:  0.83203, pr_auc:  0.81225
    Epoch:   33, Batch:   900, loss:  0.27699, roc_auc:  0.96863, pr_auc:  0.96722
    Epoch:   33, Batch:  1000, loss:  0.28213, roc_auc:  0.96250, pr_auc:  0.97801
    Epoch:   33, Batch:  1100, loss:  0.32890, roc_auc:  0.93359, pr_auc:  0.93558
    Epoch:   33, Batch:  1200, loss:  0.35052, roc_auc:  0.91342, pr_auc:  0.87247
    Validating....
    ROC AUC Score = 0.9135210267366047 / 0.9226076268789836
    Precision/Recall AUC Score = 0.9042172105203317 / 0.5405546003547956
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 34
    ***************
    Training.....
    Epoch:   34, Batch:     0, loss:  0.45576, roc_auc:  0.87698, pr_auc:  0.72950
    Epoch:   34, Batch:   100, loss:  0.30867, roc_auc:  0.95547, pr_auc:  0.96681
    Epoch:   34, Batch:   200, loss:  0.28504, roc_auc:  0.97255, pr_auc:  0.97068
    Epoch:   34, Batch:   300, loss:  0.32655, roc_auc:  0.93725, pr_auc:  0.92585
    Epoch:   34, Batch:   400, loss:  0.25747, roc_auc:  0.97083, pr_auc:  0.97971
    Epoch:   34, Batch:   500, loss:  0.37984, roc_auc:  0.90476, pr_auc:  0.92781
    Epoch:   34, Batch:   600, loss:  0.25743, roc_auc:  0.95686, pr_auc:  0.94868
    Epoch:   34, Batch:   700, loss:  0.32047, roc_auc:  0.94048, pr_auc:  0.94182
    Epoch:   34, Batch:   800, loss:  0.40489, roc_auc:  0.91406, pr_auc:  0.92171
    Epoch:   34, Batch:   900, loss:  0.49000, roc_auc:  0.85714, pr_auc:  0.81985
    Epoch:   34, Batch:  1000, loss:  0.47603, roc_auc:  0.81818, pr_auc:  0.71484
    Epoch:   34, Batch:  1100, loss:  0.14408, roc_auc:  1.00000, pr_auc:  1.00000
    Epoch:   34, Batch:  1200, loss:  0.46649, roc_auc:  0.83922, pr_auc:  0.80915
    Validating....
    ROC AUC Score = 0.9147808086560071 / 0.9232091191890187
    Precision/Recall AUC Score = 0.9062632869938363 / 0.5427760206043718
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 35
    ***************
    Training.....
    Epoch:   35, Batch:     0, loss:  0.17190, roc_auc:  0.99609, pr_auc:  0.99621
    Epoch:   35, Batch:   100, loss:  0.40651, roc_auc:  0.88889, pr_auc:  0.92073
    Epoch:   35, Batch:   200, loss:  0.44735, roc_auc:  0.86111, pr_auc:  0.88100
    Epoch:   35, Batch:   300, loss:  0.43457, roc_auc:  0.89069, pr_auc:  0.91871
    Epoch:   35, Batch:   400, loss:  0.30412, roc_auc:  0.95000, pr_auc:  0.96736
    Epoch:   35, Batch:   500, loss:  0.29997, roc_auc:  0.94510, pr_auc:  0.93340
    Epoch:   35, Batch:   600, loss:  0.34284, roc_auc:  0.92713, pr_auc:  0.94972
    Epoch:   35, Batch:   700, loss:  0.28664, roc_auc:  0.96078, pr_auc:  0.95978
    Epoch:   35, Batch:   800, loss:  0.23776, roc_auc:  0.97571, pr_auc:  0.96564
    Epoch:   35, Batch:   900, loss:  0.35267, roc_auc:  0.92208, pr_auc:  0.85318
    Epoch:   35, Batch:  1000, loss:  0.19216, roc_auc:  1.00000, pr_auc:  1.00000
    Epoch:   35, Batch:  1100, loss:  0.44658, roc_auc:  0.84167, pr_auc:  0.86813
    Epoch:   35, Batch:  1200, loss:  0.42212, roc_auc:  0.86640, pr_auc:  0.89144
    Validating....
    ROC AUC Score = 0.9149801222764028 / 0.923019861210816
    Precision/Recall AUC Score = 0.9067256878492524 / 0.5439014829045359
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 36
    ***************
    Training.....
    Epoch:   36, Batch:     0, loss:  0.22548, roc_auc:  0.97255, pr_auc:  0.96160
    Epoch:   36, Batch:   100, loss:  0.28003, roc_auc:  0.94902, pr_auc:  0.93014
    Epoch:   36, Batch:   200, loss:  0.28865, roc_auc:  0.98431, pr_auc:  0.98396
    Epoch:   36, Batch:   300, loss:  0.31863, roc_auc:  0.93750, pr_auc:  0.90022
    Epoch:   36, Batch:   400, loss:  0.29708, roc_auc:  0.93750, pr_auc:  0.88468
    Epoch:   36, Batch:   500, loss:  0.39855, roc_auc:  0.90873, pr_auc:  0.87515
    Epoch:   36, Batch:   600, loss:  0.29768, roc_auc:  0.96032, pr_auc:  0.97344
    Epoch:   36, Batch:   700, loss:  0.50938, roc_auc:  0.82996, pr_auc:  0.70126
    Epoch:   36, Batch:   800, loss:  0.48640, roc_auc:  0.82996, pr_auc:  0.80023
    Epoch:   36, Batch:   900, loss:  0.39128, roc_auc:  0.90079, pr_auc:  0.89568
    Epoch:   36, Batch:  1000, loss:  0.27811, roc_auc:  0.96094, pr_auc:  0.95776
    Epoch:   36, Batch:  1100, loss:  0.31760, roc_auc:  0.91903, pr_auc:  0.93790
    Epoch:   36, Batch:  1200, loss:  0.39015, roc_auc:  0.91787, pr_auc:  0.85486
    Validating....
    ROC AUC Score = 0.915003664422057 / 0.9233079296477767
    Precision/Recall AUC Score = 0.9066989529016909 / 0.5450238279940698
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 37
    ***************
    Training.....
    Epoch:   37, Batch:     0, loss:  0.39917, roc_auc:  0.91016, pr_auc:  0.92318
    Epoch:   37, Batch:   100, loss:  0.41324, roc_auc:  0.88889, pr_auc:  0.82381
    Epoch:   37, Batch:   200, loss:  0.30353, roc_auc:  0.95142, pr_auc:  0.92699
    Epoch:   37, Batch:   300, loss:  0.37576, roc_auc:  0.96078, pr_auc:  0.96446
    Epoch:   37, Batch:   400, loss:  0.60746, roc_auc:  0.78968, pr_auc:  0.77659
    Epoch:   37, Batch:   500, loss:  0.33865, roc_auc:  0.92969, pr_auc:  0.93164
    Epoch:   37, Batch:   600, loss:  0.30326, roc_auc:  0.96032, pr_auc:  0.97185
    Epoch:   37, Batch:   700, loss:  0.30263, roc_auc:  0.95686, pr_auc:  0.96638
    Epoch:   37, Batch:   800, loss:  0.50762, roc_auc:  0.85938, pr_auc:  0.84297
    Epoch:   37, Batch:   900, loss:  0.48333, roc_auc:  0.84167, pr_auc:  0.76754
    Epoch:   37, Batch:  1000, loss:  0.33283, roc_auc:  0.94118, pr_auc:  0.93537
    Epoch:   37, Batch:  1100, loss:  0.29283, roc_auc:  0.93651, pr_auc:  0.92584
    Epoch:   37, Batch:  1200, loss:  0.19569, roc_auc:  0.99603, pr_auc:  0.99699
    Validating....
    ROC AUC Score = 0.9163930659837727 / 0.9233621208495579
    Precision/Recall AUC Score = 0.9086343614473769 / 0.5448334115544795
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 38
    ***************
    Training.....
    Epoch:   38, Batch:     0, loss:  0.25793, roc_auc:  0.97500, pr_auc:  0.98533
    Epoch:   38, Batch:   100, loss:  0.32755, roc_auc:  0.95294, pr_auc:  0.95458
    Epoch:   38, Batch:   200, loss:  0.29684, roc_auc:  0.95238, pr_auc:  0.94531
    Epoch:   38, Batch:   300, loss:  0.21714, roc_auc:  0.99219, pr_auc:  0.99217
    Epoch:   38, Batch:   400, loss:  0.45545, roc_auc:  0.86508, pr_auc:  0.88575
    Epoch:   38, Batch:   500, loss:  0.31040, roc_auc:  0.95238, pr_auc:  0.96646
    Epoch:   38, Batch:   600, loss:  0.54653, roc_auc:  0.84314, pr_auc:  0.87442
    Epoch:   38, Batch:   700, loss:  0.44146, roc_auc:  0.86508, pr_auc:  0.86573
    Epoch:   38, Batch:   800, loss:  0.50813, roc_auc:  0.84766, pr_auc:  0.83902
    Epoch:   38, Batch:   900, loss:  0.27879, roc_auc:  0.94737, pr_auc:  0.96229
    Epoch:   38, Batch:  1000, loss:  0.48133, roc_auc:  0.85547, pr_auc:  0.89590
    Epoch:   38, Batch:  1100, loss:  0.27722, roc_auc:  0.94922, pr_auc:  0.94520
    Epoch:   38, Batch:  1200, loss:  0.28796, roc_auc:  0.95951, pr_auc:  0.95129
    Validating....
    ROC AUC Score = 0.9151109964179934 / 0.922631918611818
    Precision/Recall AUC Score = 0.9056730840323759 / 0.5417519766688644
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 39
    ***************
    Training.....
    Epoch:   39, Batch:     0, loss:  0.28065, roc_auc:  0.94841, pr_auc:  0.96382
    Epoch:   39, Batch:   100, loss:  0.35959, roc_auc:  0.93359, pr_auc:  0.92895
    Epoch:   39, Batch:   200, loss:  0.54869, roc_auc:  0.78175, pr_auc:  0.80768
    Epoch:   39, Batch:   300, loss:  0.30072, roc_auc:  0.94048, pr_auc:  0.95541
    Epoch:   39, Batch:   400, loss:  0.41243, roc_auc:  0.91373, pr_auc:  0.82987
    Epoch:   39, Batch:   500, loss:  0.38654, roc_auc:  0.91406, pr_auc:  0.93095
    Epoch:   39, Batch:   600, loss:  0.43149, roc_auc:  0.89020, pr_auc:  0.81735
    Epoch:   39, Batch:   700, loss:  0.47797, roc_auc:  0.84921, pr_auc:  0.86345
    Epoch:   39, Batch:   800, loss:  0.26851, roc_auc:  0.95686, pr_auc:  0.96121
    Epoch:   39, Batch:   900, loss:  0.24004, roc_auc:  0.99608, pr_auc:  0.99569
    Epoch:   39, Batch:  1000, loss:  0.41216, roc_auc:  0.86640, pr_auc:  0.88574
    Epoch:   39, Batch:  1100, loss:  0.40550, roc_auc:  0.90476, pr_auc:  0.86697
    Epoch:   39, Batch:  1200, loss:  0.39711, roc_auc:  0.88492, pr_auc:  0.88785
    Validating....
    ROC AUC Score = 0.9133215152122023 / 0.9230462886376865
    Precision/Recall AUC Score = 0.9043455088394312 / 0.5445526492990672
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 40
    ***************
    Training.....
    Epoch:   40, Batch:     0, loss:  0.36508, roc_auc:  0.91373, pr_auc:  0.90046
    Epoch:   40, Batch:   100, loss:  0.37740, roc_auc:  0.89474, pr_auc:  0.87363
    Epoch:   40, Batch:   200, loss:  0.40523, roc_auc:  0.89020, pr_auc:  0.88702
    Epoch:   40, Batch:   300, loss:  0.45818, roc_auc:  0.85938, pr_auc:  0.85003
    Epoch:   40, Batch:   400, loss:  0.42836, roc_auc:  0.91498, pr_auc:  0.92878
    Epoch:   40, Batch:   500, loss:  0.41711, roc_auc:  0.89069, pr_auc:  0.84863
    Epoch:   40, Batch:   600, loss:  0.42718, roc_auc:  0.88492, pr_auc:  0.85506
    Epoch:   40, Batch:   700, loss:  0.43807, roc_auc:  0.83730, pr_auc:  0.75680
    Epoch:   40, Batch:   800, loss:  0.25200, roc_auc:  0.96761, pr_auc:  0.97657
    Epoch:   40, Batch:   900, loss:  0.39359, roc_auc:  0.91270, pr_auc:  0.90501
    Epoch:   40, Batch:  1000, loss:  0.18287, roc_auc:  1.00000, pr_auc:  1.00000
    Epoch:   40, Batch:  1100, loss:  0.43420, roc_auc:  0.86905, pr_auc:  0.81753
    Epoch:   40, Batch:  1200, loss:  0.19773, roc_auc:  0.98431, pr_auc:  0.98738
    Validating....
    ROC AUC Score = 0.9153496785504794 / 0.9230516947151989
    Precision/Recall AUC Score = 0.9080045891553858 / 0.5457251769068262
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 41
    ***************
    Training.....
    Epoch:   41, Batch:     0, loss:  0.40423, roc_auc:  0.90588, pr_auc:  0.91318
    Epoch:   41, Batch:   100, loss:  0.49453, roc_auc:  0.88235, pr_auc:  0.89856
    Epoch:   41, Batch:   200, loss:  0.33555, roc_auc:  0.92549, pr_auc:  0.94940
    Epoch:   41, Batch:   300, loss:  0.37808, roc_auc:  0.88750, pr_auc:  0.88137
    Epoch:   41, Batch:   400, loss:  0.30561, roc_auc:  0.97585, pr_auc:  0.99093
    Epoch:   41, Batch:   500, loss:  0.36009, roc_auc:  0.92308, pr_auc:  0.95393
    Epoch:   41, Batch:   600, loss:  0.32729, roc_auc:  0.91765, pr_auc:  0.87317
    Epoch:   41, Batch:   700, loss:  0.45738, roc_auc:  0.88672, pr_auc:  0.90627
    Epoch:   41, Batch:   800, loss:  0.41058, roc_auc:  0.89804, pr_auc:  0.88550
    Epoch:   41, Batch:   900, loss:  0.35578, roc_auc:  0.91406, pr_auc:  0.92284
    Epoch:   41, Batch:  1000, loss:  0.53675, roc_auc:  0.84375, pr_auc:  0.75353
    Epoch:   41, Batch:  1100, loss:  0.25986, roc_auc:  0.98785, pr_auc:  0.98321
    Epoch:   41, Batch:  1200, loss:  0.38352, roc_auc:  0.91016, pr_auc:  0.89061
    Validating....
    ROC AUC Score = 0.9184844440252167 / 0.9231995932328741
    Precision/Recall AUC Score = 0.9099696748267863 / 0.544766347283947
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 42
    ***************
    Training.....
    Epoch:   42, Batch:     0, loss:  0.47324, roc_auc:  0.87302, pr_auc:  0.89416
    Epoch:   42, Batch:   100, loss:  0.27156, roc_auc:  0.97647, pr_auc:  0.97643
    Epoch:   42, Batch:   200, loss:  0.31835, roc_auc:  0.94118, pr_auc:  0.95694
    Epoch:   42, Batch:   300, loss:  0.48018, roc_auc:  0.85714, pr_auc:  0.91634
    Epoch:   42, Batch:   400, loss:  0.34297, roc_auc:  0.90476, pr_auc:  0.85401
    Epoch:   42, Batch:   500, loss:  0.33127, roc_auc:  0.92641, pr_auc:  0.85521
    Epoch:   42, Batch:   600, loss:  0.44213, roc_auc:  0.86250, pr_auc:  0.76549
    Epoch:   42, Batch:   700, loss:  0.36157, roc_auc:  0.91797, pr_auc:  0.92257
    Epoch:   42, Batch:   800, loss:  0.24928, roc_auc:  0.96875, pr_auc:  0.97064
    Epoch:   42, Batch:   900, loss:  0.35875, roc_auc:  0.92917, pr_auc:  0.90081
    Epoch:   42, Batch:  1000, loss:  0.30939, roc_auc:  0.92157, pr_auc:  0.91930
    Epoch:   42, Batch:  1100, loss:  0.30958, roc_auc:  0.94048, pr_auc:  0.92730
    Epoch:   42, Batch:  1200, loss:  0.36163, roc_auc:  0.91903, pr_auc:  0.88308
    Validating....
    ROC AUC Score = 0.9176765527304627 / 0.922785362964228
    Precision/Recall AUC Score = 0.9091125967585802 / 0.5429725763770256
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 43
    ***************
    Training.....
    Epoch:   43, Batch:     0, loss:  0.21964, roc_auc:  0.97255, pr_auc:  0.96160
    Epoch:   43, Batch:   100, loss:  0.35305, roc_auc:  0.92208, pr_auc:  0.87700
    Epoch:   43, Batch:   200, loss:  0.27659, roc_auc:  0.95833, pr_auc:  0.95613
    Epoch:   43, Batch:   300, loss:  0.23100, roc_auc:  0.97255, pr_auc:  0.96966
    Epoch:   43, Batch:   400, loss:  0.49733, roc_auc:  0.88235, pr_auc:  0.83707
    Epoch:   43, Batch:   500, loss:  0.27453, roc_auc:  0.94531, pr_auc:  0.93839
    Epoch:   43, Batch:   600, loss:  0.47163, roc_auc:  0.86667, pr_auc:  0.88840
    Epoch:   43, Batch:   700, loss:  0.30282, roc_auc:  0.95833, pr_auc:  0.94839
    Epoch:   43, Batch:   800, loss:  0.45742, roc_auc:  0.85882, pr_auc:  0.88662
    Epoch:   43, Batch:   900, loss:  0.42943, roc_auc:  0.88095, pr_auc:  0.90928
    Epoch:   43, Batch:  1000, loss:  0.32294, roc_auc:  0.93927, pr_auc:  0.95705
    Epoch:   43, Batch:  1100, loss:  0.21940, roc_auc:  0.98431, pr_auc:  0.98738
    Epoch:   43, Batch:  1200, loss:  0.18757, roc_auc:  0.99216, pr_auc:  0.99109
    Validating....
    ROC AUC Score = 0.9157669729457146 / 0.9233588577809684
    Precision/Recall AUC Score = 0.9071913872896366 / 0.54675864508515
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 44
    ***************
    Training.....
    Epoch:   44, Batch:     0, loss:  0.34244, roc_auc:  0.92157, pr_auc:  0.93387
    Epoch:   44, Batch:   100, loss:  0.43263, roc_auc:  0.87449, pr_auc:  0.73787
    Epoch:   44, Batch:   200, loss:  0.21294, roc_auc:  0.99219, pr_auc:  0.99285
    Epoch:   44, Batch:   300, loss:  0.46149, roc_auc:  0.85830, pr_auc:  0.87228
    Epoch:   44, Batch:   400, loss:  0.23485, roc_auc:  0.98039, pr_auc:  0.98097
    Epoch:   44, Batch:   500, loss:  0.24738, roc_auc:  0.98828, pr_auc:  0.98986
    Epoch:   44, Batch:   600, loss:  0.36279, roc_auc:  0.91765, pr_auc:  0.93866
    Epoch:   44, Batch:   700, loss:  0.41786, roc_auc:  0.88235, pr_auc:  0.90922
    Epoch:   44, Batch:   800, loss:  0.46328, roc_auc:  0.85425, pr_auc:  0.85476
    Epoch:   44, Batch:   900, loss:  0.51603, roc_auc:  0.82745, pr_auc:  0.83099
    Epoch:   44, Batch:  1000, loss:  0.22829, roc_auc:  0.97917, pr_auc:  0.98979
    Epoch:   44, Batch:  1100, loss:  0.49993, roc_auc:  0.82996, pr_auc:  0.80931
    Epoch:   44, Batch:  1200, loss:  0.18591, roc_auc:  0.99603, pr_auc:  0.99699
    Validating....
    ROC AUC Score = 0.9158899446161228 / 0.9233173188037349
    Precision/Recall AUC Score = 0.9071716859142197 / 0.5458011986485168
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 45
    ***************
    Training.....
    Epoch:   45, Batch:     0, loss:  0.31533, roc_auc:  0.95142, pr_auc:  0.95549
    Epoch:   45, Batch:   100, loss:  0.42841, roc_auc:  0.88281, pr_auc:  0.87860
    Epoch:   45, Batch:   200, loss:  0.42966, roc_auc:  0.87451, pr_auc:  0.87606
    Epoch:   45, Batch:   300, loss:  0.50438, roc_auc:  0.82353, pr_auc:  0.78374
    Epoch:   45, Batch:   400, loss:  0.34145, roc_auc:  0.91667, pr_auc:  0.82681
    Epoch:   45, Batch:   500, loss:  0.42897, roc_auc:  0.89062, pr_auc:  0.88118
    Epoch:   45, Batch:   600, loss:  0.49737, roc_auc:  0.89545, pr_auc:  0.81407
    Epoch:   45, Batch:   700, loss:  0.43360, roc_auc:  0.89069, pr_auc:  0.93409
    Epoch:   45, Batch:   800, loss:  0.42862, roc_auc:  0.87500, pr_auc:  0.92425
    Epoch:   45, Batch:   900, loss:  0.36132, roc_auc:  0.95142, pr_auc:  0.97090
    Epoch:   45, Batch:  1000, loss:  0.27887, roc_auc:  0.94902, pr_auc:  0.93608
    Epoch:   45, Batch:  1100, loss:  0.27931, roc_auc:  0.95142, pr_auc:  0.93274
    Epoch:   45, Batch:  1200, loss:  0.48735, roc_auc:  0.82937, pr_auc:  0.86083
    Validating....
    ROC AUC Score = 0.9148731336302369 / 0.9235499323900628
    Precision/Recall AUC Score = 0.9065526904512385 / 0.5466409565716895
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 46
    ***************
    Training.....
    Epoch:   46, Batch:     0, loss:  0.41210, roc_auc:  0.89020, pr_auc:  0.91351
    Epoch:   46, Batch:   100, loss:  0.31765, roc_auc:  0.92857, pr_auc:  0.91349
    Epoch:   46, Batch:   200, loss:  0.35160, roc_auc:  0.90980, pr_auc:  0.91202
    Epoch:   46, Batch:   300, loss:  0.30361, roc_auc:  0.93522, pr_auc:  0.95627
    Epoch:   46, Batch:   400, loss:  0.54334, roc_auc:  0.83550, pr_auc:  0.69292
    Epoch:   46, Batch:   500, loss:  0.55925, roc_auc:  0.83806, pr_auc:  0.72646
    Epoch:   46, Batch:   600, loss:  0.26107, roc_auc:  0.96471, pr_auc:  0.96400
    Epoch:   46, Batch:   700, loss:  0.31295, roc_auc:  0.92157, pr_auc:  0.91509
    Epoch:   46, Batch:   800, loss:  0.39484, roc_auc:  0.91775, pr_auc:  0.86696
    Epoch:   46, Batch:   900, loss:  0.35091, roc_auc:  0.93117, pr_auc:  0.95657
    Epoch:   46, Batch:  1000, loss:  0.29296, roc_auc:  0.95238, pr_auc:  0.95973
    Epoch:   46, Batch:  1100, loss:  0.39970, roc_auc:  0.90196, pr_auc:  0.86244
    Epoch:   46, Batch:  1200, loss:  0.28231, roc_auc:  0.94902, pr_auc:  0.95077
    Validating....
    ROC AUC Score = 0.916113518507504 / 0.9233892751136451
    Precision/Recall AUC Score = 0.9078635824944672 / 0.5456020806944183
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 47
    ***************
    Training.....
    Epoch:   47, Batch:     0, loss:  0.26910, roc_auc:  0.96863, pr_auc:  0.97302
    Epoch:   47, Batch:   100, loss:  0.57226, roc_auc:  0.80000, pr_auc:  0.82991
    Epoch:   47, Batch:   200, loss:  0.27463, roc_auc:  0.94118, pr_auc:  0.93837
    Epoch:   47, Batch:   300, loss:  0.42171, roc_auc:  0.89683, pr_auc:  0.85849
    Epoch:   47, Batch:   400, loss:  0.30270, roc_auc:  0.96484, pr_auc:  0.96456
    Epoch:   47, Batch:   500, loss:  0.39038, roc_auc:  0.91818, pr_auc:  0.84395
    Epoch:   47, Batch:   600, loss:  0.40023, roc_auc:  0.89804, pr_auc:  0.94169
    Epoch:   47, Batch:   700, loss:  0.31809, roc_auc:  0.95909, pr_auc:  0.82756
    Epoch:   47, Batch:   800, loss:  0.36135, roc_auc:  0.89583, pr_auc:  0.87874
    Epoch:   47, Batch:   900, loss:  0.53209, roc_auc:  0.81349, pr_auc:  0.87010
    Epoch:   47, Batch:  1000, loss:  0.33310, roc_auc:  0.92857, pr_auc:  0.95244
    Epoch:   47, Batch:  1100, loss:  0.34028, roc_auc:  0.92460, pr_auc:  0.92046
    Epoch:   47, Batch:  1200, loss:  0.51071, roc_auc:  0.83203, pr_auc:  0.78248
    Validating....
    ROC AUC Score = 0.9149511647292892 / 0.9235191759294874
    Precision/Recall AUC Score = 0.905601416651975 / 0.5467374271460061
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 48
    ***************
    Training.....
    Epoch:   48, Batch:     0, loss:  0.35962, roc_auc:  0.89683, pr_auc:  0.89943
    Epoch:   48, Batch:   100, loss:  0.37002, roc_auc:  0.92063, pr_auc:  0.94392
    Epoch:   48, Batch:   200, loss:  0.34648, roc_auc:  0.92578, pr_auc:  0.93957
    Epoch:   48, Batch:   300, loss:  0.40211, roc_auc:  0.89683, pr_auc:  0.91368
    Epoch:   48, Batch:   400, loss:  0.51255, roc_auc:  0.83529, pr_auc:  0.83495
    Epoch:   48, Batch:   500, loss:  0.34106, roc_auc:  0.94141, pr_auc:  0.94931
    Epoch:   48, Batch:   600, loss:  0.15152, roc_auc:  1.00000, pr_auc:  1.00000
    Epoch:   48, Batch:   700, loss:  0.33449, roc_auc:  0.93506, pr_auc:  0.88227
    Epoch:   48, Batch:   800, loss:  0.54557, roc_auc:  0.82186, pr_auc:  0.69324
    Epoch:   48, Batch:   900, loss:  0.23900, roc_auc:  0.96863, pr_auc:  0.97199
    Epoch:   48, Batch:  1000, loss:  0.19127, roc_auc:  1.00000, pr_auc:  1.00000
    Epoch:   48, Batch:  1100, loss:  0.65825, roc_auc:  0.75294, pr_auc:  0.67114
    Epoch:   48, Batch:  1200, loss:  0.41734, roc_auc:  0.89453, pr_auc:  0.91682
    Validating....
    ROC AUC Score = 0.9161602406892106 / 0.9235213523851344
    Precision/Recall AUC Score = 0.9070121221753986 / 0.5465805880189605
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 49
    ***************
    Training.....
    Epoch:   49, Batch:     0, loss:  0.30382, roc_auc:  0.96135, pr_auc:  0.98402
    Epoch:   49, Batch:   100, loss:  0.18757, roc_auc:  0.99595, pr_auc:  0.99429
    Epoch:   49, Batch:   200, loss:  0.27796, roc_auc:  0.96032, pr_auc:  0.95502
    Epoch:   49, Batch:   300, loss:  0.37448, roc_auc:  0.91093, pr_auc:  0.89347
    Epoch:   49, Batch:   400, loss:  0.37446, roc_auc:  0.91818, pr_auc:  0.81681
    Epoch:   49, Batch:   500, loss:  0.31997, roc_auc:  0.93750, pr_auc:  0.91586
    Epoch:   49, Batch:   600, loss:  0.18056, roc_auc:  0.99583, pr_auc:  0.99756
    Epoch:   49, Batch:   700, loss:  0.28823, roc_auc:  0.94841, pr_auc:  0.94131
    Epoch:   49, Batch:   800, loss:  0.46224, roc_auc:  0.86275, pr_auc:  0.88483
    Epoch:   49, Batch:   900, loss:  0.32799, roc_auc:  0.95686, pr_auc:  0.96402
    Epoch:   49, Batch:  1000, loss:  0.39253, roc_auc:  0.90588, pr_auc:  0.92953
    Epoch:   49, Batch:  1100, loss:  0.38434, roc_auc:  0.90283, pr_auc:  0.92413
    Epoch:   49, Batch:  1200, loss:  0.23362, roc_auc:  0.97571, pr_auc:  0.98360
    Validating....
    ROC AUC Score = 0.9167010019698744 / 0.9232542424193279
    Precision/Recall AUC Score = 0.9084521907027379 / 0.5460388731313157
    Epoch completed in 1 minutes, 23 seconds
    ***************
    EPOCH: 50
    ***************
    Training.....
    Epoch:   50, Batch:     0, loss:  0.41816, roc_auc:  0.89879, pr_auc:  0.93912
    Epoch:   50, Batch:   100, loss:  0.25339, roc_auc:  0.95000, pr_auc:  0.97365
    Epoch:   50, Batch:   200, loss:  0.54151, roc_auc:  0.80000, pr_auc:  0.68370
    Epoch:   50, Batch:   300, loss:  0.37963, roc_auc:  0.91093, pr_auc:  0.87017
    Epoch:   50, Batch:   400, loss:  0.25648, roc_auc:  0.97255, pr_auc:  0.97515
    Epoch:   50, Batch:   500, loss:  0.25735, roc_auc:  0.96667, pr_auc:  0.97944
    Epoch:   50, Batch:   600, loss:  0.66222, roc_auc:  0.77778, pr_auc:  0.74615
    Epoch:   50, Batch:   700, loss:  0.52324, roc_auc:  0.85098, pr_auc:  0.80646
    Epoch:   50, Batch:   800, loss:  0.33803, roc_auc:  0.91775, pr_auc:  0.95220
    Epoch:   50, Batch:   900, loss:  0.29116, roc_auc:  0.94902, pr_auc:  0.94626
    Epoch:   50, Batch:  1000, loss:  0.34607, roc_auc:  0.93333, pr_auc:  0.95989
    Epoch:   50, Batch:  1100, loss:  0.29543, roc_auc:  0.96667, pr_auc:  0.98166
    Epoch:   50, Batch:  1200, loss:  0.32796, roc_auc:  0.92083, pr_auc:  0.85993
    Validating....
    ROC AUC Score = 0.9141030697962029 / 0.9240210547654226
    Precision/Recall AUC Score = 0.9055538847679993 / 0.5471351727911192
    Epoch completed in 1 minutes, 23 seconds
    Training completed in 70 minutes, 36 seconds
    

# Test
**Attention**: First lines in this cell serve as a means to select which saved version to run test with. Pleas, make sure you're testing the model you intend to.


```python
#from_file = "./models/bert-base-multilingual-cased_1603075225.pt"
from_file = Path(WORKING_DIR / CONFIG["predict"]["model"]["from_file"]).resolve()

model, tokenizer = get_model_artifacts(CONFIG["train"]["model"], from_folder=from_folder)
model.load_state_dict(torch.load(from_file))
model.eval()

print("CUDA available:", torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Loading model to '{device}' device...")
model = model.to(device)
print("Model loaded successfully!")

test_data_loader = get_data_loader(df=df_test, tokenizer=tokenizer, config=CONFIG["train"])

# No gradient evaluation needed for validation
with torch.no_grad():
            
    print('Testing....')
    
    # Evaluation mode
    model.eval()
    
    ev_targets = []
    ev_outputs = []
    
    # Iterate through batches
    for bi, d in enumerate(test_data_loader):

        # BERT-friendly input data
        ids, token_type_ids, mask, targets = get_input_tensors(d, device)
        
        # Forward step
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        # Retrieve and store batch-level results back from GPU/TPU
        targets_np = targets.cpu().detach().numpy().tolist()
        outputs_np = torch.sigmoid(outputs).cpu().detach().numpy().tolist() # Sigmoid needed (defined output activation function was linear to take advantage of BCELogitsLoss)
        
        ev_targets.extend(targets_np)
        ev_outputs.extend(outputs_np)

    # Test aggregated metrics calculation
    try:
        ev_targets = np.array(ev_targets) >= 0.5
        ev_roc_auc_sc = roc_auc_score(ev_targets, ev_outputs)
        ev_precision, ev_recall, _ = precision_recall_curve(ev_targets, ev_outputs)
        ev_pr_auc_sc = auc(ev_recall, ev_precision)
    except ValueError as e:
        print(ev_targets.shape)
        print(ev_outputs.shape)
        raise e
        
ev_targets = np.array(ev_targets).reshape(1, -1)[0] >= 0.5
ev_outputs = np.array(ev_outputs).reshape(1, -1)[0]

# Aggregated Test metrics
print(f"ROC AUC Score = {ev_roc_auc_sc}")
print(f"Precision/Recall AUC Score = {ev_pr_auc_sc}")
```

    CUDA available: True
    Loading model to 'cuda' device...
    Model loaded successfully!
    Testing....
    ROC AUC Score = 0.9192519541020352
    Precision/Recall AUC Score = 0.5530386874501412
    

# Preparing the classifier (optimal cutoff point)

Now I've validated my best trained model performs good also for test data, it's time to find the optimal cutoff point, for which the criteria is dictated by the use case: `Factual claim classification`.

Arguably enough, this use case is expected to deliver results that, unless we're planning to apply further heuristical methods over those, will necessarily need to be checked by a human agent, who will seamlessly be able to perform a second check over those at the same time he/she is dedicated to check the claim itself.

This consideration alone **favors minimizing False Negatives over minimizing False Positives**. Equivalently, it favors **recall over precision** as a measure to optimize when choosing our cutoff point.

However, as **human labor is a limited asset** and the ratio of False Positives burdens that same exact resource, it's also wise to **impose a limitation in precisision** that avoids an excessive additional load of work for the human agent. For that reason, we'll try 2 different scenarios with regards to the maximum accepted amount of work overload generated:

- **Scenario 1**: In this scenario we find unacceptable that the human agent finds a ratio of misclassified non-factual claims of 33%. This is equivalent to imposing a **minimum precision of 2/3**.
- **Scenario 2**: Here we relax the condition and impose a higher maximum ratio of misclassified non-factual claims of 66%. This is equivalent to imposing a **minimum precision of 1/3**.

Attending to the pictures below, representing both the ROC curve (misleading with class-imbalanced problems like this, but still important) and Precision/Recall curve (better fit for choosing the optimal cutoff point), we may see how we can easily find the optimal empirical cutoff points for both scenarios.


```python
scenarios = {
    "Scenario 1": dict(min_precision=2./3),
    "Scenario 2": dict(min_precision=1./3)
}
```


```python
from sklearn.metrics import roc_curve, precision_recall_curve

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def change_of_basis(value, ax1, ax2, axis=1):
    """
    Performs a change of basis `value` from `ax1` to `ax2`.
    
    Args:
      value (float): Value to be changed basis.
      ax1 (matplotlib.axes.Axes.axis): Matplotlib axis from which to change basis
      ax2 (matplotlib.axes.Axes.axis): Matplotlib axis to which to change basis
      axis (optional, int): Whether the basis change must be made over xaxis (0) or yaxis (1).
    
    Returns:
      (float): Result from basis change of `value`.
    """
    
    # Retrieve dynamic ranges for each axis
    ax1_range = ax1.get_ylim() if axis == 1 else ax1.get_xlim()
    ax2_range = ax2.get_ylim() if axis == 1 else ax2.get_xlim()
    
    # Return basis-changed value
    return (value - ax1_range[0]) / (ax1_range[1] - ax1_range[0]) * (ax2_range[1] - ax2_range[0]) + ax2_range[0]


def calculate_general_keypoints(ev_targets, ev_outputs):
    """
    Calculates all non-scenario-specific keypoints to be represented on optimal cutoff point
    search figures.
    
    Args:
      ev_targets (np.array): Target values for the classification at hand.
      ev_outputs (np.array): Values returned by our current model.
    
    Returns:
      fpr (np.array): Array holding xaxis datapoints (false positive rates) for ROC curve.
      tpr (np.array): Array holding yaxis datapoints (true positive rates) for ROC curve.
      thresholds_roc (np.array): Array holding yaxis datapoints for threshold values of the ROC curve.
      lr_precision (np.array): Array holding yaxis datapoints (precision) for Precision/Recall curve.
      lr_recall (np.array): Array holding xaxis datapoints (recall) for Precision/Recall curve.
      thresholds_pr (np.array): Array holding yaxis datapoints for threshold values of the Precision/Recall curve.
      no_skill (float): No-skill precision value for Precision/Recall curve.
    """

    # Calculate both curves
    fpr, tpr, thresholds_roc = roc_curve(ev_targets, ev_outputs)
    lr_precision, lr_recall, thresholds_pr = precision_recall_curve(ev_targets, ev_outputs)

    # Sort indices for Precision / Recall curve resulting arrays
    sorted_pr_ixs = np.argsort(lr_recall)
    lr_recall = lr_recall[sorted_pr_ixs]
    lr_precision = lr_precision[sorted_pr_ixs]
    thresholds_pr = np.concatenate([[thresholds_pr.min()], thresholds_pr])[sorted_pr_ixs]

    # No-skill precision
    no_skill = len(ev_targets[ev_targets]) / len(ev_targets)
    
    return fpr, tpr, thresholds_roc, lr_precision, lr_recall, thresholds_pr, no_skill


def calculate_scenario_keypoints(lr_precision, no_skill, thresholds_pr, thresholds_roc, min_precision):
    """
    Calculates all scenario-specific keypoints to be represented on optimal cutoff point
    search figures.
    
    Args:
      lr_precision (np.array): Array holding yaxis datapoints (precision) for Precision/Recall curve.
      no_skill (np.array): No-skill precision value for Precision/Recall curve.
      thresholds_pr (np.array): Array holding yaxis datapoints for threshold values of the Precision/Recall curve.
      thresholds_roc (np.array): Array holding yaxis datapoints for threshold values of the ROC curve.
      min_precision (float): Minimum desired precision for the current scenario.
    
    Returns:
      optimal_recall_ix (int): Index in recall array (xaxis for Precision/Recall curve) at which the cutoff point is.
      optimal_cutoff (float): Optimal cutoff point value for the current scenario.
      optimal_cutoff_roc_ix (int): Index in `thresholds_roc` at which the cutoff point is.
    """

    # Look in the margin of thresholds that makes sense
    min_ix_recall = np.argmax(lr_precision[1:]) + 1
    max_ix_recall = min_ix_recall + np.argmin((lr_precision[min_ix_recall:] - no_skill)**2)

    # Find the optimal point
    optimal_recall_ix = min_ix_recall + np.argmin((lr_precision[min_ix_recall:max_ix_recall]-min_precision)**2)
    optimal_cutoff = thresholds_pr[optimal_recall_ix]

    optimal_cutoff_roc_ix = np.argmin((thresholds_roc - optimal_cutoff)**2)
    
    return optimal_recall_ix, optimal_cutoff, optimal_cutoff_roc_ix


def test_and_plot_scenarios(ev_targets, ev_outputs, scenarios):
    """
    Calculates the optimal cutoff points for all different scenarios defined and plots a
    representation of the ROC curve and Precision/Recall curve which shows how those cutoff
    points were calculated and where in those curves are placed the points representing the
    expected behaviour of the resulting classification model per scenario.
    
    Args:
      ev_targets (np.array): Target values for the classification at hand.
      ev_outputs (np.array): Values returned by our current model.
      scenarios (dict): Dictionary holding the name of each scenario as key and their
          constraints as values.
    """
    
    # General keypoints
    fpr, tpr, thresholds_roc, lr_precision, lr_recall, thresholds_pr, no_skill = \
        calculate_general_keypoints(ev_targets, ev_outputs)
    
    # Scenario-specific keypoints
    for sc_k in scenarios.keys():
        (scenarios[sc_k]['optimal_recall_ix'], scenarios[sc_k]['optimal_cutoff'],
         scenarios[sc_k]['optimal_cutoff_roc_ix']) = calculate_scenario_keypoints(
            lr_precision, no_skill, thresholds_pr, thresholds_roc, scenarios[sc_k]['min_precision'])

    # Plot setup
    fig = plt.figure(figsize=(16, 20), constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig)

    # First subplot (ROC curve + thresholds)
    ax1 = fig.add_subplot(gs[0])

    # ROC curve + scenario-specific cutoff points
    ax1.plot(fpr, tpr)
    ax1.set_title(f"ROC curve (AUC Score={auc(fpr, tpr)})")
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlim([0., 1.])

    # Additional y-axis on first subplot
    ax2 = ax1.twinx()

    # Thresholds + scenario-specific cutoff points
    ax2.plot(fpr, thresholds_roc, color='orange', linestyle='--')
    ax2.set_ylabel('Thresholds')
    ax2.set_ylim(ax2.get_ylim())
    
    for sc_k, sc_v in scenarios.items():
        
        x = fpr[sc_v['optimal_cutoff_roc_ix']]
        
        ax1_y = tpr[sc_v['optimal_cutoff_roc_ix']]
        ax2_y = thresholds_roc[sc_v['optimal_cutoff_roc_ix']]
        ax2_y_line = max(ax2_y, change_of_basis(ax1_y, ax1, ax2))
        
        ax1.scatter([x], [ax1_y], marker='o', color='black')
        ax1.text(x + .01, ax1_y, f"{sc_k} TP/FP tradeoff", fontsize=12, verticalalignment='center')
        
        ax2.plot([x]*2, [ax2.get_ylim()[0], ax2_y_line], color='black', linestyle=':')
        ax2.scatter([x], [ax2_y], marker='o', color='black')

    # Second subplot (Precision / Recall curve + thresholds)
    ax3 = fig.add_subplot(gs[1])

    # Precision / Recall curve + scenario-specific cutoff points
    ax3.plot(lr_recall, lr_precision, label='Performance')
    ax3.axline([0., no_skill], [1., no_skill], linestyle='--', label='No Skill')
    ax3.set_title(f"Precision / Recall Curve (AUC Score={auc(lr_recall, lr_precision)})")
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_xlim([0., 1.])

    # Additional y-axis on second subplot
    ax4 = ax3.twinx()

    # Thresholds + scenario-specific cutoff points
    ax4.plot(lr_recall, thresholds_pr, color='orange', linestyle='--')
    ax4.set_ylabel('Thresholds')
    
    for sc_k, sc_v in scenarios.items():
        
        x = lr_recall[sc_v['optimal_recall_ix']]
        ax3_y_min_pr = sc_v['min_precision']
        ax4_y_min_pr = change_of_basis(lr_precision[sc_v['optimal_recall_ix']], ax3, ax4)
        ax4_y_cutoff = sc_v['optimal_cutoff']
        
        ax3.axline([0., ax3_y_min_pr], [1., ax3_y_min_pr], color='red', linestyle='-.', label='Minimum precision')
        ax3.text(.15, ax3_y_min_pr + .02, f"{sc_k} min. precision", fontsize=12, fontdict=dict(color='red'))
        ax3.scatter([x], [ax3_y_min_pr], marker='o', color='black')

        ax4.scatter([x], [ax4_y_cutoff], marker='o', color='black')
        ax4.plot([x]*2 + [1.], [ax4_y_min_pr] + [ax4_y_cutoff]*2, color='black', linestyle=':')
        ax4.text(x-.03, ax4_y_cutoff, f"{sc_k} optimal cutoff point = {ax4_y_cutoff:.4f}", fontsize=14,
                 horizontalalignment='right', verticalalignment='center')

    plt.show()
```


```python
test_and_plot_scenarios(ev_targets, ev_outputs, scenarios)
```


    
![png](output_39_0.png)
    


Now I can proceed with the final classification metrics for each scenario:


```python
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

# Retrieve and represent all metrics for each of the scenarios given
for sc_k, sc_v in scenarios.items():

    scenario_title = f"{sc_k} (Precision >= {sc_v['min_precision']:.2f}):"
    print(scenario_title)
    print("=" * len(scenario_title), end="\n\n")

    support = len(ev_targets)
    output_decisions = ev_outputs >= sc_v['optimal_cutoff']

    cm = confusion_matrix(ev_targets, output_decisions)
    tn, fp, fn, tp = cm.ravel()

    pr_sc = tp / (tp + fp)
    re_sc = tp / (tp + fn)
    ac_sc = (tp + tn) / support
    f1_sc = 2 * pr_sc * re_sc / (pr_sc + re_sc)

    print("Confusion Matrix:")
    print("-----------------")
    print(cm, end="\n\n\n")

    print("Metrics:")
    print("--------")
    print(f"Precision: {pr_sc:.4f}")
    print(f"Recall:    {re_sc:.4f}")
    print(f"Accuracy:  {ac_sc:.4f}")
    print(f"F1 Score:  {f1_sc:.4f}", end="\n\n\n")

    print("Case-specific metrics:")
    print("----------------------")
    print(f"Non-factual sentences reaching human agent: {100 * (1 - pr_sc):.2f}% ({fp}/{tp + fp})")
    print(f"Factual sentences not reaching human agent: {100 * (1 - re_sc):.2f}% ({fn}/{tp + fn})")
    print(f"Sentences filtered out:                     {100 * (tn + fn)/support:.2f}% ({tn + fn}/{support})")
    
    print("\n")

```

    Scenario 1 (Precision >= 0.67):
    ===============================
    
    Confusion Matrix:
    -----------------
    [[5848   85]
     [ 332  167]]
    
    
    Metrics:
    --------
    Precision: 0.6627
    Recall:    0.3347
    Accuracy:  0.9352
    F1 Score:  0.4447
    
    
    Case-specific metrics:
    ----------------------
    Non-factual sentences reaching human agent: 33.73% (85/252)
    Factual sentences not reaching human agent: 66.53% (332/499)
    Sentences filtered out:                     96.08% (6180/6432)
    
    
    Scenario 2 (Precision >= 0.33):
    ===============================
    
    Confusion Matrix:
    -----------------
    [[5160  773]
     [ 113  386]]
    
    
    Metrics:
    --------
    Precision: 0.3330
    Recall:    0.7735
    Accuracy:  0.8623
    F1 Score:  0.4656
    
    
    Case-specific metrics:
    ----------------------
    Non-factual sentences reaching human agent: 66.70% (773/1159)
    Factual sentences not reaching human agent: 22.65% (113/499)
    Sentences filtered out:                     81.98% (5273/6432)
    
    
    

# Next steps

With the results at hand, it's a **matter of business needs** to decide if the performance obtained is enough to implement the resulting model and which the cutoff point must be.

However there are some potential further actions I can highlight, which arguably may to help render a better performant, more robust solution, namely:

- **Train with more data**: There is no doubt Deep Learning models need huge amounts of data to perform at their best, and the current use case is still far off that point. We can do this through the addition of **more data sources** containing more valuable real world data or even the design of **data augmentation** pipelines which help incorporating additional engineered samples (e.g.: automatic translation via NMT, through the creation of claims based on knowledge graphs and known syntactic structures...).
- **Further fine-tuning strategies**: In this exercise I've chosen to fine-tune BERT by freezing only the non-classifier layers. However, it'd be also interesting to try if fine-tuning just for the last layer or a different subset works better.
- **Ensemble modelling**: Transformers and attention based systems are very good at incorporating all kinds of nuances in language, up to the point that they are proven to perform seamlessly good on tasks which involve some degree of understanding of the knowledge contained in text. However it's also proved that they cannot still perform at such good levels as purpose-specific models for specific tasks. Thus, there's chance an ensemble architecture made up with a purpose-specific model and this fine-tuned BERT will show better performance. For this, some good option based on latest advances in NLP could be a **BiLSTM** network and/or **ad-hoc embeddings** followed by a **shallow Feedforward Neural Network** whose output could be concatenated to the last layer taken from BERT, or even with help of a **Tree-based model**.
- **Broaden the context**: Although so far we've focused on data only on a sentence level, many times language requires more context to clear up uncertainties regarding the message (e.g.: how cutting someones message may serve as a means for manipulation). Thus, there's good chance that incorporating further context awareness will help with better overall performance and robustness of the solution, although functional feasibility must be taken into account. In case it's feasible, an interesting way to try this will be to **incorporate a Recurrent Neural Network at the last layer of our fine-tuned BERT** model. In addition this approach will demand making all necessary changes on training and prediction pipelines, and some other context-specific additions like incorporating tokens for the new broader context at hand (e.g.:`[MESSAGE_BEGINS]`, `[MESSAGE_ENDS]`, `[PARAGRAPH_CHANGE]`).
