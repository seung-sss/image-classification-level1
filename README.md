# image_classification

## Getting Started    
### Dependencies
- torch==1.10.1
- torchvision==0.11.2
- Language : Python 3.8.5
- Ubuntu 18.04.5 LTS
- Server : V100
                                                              

### Install Requirements
- `pip3 install -r requirements.txt`

### Prepare Images
```
data
  +- eval
  |  +- images
  +- train
  |  +- images
```

### Training
- `python3 train.py --config [yaml file path]`
- 모델 학습을 위한 명령어
- config의 옵션으로 모델 학습 위해 생성한 yaml 파일의 경로를 붙여주면 된다 (default=sample.yaml)

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
efficientnet-b7 | V100 | 512/2, 384/2 | 15 | 3 hours