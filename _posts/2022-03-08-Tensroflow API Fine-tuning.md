---
layout: post
title: Tensorflow Object Detection API Fine-tunning
date: 2022-06-08 14:30:23 +0900
category: Tensorflow
---
&nbsp;  
**텐서플로우의 pre-trained 모델을 사용하여 미세조정(Fine-tunning)을 통해 목적에 맞는 custom model 제작**  
선행 : Anaconda, Tensorflow, VS code,  NVIDIA CUDA 설치  
&nbsp;  

**1. [텐서플로우 API GIT](https://github.com/tensorflow/models)에서 ZIP으로 다운로드**  
원하는 경로에 압축 해제 후, research 폴더만 남기고 전부 삭제  
![oddownload](/images/oddownload.jpg)  
&nbsp;  
**2. [제너레이트 파일 GIT](https://github.com/hojihun5516/object_detection_setting)에서 ZIP으로 다운로드**   
../research/object_detection 폴더에 압축 해제  
&nbsp;  
**3. xml 파일을 csv로 컨버팅**  
이미지에 대한 라벨링 파일인 xml을 csv로 변환시켜 준다.  
Anaconda prompt, CMD 등을 이용하여 다음 명령을 실행  
```ruby
cd ../research/object_detection
python xml_to_csv.py
```
![xmltocsv](/images/xmltocsv.jpg)  
위와 같이 좌표, lable 값이 정리된 csv 파일로 변환된다.  
&nbsp;  
**4. train.record, test.record 생성**  
```ruby
cd ../research/object_detection
# train.record 생성
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record  
# test.record 생성
python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record  
```
![tfrecord](/images/tfrecord.jpg)  
위와 같이 train.record, test.record 파일이 생성된다.  
&nbsp;  
**5. labelmap 생성**  
```ruby
cd ../research/object_detection
python generate_labelmap.py
```
![labelmap](/images/labelmap.jpg)  
위와 같이 labelmap.pbtxt 파일이 생성된다.  
&nbsp;  
**6. pre-trained model 가져오기**  
[Tensorflow 2 Model ZOO](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)에서 학습이 이루어진 model을 가져올 수 있다.  
다운로드 받은 후 object detection 폴더에 압축을 해제 한다.  
&nbsp;  
**7. 파라미터 수정**  
object_detection/configs/tf2 경로의 6에서 받은 모델의 config 파일을 가져와  
images 폴더에 붙여넣기 해주고 VS code 등을 이용하여 편집 해준다.  
```ruby
num_classes : #원하는 Detection 클래스(label) 수
model {
  ssd {
    inplace_batchnorm_update: true
    freeze_batchnorm: false
    num_classes: 4
    ...

train_config 
#가져온 pre-trained 모델의 체크포인트 경로(object_detection 폴더 기준)
#detection으로 변경, batch_size 조정

train_config: {
  fine_tune_checkpoint: "pre-trained model name/checkpoint/ckpt-0"
  fine_tune_checkpoint_version: V2
  fine_tune_checkpoint_type: "detection"
  batch_size: 4
  ...
# learning_rate_base, total_steps 조정
optimizer {
    momentum_optimizer: {
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: 8e-3
          total_steps: 300000
          warmup_learning_rate: .0001
          warmup_steps: 2500
        }
      }

train_input_reader : # labelmap의 경로와 train.record 경로 입력

train_input_reader: {
  label_map_path: "images/labelmap.pbtxt"
  tf_record_input_reader {
    input_path: "train.record"
  }
}

eval_input_reader :  # labelmap의 경로와 test.record 경로 입력

eval_input_reader: {
  label_map_path: "images/labelmap.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "test.record"
  }
}
```













