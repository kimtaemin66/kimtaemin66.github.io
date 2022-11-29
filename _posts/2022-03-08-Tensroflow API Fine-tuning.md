---
layout: post
title: Tensorflow Object Detection API Fine-tunning
date: 2022-06-08 14:30:23 +0900
category: Tensorflow
---
&nbsp;  
선행 : Anaconda, Tensorflow, NVIDIA CUDA 설치  
&nbsp;  

**1. [텐서플로우 API GIT](https://github.com/tensorflow/models)에서 ZIP으로 다운로드**  
원하는 경로에 압축 해제 후, research 폴더만 남기고 전부 삭제  
![oddownload](/images/oddownload.jpg)  
&nbsp;  
**2. [제너레이트 파일 GIT](https://github.com/hojihun5516/object_detection_setting)에서 ZIP으로 다운로드**   
../research/object detection 폴더에 압축 해제  
&nbsp;  
**3. xml 파일을 csv로 컨버팅**  
이미지에 대한 라벨링 파일인 xml을 csv로 변환시켜 준다.  
Anaconda prompt, CMD 등을 이용하여 다음 명령을 실행  
```ruby
cd ../research/object detection
python xml_to_csv.py
```
![xmltocsv](/images/xmltocsv.jpg)  
위와 같이 좌표, lable 값이 정리된 csv 파일로 변환된다.  
&nbsp;  
**4. train.record, test.record 생성**  
```ruby
cd ../research/object detection
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
cd ../research/object detection
python generate_labelmap.py
```
![labelmap](/images/labelmap.jpg)  
위와 같이 labelmap.pbtxt 파일이 생성된다.  









