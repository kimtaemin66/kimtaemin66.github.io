---
layout: post
title: Transfer Learning을 위한 사전 준비 - 데이터 라벨링
date: 2022-06-01 17:18:23 +0900
category: Tensorflow
---
_선행 : Anaconda, Tensorflow, VS code,  NVIDIA CUDA 설치_  
**[kaggle](https://www.kaggle.com/)에서 데이터 수집**  
수집한 데이터는 images 폴더를 생성하여 train, test 각각 9:1 비율로 나눔  
![image](/images/images.jpg)  
&nbsp;  
anaconda prompt에서 labelImg 설치  
[labelImg 사용법](https://inf-coding.tistory.com/12)
```ruby
pip install labelImg
```
&nbsp;  
labelImg 실행 후 open Dir로 train 폴더를 열어줌(test도 동일), 포맷을 Pascal/VOC로 변경
![labelImg](/images/labelImg.jpg)
&nbsp;  
라벨링이 완료된 이미지는 .xml 파일이 생성된다. 
![labelled](/images/labelled.jpg)

