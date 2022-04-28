# Lane Detection using Deep Learning (YoloV3)

> member : Jae Ho Yoon
term : 2022.04.27 ~ 2022.05.11 ( about 14 week )
> 

<br>

```markdown
lane_detect_yolov3
    ⊢ config
        ⊢ requirements.txt
        ∟ yolov3.cfg
    ⊢ datasets
        ⊢ BDD100K
        ∟ KITTI
            ⊢ train
                ⊢ Annotations
                ⊢ ImageSets
                ∟ JPEGimages
            ⊢ valid
                ⊢ Annotations
                ⊢ ImageSets
                ∟ JPEGimages
            ⊢ test
                ⊢ ImageSets
                ∟ JPEGimages
            ⊢ img_kitti.zip
            ⊢ kitti.names
            ∟ lab_kitti.zip
    ⊢ model
        ⊢ __init__.py
        ∟ yolov3.py
    ⊢ utils
        ⊢ __init__.py
        ⊢ activations.py
        ⊢ augmentations.py
        ⊢ convert2yolo.py
        ⊢ install_dataset.py
        ⊢ loss.py
        ⊢ tools.py
        ⊢ yolo_datasets.py
        ∟ plots.py
    ⊢ READMD.md
    ⊢ output
        ⊢ weights
        ∟ tensorboard
    ⊢ env
    ∟ imgs
        ∟ gantt_chart.png
```

<br>

### 가상 환경 설정

```bash
pip install virtualenv

virtualenv env --python=python3.9
source env/Scripts/activate
```

### 패키지 설치

```bash
pip install -r requirements.txt
```

- pytorch 설치

https://pytorch.org/

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

<br>

<br>

# Project report

## 제목 : 딥러닝 기반 차선인식

기간 : 2022.04.28 ~ 2022.05.11

인원 : 1명

### 엔지니어링 요구사항

1. 관련 논문을 읽고, 딥러닝 기반 차선 인식 프로세스 확인
2. 데이터셋은 오픈 소스 차선 인식 데이터셋을 사용한다.
    1. BDD100K, KITTI, tuSimple, CULane
    2. 사용한 데이터셋 포맷과 yolov3 포맷을 맞춘다. 
        1. convert2yolo.py
        ROI를 따서 넣을 건데, bbox point들을 이용해서 dacon keypoint 대회에서 사용했던 방식으로 min,max 값을 사용해서 잘라낼 것

        2. img_crop.py
        [Key Point를 이용하여 손 동작 이미지만 크롭](https://dacon.io/competitions/official/235805/codeshare/3362?page=2&dtype=recent)

3. yolov3 딥러닝 모델을 활용하여 차선 위치 화면에 표시
    1. 차선 인식 bounding box를 그려본다.
    2. 차선 인식된 위치를 기반으로 polylines 그려서 표기해본다.
4. 인식된 차선의 위치를 기반으로 핸들링 조작
    1. 차선 중앙도 화면에 표시
    2. 차선의 중앙과 화면의 중앙의 차 및 핸들링 값 csv로 저장
5. 이미지가 아닌 video를 사용해서 cv2로 해보기

    <details open>
        <summary> 이게 안되면 그냥 video를 frame마다 저장해서 train </summary>

        ```python
        import cv2
        import os
        from glob import glob
        
        video = ["./src/video/track1.avi","./src/video/xycar_track1.mp4","./src/video/base_camera_dark.avi"]
        train_cnt = 0
        test_cnt = 0
        
        for v in video:
            cap = cv2.VideoCapture(v)
        
            while True:
                ok, img = cap.read()
                if not ok: break
                os.makedirs("./src/img/track/train/", exist_ok=True)
                os.makedirs("./src/img/track/test/", exist_ok=True)
                
                if len(glob("./src/img/track/train/*")) < 3000:
                    cv2.imwrite("./src/img/track/train/%06d.jpg" % train_cnt, img)
                    train_cnt += 1
                else:
                    cv2.imwrite("./src/img/track/test/%06d.jpg" % test_cnt, img)
                    test_cnt += 1
        
            print("finish!")
        ```
    </details>

### 산출물

1. 깃허브
    1. 버전 관리
    2. 코드 저장
2. 개발 문서
    1. 아키텍쳐 플로우 설명
    2. 중요 포인트
    3. 참고 문헌
        - loss, predict graph화하기
3. 가이드 문서(README.md)
    1. 각각의 폴더마다 readme 생성
    2. 사용 방법 설명
    3. 구조 간략히 설명
4. 피드백
    1. 부족했던 점
    2. 아쉬웠던 점
    3. 완성, 미완성 구분

![](2022-04-28-18-24-15.png)

<details open> 
    <summary> 보고서 포맷 </summary> 
        1.제목 : 프로젝트 이름
        
        2.기간 : 기간 +- 몇 주(gantt chart)
        3.인원 : 1,2,3,4명
        4.엔지니어링 요구사항 (숫자로 표현할 수 있거나, 성공과 실패를 명확하게 나올 수 있도록 아주 자세하게 표현)
        특정 프로그램을 만든다.
        데이터셋을 어떻게 구축할지
        어떤 모델을 사용할지
        학습을 어떻게 하고 이를 어떻게 출력할지
        혼자만의 간직하는 프로그램이 아닌 출시할 수 있는 정도가 되어야 할 것	
        필요한 기술 스택이 어떤것인지
        어떻게 연결하고, 모르는 부분은 어떻게 찾아볼지
        태스크 전체 정리
        어떤 태스크들을 사용할지
        5.산출물
        github에 repo를 만들어 가이드를 작성한다.
        버전관리를 통해 다양한 기록을 남긴다.

</details>

<details open>
    <summary> 예시 </summary>

        1.제목 : visual slam 개발

        2.기간 : 기간 +- 몇 주(gantt chart)

        3.인원 : 1명

        4.엔지니어링 요구사항 (숫자로 표현할 수 있거나, 성공과 실패를 명확하게 나올 수 있도록 아주 자세하게 표현)
        내 핸드폰에서 작동하는 visual slam 데모 앱을 만든다.
        앱을 실행하면 자동으로 카메라가 켜지고 시작버튼이 생성된다.
        slam 시작을 누르면 visual slam이 시작되어야 한다.
        visual slam을 작동하는 동안 자연스럽게 걷는 움직임에는 프로그램이 버그로 종료되면 안된다.
        모바일 앱이 정상작동하지 않는 경우는 예외(강제 홀드, 배터리 나가서 강제 종료)
        visual slam이 정상작동하지 않는 경우는 예외(너무 어두운 환경, 유저가 카메라 가리는 경우)
        visual slam은 최소 20초 이상 작동할 수 있어야 한다.
        visual slam은 유저가 동작중지시킬 수 있어야 하고, 아웃풋인 지도 정보를 스마트폰에 파일로 저장할 수 있어야 한다.

        5.산출물
        github에 repo를 만들어 소스코드, 앱 생성 코드, 앱 생성 가이드를 작성한다.
        버전관리를 통해 다양한 기록을 남긴다.

</details>