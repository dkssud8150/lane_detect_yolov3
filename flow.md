## task 1. deep learning

1. 딥러닝 차선 인식 논문을 읽고, 그의 방법을 익힌 후 github를 참고
2. tusimple의 데이터 포맷이 어떤지를 중점적으로 파악
3. model에 있는 부분만 yolov3로 변환하면 거의 완성

3. ./datasets/tusimple/install_data.sh
4. convert2yolo.py
5. train.py
  - def argparse
  - def collate_fn
  - def train
6. utils
  - loss.py
  - optimizer.py
  - get_hyperparam.py
  - dataset.py
  - dataloader.py
  - my_transform.py
7. model
  - yolov3.py
  - def initialize_weight
8. test.py
+ export.py

---

## task 2. hough transform + ransac

1. 찾은 차선 위치를 통해 ransac을 통해 필터링 한 후 위치를 특정하고, 그것들의 직선을 그림
2. 차선 중앙 표시, 화면 중앙 표시
3. 둘의 차를 비교해서 핸들링
4. hough.py or hough.cpp
5. def ransac
