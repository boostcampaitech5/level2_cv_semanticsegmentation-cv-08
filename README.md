# baseline code for bone segmentation task

##baseline을 한번 만들어봤습니다.
1. train.py : 학습을 진행하는 main 스크립트
2. dataset.py : dataset을 정의해 놓는 스크립트(현재 augmentation이 dataset안에서 조정하도록 만들어져 있습니다.)
3. inference.py : 학습한 결과를 최종 제출할 파일로 만드는 스크립트(CSVs 디렉토리 아래에 csv파일이 생성됩니다.)
4. model.py : 모델을 정의하는 부분
5. parse_config.py : config.json를 읽어와서 dictionary 형태로 접근할 수 있도록 만드는 class가 정의되어 있습니다.
6. metric.py : validation 과정에서 사용되는 metric을 정의해놓는 곳(현재는 dice_coef만 있어요!)
7. utils.py : 학습과 상관없는 json을 읽어오는 등 자잘한 기능들을 모아놓는 곳
8. loss.py : loss를 모아놓는 곳

##사용방법
1. python train.py -c [json file path]를 실행시키시면 학습이 진행됩니다.
2. 수정하고 싶은 파라미터가 있으시다면 config.json을 이용해보세요!
    - augmentation이 있긴하지만 현재 적용은 되지 않아요!

##TODO
1. augmentation이 현재는 dataset안에 들어가 있는데, 이걸 json으로 조절할 수 있도록 만들어야함
2. validation in train loop(o)
3. logging(wandb)(o)
4. 학습을 이어서 하기 위한 코드 만들기
5. 설명을 위한 README.md 만들기(진행중)
6. inference.py 만들기(o)