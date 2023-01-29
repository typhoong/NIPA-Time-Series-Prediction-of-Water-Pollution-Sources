# 2022 인공지능 온라인 경진대회
## [수치] 수질 오염 요인 시계열 예측 문제

### 데이터 구조

${PROJECT}
├── train/

│   └── train.json

├── test/

│   └── sample_submission.json


### 사용 모델

Prophet

### 학습 및 추론

1. train.json, sample_submission.json, prediction.json 경로 지정
2. train.json, sample_submission.json 로드
3. 사업장별 dataframe 제작
4. Prophet 예측을 위한 dataframe 포맷 제작
5. 추론 및 추론 파일 저장
