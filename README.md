# 2024-recommendation-system
멜로니어리(melon+diary) : 오늘의 일기를 입력하면 분위기에 맞는 노래를 추천해주는 알고리즘

## Data

##### Melon Crawling
- 인기차트
- 장르음악
- 멜론 DJ 플레이리스트 

## Modeling

##### 일기 태그 추출 Tagging
- OpenAI 활용
- 키워드 3개, 감정 3개 추출

##### 벡터화 Word Embedding
- fastText Korean Pretrained Model 활용 
- 유사 단어 검색, 가사-태그리스트 벡터화

##### 취향 필터링 Filtering
- 장르, 대중성, 최신가요 선호도의 정보를 받아 1차 곡 필터링
