1) Cosine similarity cutoff
어떻게 Meloniary에 적용됨?

유사어 확장은 보통 아래 과정으로 진행되지?

감성 태그: “우울”, “답답함”, “불안”

FastText에서 top-k 유사 벡터 뽑기

이걸 추천 입력 감성 벡터로 사용

문제는 top-k 안에 문맥이 틀어진 단어가 자주 끼어든다는 점.
예: “슬픔 → 비애 → 상실 → 결핍 → 부족 → ..."
마지막 두 개는 감성 문맥과 동떨어진 케이스.

이때 cosine similarity cutoff(예: 0.55 미만 제외)를 두면
노이즈 단어 필터링이 바로 가능해.

→ 사용자 로그 없이도 embedding 간 관계만으로 수행 가능
→ 토이 수준에서도 100% 가능

Meloniary 버전 서술 예시

"FastText 기반 유사어 확장 과정에서 코사인 유사도 기준치(0.55)를 적용해 off-topic 단어를 제거하여 semantic drift를 최소화함."

2) PCA 방향 보정 (semantic centroid alignment)

이거는 딥러닝이 아니어도, FastText 벡터만 있으면 가능해.

어떻게 적용됨?

LLM이 뽑은 감성 태그 3~7개를 벡터로 변환

이 벡터들의 centroid를 구함

PCA 1~2 principal direction으로 “주감성 축”을 만들고

이와 orthogonal하지 않은 방향으로 벡터가 너무 튀면 제거

이건 아주 간단한 계산인데, 효과는 큼.

예: "긴장"을 확장했는데

“긴급”, “긴박” → 근접

“장력”, “물리적 tension” → 의미는 비슷하지만 음악 감성과 무관

이런 오프토픽 벡터를 벡터 방향 기준으로 제거하는 방식이야.

→ 가사 데이터 필요 없음
→ 사용자 데이터 필요 없음
→ “semantic consistency 유지 경험”으로 해석됨

Meloniary 버전 서술 예시

"확장된 유사어 벡터의 방향성을 PCA 기반으로 정규화하여, 감정 축과 거리가 먼 off-topic 단어들을 제거함으로써 태그의 의미 일관성을 유지."
