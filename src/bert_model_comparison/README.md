# BERT 모델 비교 실험

이 폴더는 5개의 주요 BERT 계열 한국어 모델 성능을 비교하기 위한 실험 공간입니다.

## 모델 목록

1. **klue_roberta** - KLUE RoBERTa Base
2. **klue_bert** - KLUE BERT Base  
3. **kykim_bert** - Kykim BERT Korean Base
4. **kcbert** - KcBERT Base
5. **koelectra** - KoELECTRA Base v3 Discriminator

## 실험 순서

1. 각 모델을 동일한 설정으로 훈련함
2. 검증 데이터로 성능을 평가함
3. 테스트 데이터로 최종 예측함
4. 결과를 비교‧분석함

## 실험 결과

| 모델명                    | 정확도   | 실행 일시           | 상태   |
|--------------------------|---------|---------------------|--------|
| monologg_koelectra-base-v3-discriminator | 0.8193  | 2025.10.23 03:04  | 완료   |
| beomi_kcbert-base        | 0.8076  | 2025.10.23 00:58    | 완료   |
| kykim_bert-kor-base      | 0.8216  | 2025.10.22 23:15    | 완료   |
| klue_bert-base           | 0.8131  | 2025.10.22 21:32    | 완료   |
| klue_roberta-base        | 0.8193  | 2025.10.22 19:50    | 완료   |

## 결과 분석

- 5종의 BERT 계열 한국어 모델이 전반적으로 유사한 정확도(0.81~0.82)를 보임
- `kykim_bert-kor-base`가 최고 정확도(0.8216)를 기록했으며, `monologg_koelectra-base-v3-discriminator`와 `klue_roberta-base`도 근접한 정확도를 달성함
- 모든 실험은 동일한 환경(데이터 전처리, 하이퍼파라미터 등)에서 수행함
- 각 모델의 세부 예측 분포 및 차이는 `compare_results.py`를 통해 추가 분석할 수 있음

### 주요 관찰 및 시사점

- **kykim_bert-kor-base**는 뛰어난 성능과 학습 속도를 모두 보임
- **koelectra** 계열은 일관된 강점이 있으며, 실제 서비스 적용에도 적합함
- 라벨별 예측 분포, 혼동행렬 등 추가적인 성능 분석도 필요함

## 모델 선정 제안

- `klue_bert-base` 대비 `klue_roberta-base`가 전반적으로 더 좋은 결과를 보임
- `beomi_kcbert-base`는 실제 인터넷 환경 언어(뉴스/커뮤니티/악성댓글)에 강점이 있을 것으로 기대했으나, 데이터 전처리가 필요 없는 구조 · 이미 많은 최적화가 적용된 상태임에도 낮은 정확도를 보여 최종 후보에서는 제외함
- `monologg_koelectra-base-v3-discriminator`는 경량화에 중점을 둔 모델로 효율 측면에서는 우수하지만, 최고 성능이 필요한 상황에서는 제외할 수 있음
- 이번 실험에서 `kykim_bert-kor-base`가 가장 높은 정확도(0.8216)를 달성해, 동일 조건에서 실험한 모든 모델 중 최고 성능임

따라서 `kykim_bert-kor-base`와 `klue_roberta-base`를 우선 후보로 선정했고, 추가적인 데이터 전처리 및 모델 개선 작업 후 재평가를 통하여 최종 모델을 선정할 계획임
