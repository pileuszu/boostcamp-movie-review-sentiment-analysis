#!/bin/bash
# 모든 BERT 모델 비교 실험 실행 스크립트

echo "🚀 모든 BERT 모델 비교 실험 시작"
echo "=================================="

# 각 모델별로 실험 실행
models=("klue_roberta" "klue_bert" "kykim_bert" "kcbert" "koelectra")

for model in "${models[@]}"; do
    echo ""
    echo "📊 $model 모델 실험 시작..."
    echo "----------------------------------------"
    
    cd "$model"
    python train.py
    cd ..
    
    echo "✅ $model 모델 실험 완료"
done

echo ""
echo "🎉 모든 모델 실험 완료!"
echo "📁 결과 파일은 ../../output/bert_model_comparison/ 폴더에 저장되었습니다."
echo ""
echo "생성된 파일들:"
ls -la ../../output/bert_model_comparison/submission_*.csv
