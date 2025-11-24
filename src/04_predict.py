#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습된 모델로 테스트 데이터 예측
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

# 설정
MODEL_PATH = "model/finetuned/checkpoint-39330"  # 파인튜닝된 모델
MAX_LENGTH = 256
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

print("📁 데이터 로드 중...")
test_df = pd.read_csv("data/raw/test.csv")
print(f"테스트 데이터: {len(test_df):,} 샘플")

print(f"\n📦 모델 로드 중: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print(f"✓ 모델 로드 완료")

print("\n🔮 예측 중...")
predictions = []
for i in range(0, len(test_df), BATCH_SIZE):
    batch_texts = test_df['review'].iloc[i:i+BATCH_SIZE].tolist()
    batch_texts = [str(text) if pd.notna(text) else "" for text in batch_texts]
    
    inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, 
                       padding=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        predictions.extend(preds.tolist())
    
    if (i // BATCH_SIZE + 1) % 100 == 0:
        print(f"진행: {i + BATCH_SIZE}/{len(test_df)} 샘플")

predictions = np.array(predictions)

print("\n💾 결과 저장 중...")
submission_df = pd.read_csv("data/raw/sample_submission.csv")
submission_df['pred'] = predictions
output_path = "output/bert_model_comparison/tapt_predictions.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
submission_df.to_csv(output_path, index=False)
print(f"✅ 결과 저장 완료: {output_path}")

print(f"\n📊 예측 분포:")
for i in range(4):
    count = np.sum(predictions == i)
    print(f"클래스 {i}: {count:,}개 ({count/len(predictions)*100:.1f}%)")

print("\n🎉 예측 완료!")
