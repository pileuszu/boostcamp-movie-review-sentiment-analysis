#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kykim BERT Korean Base 모델을 사용한 영화 리뷰 감정 분석
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch

# WandB 디렉토리 설정
os.environ['WANDB_DIR'] = '../../../wandb'
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

# 설정
MODEL_NAME = "kykim/bert-kor-base"
NUM_LABELS = 4
RANDOM_SEED = 42
NUM_EPOCHS = 3
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128

# 시드 설정
set_seed(RANDOM_SEED)

class ReviewDataset(Dataset):
    """리뷰 텍스트 데이터셋 클래스"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

def compute_metrics(eval_pred):
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def main():
    print(f"🚀 {MODEL_NAME} 모델 학습 시작")
    print("=" * 50)
    
    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 로드
    print("📁 데이터 로드 중...")
    train_df = pd.read_csv("../../../data/raw/train.csv")
    test_df = pd.read_csv("../../../data/raw/test.csv")
    
    print(f"훈련 데이터: {len(train_df):,} 샘플")
    print(f"테스트 데이터: {len(test_df):,} 샘플")
    
    # 토크나이저 및 모델 로드
    print("🤖 모델 및 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )
    
    # 데이터셋 생성
    print("📊 데이터셋 생성 중...")
    train_dataset = ReviewDataset(
        train_df['review'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    
    # 훈련/검증 분할
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"훈련 세트: {len(train_dataset):,} 샘플")
    print(f"검증 세트: {len(val_dataset):,} 샘플")
    
    # 훈련 설정
    training_args = TrainingArguments(
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to=None,
        seed=RANDOM_SEED,
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 모델 훈련
    print("🏋️ 모델 훈련 시작...")
    trainer.train()
    
    # 검증 데이터 평가
    print("📈 검증 데이터 평가...")
    eval_results = trainer.evaluate()
    print(f"검증 정확도: {eval_results['eval_accuracy']:.4f}")
    
    # 테스트 데이터 예측
    print("🔮 테스트 데이터 예측...")
    test_dataset = ReviewDataset(
        test_df['review'].tolist(),
        None,
        tokenizer,
        MAX_LENGTH
    )
    
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    
    # 결과 저장
    submission_df = pd.DataFrame({
        'ID': test_df['ID'],
        'pred': predicted_labels
    })
    
    output_path = f"../../../output/bert_model_comparison/submission_{MODEL_NAME.replace('/', '_')}.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"✅ 결과 저장 완료: {output_path}")
    
    print("🎉 학습 완료!")

if __name__ == "__main__":
    main()
