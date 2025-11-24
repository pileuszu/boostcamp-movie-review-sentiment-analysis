#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrastive TAPT 모델을 활용한 영화 리뷰 감정 분석 파인튜닝
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings("ignore")

# 설정
BASE_MODEL = "kykim/bert-kor-base"

# 절대 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 모델은 /data/ephemeral/repos/model/에 저장되어 있음
TAPT_MODEL_PATH = "/data/ephemeral/repos/model/tapt"  # 일반 TAPT 모델
CONTRASTIVE_TAPT_MODEL_PATH = "/data/ephemeral/repos/model/contrastive_tapt_base"  # Contrastive TAPT 모델
USE_CONTRASTIVE_TAPT = False  # True: Contrastive TAPT 사용, False: 일반 TAPT 사용

NUM_LABELS = 4
RANDOM_SEED = 42
NUM_EPOCHS = 5
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256  # TAPT에서 사용한 길이와 동일
SAVE_DIR = os.path.join(project_root, "model", "finetuned")

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
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}


def load_model_and_tokenizer():
    """TAPT 모델과 토크나이저 로드"""
    from transformers import AutoModelForMaskedLM
    import os
    
    if USE_CONTRASTIVE_TAPT:
        model_path = CONTRASTIVE_TAPT_MODEL_PATH
        print(f"📦 Contrastive TAPT 모델 로드 중: {model_path}")
    else:
        model_path = TAPT_MODEL_PATH
        print(f"📦 TAPT 모델 로드 중: {model_path}")
    
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # Contrastive TAPT인 경우 special 로드
        if USE_CONTRASTIVE_TAPT:
            # contrastive_model.pt 파일에서 모델 로드
            contrastive_model_file = os.path.join(model_path, 'contrastive_model.pt')
            if os.path.exists(contrastive_model_file):
                print(f"✓ Contrastive 모델 파일 발견: {contrastive_model_file}")
                model_state = torch.load(contrastive_model_file, map_location='cpu', weights_only=False)
                
                # Base 모델 생성
                model = AutoModelForSequenceClassification.from_pretrained(
                    BASE_MODEL,
                    num_labels=NUM_LABELS
                )
                
                # BERT 가중치만 로드
                bert_state_dict = model_state['bert_state_dict']
                model.bert.load_state_dict(bert_state_dict, strict=False)
                print("✓ Contrastive TAPT BERT 가중치 적용 완료")
                
                # hyperparameters 출력
                if 'hyperparameters' in model_state:
                    print(f"  하이퍼파라미터: {model_state['hyperparameters']}")
            else:
                # fallback: AutoModelForMaskedLM로 로드
                print("⚠️ contrastive_model.pt 파일을 찾을 수 없습니다. fallback 로드...")
                tapt_model = AutoModelForMaskedLM.from_pretrained(model_path)
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    BASE_MODEL,
                    num_labels=NUM_LABELS
                )
                
                if hasattr(model, 'bert') and hasattr(tapt_model, 'bert'):
                    model.bert.load_state_dict(tapt_model.bert.state_dict(), strict=False)
                    print("✓ TAPT 가중치 적용 완료")
        else:
            # 일반 TAPT 모델 로드
            tapt_model = AutoModelForMaskedLM.from_pretrained(model_path)
            
            # Base 모델에서 분류 모델 생성
            model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=NUM_LABELS
            )
            
            # TAPT 가중치를 분류 모델에 적용 (BERT 부분만)
            if hasattr(model, 'bert') and hasattr(tapt_model, 'bert'):
                model.bert.load_state_dict(tapt_model.bert.state_dict(), strict=False)
                print("✓ TAPT 가중치 적용 완료")
        
        model_type_str = "Contrastive TAPT" if USE_CONTRASTIVE_TAPT else "TAPT"
        print(f"✓ {model_type_str} 기반 분류 모델 준비 완료")
        
    except Exception as e:
        print(f"⚠️ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        print("Base 모델 사용")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=NUM_LABELS
        )
    
    return model, tokenizer


def main():
    """메인 실행 함수"""
    # 모델 타입 정의 (함수 내에서 사용)
    global model_type
    model_type = "Contrastive TAPT" if USE_CONTRASTIVE_TAPT else "TAPT"
    print(f"🚀 {model_type} 기반 감정 분석 모델 파인튜닝 시작")
    print("=" * 70)
    
    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 로드
    print("\n📁 데이터 로드 중...")
    train_path = os.path.join(project_root, "data", "raw", "train.csv")
    test_path = os.path.join(project_root, "data", "raw", "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # type='original'인 데이터만 사용
    if 'type' in train_df.columns:
        original_count = len(train_df[train_df['type'] == 'original'])
        print(f"전체 훈련 데이터: {len(train_df):,} 샘플")
        train_df = train_df[train_df['type'] == 'original'].copy()
        print(f"Original 데이터만 사용: {len(train_df):,} 샘플")
    
    print(f"훈련 데이터: {len(train_df):,} 샘플")
    print(f"테스트 데이터: {len(test_df):,} 샘플")
    
    # 모델 및 토크나이저 로드
    print("\n🤖 모델 및 토크나이저 로드 중...")
    model, tokenizer = load_model_and_tokenizer()
    
    # 데이터셋 생성
    print("\n📊 데이터셋 생성 중...")
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
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    print(f"훈련 세트: {len(train_dataset):,} 샘플")
    print(f"검증 세트: {len(val_dataset):,} 샘플")
    
    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        seed=RANDOM_SEED,
        dataloader_num_workers=0,
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
    print("\n🏋️ 모델 파인튜닝 시작...")
    trainer.train()
    
    # 검증 데이터 평가
    print("\n📈 검증 데이터 평가...")
    eval_results = trainer.evaluate()
    print(f"검증 정확도: {eval_results['eval_accuracy']:.4f}")
    print(f"검증 F1-Score: {eval_results['eval_f1']:.4f}")
    
    # 상세 분류 리포트
    print("\n📊 상세 분류 리포트:")
    val_predictions = trainer.predict(val_dataset)
    val_pred_labels = np.argmax(val_predictions.predictions, axis=1)
    
    # 검증 데이터의 실제 라벨 추출
    val_labels = []
    for i in range(len(val_dataset)):
        val_labels.append(val_dataset[i]['labels'].item())
    
    print(classification_report(val_labels, val_pred_labels, 
                              target_names=['0 (부정)', '1 (중립)', '2 (긍정)', '3 (강한긍정)']))
    
    # 테스트 데이터 예측
    print("\n🔮 테스트 데이터 예측...")
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
    
    # 출력 디렉토리 생성
    output_dir = os.path.join(project_root, "output", "bert_model_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"contrastive_tapt_finetuned.csv" if USE_CONTRASTIVE_TAPT else "tapt_finetuned.csv"
    output_path = os.path.join(output_dir, output_file)
    submission_df.to_csv(output_path, index=False)
    print(f"✅ 결과 저장 완료: {output_path}")
    
    # 예측 분포 확인
    print(f"\n📊 예측 분포:")
    pred_counts = pd.Series(predicted_labels).value_counts().sort_index()
    for i, count in pred_counts.items():
        print(f"클래스 {i}: {count:,}개 ({count/len(predicted_labels)*100:.1f}%)")
    
    print("\n🎉 파인튜닝 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
