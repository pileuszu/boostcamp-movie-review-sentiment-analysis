#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kykim BERT Korean Base 모델을 사용한 영화 리뷰 감정 분석
- 고급 텍스트 전처리 파이프라인 포함
- TAPT (Task-Adaptive Pre-Training) 적용
"""

import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# WandB 디렉토리 설정
os.environ['WANDB_DIR'] = '../../../wandb'
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings("ignore")

# 설정
MODEL_NAME = "kykim/bert-kor-base"
NUM_LABELS = 4
RANDOM_SEED = 42
NUM_EPOCHS = 3
TAPT_EPOCHS = 2  # TAPT 에포크 수
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
LEARNING_RATE = 2e-5
TAPT_LEARNING_RATE = 5e-5  # TAPT용 학습률
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128

# 시드 설정
set_seed(RANDOM_SEED)


class TextPreprocessingPipeline:
    """고급 텍스트 전처리 파이프라인"""
    
    def __init__(self):
        self.is_fitted = False
        self.vocab_info = {}
        self.label_patterns = {}
        self.advanced_rules = {}

    def basic_preprocess(self, texts):
        """기본 전처리 (clean_text + normalize 기능)"""
        processed_texts = []
        for text in texts:
            cleaned = self._clean_text(text)
            normalized = self._normalize_text(cleaned)
            processed_texts.append(normalized)
        return processed_texts

    def advanced_preprocess(self, texts):
        """고급 전처리 적용"""
        processed_texts = []
        for text in texts:
            text = self._normalize_punctuation(text)
            text = self._clean_special_chars(text)
            text = self._clean_text(text)
            processed_texts.append(text)
        return processed_texts

    def _normalize_punctuation(self, text):
        """구두점 정규화"""
        if pd.isna(text) or not isinstance(text, str):
            return text
        
        # 여러 개의 구두점을 하나로 정규화
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[,]{2,}", ",", text)

        # 구두점 주변 공백 정리
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"([.,!?])\s+", r"\1 ", text)

        return text

    def _clean_special_chars(self, text):
        """특수문자 및 노이즈 패턴 제거"""
        if pd.isna(text) or not isinstance(text, str):
            return text
        
        # URL 패턴 제거
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "", text,
        )
        text = re.sub(r"www\.[a-zA-Z0-9\-_~:/?#\[\]@!$&'()*+,;=.]+", "", text)
        text = re.sub(r"p://", "", text)

        # 도메인 패턴 제거
        tlds = [
            'com', 'net', 'org', 'co', 'kr', 'io', 'me', 'info', 'biz', 'tv', 'ai', 'app', 'dev',
            'xyz', 'us', 'uk', 'jp', 'cn', 'ru', 'site', 'store', 'online', 'top', 'tech', 'shop', 'cloud'
        ]
        tld_pattern = "|".join(tlds)
        text = re.sub(
            rf"\b[a-zA-Z0-9\-_]+(?:\.[a-zA-Z0-9\-_]+)*\.({tld_pattern})\b",
            "", text)

        # 이메일 패턴 제거
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\b[\w\.\-]+@\b", "", text)
        text = re.sub(r"\b[\w\.\-]+@(?=\s|$)", "", text)
        text = re.sub(r"@\b", "", text)
        text = re.sub(r"\b@\b", "", text)
        text = re.sub(r"\b@\s", " ", text)
        text = re.sub(r"@\w+", "", text)

        # 과도한 공백 정리
        text = re.sub(r"\s+", " ", text)

        # 특수 괄호로 둘러싸인 텍스트 제거
        special_bracket_pattern = r"[『【《「〈｢\"''](.*?)[』】》」〉｣\"'']"
        prev_text = None
        while prev_text != text:
            prev_text = text
            text = re.sub(special_bracket_pattern, "", text, flags=re.DOTALL)

        # 날짜 제거
        date_patterns = [
            r'\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b',
            r'\b\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b',
            r'\b\d{1,2}[-/.]\d{1,2}\b',
            r'\b\d{4}년\s?\d{1,2}월\s?\d{1,2}일\b',
            r'\b\d{2}년\s?\d{1,2}월\s?\d{1,2}일\b',
            r'\b\d{4}년\s?\d{1,2}월\b',
            r'\b\d{4}년\b',
        ]
        for pat in date_patterns:
            text = re.sub(pat, "", text)

        # 전화번호 제거
        phone_patterns = [
            r'\b01[016789][ -]?\d{3,4}[ -]?\d{4}\b',
            r'\b\d{2,4}[ -]?\d{3,4}[ -]?\d{4}\b',
            r'\b\d{4}[ -]?\d{4}\b',
        ]
        for pat in phone_patterns:
            text = re.sub(pat, "", text)

        # 금액 패턴 제거
        price_patterns = [
            r'\b\d+원\b', r'\b\d+,\d+원\b', r'\b\d+\.\d+원\b',
            r'\b\d+만원\b', r'\b\d+천원\b', r'\b\d+억원\b',
            r'\$\d+', r'\b\d+달러\b',
        ]
        for pat in price_patterns:
            text = re.sub(pat, "", text)

        # 시간 패턴 제거
        time_patterns = [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\b',
            r'\b\d{1,2}시\s?\d{1,2}분\b',
            r'\b\d{1,2}시간\b', r'\b\d{1,2}분\b', r'\b\d{1,2}초\b',
            r'\b오전\s?\d{1,2}시\b', r'\b오후\s?\d{1,2}시\b',
        ]
        for pat in time_patterns:
            text = re.sub(pat, "", text)

        # 영화 관련 패턴 제거
        movie_patterns = [
            r'\b\d+편\b', r'\b\d+부작\b', r'\b\d+기\b',
            r'\b\d+회차\b', r'\b\d+화\b', r'\b\d+분\s?\d+초\b',
            r'\b\d+분\b', r'\b\d+등급\b', r'\b\d+세\s?이상\b',
        ]
        for pat in movie_patterns:
            text = re.sub(pat, "", text)

        # SNS/플랫폼 패턴 제거
        sns_patterns = [
            r'\b#\w+\b', r'\b@\w+\b', r'\bRT\b',
            r'\b좋아요\s?\d+\b', r'\b댓글\s?\d+\b',
            r'\b공유\s?\d+\b', r'\b조회수\s?\d+\b', r'\b구독자\s?\d+\b',
        ]
        for pat in sns_patterns:
            text = re.sub(pat, "", text)

        # 기타 노이즈 패턴 제거
        noise_patterns = [
            r'\b\d+번\b', r'\b\d+개\b', r'\b\d+명\b',
            r'\b\d+장\b', r'\b\d+회\b', r'\b\d+차\b',
            r'\b\d+번째\b', r'\b\d+위\b', r'\b\d+등\b',
            r'\b\d+점\b', r'\b\d+점대\b', r'\b\d+점만점\b',
            r'\b\d+점\s?만점\b',
        ]
        for pat in noise_patterns:
            text = re.sub(pat, "", text)

        # 특수 문자 제거
        special_chars = [
            r'[★☆♥♡♠♣♦]', r'[♪♫♬♩]', r'[→←↑↓]',
            r'[①②③④⑤⑥⑦⑧⑨⑩]', r'[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]',
            r'[❶❷❸❹❺❻❼❽❾❿]', r'[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙ]',
        ]
        for pat in special_chars:
            text = re.sub(pat, "", text)

        return text.strip()

    def _clean_text(self, text):
        """한국어 텍스트를 위한 기본 텍스트 정리"""
        if pd.isna(text):
            return ""

        text = str(text).strip()

        # 한국어 특화 전처리
        text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", "", text)  # 불완전한 한글 제거
        text = re.sub(r"([ㅋㅎ])\1{2,}", r"\1\1", text)  # 웃음 표현 정규화
        text = re.sub(r"([ㅠㅜㅡ])\1{2,}", r"\1\1", text)  # 슬픔 표현 정규화
        text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)  # 과도한 반복 축소
        text = re.sub(r"[^\w\s가-힣.,!?ㅋㅎㅠㅜㅡ~\-]", " ", text)  # 필요한 문자만 유지
        text = re.sub(r"\s+", " ", text)  # 다중 공백을 단일 공백으로

        return text.strip()

    def _normalize_text(self, text):
        """텍스트 정규화"""
        if pd.isna(text):
            return ""
        
        # 기본 정규화
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)  # 공백 정규화
        
        return text

    def fit(self, texts, labels=None):
        """학습 데이터로부터 전처리 정보 학습"""
        print("학습 데이터 기반 전처리 정보 수집 중...")
        
        # 라벨별 텍스트 특성 분석
        if labels is not None:
            for label in range(4):
                label_texts = [text for text, lbl in zip(texts, labels) if lbl == label]
                self.label_patterns[label] = {
                    'count': len(label_texts),
                    'avg_length': np.mean([len(str(text)) for text in label_texts])
                }
        
        self.is_fitted = True
        print("✓ 전처리 파이프라인 학습 완료")

    def transform(self, texts):
        """전처리 적용"""
        if not self.is_fitted:
            print("Warning: 파이프라인이 학습되지 않았습니다. 기본 전처리만 적용합니다.")
        return self.advanced_preprocess(texts)

    def fit_transform(self, texts, labels=None):
        """학습과 변환을 동시에 수행"""
        self.fit(texts, labels)
        return self.transform(texts)


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


class MLMDataset(Dataset):
    """Masked Language Modeling을 위한 데이터셋"""
    
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
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
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def compute_metrics(eval_pred):
    """평가 메트릭 계산"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}


def perform_tapt(model, tokenizer, train_texts, tapt_epochs=2):
    """TAPT (Task-Adaptive Pre-Training) 수행"""
    print("🔄 TAPT (Task-Adaptive Pre-Training) 시작...")
    
    # MLM 모델로 변환
    mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    
    # MLM 데이터셋 생성
    mlm_dataset = MLMDataset(train_texts, tokenizer, MAX_LENGTH)
    
    # TAPT 훈련 설정
    tapt_args = TrainingArguments(
        output_dir="./tapt_model",
        num_train_epochs=tapt_epochs,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        learning_rate=TAPT_LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        seed=RANDOM_SEED,
    )
    
    # MLM 데이터 콜레이터
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # TAPT 트레이너
    tapt_trainer = Trainer(
        model=mlm_model,
        args=tapt_args,
        train_dataset=mlm_dataset,
        data_collator=mlm_collator,
    )
    
    # TAPT 훈련
    print(f"TAPT 훈련 중... ({tapt_epochs} 에포크)")
    tapt_trainer.train()
    
    print("✓ TAPT 완료!")
    return mlm_model


def main():
    print(f"🚀 {MODEL_NAME} 모델 학습 시작 (고급 전처리 + TAPT)")
    print("=" * 60)
    
    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 로드
    print("📁 데이터 로드 중...")
    train_df = pd.read_csv("../../../data/raw/train.csv")
    test_df = pd.read_csv("../../../data/raw/test.csv")
    
    print(f"훈련 데이터: {len(train_df):,} 샘플")
    print(f"테스트 데이터: {len(test_df):,} 샘플")
    
    # 텍스트 전처리 파이프라인 적용
    print("🔧 고급 텍스트 전처리 적용 중...")
    preprocessor = TextPreprocessingPipeline()
    
    # 훈련 데이터 전처리
    train_texts_processed = preprocessor.fit_transform(
        train_df['review'].tolist(), 
        train_df['label'].tolist()
    )
    
    # 테스트 데이터 전처리
    test_texts_processed = preprocessor.transform(test_df['review'].tolist())
    
    print(f"전처리 완료 - 훈련: {len(train_texts_processed):,} 샘플")
    print(f"전처리 완료 - 테스트: {len(test_texts_processed):,} 샘플")
    
    # 토크나이저 및 모델 로드
    print("🤖 모델 및 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # TAPT 수행
    mlm_model = perform_tapt(
        model=None,  # 모델은 TAPT 함수 내에서 로드
        tokenizer=tokenizer,
        train_texts=train_texts_processed,
        tapt_epochs=TAPT_EPOCHS
    )
    
    # 분류 모델 로드 (TAPT된 가중치 사용)
    print("📊 분류 모델 초기화 중...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )
    
    # TAPT된 가중치를 분류 모델에 적용 (BERT 부분만)
    if hasattr(mlm_model, 'bert') and hasattr(model, 'bert'):
        model.bert.load_state_dict(mlm_model.bert.state_dict())
        print("✓ TAPT된 가중치를 분류 모델에 적용 완료")
    
    # 데이터셋 생성
    print("📊 데이터셋 생성 중...")
    train_dataset = ReviewDataset(
        train_texts_processed,
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
        output_dir="./classification_model",
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
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
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
    print("🏋️ 분류 모델 훈련 시작...")
    trainer.train()
    
    # 검증 데이터 평가
    print("📈 검증 데이터 평가...")
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
                              target_names=['부정', '중립', '긍정', '강한긍정']))
    
    # 테스트 데이터 예측
    print("🔮 테스트 데이터 예측...")
    test_dataset = ReviewDataset(
        test_texts_processed,
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
    os.makedirs("../../../output/bert_model_comparison", exist_ok=True)
    
    output_path = f"../../../output/bert_model_comparison/submission_{MODEL_NAME.replace('/', '_')}_advanced_tapt.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"✅ 결과 저장 완료: {output_path}")
    
    # 예측 분포 확인
    print(f"\n📊 예측 분포:")
    pred_counts = pd.Series(predicted_labels).value_counts().sort_index()
    for i, count in pred_counts.items():
        print(f"클래스 {i}: {count:,}개 ({count/len(predicted_labels)*100:.1f}%)")
    
    print("\n🎉 고급 전처리 + TAPT 학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()