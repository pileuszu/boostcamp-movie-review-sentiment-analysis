#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAPT (Task-Adaptive Pre-Training) 전용 스크립트
영화 리뷰 감정 분석을 위한 도메인 특화 사전 훈련
"""

import os
import re
import warnings
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoConfig,
)

warnings.filterwarnings("ignore")

# TAPT 설정 (도메인 특화 사전 훈련에 최적화)
MODEL_NAME = "kykim/bert-kor-base"
RANDOM_SEED = 42
TAPT_EPOCHS = 2  # TAPT는 짧은 에포크로 충분
BATCH_SIZE = 32  # 메모리 효율성을 위해 증가
LEARNING_RATE = 3e-5  # TAPT에 적합한 학습률
WARMUP_STEPS = 100  # 짧은 warmup
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256  # 영화 리뷰는 더 긴 문장이 많음
MLM_PROBABILITY = 0.15  # 표준 MLM 확률
SAVE_DIR = "../model/tapt"

def seed_everything(seed: int = RANDOM_SEED):
    """모든 시드를 고정하여 재현성 확보"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    print(f"✓ 모든 시드 고정: {seed}")


class TextPreprocessingPipeline:
    """영화 리뷰 텍스트 전처리 파이프라인 (TAPT용 고급 전처리)"""
    
    def __init__(self):
        self.is_fitted = False

    def preprocess(self, texts):
        """텍스트 전처리 적용 (감정 표현 보존)"""
        processed_texts = []
        for text in texts:
            text = self._normalize_punctuation(text)
            text = self._clean_special_chars(text)
            text = self._clean_text(text)
            processed_texts.append(text)
        return processed_texts

    def _normalize_punctuation(self, text):
        """구두점 정규화 (감정 강도 보존)"""
        if pd.isna(text) or not isinstance(text, str):
            return text
        
        # 마침표만 정규화 (감정 표현이 아닌 경우)
        text = re.sub(r"[.]{3,}", "...", text)  # 3개 이상의 마침표는 ...으로 정규화
        
        # 쉼표만 정규화 (감정 표현이 아닌 경우)
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

        # 도메인 패턴 제거 (신중하게)
        # "naver.com 영화평" 같은 리뷰에서 정보성 단어일 수 있으므로
        # 명확한 URL 형태만 제거
        tlds = [
            'com', 'net', 'org', 'co', 'kr', 'io', 'me', 'info', 'biz', 'tv', 'ai', 'app', 'dev',
            'xyz', 'us', 'uk', 'jp', 'cn', 'ru', 'site', 'store', 'online', 'top', 'tech', 'shop', 'cloud'
        ]
        tld_pattern = "|".join(tlds)
        # http:// 또는 www.로 시작하는 명확한 URL만 제거
        text = re.sub(
            rf"(?:http://|https://|www\.)[a-zA-Z0-9\-_]+(?:\.[a-zA-Z0-9\-_]+)*\.({tld_pattern})\b",
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

        # 영화 관련 패턴 제거 (감정 신호 보존)
        # 일부 문맥에서는 평점과 연관될 수 있으므로 신중하게 제거
        movie_patterns = [
            r'\b\d+편\b',  # 1편, 2편
            r'\b\d+부작\b',  # 1부작, 2부작
            r'\b\d+기\b',  # 1기, 2기
            r'\b\d+회차\b',  # 1회차, 2회차
            r'\b\d+화\b',  # 1화, 2화
            r'\b\d+분\s?\d+초\b',  # 120분 30초 (러닝타임)
            r'\b\d+등급\b',  # 15등급, 18등급
            r'\b\d+세\s?이상\b',  # 15세 이상
        ]
        for pat in movie_patterns:
            text = re.sub(pat, "", text)
        
        # "120분" 같은 표현은 보존 (감정 표현일 수 있음)

        # SNS/플랫폼 패턴 제거 (감정 신호 보존)
        # 해시태그와 멘션은 제거하되, 감정 표현은 보존
        text = re.sub(r'\b#\w+\b', "", text)  # 해시태그만 제거
        text = re.sub(r'\b@\w+\b', "", text)  # 멘션만 제거
        text = re.sub(r'\bRT\b', "", text)    # 리트윗 표시만 제거
        
        # 감정 신호가 될 수 있는 SNS 표현은 보존
        # "좋아요 100개" 같은 표현은 긍정 신호일 수 있음

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

        # 특수 문자 제거 (감정 표현 보존)
        # 이모티콘과 감정 표현은 보존하고, 순수 장식용 특수문자만 제거
        special_chars = [
            r'[♪♫♬♩]',  # 음악 기호만 제거
            r'[→←↑↓]',  # 화살표만 제거
            r'[①②③④⑤⑥⑦⑧⑨⑩]',  # 원 숫자만 제거
            r'[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]',  # 괄호 숫자만 제거
            r'[❶❷❸❹❺❻❼❽❾❿]',  # 검은 원 숫자만 제거
            r'[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙ]',  # 원 문자만 제거
        ]
        for pat in special_chars:
            text = re.sub(pat, "", text)
        
        # 감정 표현은 보존: ★☆♥♡♠♣♦ 등은 제거하지 않음

        return text.strip()

    def _clean_text(self, text):
        """한국어 텍스트를 위한 기본 텍스트 정리 (감정 표현 보존)"""
        if pd.isna(text):
            return ""

        text = str(text).strip()

        # 한국어 특화 전처리 (감정 표현 보존)
        # 불완전한 한글 중에서 너무 긴 반복만 제한 (20자 이상)
        text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]{20,}", "", text)  # 20자 이상의 불완전한 한글만 제거
        
        # 감정 표현 정규화 (과도한 반복만 축소)
        text = re.sub(r"([ㅋㅎ])\1{4,}", r"\1\1\1", text)  # 4개 이상 → 3개로
        text = re.sub(r"([ㅠㅜㅡ])\1{4,}", r"\1\1\1", text)  # 4개 이상 → 3개로
        
        # 일반 문자 반복 축소 (과도한 반복만)
        text = re.sub(r"(.)\1{5,}", r"\1\1\1", text)  # 5개 이상 → 3개로
        
        # 필요한 문자만 유지 (감정 표현 포함)
        text = re.sub(r"[^\w\s가-힣.,!?ㅋㅎㅠㅜㅡ~\-★☆♥♡♠♣♦❤️😭⭐️]", " ", text)
        text = re.sub(r"\s+", " ", text)  # 다중 공백을 단일 공백으로

        return text.strip()


    def fit(self, texts, labels=None):
        """학습 데이터로부터 전처리 정보 학습 (TAPT용 - 라벨 불필요)"""
        print("학습 데이터 기반 전처리 정보 수집 중...")
        
        # TAPT에서는 라벨 정보가 필요 없으므로 단순히 fitted 상태만 설정
        self.is_fitted = True
        print("✓ 전처리 파이프라인 학습 완료")

    def transform(self, texts):
        """전처리 적용"""
        if not self.is_fitted:
            print("Warning: 파이프라인이 학습되지 않았습니다. 기본 전처리만 적용합니다.")
        return self.preprocess(texts)

    def fit_transform(self, texts, labels=None):
        """학습과 변환을 동시에 수행"""
        self.fit(texts, labels)
        return self.transform(texts)


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


def load_data():
    """데이터 로드 및 전처리 (TAPT용 - 라벨 불필요)"""
    print("📁 데이터 로드 중...")
    
    # 절대 경로 사용
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    train_path = os.path.join(project_root, "data", "raw", "train.csv")
    test_path = os.path.join(project_root, "data", "raw", "test.csv")
    
    # 데이터 로드
    train_df = pd.read_csv(train_path)
    print(f"📊 원본 데이터 크기:")
    print(f"   - Train: {len(train_df):,} 샘플")
    
    # test.csv 파일이 있으면 로드
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"   - Test: {len(test_df):,} 샘플")
    else:
        print(f"   - Test: 파일 없음 (train만 사용)")
        test_df = None
    
    # type='original'인 데이터만 사용
    if 'type' in train_df.columns:
        train_df = train_df[train_df['type'] == 'original'].copy()
        print(f"✓ Original 필터링 후: {len(train_df):,} 샘플")
    
    # 모든 리뷰 데이터 통합 (TAPT는 라벨이 필요 없음)
    # train과 test 데이터를 concat하여 TAPT용 전체 텍스트 구성
    if test_df is not None and len(test_df) > 0:
        all_texts = pd.concat([train_df['review'], test_df['review']], ignore_index=True)
        print(f"✓ Train + Test 데이터 concat 완료: {len(all_texts):,} 샘플")
    else:
        all_texts = train_df['review']
        print(f"✓ Train 데이터만 사용: {len(all_texts):,} 샘플")
    
    # 결측치 제거
    initial_count = len(all_texts)
    all_texts = all_texts.dropna()
    print(f"✓ 결측치 제거: {initial_count - len(all_texts):,}개 제거, {len(all_texts):,}개 남음")
    
    # 중복 제거
    initial_count = len(all_texts)
    all_texts = all_texts.drop_duplicates()
    print(f"✓ 중복 제거: {initial_count - len(all_texts):,}개 제거, {len(all_texts):,}개 남음")
    
    # 리스트로 변환
    all_texts = all_texts.tolist()
    
    # 텍스트 전처리
    print("🔧 고급 텍스트 전처리 적용 중...")
    preprocessor = TextPreprocessingPipeline()
    all_processed_texts = preprocessor.fit_transform(all_texts)
    
    # 빈 텍스트 제거
    all_processed_texts = [text for text in all_processed_texts if text.strip()]
    
    print(f"✓ 전처리 완료: {len(all_processed_texts):,} 최종 샘플")
    print(f"  (전처리 전: {len(all_texts):,} → 전처리 후: {len(all_processed_texts):,})")
    
    return all_processed_texts


def perform_tapt(texts, model_name, tapt_epochs=3):
    """TAPT (Task-Adaptive Pre-Training) 수행"""
    print("🔄 TAPT (Task-Adaptive Pre-Training) 시작...")
    print(f"모델: {model_name}")
    print(f"에포크: {tapt_epochs}")
    print(f"텍스트 샘플 수: {len(texts):,}")
    
    # 5️⃣ 모델 및 토크나이저 로드
    print("🤖 모델 및 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Config에 return_dict=True 설정
    config = AutoConfig.from_pretrained(model_name)
    config.return_dict = True
    
    # MLM 모델 로드 시 경고 무시 (일부 가중치가 사용되지 않는 것은 정상)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlm_model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
    
    # 6️⃣ MLM 데이터셋 생성
    print("📊 MLM 데이터셋 생성 중...")
    train_dataset = MLMDataset(texts, tokenizer, MAX_LENGTH)
    print(f"✓ Train: {len(train_dataset):,} 샘플")
    
    # 8️⃣ TrainingArguments 설정
    print("⚙️ 훈련 설정 중...")
    tapt_args = TrainingArguments(
        output_dir=SAVE_DIR,
        num_train_epochs=tapt_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        seed=RANDOM_SEED,
    )
    
    # 7️⃣ DataCollator 정의
    print("🔧 MLM DataCollator 생성 중...")
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY
    )
    
    # 9️⃣ Trainer 구성 및 학습 실행
    print("🤖 Trainer 구성 중...")
    tapt_trainer = Trainer(
        model=mlm_model,
        args=tapt_args,
        train_dataset=train_dataset,
        data_collator=mlm_collator,
    )
    
    # TAPT 훈련
    print(f"🏋️ TAPT 훈련 시작... ({tapt_epochs} 에포크)")
    train_result = tapt_trainer.train()
    
    # 🔟 모델 저장 및 검증
    print("💾 모델 저장 중...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    tapt_trainer.save_model()
    tokenizer.save_pretrained(SAVE_DIR)
    
    # MLM loss에서 perplexity 계산 (exp(loss))
    train_loss = train_result.training_loss
    train_perplexity = np.exp(train_loss)
    
    print(f"\n{'='*60}")
    print(f"📊 TAPT 완료 - 최종 성능")
    print(f"{'='*60}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Perplexity: {train_perplexity:.2f}")
    print(f"{'='*60}")
    
    print("✅ TAPT 완료!")
    return mlm_model, tokenizer


def main():
    """메인 실행 함수 (TAPT 최적화)"""
    print("=" * 60)
    print("🚀 TAPT (Task-Adaptive Pre-Training) 시작")
    print("=" * 60)
    
    # 1️⃣ 설정 초기화
    print("\n[1/10] ⚙️ 설정 초기화")
    print(f"목적: 영화 리뷰 도메인 특화 사전 훈련")
    print(f"모델: {MODEL_NAME}")
    print(f"에포크: {TAPT_EPOCHS}")
    print(f"배치 크기: {BATCH_SIZE}")
    print(f"최대 길이: {MAX_LENGTH}")
    print(f"학습률: {LEARNING_RATE}")
    print(f"MLM 확률: {MLM_PROBABILITY}")
    
    # 시드 고정
    seed_everything(RANDOM_SEED)
    
    # 2️⃣ 환경 확인
    print("\n[2/10] 🔍 환경 확인")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"PyTorch 버전: {torch.__version__}")
    else:
        print("⚠️ GPU가 사용 불가능합니다. CPU로 실행됩니다.")
    
    # 3️⃣ 데이터 로드 및 통합
    print("\n[3/10] 📁 데이터 로드 및 통합")
    texts = load_data()
    
    # 4️⃣ 텍스트 전처리는 load_data() 내부에서 수행됨
    
    # 5️⃣~🔟 TAPT 수행 (모델 로드부터 평가까지)
    print("\n[5-10/10] 🎯 TAPT 수행")
    tapt_model, tokenizer = perform_tapt(
        texts=texts,
        model_name=MODEL_NAME,
        tapt_epochs=TAPT_EPOCHS
    )
    
    print("\n" + "=" * 60)
    print("🎉 TAPT 사전 훈련 완료!")
    print("=" * 60)
    print(f"저장된 모델 경로: {SAVE_DIR}")
    print("💡 다음 단계: 이 모델을 분류 작업(03_train.py)에 사용할 수 있습니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
