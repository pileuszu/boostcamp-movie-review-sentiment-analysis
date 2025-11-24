"""
데이터 전처리 모듈
"""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import warnings
import random

warnings.filterwarnings('ignore')


def normalize_punctuation(text: str) -> str:
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


def clean_special_chars(text: str) -> str:
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
        "", text
    )

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
    special_bracket_pattern = r"[『【《「〈｢\"\"''](.*?)[』】》」〉｣\"\"'']"
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = re.sub(special_bracket_pattern, "", text, flags=re.DOTALL)

    # 날짜 패턴 제거
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

    # 전화번호 패턴 제거
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
        r'\b좋아요\s?\d+\b', r'\b댓글\s?\d+\b', r'\b공유\s?\d+\b',
        r'\b조회수\s?\d+\b', r'\b구독자\s?\d+\b',
    ]
    for pat in sns_patterns:
        text = re.sub(pat, "", text)

    # 기타 노이즈 패턴 제거
    noise_patterns = [
        r'\b\d+번\b', r'\b\d+개\b', r'\b\d+명\b', r'\b\d+장\b',
        r'\b\d+회\b', r'\b\d+차\b', r'\b\d+번째\b',
        r'\b\d+위\b', r'\b\d+등\b', r'\b\d+점\b',
        r'\b\d+점대\b', r'\b\d+점만점\b', r'\b\d+점\s?만점\b',
    ]
    for pat in noise_patterns:
        text = re.sub(pat, "", text)

    # 특수 문자 및 기호 제거
    special_chars = [
        r'[★☆♥♡♠♣♦]', r'[♪♫♬♩]', r'[→←↑↓]',
        r'[①②③④⑤⑥⑦⑧⑨⑩]', r'[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]',
        r'[❶❷❸❹❺❻❼❽❾❿]', r'[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙ]',
    ]
    for pat in special_chars:
        text = re.sub(pat, "", text)

    return text.strip()


def clean_text(text: str) -> str:
    """기본 텍스트 정리 함수"""
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


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """데이터 로드 및 전처리"""
    print("데이터 로드 중...")
    df = pd.read_csv(data_path)
    print(f"전체 데이터 크기: {len(df):,}개")
    
    # Original 데이터만 사용
    df_original = df[df['type'] == 'original'].copy()
    print(f"Original 데이터 크기: {len(df_original):,}개")
    print(f"전체 대비 비율: {len(df_original)/len(df)*100:.1f}%")
    
    # 라벨 분포 확인
    print("\n라벨 분포:")
    label_counts = df_original['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"라벨 {label}: {count:,}개 ({count/len(df_original)*100:.1f}%)")
    
    # 필요한 컬럼만 선택
    df_processed = df_original[["ID", "label", "review", "type"]].copy()
    
    print("\n텍스트 정규화 과정 시작")
    print("=" * 50)
    
    # 1단계: 대소문자 정규화
    print("1단계: 대소문자 정규화 수행 중...")
    df_processed["review_normalized"] = df_processed["review"].str.lower()
    print("✓ 소문자 변환 완료")
    
    # 2단계: 구두점 정규화
    print("2단계: 구두점 정규화 수행 중...")
    df_processed["review_normalized"] = df_processed["review_normalized"].apply(normalize_punctuation)
    print("✓ 구두점 정규화 완료")
    
    # 3단계: 특수문자 정리
    print("3단계: 특수문자 정리 수행 중...")
    df_processed["review_normalized"] = df_processed["review_normalized"].apply(clean_special_chars)
    print("✓ URL/이메일/멘션 제거 완료")
    
    # 4단계: 빈 텍스트 처리
    print("4단계: 정규화 후 빈 텍스트 확인 중...")
    empty_after_normalization = df_processed["review_normalized"].str.strip().eq("").sum()
    if empty_after_normalization > 0:
        df_processed = df_processed[df_processed["review_normalized"].str.strip() != ""]
        print(f"✓ 빈 텍스트 {empty_after_normalization}개 제거")
    else:
        print("✓ 빈 텍스트 없음")
    
    # 5단계: 기본 텍스트 정리
    print("5단계: 기본 텍스트 정리 수행 중...")
    initial_size = len(df_processed)
    df_processed["review_cleaned"] = df_processed["review_normalized"].apply(clean_text)
    print("✓ 한글 자음/모음 정리, 반복 표현 정규화, 특수문자 제거 완료")
    
    # 6단계: 빈 텍스트 제거
    print("6단계: 빈 텍스트 제거 중...")
    empty_count = df_processed["review_cleaned"].str.strip().eq("").sum()
    if empty_count > 0:
        df_processed = df_processed[df_processed["review_cleaned"].str.strip() != ""]
        print(f"✓ 빈 텍스트 {empty_count}개 제거")
    else:
        print("✓ 빈 텍스트 없음")
    
    # 7단계: 중복 제거
    print("7단계: 중복 데이터 제거 중...")
    duplicates_count = df_processed.duplicated(subset=["review_cleaned", "label"]).sum()
    if duplicates_count > 0:
        df_processed = df_processed.drop_duplicates(subset=["review_cleaned", "label"])
        print(f"✓ 중복 데이터 {duplicates_count}개 제거")
    else:
        print("✓ 중복 데이터 없음")
    
    # 최종 결과 요약
    final_size = len(df_processed)
    removed = initial_size - final_size
    print("\n" + "=" * 50)
    print("전처리 결과 요약:")
    print(f"최종 데이터 크기: {initial_size:,} → {final_size:,}")
    print(f"제거된 데이터: {removed:,}개 ({removed / initial_size * 100:.1f}%)")
    print(f"평균 길이 - 정규화됨: {df_processed['review_normalized'].str.len().mean():.1f}자")
    print("=" * 50)
    
    return df_processed


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """훈련/검증 데이터 분할"""
    print(f"\n데이터 분할 중... (test_size={test_size})")
    
    train_data, val_data = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']  # 라벨 비율 유지
    )
    
    print(f"훈련 데이터: {len(train_data):,}개")
    print(f"검증 데이터: {len(val_data):,}개")
    
    # 라벨 분포 확인
    print("\n훈련 데이터 라벨 분포:")
    train_label_counts = train_data['label'].value_counts().sort_index()
    for label, count in train_label_counts.items():
        print(f"라벨 {label}: {count:,}개 ({count/len(train_data)*100:.1f}%)")
    
    print("\n검증 데이터 라벨 분포:")
    val_label_counts = val_data['label'].value_counts().sort_index()
    for label, count in val_label_counts.items():
        print(f"라벨 {label}: {count:,}개 ({count/len(val_data)*100:.1f}%)")
    
    return train_data, val_data


def augment_text(text: str, augmentation_prob: float = 0.3) -> str:
    """
    텍스트 데이터 증강
    
    Args:
        text: 원본 텍스트
        augmentation_prob: 증강 확률
    
    Returns:
        증강된 텍스트
    """
    if random.random() > augmentation_prob:
        return text
    
    # 랜덤하게 증강 기법 선택
    augmentation_methods = [
        _random_insertion,
        _random_deletion,
        _random_swap,
        _synonym_replacement
    ]
    
    method = random.choice(augmentation_methods)
    return method(text)


def _random_insertion(text: str) -> str:
    """랜덤 삽입"""
    words = text.split()
    if len(words) < 2:
        return text
    
    # 랜덤한 위치에 랜덤한 단어 삽입
    insert_pos = random.randint(0, len(words))
    random_word = random.choice(words)
    words.insert(insert_pos, random_word)
    
    return ' '.join(words)


def _random_deletion(text: str) -> str:
    """랜덤 삭제"""
    words = text.split()
    if len(words) < 3:
        return text
    
    # 랜덤하게 단어 삭제 (최소 1개는 남김)
    num_deletions = random.randint(1, min(2, len(words) - 1))
    indices_to_delete = random.sample(range(len(words)), num_deletions)
    
    new_words = [word for i, word in enumerate(words) if i not in indices_to_delete]
    return ' '.join(new_words)


def _random_swap(text: str) -> str:
    """랜덤 교환"""
    words = text.split()
    if len(words) < 2:
        return text
    
    # 두 단어의 위치 교환
    pos1, pos2 = random.sample(range(len(words)), 2)
    words[pos1], words[pos2] = words[pos2], words[pos1]
    
    return ' '.join(words)


def _synonym_replacement(text: str) -> str:
    """동의어 교체 (간단한 버전)"""
    # 간단한 동의어 사전
    synonyms = {
        '좋다': ['훌륭하다', '훌륭한', '좋은', '훌륭한'],
        '나쁘다': ['안좋다', '안좋은', '나쁜', '안좋은'],
        '최고': ['최고의', '최고다', '최고의'],
        '최악': ['최악의', '최악이다', '최악의'],
        '재미있다': ['재미있는', '재미있다', '재미있는'],
        '재미없다': ['재미없는', '재미없다', '재미없는'],
        '멋지다': ['멋진', '멋지다', '멋진'],
        '끔찍하다': ['끔찍한', '끔찍하다', '끔찍한']
    }
    
    words = text.split()
    for i, word in enumerate(words):
        if word in synonyms:
            words[i] = random.choice(synonyms[word])
    
    return ' '.join(words)


def augment_dataset(df: pd.DataFrame, target_samples_per_class: int = None) -> pd.DataFrame:
    """
    데이터셋 증강 (클래스 불균형 해결)
    
    Args:
        df: 원본 데이터프레임
        target_samples_per_class: 클래스당 목표 샘플 수
    
    Returns:
        증강된 데이터프레임
    """
    print("데이터 증강 시작...")
    
    # 클래스별 샘플 수 확인
    class_counts = df['label'].value_counts().sort_index()
    print(f"원본 클래스별 샘플 수: {class_counts.to_dict()}")
    
    if target_samples_per_class is None:
        # 가장 많은 클래스의 샘플 수를 목표로 설정
        target_samples_per_class = class_counts.max()
    
    augmented_data = []
    
    for class_label in class_counts.index:
        class_data = df[df['label'] == class_label].copy()
        current_count = len(class_data)
        
        if current_count < target_samples_per_class:
            # 부족한 샘플 수만큼 증강
            needed_samples = target_samples_per_class - current_count
            print(f"클래스 {class_label}: {current_count}개 -> {target_samples_per_class}개 (증강 필요: {needed_samples}개)")
            
            # 원본 데이터 추가
            augmented_data.append(class_data)
            
            # 증강된 데이터 생성
            augmented_samples = []
            for _ in range(needed_samples):
                # 랜덤하게 원본 샘플 선택
                original_sample = class_data.sample(1).iloc[0]
                
                # 텍스트 증강
                augmented_text = augment_text(original_sample['review_cleaned'])
                
                # 증강된 샘플 생성
                augmented_sample = original_sample.copy()
                augmented_sample['review_cleaned'] = augmented_text
                augmented_samples.append(augmented_sample)
            
            if augmented_samples:
                augmented_df = pd.DataFrame(augmented_samples)
                augmented_data.append(augmented_df)
        else:
            # 이미 충분한 샘플이 있는 경우 원본 그대로 사용
            augmented_data.append(class_data)
    
    # 모든 클래스 데이터 합치기
    result_df = pd.concat(augmented_data, ignore_index=True)
    
    # 데이터 셔플
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 최종 클래스 분포 확인
    final_class_counts = result_df['label'].value_counts().sort_index()
    print(f"증강 후 클래스별 샘플 수: {final_class_counts.to_dict()}")
    print(f"총 샘플 수: {len(result_df):,}개")
    
    return result_df


def split_data_with_augmentation(df: pd.DataFrame, test_size: float = 0.2, 
                                random_state: int = 42, use_augmentation: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """데이터를 훈련/검증 세트로 분할 (선택적 증강 포함)"""
    print(f"데이터 분할 중... (test_size={test_size})")
    
    # 먼저 훈련/검증 분할
    train_data, val_data = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']  # 클래스 비율 유지
    )
    
    # 훈련 데이터 증강 (선택적)
    if use_augmentation:
        print("\n훈련 데이터 증강 중...")
        train_data = augment_dataset(train_data)
    
    print(f"\n훈련 데이터: {len(train_data):,}개")
    print(f"검증 데이터: {len(val_data):,}개")
    
    # 클래스 분포 확인
    print("\n훈련 데이터 클래스 분포:")
    print(train_data['label'].value_counts().sort_index())
    print("\n검증 데이터 클래스 분포:")
    print(val_data['label'].value_counts().sort_index())
    
    return train_data, val_data
