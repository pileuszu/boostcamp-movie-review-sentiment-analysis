"""
모델 정의 및 커스텀 클래스 모듈
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer
from typing import Optional, Union, List
import pandas as pd


class FocalLoss(nn.Module):
    """
    개선된 Focal Loss 구현 (클래스 가중치 포함)
    
    Focal Loss는 클래스 불균형 문제를 해결하기 위해 제안된 손실 함수입니다.
    어려운 샘플(잘못 분류된 샘플)에 더 큰 가중치를 부여하고,
    쉬운 샘플(올바르게 분류된 샘플)의 가중치는 줄입니다.
    
    Args:
        alpha (float): 클래스 가중치 (기본값: 1.0)
        gamma (float): focusing parameter (기본값: 2.0)
        class_weights (torch.Tensor): 클래스별 가중치
        reduction (str): 손실 감소 방식 ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 class_weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss 계산
        
        Args:
            inputs: 모델의 로짓 출력 (batch_size, num_classes)
            targets: 실제 라벨 (batch_size,)
        
        Returns:
            focal_loss: 계산된 Focal Loss
        """
        # Cross Entropy Loss 계산 (클래스 가중치 없이)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 확률 계산 (소프트맥스 적용)
        pt = torch.exp(-ce_loss)
        
        # Focal Loss 계산
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 클래스 가중치 적용 (수동으로)
        if self.class_weights is not None:
            # 클래스 가중치를 현재 디바이스로 이동
            weights = self.class_weights.to(inputs.device)
            # 각 샘플에 해당하는 클래스 가중치 적용
            sample_weights = weights[targets]
            focal_loss = focal_loss * sample_weights
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossTrainer(Trainer):
    """
    Focal Loss를 사용하는 커스텀 Trainer (클래스 가중치 포함)
    """
    
    def __init__(self, focal_loss_alpha: float = 1.0, focal_loss_gamma: float = 0.5, 
                 class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(
            alpha=focal_loss_alpha, 
            gamma=focal_loss_gamma,
            class_weights=class_weights
        )
        self._class_weights = class_weights  # 원본 가중치 저장
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Focal Loss를 사용하여 손실 계산
        (Trainer 내부 호출: 추가 인자 허용)
        """
        labels = inputs.get("labels")
        
        # 모델 출력
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Focal Loss 계산 (클래스 가중치는 forward에서 처리)
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def calculate_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    클래스 가중치 계산 (역빈도 기반)
    
    Args:
        labels: 클래스 라벨 텐서
        num_classes: 클래스 수
    
    Returns:
        class_weights: 클래스별 가중치 텐서
    """
    # 각 클래스의 빈도 계산
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    
    # 전체 샘플 수
    total_samples = len(labels)
    
    # 역빈도 가중치 계산 (적은 클래스에 더 큰 가중치)
    class_weights = total_samples / (num_classes * class_counts)
    
    # 가중치 정규화 (가장 큰 가중치를 1.0으로)
    class_weights = class_weights / class_weights.max()
    
    return class_weights


class ReviewDataset(Dataset):
    """
    리뷰 데이터를 위한 PyTorch Dataset 클래스
    
    이 클래스는 텍스트 데이터를 BERT 모델이 처리할 수 있는 형태로 변환합니다.
    
    Attributes:
        texts: 리뷰 텍스트 리스트
        labels: 감정 라벨 리스트
        tokenizer: BERT 토크나이저
        max_length: 최대 시퀀스 길이
    """
    
    def __init__(self, texts: Union[List[str], pd.Series], labels: Union[List[int], pd.Series], 
                 tokenizer, max_length: int = 128):
        """
        ReviewDataset 초기화
        
        Args:
            texts: 리뷰 텍스트 리스트 또는 pandas Series
            labels: 감정 라벨 리스트 또는 pandas Series
            tokenizer: BERT 토크나이저
            max_length: 최대 시퀀스 길이 (기본값: 128)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """
        데이터셋 크기 반환
        """
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        """
        특정 인덱스의 데이터 아이템 반환
        
        Args:
            idx: 데이터 인덱스
        
        Returns:
            dict: 토크나이징된 텍스트와 라벨을 포함한 딕셔너리
        """
        # 텍스트 토크나이징 및 패딩
        encoding = self.tokenizer(
            str(self.texts.iloc[idx]) if hasattr(self.texts, 'iloc') else str(self.texts[idx]),
            truncation=True,  # 최대 길이 초과시 자르기
            padding="max_length",  # 최대 길이까지 패딩
            max_length=self.max_length,
            return_tensors="pt",  # PyTorch 텐서로 반환
        )
        
        # 기본 아이템 구성 (input_ids, attention_mask)
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        
        # 라벨 추가 (훈련용)
        if self.labels is not None:
            label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
            item["labels"] = torch.tensor(label, dtype=torch.long)
        
        return item
