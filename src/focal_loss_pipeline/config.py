"""
Focal Loss 감정 분석 파이프라인 설정 모듈
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """전체 파이프라인 설정"""
    
    # 기본 설정
    random_state: int = 42
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # 데이터 설정
    data_path: str = "/data/ephemeral/repos/boostcamp-movie-review-sentiment-analysis/data/raw/train.csv"
    test_size: float = 0.2
    max_length: int = 128
    
    # 모델 설정 (과적합 방지 및 성능 향상)
    model_name: str = "beomi/kcbert-base"
    num_classes: int = 4
    hidden_dropout_prob: float = 0.3  # 드롭아웃 증가
    attention_probs_dropout_prob: float = 0.3  # 어텐션 드롭아웃 증가
    
    # 훈련 설정 (성능 향상을 위해 조정)
    batch_size: int = 32  # 더 작은 배치로 안정적인 학습
    learning_rate: float = 2e-5  # 학습률 증가
    num_epochs: int = 10  # 에포크 수 증가
    weight_decay: float = 0.01  # 가중치 감쇠 감소
    max_grad_norm: float = 1.0
    warmup_steps: int = 500  # 워밍업 스텝 감소
    
    # Focal Loss 설정 (클래스 불균형에 맞게 조정)
    focal_alpha: float = 3.0  # 중립 클래스에 더 집중
    focal_gamma: float = 2.0  # 어려운 샘플에 더 집중
    
    # 훈련 옵션
    save_model: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 4
    logging_steps: int = 500
    eval_steps: int = 500
    
    # Early Stopping 설정 (더 관대하게 조정)
    early_stopping_patience: int = 8  # 더 많은 에포크 기다림
    early_stopping_threshold: float = 0.0005  # 더 작은 개선도 인정
    
    # Learning Rate Scheduler 설정
    lr_scheduler_type: str = "cosine"  # 더 부드러운 스케줄링
    warmup_ratio: float = 0.05  # 워밍업 비율 감소
    
    # 출력 경로
    output_dir: str = "./focal_loss_results"
    model_save_path: str = "model/focal_loss/focal_loss_model"
    log_dir: str = "./logs"
    
    # Wandb 설정
    use_wandb: bool = True
    wandb_project: str = "focal-loss-sentiment-analysis"
    
    def __post_init__(self):
        """설정 검증 및 후처리"""
        # 환경 변수 설정
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)


# 기본 설정 인스턴스
config = Config()
