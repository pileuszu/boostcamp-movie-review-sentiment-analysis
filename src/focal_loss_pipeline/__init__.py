"""
Focal Loss 감정 분석 파이프라인

이 패키지는 한국어 영화 리뷰 감정 분석을 위한 Focal Loss 기반 BERT 모델 훈련 파이프라인을 제공합니다.

주요 구성 요소:
- config: 설정 관리
- data_processing: 데이터 로드 및 전처리
- models: Focal Loss, 커스텀 Trainer, Dataset 클래스
- trainer: 모델 훈련 및 평가
- utils: 유틸리티 함수들

사용 예시:
    from focal_loss_pipeline import SentimentAnalysisTrainer, Config
    from focal_loss_pipeline.data_processing import load_and_preprocess_data, split_data
    
    # 설정 로드
    config = Config()
    
    # 데이터 전처리
    df = load_and_preprocess_data(config.data_path)
    train_data, val_data = split_data(df, config.test_size, config.random_state)
    
    # 훈련 실행
    trainer = SentimentAnalysisTrainer(config)
    results = trainer.run_full_pipeline(train_data, val_data)
"""

from config import Config, config
from data_processing import load_and_preprocess_data, split_data
from models import FocalLoss, FocalLossTrainer, ReviewDataset
from trainer import SentimentAnalysisTrainer
from utils import (
    compute_metrics, 
    ReduceLROnPlateauCallback, 
    plot_confusion_matrix, 
    analyze_class_performance,
    print_system_info,
    print_training_summary
)

__version__ = "1.0.0"
__author__ = "Boostcamp AI Tech"

__all__ = [
    "Config",
    "config", 
    "load_and_preprocess_data",
    "split_data",
    "FocalLoss",
    "FocalLossTrainer", 
    "ReviewDataset",
    "SentimentAnalysisTrainer",
    "compute_metrics",
    "ReduceLROnPlateauCallback",
    "plot_confusion_matrix",
    "analyze_class_performance",
    "print_system_info",
    "print_training_summary"
]
