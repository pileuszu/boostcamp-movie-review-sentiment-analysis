#!/usr/bin/env python3
"""
Focal Loss 감정 분석 파이프라인 메인 실행 스크립트

이 스크립트는 Jupyter 노트북의 모든 기능을 background에서 실행할 수 있도록 
모듈화된 파이프라인으로 변환한 것입니다.

사용법:
    python main.py [--config CONFIG_FILE] [--data_path DATA_PATH] [--output_dir OUTPUT_DIR]

예시:
    # 기본 설정으로 실행
    python main.py
    
    # 커스텀 설정으로 실행
    python main.py --data_path data/raw/train.csv --output_dir ./results --focal_alpha 2.0 --focal_gamma 1.5
"""

import argparse
import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import Config
from data_processing import load_and_preprocess_data, split_data_with_augmentation
from trainer import SentimentAnalysisTrainer
from utils import plot_confusion_matrix, analyze_class_performance


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Focal Loss 감정 분석 파이프라인")
    
    # 데이터 관련 인자
    parser.add_argument("--data_path", type=str, default="/data/ephemeral/repos/boostcamp-movie-review-sentiment-analysis/data/raw/train.csv",
                       help="훈련 데이터 경로")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="검증 데이터 비율")
    
    # 모델 관련 인자
    parser.add_argument("--model_name", type=str, default="beomi/kcbert-base",
                       help="사전 훈련된 모델 이름")
    parser.add_argument("--max_length", type=int, default=128,
                       help="최대 시퀀스 길이")
    
    # 훈련 관련 인자
    parser.add_argument("--batch_size", type=int, default=64,
                       help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="학습률")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="에포크 수")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="가중치 감쇠")
    
    # Focal Loss 관련 인자
    parser.add_argument("--focal_alpha", type=float, default=2.0,
                       help="Focal Loss alpha 파라미터")
    parser.add_argument("--focal_gamma", type=float, default=1.5,
                       help="Focal Loss gamma 파라미터")
    
    # 출력 관련 인자
    parser.add_argument("--output_dir", type=str, default="./focal_loss_results",
                       help="출력 디렉토리")
    parser.add_argument("--model_save_path", type=str, default="model/focal_loss/focal_loss_model",
                       help="모델 저장 경로")
    
    # 기타 인자
    parser.add_argument("--random_state", type=int, default=42,
                       help="랜덤 시드")
    parser.add_argument("--save_model", action="store_true", default=True,
                       help="모델 저장 여부")
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Wandb 사용 여부")
    parser.add_argument("--wandb_project", type=str, default="focal-loss-sentiment-analysis",
                       help="Wandb 프로젝트 이름")
    
    return parser.parse_args()


def create_config_from_args(args):
    """명령행 인자로부터 Config 객체 생성"""
    config = Config()
    
    # 데이터 관련 설정
    config.data_path = args.data_path
    config.test_size = args.test_size
    
    # 모델 관련 설정
    config.model_name = args.model_name
    config.max_length = args.max_length
    
    # 훈련 관련 설정
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.weight_decay = args.weight_decay
    
    # Focal Loss 관련 설정
    config.focal_alpha = args.focal_alpha
    config.focal_gamma = args.focal_gamma
    
    # 출력 관련 설정
    config.output_dir = args.output_dir
    config.model_save_path = args.model_save_path
    
    # 기타 설정
    config.random_state = args.random_state
    config.save_model = args.save_model
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    
    return config


def main():
    """메인 실행 함수"""
    print("🚀 Focal Loss 감정 분석 파이프라인 시작")
    print("=" * 60)
    
    # 명령행 인자 파싱
    args = parse_args()
    
    # 설정 생성
    config = create_config_from_args(args)
    
    print(f"📊 설정 정보:")
    print(f"  - 데이터 경로: {config.data_path}")
    print(f"  - 모델: {config.model_name}")
    print(f"  - 배치 크기: {config.batch_size}")
    print(f"  - 학습률: {config.learning_rate}")
    print(f"  - 에포크 수: {config.num_epochs}")
    print(f"  - Focal Alpha: {config.focal_alpha}")
    print(f"  - Focal Gamma: {config.focal_gamma}")
    print(f"  - 출력 디렉토리: {config.output_dir}")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드 및 전처리
        print("\n📁 1단계: 데이터 로드 및 전처리")
        df = load_and_preprocess_data(config.data_path)
        
        # 2. 데이터 분할 (증강 포함)
        print("\n✂️ 2단계: 데이터 분할 및 증강")
        train_data, val_data = split_data_with_augmentation(df, config.test_size, config.random_state, use_augmentation=True)
        
        # 3. 훈련기 초기화
        print("\n🤖 3단계: 훈련기 초기화")
        trainer = SentimentAnalysisTrainer(config)
        
        # 4. 전체 파이프라인 실행
        print("\n🏃 4단계: 모델 훈련 및 평가")
        results = trainer.run_full_pipeline(train_data, val_data)
        
        # 5. 추가 분석 (선택사항)
        print("\n📈 5단계: 추가 분석")
        
        # 혼동 행렬 생성
        class_names = ["부정", "중립", "긍정", "강한 긍정"]
        confusion_matrix_path = os.path.join(config.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            results['trainer'], 
            results['val_dataset'], 
            class_names, 
            "Focal Loss - Confusion Matrix",
            save_path=confusion_matrix_path
        )
        
        # 클래스별 성능 분석
        analyze_class_performance(
            results['trainer'], 
            results['val_dataset'], 
            "Focal Loss",
            class_names
        )
        
        print("\n✅ 파이프라인 실행 완료!")
        print(f"📁 결과 저장 위치: {config.output_dir}")
        print(f"💾 모델 저장 위치: {config.model_save_path}")
        
        # 최종 성능 요약
        eval_results = results['eval_results']
        print(f"\n🎯 최종 성능:")
        print(f"  - 정확도: {eval_results['eval_accuracy']:.4f}")
        print(f"  - F1-Score: {eval_results['eval_f1']:.4f}")
        print(f"  - 검증 손실: {eval_results['eval_loss']:.4f}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
