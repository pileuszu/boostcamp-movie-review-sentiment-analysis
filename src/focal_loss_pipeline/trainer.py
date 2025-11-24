"""
모델 훈련 및 평가 모듈
"""

import os
import wandb
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from typing import Tuple, Dict, Any

from config import Config
from models import FocalLossTrainer, ReviewDataset, calculate_class_weights
from utils import compute_metrics, ReduceLROnPlateauCallback, print_training_summary


class SentimentAnalysisTrainer:
    """감정 분석 모델 훈련 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # 랜덤 시드 설정
        set_seed(config.random_state)
        print(f"랜덤 시드 {config.random_state}로 설정 완료")
        
        # 시스템 정보 출력
        from utils import print_system_info
        print_system_info()
    
    def setup_model_and_tokenizer(self) -> None:
        """모델 및 토크나이저 설정"""
        print(f"\n🤖 모델 로딩: {self.config.model_name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # 모델 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        )
        
        print(f"토크나이저 로드 완료: {self.tokenizer.__class__.__name__}")
        print(f"모델 로드 완료: {self.model.__class__.__name__}")
        print(f"분류 클래스 수: {self.config.num_classes}")
        print(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_datasets(self, train_data, val_data) -> Tuple[ReviewDataset, ReviewDataset]:
        """데이터셋 생성"""
        print(f"\n데이터셋 생성 중...")
        
        train_dataset = ReviewDataset(
            train_data["review_cleaned"],
            train_data["label"],
            self.tokenizer,
            max_length=self.config.max_length
        )
        
        val_dataset = ReviewDataset(
            val_data["review_cleaned"],
            val_data["label"],
            self.tokenizer,
            max_length=self.config.max_length
        )
        
        print(f"훈련 데이터셋 크기: {len(train_dataset):,}")
        print(f"검증 데이터셋 크기: {len(val_dataset):,}")
        
        # 데이터셋 샘플 확인
        sample = train_dataset[0]
        print(f"\n데이터셋 샘플:")
        print(f"input_ids shape: {sample['input_ids'].shape}")
        print(f"attention_mask shape: {sample['attention_mask'].shape}")
        print(f"label: {sample['labels'].item()}")
        
        return train_dataset, val_dataset
    
    def setup_wandb(self) -> None:
        """Wandb 설정"""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=f"focal_loss_alpha{self.config.focal_alpha}_gamma{self.config.focal_gamma}",
                config={
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "epochs": self.config.num_epochs,
                    "focal_alpha": self.config.focal_alpha,
                    "focal_gamma": self.config.focal_gamma,
                    "model_name": self.config.model_name,
                    "max_grad_norm": self.config.max_grad_norm,
                    "weight_decay": self.config.weight_decay,
                }
            )
    
    def create_training_args(self) -> TrainingArguments:
        """훈련 인자 설정"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=self.config.log_dir,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="epoch" if self.config.save_model else "no",
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2 if self.config.save_model else 0,
            seed=self.config.random_state,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=False,
            max_grad_norm=self.config.max_grad_norm,
            report_to="wandb" if self.config.use_wandb else None,
            run_name=f"focal_loss_alpha{self.config.focal_alpha}_gamma{self.config.focal_gamma}",
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            optim="adamw_torch",
        )
        
        print("TrainingArguments 설정 완료!")
        return training_args
    
    def create_trainer(self, train_dataset: ReviewDataset, val_dataset: ReviewDataset, 
                      training_args: TrainingArguments) -> FocalLossTrainer:
        """훈련기 생성"""
        # 클래스 가중치 계산
        print("클래스 가중치 계산 중...")
        train_labels = torch.tensor([train_dataset[i]['labels'].item() for i in range(len(train_dataset))])
        class_weights = calculate_class_weights(train_labels, self.config.num_classes)
        
        # 클래스 가중치를 모델과 같은 디바이스로 이동
        device = next(self.model.parameters()).device
        class_weights = class_weights.to(device)
        
        print(f"클래스 가중치: {class_weights.tolist()}")
        print(f"클래스별 샘플 수: {torch.bincount(train_labels).tolist()}")
        print(f"클래스 가중치 디바이스: {class_weights.device}")
        
        # 콜백 설정
        callbacks = [
            ReduceLROnPlateauCallback(factor=0.5, patience=2),
            EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            )
        ]
        
        # Focal Loss를 사용하는 Trainer 초기화
        trainer = FocalLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
            focal_loss_alpha=self.config.focal_alpha,
            focal_loss_gamma=self.config.focal_gamma,
            class_weights=class_weights,
            callbacks=callbacks
        )
        
        print("Focal Loss Trainer 초기화 완료!")
        print(f"훈련 샘플: {len(train_dataset):,}개")
        print(f"검증 샘플: {len(val_dataset):,}개")
        
        return trainer
    
    def train(self, train_dataset: ReviewDataset, val_dataset: ReviewDataset) -> FocalLossTrainer:
        """모델 훈련"""
        print("\n" + "=" * 50)
        print("Focal Loss 모델 훈련 시작")
        print("=" * 50)
        
        # Wandb 설정
        self.setup_wandb()
        
        # 훈련 인자 설정
        training_args = self.create_training_args()
        
        # 훈련기 생성
        self.trainer = self.create_trainer(train_dataset, val_dataset, training_args)
        
        # 훈련 시작
        self.trainer.train()
        
        print("\n훈련 완료!")
        
        return self.trainer
    
    def evaluate(self) -> Dict[str, float]:
        """모델 평가"""
        print("\n" + "=" * 50)
        print("Focal Loss 모델 최종 평가")
        print("=" * 50)
        
        eval_results = self.trainer.evaluate()
        
        print_training_summary(self.trainer, eval_results)
        
        return eval_results
    
    def save_model(self) -> None:
        """모델 저장"""
        if self.config.save_model:
            try:
                print(f"\n모델 저장 중: {self.config.model_save_path}")
                
                # 디렉토리 생성
                os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
                
                # 모델 및 토크나이저 저장
                self.trainer.save_model(self.config.model_save_path)
                self.tokenizer.save_pretrained(self.config.model_save_path)
                
                print(f"모델 저장 완료: {self.config.model_save_path}")
                
                # 저장된 파일 확인
                if os.path.exists(self.config.model_save_path):
                    saved_files = os.listdir(self.config.model_save_path)
                    print(f"저장된 파일들: {saved_files}")
                    
            except Exception as e:
                print(f"모델 저장 실패: {str(e)}")
        else:
            print("모델 저장이 비활성화되어 있습니다.")
    
    def run_full_pipeline(self, train_data, val_data) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        # 모델 및 토크나이저 설정
        self.setup_model_and_tokenizer()
        
        # 데이터셋 생성
        train_dataset, val_dataset = self.create_datasets(train_data, val_data)
        
        # 훈련
        trainer = self.train(train_dataset, val_dataset)
        
        # 평가
        eval_results = self.evaluate()
        
        # 모델 저장
        self.save_model()
        
        print("\n모델 훈련 및 평가 완료!")
        
        return {
            'trainer': trainer,
            'eval_results': eval_results,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset
        }
