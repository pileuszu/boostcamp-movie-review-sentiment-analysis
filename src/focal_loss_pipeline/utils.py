"""
유틸리티 함수 모듈
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import TrainerCallback
from typing import Dict, Any, List
import warnings

warnings.filterwarnings('ignore')


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    평가 지표 계산 함수
    
    Args:
        eval_pred: (predictions, labels) 튜플
    
    Returns:
        dict: 계산된 평가 지표들
    """
    predictions, labels = eval_pred
    
    # 예측값을 클래스 인덱스로 변환
    predictions = np.argmax(predictions, axis=1)
    
    # 정확도 계산
    accuracy = accuracy_score(labels, predictions)
    
    # F1 점수 계산 (macro average)
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


class ReduceLROnPlateauCallback(TrainerCallback):
    """학습률 감소 콜백"""
    
    def __init__(self, factor: float = 0.5, patience: int = 2, min_lr: float = 1e-7):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_metric = None
        self.num_bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics.get("eval_loss")
        optimizer = kwargs["model"].optimizer
        if current_metric is None or optimizer is None:
            return control

        if self.best_metric is None or current_metric < self.best_metric:
            self.best_metric = current_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for group in optimizer.param_groups:
                old_lr = group["lr"]
                new_lr = max(old_lr * self.factor, self.min_lr)
                group["lr"] = new_lr
                print(f"🔽 Reduce LR: {old_lr:.6f} → {new_lr:.6f}")
            self.num_bad_epochs = 0
        return control


def plot_confusion_matrix(trainer, dataset, target_names: List[str], title: str = "Confusion Matrix", 
                         save_path: str = None) -> None:
    """혼동 행렬 시각화"""
    preds_output = trainer.predict(dataset)
    y_true = preds_output.label_ids
    y_pred = np.argmax(preds_output.predictions, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"혼동 행렬 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_class_performance(trainer, dataset, model_name: str, 
                            target_names: List[str]) -> Dict[str, Any]:
    """클래스별 성능 분석"""
    preds_output = trainer.predict(dataset)
    y_true = preds_output.label_ids
    y_pred = np.argmax(preds_output.predictions, axis=1)
    
    # 분류 리포트 생성
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    print(f"\n=== {model_name} 클래스별 성능 ===")
    
    # 클래스별 성능 출력
    for class_name in target_names:
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            
            print(f"{class_name}:")
            print(f"  정밀도: {precision:.4f}")
            print(f"  재현율: {recall:.4f}")
            print(f"  F1점수: {f1:.4f}")
            print(f"  샘플수: {support}")
            print()
    
    return report


def print_system_info() -> None:
    """시스템 정보 출력"""
    import sys
    import platform
    import torch
    import pandas as pd
    import numpy as np
    from transformers import __version__ as transformers_version
    from sklearn import __version__ as sklearn_version
    
    print("=== 라이브러리 버전 정보 ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers_version}")
    print(f"sklearn: {sklearn_version}")

    # GPU 사용 가능 여부 확인
    print("\n=== PyTorch GPU 지원 정보 ===")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"현재 GPU: {torch.cuda.current_device()}")
        print(f"GPU 이름: {torch.cuda.get_device_name()}")
    else:
        print("CPU에서 실행 중")


def print_training_summary(trainer, eval_results: Dict[str, float]) -> None:
    """훈련 결과 요약 출력"""
    print("\n" + "=" * 50)
    print("훈련 완료!")
    print("=" * 50)
    
    print(f"\n최종 성능 결과:")
    print("=" * 40)
    print(f"정확도 (Accuracy): {eval_results['eval_accuracy']:.4f}")
    print(f"F1-Score: {eval_results['eval_f1']:.4f}")
    print(f"검증 손실 (Loss): {eval_results['eval_loss']:.4f}")
    print("=" * 40)
