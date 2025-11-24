#!/usr/bin/env python3
"""
파이프라인 테스트 스크립트

파이프라인이 제대로 작동하는지 간단히 테스트합니다.
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """모듈 import 테스트"""
    print("🧪 모듈 import 테스트...")
    
    try:
        # 절대 import 사용
        import config
        import data_processing
        import models
        import trainer
        import utils
        print("✅ 모든 모듈 import 성공!")
        return True
    except ImportError as e:
        print(f"❌ Import 오류: {e}")
        return False


def test_config():
    """설정 클래스 테스트"""
    print("\n🧪 설정 클래스 테스트...")
    
    try:
        import config
        config_obj = config.Config()
        
        # 기본 설정 확인
        assert config_obj.random_state == 42
        assert config_obj.model_name == "beomi/kcbert-base"
        assert config_obj.num_classes == 4
        assert config_obj.focal_alpha == 2.0
        assert config_obj.focal_gamma == 1.5
        
        print("✅ 설정 클래스 테스트 성공!")
        return True
    except Exception as e:
        print(f"❌ 설정 클래스 테스트 실패: {e}")
        return False


def test_focal_loss():
    """Focal Loss 테스트"""
    print("\n🧪 Focal Loss 테스트...")
    
    try:
        import torch
        import models
        
        # Focal Loss 인스턴스 생성
        focal_loss = models.FocalLoss(alpha=1.0, gamma=2.0)
        
        # 테스트 데이터 생성
        batch_size, num_classes = 4, 4
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # 손실 계산
        loss = focal_loss(inputs, targets)
        
        # 결과 확인
        assert loss.item() > 0
        assert loss.shape == torch.Size([])
        
        print("✅ Focal Loss 테스트 성공!")
        return True
    except Exception as e:
        print(f"❌ Focal Loss 테스트 실패: {e}")
        return False


def test_data_processing():
    """데이터 전처리 테스트"""
    print("\n🧪 데이터 전처리 테스트...")
    
    try:
        import data_processing
        
        # 테스트 텍스트
        test_text = "안녕하세요!!! 이것은 테스트입니다... ㅋㅋㅋ"
        
        # 각 전처리 함수 테스트
        normalized = data_processing.normalize_punctuation(test_text)
        cleaned_special = data_processing.clean_special_chars(normalized)
        final_cleaned = data_processing.clean_text(cleaned_special)
        
        # 결과 확인
        assert isinstance(final_cleaned, str)
        assert len(final_cleaned) > 0
        
        print("✅ 데이터 전처리 테스트 성공!")
        return True
    except Exception as e:
        print(f"❌ 데이터 전처리 테스트 실패: {e}")
        return False


def test_utils():
    """유틸리티 함수 테스트"""
    print("\n🧪 유틸리티 함수 테스트...")
    
    try:
        import utils
        import numpy as np
        
        # 테스트 데이터 생성
        predictions = np.array([[0.1, 0.2, 0.6, 0.1], [0.7, 0.1, 0.1, 0.1]])
        labels = np.array([2, 0])
        
        # 메트릭 계산
        metrics = utils.compute_metrics((predictions, labels))
        
        # 결과 확인
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
        
        print("✅ 유틸리티 함수 테스트 성공!")
        return True
    except Exception as e:
        print(f"❌ 유틸리티 함수 테스트 실패: {e}")
        return False


def test_system_info():
    """시스템 정보 출력 테스트"""
    print("\n🧪 시스템 정보 출력 테스트...")
    
    try:
        import utils
        utils.print_system_info()
        print("✅ 시스템 정보 출력 테스트 성공!")
        return True
    except Exception as e:
        print(f"❌ 시스템 정보 출력 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🚀 Focal Loss 파이프라인 테스트 시작")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_focal_loss,
        test_data_processing,
        test_utils,
        test_system_info
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print("🏁 테스트 완료!")
    print(f"✅ 통과: {passed}/{total}")
    print(f"❌ 실패: {total - passed}/{total}")
    print(f"📊 성공률: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! 파이프라인이 정상적으로 작동합니다.")
        return True
    else:
        print(f"\n⚠️ {total - passed}개의 테스트가 실패했습니다. 문제를 확인해주세요.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
