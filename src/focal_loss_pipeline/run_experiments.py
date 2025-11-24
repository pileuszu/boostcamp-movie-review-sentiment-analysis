#!/usr/bin/env python3
"""
실험 실행 스크립트

다양한 하이퍼파라미터 조합으로 실험을 자동으로 실행하는 스크립트입니다.
"""

import os
import sys
import itertools
from pathlib import Path
import subprocess
import time
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import Config


def run_experiment(config_dict, experiment_name, base_output_dir="./experiments"):
    """단일 실험 실행"""
    print(f"\n🧪 실험 시작: {experiment_name}")
    print("=" * 60)
    
    # 출력 디렉토리 설정
    output_dir = os.path.join(base_output_dir, experiment_name)
    model_save_path = os.path.join(output_dir, "model")
    
    # 명령어 구성
    cmd = [
        "python", "main.py",
        "--output_dir", output_dir,
        "--model_save_path", model_save_path,
        "--focal_alpha", str(config_dict["focal_alpha"]),
        "--focal_gamma", str(config_dict["focal_gamma"]),
        "--learning_rate", str(config_dict["learning_rate"]),
        "--batch_size", str(config_dict["batch_size"]),
        "--num_epochs", str(config_dict["num_epochs"]),
    ]
    
    # Wandb 설정
    if config_dict.get("use_wandb", True):
        cmd.extend(["--use_wandb", "--wandb_project", f"focal-loss-experiments-{experiment_name}"])
    
    print(f"실행 명령어: {' '.join(cmd)}")
    
    # 실험 실행
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"✅ 실험 완료: {experiment_name}")
        print(f"⏱️ 실행 시간: {end_time - start_time:.2f}초")
        
        return {
            "experiment_name": experiment_name,
            "config": config_dict,
            "success": True,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"❌ 실험 실패: {experiment_name}")
        print(f"오류: {e.stderr}")
        
        return {
            "experiment_name": experiment_name,
            "config": config_dict,
            "success": False,
            "duration": end_time - start_time,
            "stdout": e.stdout,
            "stderr": e.stderr
        }


def main():
    """메인 실험 실행 함수"""
    print("🔬 Focal Loss 실험 자동 실행")
    print("=" * 60)
    
    # 실험 설정
    experiments = [
        # 기본 실험
        {
            "name": "baseline",
            "config": {
                "focal_alpha": 1.0,
                "focal_gamma": 0.0,  # Cross Entropy Loss
                "learning_rate": 1e-5,
                "batch_size": 64,
                "num_epochs": 5,
                "use_wandb": True
            }
        },
        
        # Focal Loss 실험들
        {
            "name": "focal_alpha_1_gamma_1",
            "config": {
                "focal_alpha": 1.0,
                "focal_gamma": 1.0,
                "learning_rate": 1e-5,
                "batch_size": 64,
                "num_epochs": 5,
                "use_wandb": True
            }
        },
        {
            "name": "focal_alpha_2_gamma_1",
            "config": {
                "focal_alpha": 2.0,
                "focal_gamma": 1.0,
                "learning_rate": 1e-5,
                "batch_size": 64,
                "num_epochs": 5,
                "use_wandb": True
            }
        },
        {
            "name": "focal_alpha_2_gamma_1_5",
            "config": {
                "focal_alpha": 2.0,
                "focal_gamma": 1.5,
                "learning_rate": 1e-5,
                "batch_size": 64,
                "num_epochs": 5,
                "use_wandb": True
            }
        },
        {
            "name": "focal_alpha_2_gamma_2",
            "config": {
                "focal_alpha": 2.0,
                "focal_gamma": 2.0,
                "learning_rate": 1e-5,
                "batch_size": 64,
                "num_epochs": 5,
                "use_wandb": True
            }
        },
        
        # 학습률 실험
        {
            "name": "focal_lr_2e5",
            "config": {
                "focal_alpha": 2.0,
                "focal_gamma": 1.5,
                "learning_rate": 2e-5,
                "batch_size": 64,
                "num_epochs": 5,
                "use_wandb": True
            }
        },
        {
            "name": "focal_lr_5e6",
            "config": {
                "focal_alpha": 2.0,
                "focal_gamma": 1.5,
                "learning_rate": 5e-6,
                "batch_size": 64,
                "num_epochs": 5,
                "use_wandb": True
            }
        },
        
        # 배치 크기 실험
        {
            "name": "focal_batch_32",
            "config": {
                "focal_alpha": 2.0,
                "focal_gamma": 1.5,
                "learning_rate": 1e-5,
                "batch_size": 32,
                "num_epochs": 5,
                "use_wandb": True
            }
        },
        {
            "name": "focal_batch_128",
            "config": {
                "focal_alpha": 2.0,
                "focal_gamma": 1.5,
                "learning_rate": 1e-5,
                "batch_size": 128,
                "num_epochs": 5,
                "use_wandb": True
            }
        },
    ]
    
    # 실험 결과 저장
    results = []
    start_time = datetime.now()
    
    print(f"📅 실험 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔢 총 실험 수: {len(experiments)}")
    
    # 각 실험 실행
    for i, experiment in enumerate(experiments, 1):
        print(f"\n📊 진행률: {i}/{len(experiments)}")
        
        result = run_experiment(
            experiment["config"], 
            experiment["name"],
            base_output_dir="./experiments"
        )
        results.append(result)
        
        # 실험 간 잠시 대기 (시스템 안정성을 위해)
        if i < len(experiments):
            print("⏳ 다음 실험까지 10초 대기...")
            time.sleep(10)
    
    # 실험 결과 요약
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("🏁 모든 실험 완료!")
    print("=" * 60)
    print(f"📅 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ 총 소요 시간: {total_duration/3600:.2f}시간")
    
    # 성공/실패 통계
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"\n📈 실험 결과 요약:")
    print(f"  ✅ 성공: {successful}개")
    print(f"  ❌ 실패: {failed}개")
    print(f"  📊 성공률: {successful/len(results)*100:.1f}%")
    
    # 실패한 실험 목록
    if failed > 0:
        print(f"\n❌ 실패한 실험:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['experiment_name']}")
    
    # 결과를 파일로 저장
    results_file = f"experiment_results_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Focal Loss 실험 결과\n")
        f.write("=" * 60 + "\n")
        f.write(f"실험 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"실험 종료: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 소요 시간: {total_duration/3600:.2f}시간\n")
        f.write(f"성공: {successful}개, 실패: {failed}개\n\n")
        
        for result in results:
            f.write(f"실험: {result['experiment_name']}\n")
            f.write(f"성공: {'Yes' if result['success'] else 'No'}\n")
            f.write(f"소요 시간: {result['duration']:.2f}초\n")
            f.write(f"설정: {result['config']}\n")
            if not result['success']:
                f.write(f"오류: {result['stderr']}\n")
            f.write("-" * 40 + "\n")
    
    print(f"\n💾 결과 저장: {results_file}")


if __name__ == "__main__":
    main()
