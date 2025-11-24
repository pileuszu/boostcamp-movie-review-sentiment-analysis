#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모든 BERT 모델 결과 비교 및 분석 스크립트
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def get_output_dir():
    """
    실행 경로와 상관없이 프로젝트 루트 기준 output 디렉토리를 올바르게 잡아준다.
    """
    # 이 파일의 위치에서 프로젝트 루트로 감
    this_file = Path(__file__).resolve()
    # src/bert_model_comparison 하위에 있다는 가정
    project_root = this_file.parent.parent.parent
    output_dir = project_root / "output" / "bert_model_comparison"
    return output_dir

def load_submission_results():
    """모든 모델의 제출 결과를 로드"""
    output_dir = get_output_dir()
    results = {}

    model_files = {
        "klue_roberta": "submission_klue_roberta-base.csv",
        "klue_bert": "submission_klue_bert-base.csv", 
        "kykim_bert": "submission_kykim_bert-kor-base.csv",
        "kcbert": "submission_beomi_kcbert-base.csv",
        "koelectra": "submission_monologg_koelectra-base-v3-discriminator.csv"
    }

    for model_name, filename in model_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            results[model_name] = df
            print(f"✅ {model_name}: {len(df):,} 예측 결과 로드")
        else:
            print(f"❌ {model_name}: {filename} 파일을 찾을 수 없습니다")

    return results

def compare_predictions(results):
    """모델별 예측 결과 비교"""
    print("\n" + "="*60)
    print("📊 모델별 예측 결과 비교")
    print("="*60)
    
    # 첫 번째 모델을 기준으로 비교
    if not results:
        print("❌ 비교할 결과가 없습니다")
        return

    first_model = list(results.keys())[0]
    base_df = results[first_model]

    print(f"\n기준 모델: {first_model}")
    print(f"총 예측 수: {len(base_df):,}")
    
    # 각 모델별 예측 분포
    print("\n📈 모델별 예측 분포:")
    print("-" * 40)
    
    for model_name, df in results.items():
        if len(df) == len(base_df):
            pred_counts = None
            if 'label' in df.columns:
                pred_counts = df['label'].value_counts().sort_index()
            elif 'pred' in df.columns:
                pred_counts = df['pred'].value_counts().sort_index()
            else:
                print(f"\n{model_name}: 'label' 또는 'pred' 컬럼을 찾을 수 없습니다.")
                continue

            print(f"\n{model_name}:")
            for label, count in pred_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  라벨 {label}: {count:,}개 ({percentage:.1f}%)")
        else:
            print(f"\n{model_name}: 데이터 크기 불일치 ({len(df):,} vs {len(base_df):,})")
    
    # 모델 간 일치도 분석
    if len(results) > 1:
        print("\n🔄 모델 간 예측 일치도:")
        print("-" * 40)
        
        model_names = list(results.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                df1, df2 = results[model1], results[model2]
                
                if len(df1) == len(df2):
                    # ID 기준으로 정렬하여 비교 (label 또는 pred 컬럼을 자동 판별)
                    df1_sorted = df1.sort_values('ID').reset_index(drop=True)
                    df2_sorted = df2.sort_values('ID').reset_index(drop=True)
                    
                    col1 = 'label' if 'label' in df1_sorted.columns else 'pred'
                    col2 = 'label' if 'label' in df2_sorted.columns else 'pred'
                    matches = (df1_sorted[col1] == df2_sorted[col2]).sum()
                    total = len(df1_sorted)
                    agreement = (matches / total) * 100
                    
                    print(f"{model1} vs {model2}: {matches:,}/{total:,} 일치 ({agreement:.1f}%)")

def analyze_label_distribution(results):
    """라벨 분포 분석"""
    print("\n" + "="*60)
    print("📊 전체 라벨 분포 분석")
    print("="*60)
    
    if not results:
        return
    
    # 모든 모델의 예측을 합쳐서 분석
    all_predictions = []
    for model_name, df in results.items():
        if 'label' in df.columns:
            all_predictions.extend(df['label'].tolist())
        elif 'pred' in df.columns:
            all_predictions.extend(df['pred'].tolist())

    if len(all_predictions) == 0:
        print("\n라벨 예측 정보를 찾을 수 없습니다.")
        return

    unique_labels, counts = np.unique(all_predictions, return_counts=True)
    
    print(f"\n총 예측 수: {len(all_predictions):,}")
    print("\n라벨별 분포:")
    print("-" * 20)
    
    label_names = {0: "강한부정", 1: "약한부정", 2: "약한긍정", 3: "강한긍정"}
    
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(all_predictions)) * 100
        label_name = label_names.get(label, f"라벨 {label}")
        print(f"{label_name}: {count:,}개 ({percentage:.1f}%)")

def main():
    print("🔍 BERT 모델 비교 분석 시작")
    print("="*60)
    
    # 결과 로드
    results = load_submission_results()
    
    if not results:
        print("❌ 분석할 결과 파일이 없습니다")
        print("먼저 run_all_experiments.sh를 실행하여 모델들을 학습시키세요")
        return
    
    # 예측 결과 비교
    compare_predictions(results)
    
    # 라벨 분포 분석
    analyze_label_distribution(results)
    
    print("\n" + "="*60)
    print("✅ 분석 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
