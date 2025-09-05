#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量数据分析工具
对testdata目录下所有JSON文件进行统计分析，检测异常点
"""

import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from datetime import datetime


def load_patient_measurements(json_file_path):
    """
    从JSON文件加载患者测量数据
    
    Returns:
    tuple: (measurements, patient_info) 包含所有眼睛的测量数据和患者信息
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        measurements = {}
        patient_info = {
            'name': data.get('name', 'Unknown'),
            'age': data.get('age', 'Unknown'),
            'gender': data.get('gender', 'Unknown'),
            'date': data.get('date', 'Unknown'),
            'file': os.path.basename(json_file_path)
        }
        
        for eye in ['left', 'right']:
            if eye in data and 'points' in data[eye]:
                eye_measurements = []
                for point in data[eye]['points']:
                    if point.get('result'):
                        # 使用中位数数据
                        if 'median' in point:
                            sphere = point['median']['sphere']
                        else:
                            # 如果没有预计算的中位数，计算现场中位数
                            all_spheres = [r['sphere'] for r in point['result']]
                            sphere = np.median(all_spheres)
                        
                        # 排除 (50°, 180°) 位置的数据点
                        if not (point['offaxis'] == 50 and point['meridian'] == 180):
                            eye_measurements.append({
                                'offaxis': point['offaxis'],
                                'meridian': point['meridian'],
                                'sphere': sphere,
                                'position_key': f"({point['offaxis']}°, {point['meridian']}°)"
                            })
                
                if eye_measurements:  # 只添加有数据的眼睛
                    measurements[eye] = eye_measurements
        
        return measurements, patient_info
        
    except Exception as e:
        print(f"Error loading {json_file_path}: {e}")
        return None, None


def detect_outliers_modified_zscore(values, threshold=3.5):
    """
    使用修正Z分数法检测异常点
    
    Args:
        values: 数据数组
        threshold: 异常点阈值，默认3.5
        
    Returns:
        tuple: (异常点掩码, 修正Z分数数组)
    """
    values = np.array(values)
    if len(values) < 3:
        return np.zeros(len(values), dtype=bool), np.zeros(len(values))
    
    median_val = np.median(values)
    mad = np.median(np.abs(values - median_val))  # 中位数绝对偏差
    
    if mad == 0:
        return np.zeros(len(values), dtype=bool), np.zeros(len(values))
    
    # 修正Z分数
    modified_z_scores = 0.6745 * (values - median_val) / mad
    outliers_mask = np.abs(modified_z_scores) > threshold
    
    return outliers_mask, modified_z_scores


def analyze_single_patient(measurements, patient_info):
    """
    分析单个患者的数据
    
    Returns:
    dict: 分析结果
    """
    results = {
        'patient_info': patient_info,
        'eye_results': {},
        'has_outliers': False,
        'total_outliers': 0
    }
    
    for eye, eye_data in measurements.items():
        if not eye_data:
            continue
            
        # 提取球镜度数据
        spheres = np.array([point['sphere'] for point in eye_data])
        positions = [point['position_key'] for point in eye_data]
        
        # 基础统计
        mean_sphere = np.mean(spheres)
        std_sphere = np.std(spheres, ddof=1) if len(spheres) > 1 else 0
        var_sphere = np.var(spheres, ddof=1) if len(spheres) > 1 else 0
        median_sphere = np.median(spheres)
        min_sphere = np.min(spheres)
        max_sphere = np.max(spheres)
        
        # 异常点检测
        outliers_mask, modified_z_scores = detect_outliers_modified_zscore(spheres)
        outliers_indices = np.where(outliers_mask)[0]
        
        # 收集异常点信息
        outlier_details = []
        for idx in outliers_indices:
            outlier_details.append({
                'position': positions[idx],
                'sphere_value': spheres[idx],
                'modified_z_score': modified_z_scores[idx],
                'offaxis': eye_data[idx]['offaxis'],
                'meridian': eye_data[idx]['meridian']
            })
        
        eye_result = {
            'sample_count': len(spheres),
            'mean': mean_sphere,
            'std': std_sphere,
            'variance': var_sphere,
            'median': median_sphere,
            'min': min_sphere,
            'max': max_sphere,
            'range': max_sphere - min_sphere,
            'outlier_count': len(outliers_indices),
            'outlier_details': outlier_details,
            'has_outliers': len(outliers_indices) > 0
        }
        
        results['eye_results'][eye] = eye_result
        
        if eye_result['has_outliers']:
            results['has_outliers'] = True
            results['total_outliers'] += eye_result['outlier_count']
    
    return results


def batch_analyze_all_patients(data_folder="testdata_aier"):
    """
    批量分析所有患者数据
    
    Returns:
    dict: 所有患者的分析结果
    """
    data_path = Path(data_folder)
    if not data_path.exists():
        print(f"Data folder {data_folder} does not exist!")
        return {}
    
    json_files = list(data_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_folder}")
        return {}
    
    print(f"Found {len(json_files)} JSON files for analysis")
    print("注意: 排除 (50°, 180°) 位置的数据点不参与统计分析")
    print("=" * 60)
    
    all_results = {}
    patients_with_outliers = []
    total_patients = 0
    total_eyes_analyzed = 0
    total_measurements = 0
    global_outlier_count = 0
    
    for json_file in sorted(json_files):
        measurements, patient_info = load_patient_measurements(json_file)
        
        if measurements is None:
            continue
            
        total_patients += 1
        analysis_result = analyze_single_patient(measurements, patient_info)
        all_results[json_file.name] = analysis_result
        
        # 统计信息
        for eye, eye_result in analysis_result['eye_results'].items():
            total_eyes_analyzed += 1
            total_measurements += eye_result['sample_count']
            global_outlier_count += eye_result['outlier_count']
        
        if analysis_result['has_outliers']:
            patients_with_outliers.append({
                'file': json_file.name,
                'patient': patient_info['name'],
                'total_outliers': analysis_result['total_outliers'],
                'details': analysis_result
            })
    
    # 打印分析结果
    print(f"\n[分析完成] 批量分析完成!")
    print(f"总患者数: {total_patients}")
    print(f"总分析眼数: {total_eyes_analyzed}")
    print(f"总测量点数: {total_measurements}")
    print(f"总异常点数: {global_outlier_count}")
    print(f"有异常点的患者数: {len(patients_with_outliers)}")
    
    if patients_with_outliers:
        print(f"\n[异常点检测] 检测到异常点的患者详情:")
        print("=" * 60)
        
        for patient in patients_with_outliers:
            details = patient['details']
            patient_info = details['patient_info']
            
            print(f"\n[文件] {patient['file']}")
            print(f"[患者] {patient_info['name']} (年龄: {patient_info['age']}, 性别: {patient_info['gender']})")
            print(f"[日期] {patient_info['date']}")
            print(f"[异常点总数] {patient['total_outliers']}")
            
            for eye, eye_result in details['eye_results'].items():
                if eye_result['has_outliers']:
                    print(f"\n  [{eye.capitalize()} 眼]")
                    print(f"    [统计信息]")
                    print(f"      样本数: {eye_result['sample_count']}")
                    print(f"      平均值: {eye_result['mean']:.3f} D")
                    print(f"      标准差: {eye_result['std']:.3f} D")
                    print(f"      方差: {eye_result['variance']:.3f} D^2")
                    print(f"      中位数: {eye_result['median']:.3f} D")
                    print(f"      范围: [{eye_result['min']:.3f}, {eye_result['max']:.3f}] D")
                    
                    print(f"    [异常点详情] ({eye_result['outlier_count']} 个):")
                    for i, outlier in enumerate(eye_result['outlier_details'], 1):
                        print(f"      异常点 {i}: {outlier['sphere_value']:.3f} D")
                        print(f"        位置: {outlier['position']}")
                        print(f"        修正Z分数: {outlier['modified_z_score']:.3f}")
        
    else:
        print(f"\n[正常] 所有患者数据均正常，未发现异常点")
    
    print(f"\n" + "=" * 60)
    
    return all_results


def save_analysis_report(results, output_file="batch_analysis_report_filtered.csv"):
    """
    将分析结果保存为CSV报告 - 使用pandas提供更好的数据处理
    """
    report_data = []
    
    for filename, result in results.items():
        patient_info = result['patient_info']
        
        for eye, eye_result in result['eye_results'].items():
            row = {
                'File': filename,
                'Patient_Name': patient_info['name'],
                'Age': patient_info['age'],
                'Gender': patient_info['gender'],
                'Date': patient_info['date'],
                'Eye': eye.capitalize(),
                'Sample_Count': eye_result['sample_count'],
                'Mean_Sphere': round(eye_result['mean'], 3),
                'Std_Sphere': round(eye_result['std'], 3),
                'Variance_Sphere': round(eye_result['variance'], 3),
                'Median_Sphere': round(eye_result['median'], 3),
                'Min_Sphere': round(eye_result['min'], 3),
                'Max_Sphere': round(eye_result['max'], 3),
                'Range_Sphere': round(eye_result['range'], 3),
                'Outlier_Count': eye_result['outlier_count'],
                'Has_Outliers': 'Yes' if eye_result['has_outliers'] else 'No',
                'Outlier_Positions': '; '.join([d['position'] for d in eye_result['outlier_details']]) if eye_result['outlier_details'] else '',
                'Outlier_Values': '; '.join([f"{d['sphere_value']:.3f}" for d in eye_result['outlier_details']]) if eye_result['outlier_details'] else ''
            }
            report_data.append(row)
    
    if report_data:
        # 使用pandas保存和处理数据
        df = pd.DataFrame(report_data)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n[报告] 分析报告已保存至: {output_file}")
        
        # 使用pandas提供额外的数据统计
        return generate_summary_statistics(df)
    
    return None


def generate_summary_statistics(df):
    """
    生成汇总统计信息
    """
    print(f"\n[数据统计摘要] (已排除 50°, 180° 位置)")
    print("=" * 50)
    
    # 基础统计
    total_patients = df['Patient_Name'].nunique()
    total_eyes = len(df)
    total_measurements = df['Sample_Count'].sum()
    total_outliers = df['Outlier_Count'].sum()
    
    print(f"患者总数: {total_patients}")
    print(f"眼部数据: {total_eyes}")
    print(f"测量点总数: {total_measurements}")
    print(f"异常点总数: {total_outliers}")
    
    # 异常点分析
    outlier_patients = df[df['Has_Outliers'] == 'Yes']['Patient_Name'].nunique()
    outlier_percentage = (outlier_patients / total_patients) * 100
    print(f"有异常点的患者: {outlier_patients} ({outlier_percentage:.1f}%)")
    
    # 年龄分析
    print(f"\n[年龄分布]")
    age_stats = df.groupby('Age')['Has_Outliers'].apply(lambda x: (x == 'Yes').sum()).reset_index()
    age_stats['Total'] = df.groupby('Age').size().reset_index(drop=True)
    age_stats['Outlier_Rate'] = (age_stats['Has_Outliers'] / age_stats['Total'] * 100).round(1)
    print(age_stats.to_string(index=False))
    
    # 性别分析
    print(f"\n[性别分析]")
    gender_stats = df.groupby('Gender').agg({
        'Has_Outliers': lambda x: (x == 'Yes').sum(),
        'Patient_Name': 'count'
    }).reset_index()
    gender_stats['Outlier_Rate'] = (gender_stats['Has_Outliers'] / gender_stats['Patient_Name'] * 100).round(1)
    gender_stats.columns = ['Gender', 'Outliers', 'Total', 'Outlier_Rate(%)']
    print(gender_stats.to_string(index=False))
    
    # 眼别分析
    print(f"\n[眼别分析]")
    eye_stats = df.groupby('Eye').agg({
        'Has_Outliers': lambda x: (x == 'Yes').sum(),
        'Patient_Name': 'count'
    }).reset_index()
    eye_stats['Outlier_Rate'] = (eye_stats['Has_Outliers'] / eye_stats['Patient_Name'] * 100).round(1)
    eye_stats.columns = ['Eye', 'Outliers', 'Total', 'Outlier_Rate(%)']
    print(eye_stats.to_string(index=False))
    
    # 球镜度统计
    print(f"\n[球镜度统计]")
    sphere_stats = df[['Mean_Sphere', 'Std_Sphere', 'Min_Sphere', 'Max_Sphere']].describe()
    print(sphere_stats.round(3))
    
    # 最严重异常点
    print(f"\n[最严重异常点 (Top 10)]")
    outlier_df = df[df['Has_Outliers'] == 'Yes'].copy()
    if not outlier_df.empty:
        # 解析异常值
        outlier_values = []
        for idx, row in outlier_df.iterrows():
            if row['Outlier_Values']:
                values = [float(v) for v in row['Outlier_Values'].split('; ')]
                max_val = max(values, key=abs)  # 绝对值最大的异常值
                outlier_values.append({
                    'File': row['File'],
                    'Patient': row['Patient_Name'],
                    'Eye': row['Eye'],
                    'Max_Outlier': max_val,
                    'Abs_Outlier': abs(max_val)
                })
        
        if outlier_values:
            outlier_df_sorted = pd.DataFrame(outlier_values).sort_values('Abs_Outlier', ascending=False).head(10)
            print(outlier_df_sorted[['File', 'Patient', 'Eye', 'Max_Outlier']].to_string(index=False))
    
    return df


def main():
    """
    主函数
    """
    print("批量患者数据异常点分析工具")
    print("注意: 本分析将排除 (50°, 180°) 位置的数据点")
    print("=" * 60)
    
    # 执行批量分析
    results = batch_analyze_all_patients("testdata_aier")
    
    if results:
        # 保存报告并生成详细统计
        df = save_analysis_report(results)
        
        # 额外的pandas分析功能
        if df is not None:
            print(f"\n[高级分析]")
            print("=" * 50)
            
            # 异常点位置分布分析
            outlier_positions = []
            for _, row in df[df['Has_Outliers'] == 'Yes'].iterrows():
                if row['Outlier_Positions']:
                    positions = row['Outlier_Positions'].split('; ')
                    outlier_positions.extend(positions)
            
            if outlier_positions:
                print(f"\n[异常点位置分布]")
                position_counts = pd.Series(outlier_positions).value_counts()
                print(position_counts.to_string())
                
                # 分析50度位置的异常点比例
                fifty_degree_positions = [pos for pos in outlier_positions if '50°' in pos]
                fifty_percent = len(fifty_degree_positions) / len(outlier_positions) * 100
                print(f"\n50°位置异常点占比: {fifty_percent:.1f}% ({len(fifty_degree_positions)}/{len(outlier_positions)})")
            
            # 保存详细分析结果
            analysis_summary_file = "batch_analysis_summary_filtered.txt"
            with open(analysis_summary_file, 'w', encoding='utf-8') as f:
                f.write("批量患者数据异常点分析汇总报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("注意: 本分析排除了 (50°, 180°) 位置的数据点\n\n")
                
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据文件数: {len(results)}\n")
                f.write(f"患者总数: {df['Patient_Name'].nunique()}\n")
                f.write(f"眼部数据: {len(df)}\n")
                f.write(f"异常点总数: {df['Outlier_Count'].sum()}\n\n")
                
                if outlier_positions:
                    f.write("异常点位置分布:\n")
                    for pos, count in position_counts.items():
                        f.write(f"  {pos}: {count}次\n")
                
            print(f"\n[报告] 详细分析摘要已保存至: {analysis_summary_file}")


if __name__ == "__main__":
    main()