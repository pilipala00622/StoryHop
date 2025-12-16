#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
答案和问题去重脚本
对JSONL文件中的answer和question字段进行去重，保留answer和question都不相似的行数
使用编辑距离计算相似度，去除相似度超过阈值的重复答案和问题
"""

import json
import argparse
from typing import List, Dict, Any
from difflib import SequenceMatcher
import sys
from tqdm import tqdm


def calculate_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的相似度
    使用SequenceMatcher计算相似度，返回0-1之间的值
    """
    return SequenceMatcher(None, text1, text2).ratio()


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的Jaccard相似度
    基于字符级别的n-gram
    """
    def get_ngrams(text: str, n: int = 2) -> set:
        """获取文本的n-gram集合"""
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    ngrams1 = get_ngrams(text1)
    ngrams2 = get_ngrams(text2)
    
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    
    return intersection / union if union > 0 else 0.0


def deduplicate_answers(input_file: str, output_file: str, answer_similarity_threshold: float = 0.6, 
                       question_similarity_threshold: float = 0.5, method: str = 'sequence') -> None:
    """
    对JSONL文件中的answer和question进行去重
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        answer_similarity_threshold: answer相似度阈值，超过此值的答案将被去重
        question_similarity_threshold: question相似度阈值，超过此值的问题将被去重
        method: 相似度计算方法 ('sequence' 或 'jaccard')
    """
    print(f"开始处理文件: {input_file}")
    print(f"Answer相似度阈值: {answer_similarity_threshold}")
    print(f"Question相似度阈值: {question_similarity_threshold}")
    print(f"相似度计算方法: {method}")
    
    # 读取所有数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    print(f"原始数据条数: {len(data)}")
    
    # 去重处理
    unique_data = []
    removed_count = 0
    
    # 选择相似度计算方法
    if method == 'jaccard':
        similarity_func = jaccard_similarity
    else:
        similarity_func = calculate_similarity
    
    print("开始去重处理...")
    
    for i, current_item in enumerate(tqdm(data, desc="处理进度")):
        current_answer = current_item.get('answer', '')
        current_question = current_item.get('question', '')
        
        # 检查是否与已保留的答案或问题相似
        is_duplicate = False
        for unique_item in unique_data:
            unique_answer = unique_item.get('answer', '')
            unique_question = unique_item.get('question', '')
            
            # 计算answer相似度
            answer_similarity = similarity_func(current_answer, unique_answer)
            
            # 计算question相似度
            question_similarity = similarity_func(current_question, unique_question)
            
            # 如果answer或question相似度超过阈值，则认为是重复
            if answer_similarity >= answer_similarity_threshold or question_similarity >= question_similarity_threshold:
                is_duplicate = True
                removed_count += 1
                print(f"发现重复项:")
                if answer_similarity >= answer_similarity_threshold:
                    print(f"  Answer相似度: {answer_similarity:.3f}")
                if question_similarity >= question_similarity_threshold:
                    print(f"  Question相似度: {question_similarity:.3f}")
                print(f"  保留Answer: {unique_answer[:100]}...")
                print(f"  删除Answer: {current_answer[:100]}...")
                print(f"  保留Question: {unique_question[:100]}...")
                print(f"  删除Question: {current_question[:100]}...")
                break
        
        if not is_duplicate:
            unique_data.append(current_item)
    
    # 保存去重后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in unique_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n去重完成!")
    print(f"原始数据: {len(data)} 条")
    print(f"去重后数据: {len(unique_data)} 条")
    print(f"删除重复: {removed_count} 条")
    print(f"去重率: {removed_count/len(data)*100:.2f}%")
    print(f"结果保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='对JSONL文件中的answer和question字段进行去重')
    parser.add_argument('input_file', help='输入JSONL文件路径')
    parser.add_argument('-o', '--output', help='输出JSONL文件路径', 
                       default=None)
    parser.add_argument('-at', '--answer-threshold', type=float, default=0.6,
                       help='answer相似度阈值 (0-1)，默认0.6')
    parser.add_argument('-qt', '--question-threshold', type=float, default=0.5,
                       help='question相似度阈值 (0-1)，默认0.5')
    parser.add_argument('-m', '--method', choices=['sequence', 'jaccard'], 
                       default='sequence',
                       help='相似度计算方法: sequence(编辑距离) 或 jaccard(Jaccard相似度)')
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，自动生成
    if args.output is None:
        input_path = args.input_file
        if input_path.endswith('.jsonl'):
            output_path = input_path.replace('.jsonl', '_deduplicated.jsonl')
        else:
            output_path = input_path + '_deduplicated.jsonl'
    else:
        output_path = args.output
    
    try:
        deduplicate_answers(args.input_file, output_path, args.answer_threshold, 
                           args.question_threshold, args.method)
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
