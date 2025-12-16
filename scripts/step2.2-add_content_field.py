#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本功能：根据jsonl文件中的source字段，从对应的txt文件和novel开头的jsonl文件中检索原文内容，
并将原文内容添加到原jsonl的新字段content中。

作者：AI Assistant
日期：2024
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_txt_files(txt_dir: str) -> Dict[str, str]:
    """
    加载所有txt文件的内容，返回文件名到内容的映射
    
    Args:
        txt_dir: txt文件所在目录
        
    Returns:
        Dict[str, str]: 文件名到文件内容的映射
    """
    txt_content_map = {}
    txt_path = Path(txt_dir)
    
    if not txt_path.exists():
        print(f"警告：txt目录不存在: {txt_dir}")
        return txt_content_map
    
    for txt_file in txt_path.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                txt_content_map[txt_file.name] = content
                print(f"已加载txt文件: {txt_file.name}")
        except Exception as e:
            print(f"加载txt文件失败 {txt_file.name}: {e}")
    
    return txt_content_map


def load_novel_jsonl(novel_jsonl_path: str) -> Dict[str, str]:
    """
    加载novel开头的jsonl文件，返回source到full_content的映射
    
    Args:
        novel_jsonl_path: novel jsonl文件路径
        
    Returns:
        Dict[str, str]: source到full_content的映射
    """
    novel_content_map = {}
    
    if not os.path.exists(novel_jsonl_path):
        print(f"警告：novel jsonl文件不存在: {novel_jsonl_path}")
        return novel_content_map
    
    try:
        with open(novel_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    source = data.get('source', '')
                    full_content = data.get('full_content', '')
                    
                    if source and full_content:
                        novel_content_map[source] = full_content
                        
                except json.JSONDecodeError as e:
                    print(f"解析jsonl第{line_num}行失败: {e}")
                    continue
                    
        print(f"已加载novel jsonl文件，共{len(novel_content_map)}条记录")
        
    except Exception as e:
        print(f"加载novel jsonl文件失败: {e}")
    
    return novel_content_map


def find_matching_content(source: str, txt_content_map: Dict[str, str], 
                         novel_content_map: Dict[str, str]) -> Optional[str]:
    """
    根据source字段查找匹配的原文内容
    
    Args:
        source: 源文件名
        txt_content_map: txt文件内容映射
        novel_content_map: novel jsonl内容映射
        
    Returns:
        Optional[str]: 匹配的原文内容，如果未找到则返回None
    """
    # 首先尝试在novel_content_map中查找完全匹配
    if source in novel_content_map:
        return novel_content_map[source]
    
    # 如果完全匹配失败，尝试在txt_content_map中查找
    if source in txt_content_map:
        return txt_content_map[source]
    
    # 尝试模糊匹配（去掉一些常见的后缀或前缀）
    source_clean = source.strip()
    
    # 在novel_content_map中查找模糊匹配
    for novel_source, content in novel_content_map.items():
        if source_clean in novel_source or novel_source in source_clean:
            print(f"模糊匹配成功: {source} -> {novel_source}")
            return content
    
    # 在txt_content_map中查找模糊匹配
    for txt_source, content in txt_content_map.items():
        if source_clean in txt_source or txt_source in source_clean:
            print(f"模糊匹配成功: {source} -> {txt_source}")
            return content
    
    return None


def process_jsonl_file(input_file: str, output_file: str, txt_dir: str, novel_jsonl_path: str):
    """
    处理jsonl文件，添加content字段
    
    Args:
        input_file: 输入jsonl文件路径
        output_file: 输出jsonl文件路径
        txt_dir: txt文件目录
        novel_jsonl_path: novel jsonl文件路径
    """
    # 加载所有内容映射
    print("正在加载txt文件...")
    txt_content_map = load_txt_files(txt_dir)
    
    print("正在加载novel jsonl文件...")
    novel_content_map = load_novel_jsonl(novel_jsonl_path)
    
    # 统计信息
    total_records = 0
    matched_records = 0
    unmatched_sources = set()
    
    # 处理jsonl文件
    print(f"正在处理文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                total_records += 1
                
                # 获取source字段
                source = data.get('source', '')
                if not source:
                    print(f"第{line_num}行缺少source字段")
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    continue
                
                # 查找匹配的内容
                content = find_matching_content(source, txt_content_map, novel_content_map)
                
                if content:
                    data['content'] = content
                    matched_records += 1
                    print(f"第{line_num}行匹配成功: {source}")
                else:
                    unmatched_sources.add(source)
                    print(f"第{line_num}行未找到匹配内容: {source}")
                
                # 写入输出文件
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
                
            except json.JSONDecodeError as e:
                print(f"解析第{line_num}行失败: {e}")
                continue
    
    # 输出统计信息
    print(f"\n处理完成！")
    print(f"总记录数: {total_records}")
    print(f"成功匹配: {matched_records}")
    print(f"未匹配: {total_records - matched_records}")
    print(f"匹配率: {matched_records/total_records*100:.2f}%")
    
    if unmatched_sources:
        print(f"\n未匹配的source列表:")
        for source in sorted(unmatched_sources):
            print(f"  - {source}")


def main():
    """主函数"""
    # 文件路径配置
    base_dir = Path(__file__).parent
    input_file = base_dir / "filtered_507_v3_0919.jsonl"
    output_file = base_dir / "filtered_507_v3_0919_with_content.jsonl"
    txt_dir = base_dir / "raw_data"
    novel_jsonl_path = base_dir / "raw_data" / "novel_content-128k_24year_371.jsonl"
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"错误：输入文件不存在: {input_file}")
        return
    
    # 处理文件
    process_jsonl_file(
        input_file=str(input_file),
        output_file=str(output_file),
        txt_dir=str(txt_dir),
        novel_jsonl_path=str(novel_jsonl_path)
    )
    
    print(f"\n输出文件已保存到: {output_file}")


if __name__ == "__main__":
    main()
