# project_validator.py
import re
import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class ProjectLevelValidator:
    """小说QA验证器：专门用于验证小说多跳推理QA对的质量"""
    
    def __init__(self, validation_config: Optional[Dict] = None):
        """
        初始化小说QA验证器
        :param validation_config: 验证配置字典
        """
        self.config = validation_config or self._get_default_config()
        self.validation_stats = {
            'total_qa': 0,
            'final_valid': 0
        }
    
    def _get_default_config(self) -> Dict:
        """获取默认验证配置"""
        return {
            'check_chain': True,  # 是否检查推理链
            'check_fields': True,  # 是否检查四个字段
            'check_answer_in_content': True,  # 是否检查答案在内容中存在
            'output_config': {
                'save_invalid_qa': True,
                'invalid_qa_path': 'invalid_novel_qa_debug.json'
            }
        }
    
    def validate_fields(self, qa: Dict) -> bool:
        """
        验证四个必要字段是否存在
        :param qa: QA字典
        :return: 是否通过验证
        """
        if not self.config['check_fields']:
            return True
        
        required_fields = ['hop_depth', 'question', 'answer', 'chain']
        missing_fields = []
        
        for field in required_fields:
            if field not in qa or not qa[field]:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ 缺少必要字段：{missing_fields}，问题：{qa.get('question', '')[:50]}...")
            return False
        
        print(f"✅ 字段验证通过")
        return True
    
    def validate_chain(self, qa: Dict) -> bool:
        """
        验证推理链是否存在且格式正确
        :param qa: QA字典
        :return: 是否通过验证
        """
        if not self.config['check_chain']:
            return True
        
        chain = qa.get("chain", "")
        hop_depth = qa.get("hop_depth", 0)
        
        # 检查推理链是否存在
        if not chain:
            print(f"❌ 缺少推理链，问题：{qa.get('question', '')[:50]}...")
            return False
        
        # 检查推理链格式（是否包含→）
        if "→" not in chain:
            print(f"❌ 推理链格式错误（缺少→），问题：{qa.get('question', '')[:50]}...")
            return False
        
        # 检查节点数量与跳数是否匹配
        nodes = chain.split("→")
        if len(nodes) != hop_depth + 1:
            print(f"❌ 推理链节点数量不匹配（跳数：{hop_depth}，节点数：{len(nodes)}），问题：{qa.get('question', '')[:50]}...")
            return False
        
        print(f"✅ 推理链验证通过（{hop_depth}跳）")
        return True
    
    def validate_answer_in_content(self, qa: Dict, content: str) -> bool:
        """
        验证答案是否在内容中存在
        :param qa: QA字典
        :param content: 内容文本
        :return: 是否通过验证
        """
        if not self.config['check_answer_in_content']:
            return True
        
        answer = qa.get("answer", "")
        if not answer:
            print(f"❌ 答案为空，问题：{qa.get('question', '')[:50]}...")
            return False
        
        # 在内容中查找答案
        answer_lower = answer.lower()
        content_lower = content.lower()
        
        # 策略1：直接匹配
        if answer_lower in content_lower:
            print(f"✅ 答案在内容中找到：{answer[:50]}...")
            return True
        
        # 策略2：去除标点符号后匹配
        answer_clean = re.sub(r'[^\w\s]', '', answer_lower)
        content_clean = re.sub(r'[^\w\s]', '', content_lower)
        if answer_clean in content_clean:
            print(f"✅ 答案在内容中找到（去标点）：{answer[:50]}...")
            return True
        
        # 策略3：关键词匹配
        answer_words = [w for w in answer_lower.split() if len(w) > 2]
        if answer_words:
            matched_words = sum(1 for word in answer_words if word in content_lower)
            if matched_words / len(answer_words) >= 0.5:
                print(f"✅ 答案关键词在内容中找到：{answer[:50]}...")
                return True
        
        print(f"❌ 答案在内容中未找到：{answer[:50]}...")
        return False
    
    def validate_single_qa(self, qa: Dict, content: str = None) -> Dict:
        """
        验证单个小说QA对
        :param qa: QA字典
        :param content: 内容文本（用于验证答案存在性）
        :return: 验证结果字典
        """
        validation_result = {
            'qa': qa,
            'valid': True,
            'errors': []
        }
        
        # 1. 验证四个字段
        if not self.validate_fields(qa):
            validation_result['valid'] = False
            validation_result['errors'].append('fields_validation_failed')
        
        # 2. 验证推理链
        if not self.validate_chain(qa):
            validation_result['valid'] = False
            validation_result['errors'].append('chain_validation_failed')
        
        # 3. 验证答案是否在内容中存在
        if content and not self.validate_answer_in_content(qa, content):
            validation_result['valid'] = False
            validation_result['errors'].append('answer_content_validation_failed')
        
        return validation_result
    
    def validate_all_qa(self, qa_list: List[Dict], content: str = None) -> List[Dict]:
        """
        验证小说QA列表，返回有效QA
        :param qa_list: QA列表
        :param content: 内容文本（用于验证答案存在性）
        :return: 有效QA列表
        """
        print(f"\n🔍 开始小说QA验证，总QA数：{len(qa_list)}")
        if content:
            print(f"📄 启用内容验证，内容长度：{len(content)} 字符")
        
        valid_qa: List[Dict] = []
        invalid_qa: List[Dict] = []
        
        self.validation_stats['total_qa'] = len(qa_list)
        
        for qa in qa_list:
            validation_result = self.validate_single_qa(qa, content)
            
            if validation_result['valid']:
                valid_qa.append(qa)
                self.validation_stats['final_valid'] += 1
            else:
                invalid_qa.append(validation_result)
        
        # 输出验证统计
        self._print_validation_stats()
        
        # 保存无效QA用于调试
        if invalid_qa and self.config['output_config']['save_invalid_qa']:
            self._save_invalid_qa(invalid_qa)
        
        if not valid_qa:
            print("❌ 所有QA均未通过验证")
            print("💡 建议检查：")
            print("   1. QA字段是否完整（hop_depth, question, answer, chain）")
            print("   2. 推理链格式是否正确（使用→连接）")
            print("   3. 答案是否在内容中存在")
            raise ValueError("❌ 所有QA均未通过验证，请检查QA生成质量")
        
        print(f"✅ 小说QA验证完成，有效QA对数：{len(valid_qa)}（总QA数：{len(qa_list)}）")
        return valid_qa
    
    def _print_validation_stats(self):
        """打印验证统计信息"""
        stats = self.validation_stats
        print(f"\n📊 验证统计：")
        print(f"   总QA数：{stats['total_qa']}")
        print(f"   最终有效：{stats['final_valid']}")
        print(f"   通过率：{stats['final_valid']/stats['total_qa']*100:.1f}%")
    
    def _save_invalid_qa(self, invalid_qa: List[Dict]):
        """保存无效QA用于调试"""
        output_path = self.config['output_config']['invalid_qa_path']
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(invalid_qa, f, ensure_ascii=False, indent=2)
            print(f"💾 无效QA已保存至：{output_path}")
        except Exception as e:
            print(f"⚠️  保存无效QA失败：{str(e)}")
    
    def update_config(self, new_config: Dict):
        """更新验证配置"""
        self.config.update(new_config)
        print("✅ 验证配置已更新")
    
    def get_config(self) -> Dict:
        """获取当前验证配置"""
        return self.config.copy()


# 便捷函数：创建不同模式的验证器
def create_strict_validator() -> ProjectLevelValidator:
    """创建严格模式验证器"""
    config = {
        'check_chain': True,
        'check_fields': True,
        'check_answer_in_content': True
    }
    return ProjectLevelValidator(config)


def create_loose_validator() -> ProjectLevelValidator:
    """创建宽松模式验证器"""
    config = {
        'check_chain': True,
        'check_fields': True,
        'check_answer_in_content': False,  # 不检查答案是否在内容中
        'output_config': {'save_invalid_qa': False}
    }
    return ProjectLevelValidator(config)


def create_custom_validator(config: Dict) -> ProjectLevelValidator:
    """创建自定义配置验证器"""
    return ProjectLevelValidator(config)