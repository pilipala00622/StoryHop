"""
对小说片段进行提问，要求必须用原文信息回答。

重要说明：关于文本截断问题
===============================================
本代码中存在多处使用 segment[:1000] 或 segment[:1500] 的情况，这会导致以下问题：

1. segment[:1000] 的含义：
   - 这是Python的切片操作，表示取segment字符串的前1000个字符
   - 例如：如果segment有5000个字符，segment[:1000]只返回前1000个字符
   - 后面的4000个字符会被忽略

2. 为什么会有截断：
   - API调用有token限制，过长的prompt会导致调用失败
   - 为了控制prompt长度，代码只使用片段的前1000-1500个字符
   - 这是一种权衡：保证API调用成功 vs 信息完整性

3. 截断导致的问题：
   - 关键词提取不完整：重要信息可能在片段后面
   - 跳链生成不准确：推理路径可能不完整
   - 答案验证失败：正确答案可能在截断的部分
   - QA对质量下降：基于不完整信息生成的问答

4. 解决方案建议：
   - 增加片段长度限制（如使用segment[:2000]）
   - 实现滑动窗口机制，多次处理长片段
   - 使用更智能的文本摘要技术
   - 分段处理后再合并结果

当前代码中的截断位置：
- extract_keywords(): segment[:1500] (第162行)
- generate_hop_chains(): segment[:1000] (第272行)  
- generate_qa_pairs(): segment[:1000] (第390行)
"""

import json
import os
import sys
from tqdm import tqdm
from typing import List, Dict, Tuple

# 添加 multiprocess_api_utils 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
multiprocess_api_path = os.path.join(current_dir, "multiprocess_api_utils")
sys.path.append(multiprocess_api_path)

# 添加项目级validator路径
sys.path.append(os.path.join(current_dir, ".."))
from project_validator import ProjectLevelValidator, create_loose_validator

try:
    from run import LLM, retry_get_model_answer
except ImportError:
    # 如果无法导入，尝试从上级目录导入
    parent_dir = os.path.dirname(current_dir)
    multiprocess_api_path = os.path.join(parent_dir, "multiprocess_api_utils")
    sys.path.append(multiprocess_api_path)
    from run import LLM, retry_get_model_answer

# -------------------------- 1. 配置参数（用户需根据实际情况修改） --------------------------
class Config:
    # 模型配置
    MODEL_NAME = "gemini-2.5-flash"  # 使用 Gemini-2.5-flash 模型

    # 文本参数
    # TARGET_TOKEN_RANGE = (64000, 128000)  # 目标上下文长度（64k-128k token）
    TARGET_TOKEN_RANGE = (320, 640)
    HOP_DEPTHS = [3, 4, 5]  # 支持的推理跳数（至少2跳以上，尽量3-5跳）
    OUTPUT_PATH = "novelhopqa_qa_pairs.jsonl"  # 最终 QA 对输出路径

    # 验证参数
    STRICT_ORACLE_VALIDATION = False  # 是否启用严格的Oracle验证（True=仅直接匹配，False=多层次匹配）
    CROSS_SEGMENT_VALIDATION = True  # 是否启用跨片段验证（True=验证整个文档，False=仅验证当前片段）

    # QA生成参数
    MIN_QA_COUNT = 5  # 每个片段最少生成的QA对数量
    MAX_RETRY_ATTEMPTS = 3  # 最大重试次数
    CHAINS_PER_HOP = [3, 4, 3, 2]  # 每个跳数生成的跳链数量 [2跳, 3跳, 4跳, 5跳]
    
    # 文本截断参数（解决segment[:2000]问题）
    MAX_PROMPT_LENGTH = 2000  # 最大prompt长度（字符数），避免API调用失败
    USE_SLIDING_WINDOW = True  # 是否使用滑动窗口处理长片段
    WINDOW_OVERLAP_RATIO = 0.2  # 滑动窗口重叠比例（20%）

    # 分词器配置（用户需替换为自己的分词器逻辑）
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        自定义分词器接口：将文本转换为 token 列表
        示例：若使用 HuggingFace Tokenizer，可替换为 `tokenizer.encode(text, add_special_tokens=False)`
        """
        # 占位逻辑：此处用空格分词（需用户替换为真实分词器！）
        return text.split()

    @staticmethod
    def count_tokens(text: str) -> int:
        """计算文本的 token 数量"""
        return len(Config.tokenize(text))


# -------------------------- 2. LLM 调用工具类 --------------------------
class LLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = LLM(model_name)

    def call(self, prompt: str, temperature: float = 0.3) -> str:
        """
        调用 LLM 执行任务
        :param prompt: 任务提示词（需明确任务目标）
        :param temperature: 生成随机性（低温度确保结果稳定，推荐 0.2-0.4）
        :return: API 返回的文本结果
        """
        try:
            result = retry_get_model_answer(self.llm, prompt)
            return result.strip()
        except Exception as e:
            print(f"LLM 调用失败：{str(e)}")
            return ""


# -------------------------- 3. 辅助函数：智能文本截断 --------------------------
def smart_text_truncation(text: str, max_length: int = None, use_sliding_window: bool = None) -> List[str]:
    """
    智能文本截断：解决segment[:1000]问题
    :param text: 输入文本
    :param max_length: 最大长度（默认使用Config.MAX_PROMPT_LENGTH）
    :param use_sliding_window: 是否使用滑动窗口（默认使用Config.USE_SLIDING_WINDOW）
    :return: 文本片段列表
    """
    if max_length is None:
        max_length = Config.MAX_PROMPT_LENGTH
    if use_sliding_window is None:
        use_sliding_window = Config.USE_SLIDING_WINDOW
    
    # 如果文本长度小于最大长度，直接返回
    if len(text) <= max_length:
        return [text]
    
    if not use_sliding_window:
        # 简单截断：只返回前max_length个字符
        print(f"警告：文本长度{len(text)}超过限制{max_length}，将截断为前{max_length}个字符")
        return [text[:max_length]]
    
    # 滑动窗口处理
    segments = []
    overlap_size = int(max_length * Config.WINDOW_OVERLAP_RATIO)
    step_size = max_length - overlap_size
    
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        segment = text[start:end]
        segments.append(segment)
        
        # 如果到达文本末尾，停止
        if end == len(text):
            break
            
        start += step_size
    
    print(f"文本长度{len(text)}超过限制{max_length}，使用滑动窗口分割为{len(segments)}个片段")
    return segments


# -------------------------- 4. 核心模块：文本预处理 --------------------------
def preprocess_novel_text(raw_text: str) -> List[str]:
    """
    小说文本预处理：分割为 64k-128k token 的连续片段，确保情节连贯性
    :param raw_text: 输入的小说纯文本
    :return: 符合长度要求的片段列表（每个片段为字符串）
    """
    # 步骤1：清理文本（去除无意义符号、统一换行）
    cleaned_text = raw_text.replace("\n\n", "\n").replace("\r", "").strip()
    if not cleaned_text:
        raise ValueError("输入文本为空或仅含空白字符")

    # 步骤2：按 token 长度分割片段
    tokens = Config.tokenize(cleaned_text)
    total_tokens = len(tokens)
    segments = []
    start_idx = 0

    print(f"原始文本总 token 数：{total_tokens}，目标片段长度：{Config.TARGET_TOKEN_RANGE}")

    while start_idx < total_tokens:
        # 计算当前片段的结束位置（优先取目标范围上限，避免超出总长度）
        end_idx = min(start_idx + Config.TARGET_TOKEN_RANGE[1], total_tokens)
        # 若当前片段过短（小于下限），合并到下一段（确保片段长度达标）
        if end_idx - start_idx < Config.TARGET_TOKEN_RANGE[0]:
            if end_idx == total_tokens:
                # 最后一段若过短，与前一段合并
                if segments:
                    segments[-1] += " " + " ".join(tokens[start_idx:end_idx])
                else:
                    segments.append(" ".join(tokens[start_idx:end_idx]))
            break

        # 还原为文本片段（用空格连接 token，需根据分词器调整）
        segment_text = " ".join(tokens[start_idx:end_idx])
        segments.append(segment_text)
        start_idx = end_idx

    # 验证片段长度
    valid_segments = []
    for seg in segments:
        seg_tokens = Config.count_tokens(seg)
        if Config.TARGET_TOKEN_RANGE[0] <= seg_tokens <= Config.TARGET_TOKEN_RANGE[1]:
            valid_segments.append(seg)
        else:
            print(f"片段长度不达标（{seg_tokens} token），已过滤")

    print(f"预处理完成，有效片段数：{len(valid_segments)}")
    return valid_segments


# -------------------------- 4. 核心模块：关键词提取（基于 LLM API） --------------------------
def extract_keywords(segment: str, api_client: LLMClient) -> Dict[str, List[str]]:
    """
    从小说片段中提取关键词（实体类+事件类），用于后续跳链生成
    :param segment: 预处理后的小说片段
    :param api_client: LLM API 客户端实例
    :return: 关键词字典（key: 关键词类型，value: 关键词列表）
    """
    # Prompt 设计：明确要求提取符合多跳推理需求的关键词，贴合小说情节
    # 注意：segment[:1500] 表示只取片段的前1500个字符，这是为了避免prompt过长导致API调用失败
    # 但这也可能导致关键词提取不完整，因为重要的关键词可能在片段的后面部分
    prompt = f"""
    任务：从以下小说片段中提取两类关键词，用于构建多跳推理链（3-5跳）。
    要求：
    1. 实体类关键词：人物（姓名/身份）、地点（场景/建筑）、物品（道具/关键物品），需明确且在片段中多次提及或为核心元素；
    2. 事件类关键词：关键动作、因果关系、情节转折（如"拒绝求婚""发现秘密""马车失事"），需与实体强关联；
    3. 每个类别提取 5-10 个关键词，避免重复或无关信息；
    4. 必须严格按照以下JSON格式输出，不要添加任何其他内容：
    
    {{
        "entity_keywords": ["关键词1", "关键词2", "关键词3"],
        "event_keywords": ["关键词1", "关键词2", "关键词3"]
    }}
    
    小说片段（注意：这里只显示了片段的前2000个字符，完整片段长度为{len(segment)}字符）：
    {segment[:2000]}...
    """

    # 调用 API 提取关键词
    api_result = api_client.call(prompt, temperature=0.2)
    if not api_result:
        return {"entity_keywords": [], "event_keywords": []}

    # 解析 JSON 结果（容错处理：若 API 返回非 JSON，手动整理）
    try:
        # 尝试清理 API 结果，移除可能的 markdown 格式
        cleaned_result = api_result.strip()
        if cleaned_result.startswith("```json"):
            cleaned_result = cleaned_result[7:]
        if cleaned_result.endswith("```"):
            cleaned_result = cleaned_result[:-3]
        cleaned_result = cleaned_result.strip()

        keywords = json.loads(cleaned_result)
        # 验证格式
        if not isinstance(keywords.get("entity_keywords"), list) or not isinstance(keywords.get("event_keywords"), list):
            raise ValueError("关键词格式错误")
        return keywords
    except Exception as e:
        print(f"关键词解析失败：{str(e)}，尝试手动整理")
        print(f"API 返回内容：{api_result[:200]}...")

        # 备用逻辑：若 API 返回非 JSON，按文本格式提取
        entity_keywords = []
        event_keywords = []

        # 尝试多种格式解析
        lines = api_result.split("\n")
        for line in lines:
            line = line.strip()
            if "实体类" in line or "entity" in line.lower():
                # 尝试多种分隔符
                for sep in ["：", ":", "：", " -", "-"]:
                    if sep in line:
                        parts = line.split(sep, 1)
                        if len(parts) > 1:
                            keywords_str = parts[1].strip()
                            entity_keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
                            break

            if "事件类" in line or "event" in line.lower():
                # 尝试多种分隔符
                for sep in ["：", ":", "：", " -", "-"]:
                    if sep in line:
                        parts = line.split(sep, 1)
                        if len(parts) > 1:
                            keywords_str = parts[1].strip()
                            event_keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
                            break

        # 如果仍然没有找到关键词，使用简单的启发式方法
        if not entity_keywords and not event_keywords:
            print("使用启发式方法提取关键词...")
            # 简单的关键词提取：找一些常见的人物、地点、物品词汇
            words = api_result.split()
            entity_keywords = [w for w in words if len(w) > 2 and any(char in w for char in "先生小姐老师教授")]
            event_keywords = [w for w in words if len(w) > 2 and any(char in w for char in "见面说话走路吃饭")]

        return {"entity_keywords": entity_keywords[:10], "event_keywords": event_keywords[:10]}


# -------------------------- 5. 核心模块：跳分离链生成（基于 LLM API） --------------------------
def generate_hop_chains(segment: str, keywords: Dict[str, List[str]],
                       hop_depths: List[int], api_client: LLMClient) -> List[Dict]:
    """
    基于关键词生成 3-5 跳的跳分离链（推理路径），贴合小说情节
    :param segment: 小说片段
    :param keywords: 提取的关键词字典
    :param hop_depths: 支持的推理跳数（[3,4,5]）
    :param api_client: LLM API 客户端实例
    :return: 跳链列表（每个跳链含"hop_depth""chain""nodes"字段）
    """
    hop_chains = []
    entity_str = ", ".join(keywords["entity_keywords"])
    event_str = ", ".join(keywords["event_keywords"])

    for depth in tqdm(hop_depths, desc="生成各跳数的推理链"):
        # 获取当前跳数应该生成的链数量
        # 由于HOP_DEPTHS从2开始，需要调整索引
        hop_index = Config.HOP_DEPTHS.index(depth) if depth in Config.HOP_DEPTHS else 0
        chains_needed = Config.CHAINS_PER_HOP[hop_index] if hop_index < len(Config.CHAINS_PER_HOP) else 2

        # 为当前跳数生成指定数量的跳链
        for chain_idx in range(chains_needed):
            # Prompt 设计：明确跳数要求，确保跳链逻辑连贯、节点明确
            # 注意：segment[:1000] 表示只取片段的前1000个字符，这是为了避免prompt过长导致API调用失败
            # 但这也可能导致跳链生成不完整，因为重要的推理信息可能在片段的后面部分
            prompt = f"""
            任务：基于以下小说片段和关键词，生成第 {chain_idx + 1} 条 {depth} 跳的"跳分离链"（推理路径）。
            
            要求：
            1. 跳链定义：{depth} 跳链需包含 {depth+1} 个信息节点，用"→"连接
            2. 节点要求：每个节点必须是片段中的具体信息，基于提供的关键词
            3. 推理逻辑：确保每个节点之间有明确的逻辑关系，能够支持多步推理
            4. 起始和终点：第一个节点作为起始信息，最后一个节点作为目标信息
            5. 多样性要求：请生成与之前不同的推理路径，避免重复
            6. 输出格式：严格按照以下JSON格式，不要添加任何其他内容
            
            {{
                "hop_depth": {depth},
                "chain": "节点1→节点2→节点3",
                "nodes": ["节点1", "节点2", "节点3"]
            }}
            
            小说片段（注意：这里只显示了片段的前2000个字符，完整片段长度为{len(segment)}字符）：
            {segment[:2000]}...
            
            可用关键词：
            实体类：{entity_str}
            事件类：{event_str}
            """

            api_result = api_client.call(prompt, temperature=0.3)
            if not api_result:
                continue

            # 解析跳链（支持多跳链批量解析）
            try:
                # 尝试清理 API 结果，移除可能的 markdown 格式
                cleaned_result = api_result.strip()
                if cleaned_result.startswith("```json"):
                    cleaned_result = cleaned_result[7:]
                if cleaned_result.endswith("```"):
                    cleaned_result = cleaned_result[:-3]
                cleaned_result = cleaned_result.strip()

                # 尝试直接解析 JSON
                try:
                    chain = json.loads(cleaned_result)
                    # 验证跳链格式
                    if (chain.get("hop_depth") == depth and
                        isinstance(chain.get("chain"), str) and
                        isinstance(chain.get("nodes"), list) and
                        len(chain["nodes"]) == depth + 1):
                        hop_chains.append(chain)
                        print(f"成功解析 {depth} 跳链 #{chain_idx + 1}：{chain}")
                        continue
                except:
                    pass

                # 如果直接解析失败，尝试逐行解析
                for line in api_result.split("\n"):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            chain = json.loads(line)
                            # 验证跳链格式
                            if (chain.get("hop_depth") == depth and
                                isinstance(chain.get("chain"), str) and
                                isinstance(chain.get("nodes"), list) and
                                len(chain["nodes"]) == depth + 1):
                                hop_chains.append(chain)
                                print(f"成功解析 {depth} 跳链 #{chain_idx + 1}：{chain}")
                                break
                        except:
                            continue

                # 如果仍然没有找到有效跳链，输出调试信息
                if not any(c.get("hop_depth") == depth for c in hop_chains[-chains_needed:]):
                    print(f"{depth} 跳链 #{chain_idx + 1} 解析失败，API 返回内容：{api_result[:200]}...")

            except Exception as e:
                print(f"{depth} 跳链 #{chain_idx + 1} 解析失败：{str(e)}")
                print(f"API 返回内容：{api_result[:200]}...")

    print(f"跳链生成完成，共生成 {len(hop_chains)} 条有效跳链")
    return hop_chains


# -------------------------- 6. 核心模块：QA 对生成（基于 LLM API） --------------------------
def generate_qa_pairs(hop_chains: List[Dict], segment: str, api_client: LLMClient, source_file: str = None) -> List[Dict]:
    """
    基于跳分离链生成 QA 对，确保问题贴合跳深度、答案明确可验证
    :param hop_chains: 跳链列表
    :param segment: 小说片段（用于验证答案存在性）
    :param api_client: LLM API 客户端实例
    :param source_file: 源文件名（用于记录来源）
    :return: QA 对列表（每个 QA 含"hop_depth""question""answer""chain""source"字段）
    """
    qa_pairs = []

    # 如果跳链数量不足，尝试生成更多QA对
    if len(hop_chains) < Config.MIN_QA_COUNT:
        print(f"跳链数量({len(hop_chains)})不足，尝试重复使用跳链生成更多QA对")
        # 重复使用跳链直到达到最小数量
        extended_chains = []
        while len(extended_chains) < Config.MIN_QA_COUNT:
            extended_chains.extend(hop_chains)
        hop_chains = extended_chains[:Config.MIN_QA_COUNT]

    for chain in tqdm(hop_chains, desc="基于跳链生成 QA 对"):
        depth = chain["hop_depth"]
        chain_str = chain["chain"]
        nodes = chain["nodes"]

        # 重试机制：确保每个跳链都能生成有效的QA对
        qa_generated = False
        for attempt in range(Config.MAX_RETRY_ATTEMPTS):
            # Prompt 设计：明确 QA 生成规则，确保答案唯一、跳深度对齐
            # 注意：segment[:1000] 表示只取片段的前1000个字符，这是为了避免prompt过长导致API调用失败
            # 但这也可能导致答案验证不完整，因为正确的答案可能在片段的后面部分
            prompt = f"""
            任务：基于以下 {depth} 跳推理链，生成对应的 QA 对（问题-答案）。
            
            要求：
            1. 问题设计：必须包含推理链的第一个节点（起始信息）和最后一个节点（目标信息）。
            2. 问题表述：引导模型通过多步推理从起始信息推导到目标信息，表述清晰无歧义，并且语义清晰连贯。
            3. 答案：必须是跳链最后一个节点的具体信息，需在小说片段中明确存在，并且是原文信息、语义清晰连贯。
            4. 多样性：请生成与之前不同的问题和答案，避免重复。
            5. 输出格式：严格按照以下JSON格式，不要添加任何其他内容。
            6. 保证问题和答案的连贯性，不要出现无关信息。例如问题是关于人物的，回答一定是有具体人物信息。
            反面例子。问题是：考虑到尾田和风间在两年前曾于纽约有过联系，这最终使得柳生获得了怎样的保护？。答案: 柳生受到了刑警的保护。这就是不正确的问题和答案，提问应该是受到谁的保护？
            正面例子。问题是：当探险队在山中前进时，他们最终做了什么以便能够烹饪？答案: 他们很快就生起了一堆熊熊燃烧的干枯树枝火。语义连贯正确，并且回复的是原文。
            {{
                "hop_depth": {depth},
                "question": "问题内容",
                "answer": "答案内容",
                "chain": "{chain_str}"
            }}
            
            推理链：{chain_str}
            
            小说片段（注意：这里只显示了片段的前2000个字符，完整片段长度为{len(segment)}字符，用于确认答案）：
            {segment[:2000]}...
            """

            api_result = api_client.call(prompt, temperature=0.2)
            if not api_result:
                continue

            # 解析 QA 对
            try:
                # 尝试清理 API 结果，移除可能的 markdown 格式
                cleaned_result = api_result.strip()
                if cleaned_result.startswith("```json"):
                    cleaned_result = cleaned_result[7:]
                if cleaned_result.endswith("```"):
                    cleaned_result = cleaned_result[:-3]
                cleaned_result = cleaned_result.strip()

                # 尝试直接解析 JSON
                try:
                    qa = json.loads(cleaned_result)
                    # 基础验证：字段完整+答案非空
                    if (qa.get("hop_depth") == depth and
                        qa.get("question") and qa.get("answer") and
                        qa.get("chain") == chain_str):
                        # 添加source字段
                        qa["source"] = source_file if source_file else "unknown"
                        qa_pairs.append(qa)
                        print(f"成功解析 QA：{qa}")
                        qa_generated = True
                        break
                except:
                    pass

                # 如果直接解析失败，尝试逐行解析
                for line in api_result.split("\n"):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            qa = json.loads(line)
                            # 基础验证：字段完整+答案非空
                            if (qa.get("hop_depth") == depth and
                                qa.get("question") and qa.get("answer") and
                                qa.get("chain") == chain_str):
                                # 添加source字段
                                qa["source"] = source_file if source_file else "unknown"
                                qa_pairs.append(qa)
                                print(f"成功解析 QA：{qa}")
                                qa_generated = True
                                break
                        except:
                            continue

                # 如果成功生成QA，跳出重试循环
                if qa_generated:
                    break

            except Exception as e:
                print(f"QA 解析失败（跳链：{chain_str}，尝试 {attempt + 1}）：{str(e)}")
                if attempt == Config.MAX_RETRY_ATTEMPTS - 1:
                    print(f"API 返回内容：{api_result[:200]}...")

        # 如果重试后仍未生成QA，输出警告
        if not qa_generated:
            print(f"警告：跳链 {chain_str} 经过 {Config.MAX_RETRY_ATTEMPTS} 次尝试后仍无法生成有效QA")

    print(f"QA 生成完成，共生成 {len(qa_pairs)} 条初步 QA 对")
    return qa_pairs


# -------------------------- 7. 核心模块：质量验证（Oracle-Context 过滤） --------------------------
def validate_qa_pairs(qa_pairs: List[Dict], segment: str, full_document: str = None) -> List[Dict]:
    """
    QA 对质量验证：
    1. Oracle-Context 过滤：确保答案在文档中存在（支持跨片段验证）；
    :param qa_pairs: 初步生成的 QA 对列表
    :param segment: 当前小说片段
    :param full_document: 完整文档内容（用于跨片段验证，如果为None则只验证当前片段）
    :return: 最终通过验证的 QA 对列表
    """
    # Oracle-Context 过滤（支持跨片段验证）
    oracle_valid_qa = []

    # 确定验证范围：优先使用完整文档，否则使用当前片段
    if full_document:
        verification_text = full_document.lower()
        verification_scope = "完整文档"
    else:
        verification_text = segment.lower()
        verification_scope = "当前片段"

    print(f"Oracle验证范围：{verification_scope}")

    for qa in qa_pairs:
        answer = qa["answer"].lower()

        # 改进的匹配策略：多层次验证
        is_valid = False

        # 策略1：直接子字符串匹配
        if answer in verification_text:
            is_valid = True

        # 如果启用严格验证，只使用直接匹配
        if Config.STRICT_ORACLE_VALIDATION:
            pass  # 只使用策略1
        else:
            # 策略2：去除标点符号后匹配
            if not is_valid:
                import re
                answer_clean = re.sub(r'[^\w\s]', '', answer)  # 移除标点符号
                verification_clean = re.sub(r'[^\w\s]', '', verification_text)
                if answer_clean in verification_clean:
                    is_valid = True

            # 策略3：关键词匹配（至少包含答案中的主要词汇）
            if not is_valid:
                answer_words = [w for w in answer.split() if len(w) > 2]  # 过滤短词
                if answer_words:
                    matched_words = sum(1 for word in answer_words if word in verification_text)
                    # 如果答案中超过50%的关键词在验证文本中，认为有效
                    if matched_words / len(answer_words) >= 0.5:
                        is_valid = True

            # 策略4：部分匹配（答案长度大于10时，检查是否包含主要部分）
            if not is_valid and len(answer) > 10:
                # 尝试匹配答案的主要部分（去除首尾各20%）
                main_part_start = len(answer) // 5
                main_part_end = len(answer) - len(answer) // 5
                main_part = answer[main_part_start:main_part_end]
                if main_part in verification_text:
                    is_valid = True

        if is_valid:
            oracle_valid_qa.append(qa)
        else:
            print(f"Oracle 过滤：QA 答案'{qa['answer']}'不在{verification_scope}中，已剔除")

    print(f"\n质量验证完成，最终通过 QA 对数量：{len(oracle_valid_qa)}")
    return oracle_valid_qa


# -------------------------- 8. 主函数：串联完整流程 --------------------------
def novelhopqa_constructor(raw_novel_text: str, source_file: str = None) -> List[Dict]:
    """
    NovelHopQA 问题构造主函数：输入小说纯文本，输出高质量 QA 对
    :param raw_novel_text: 输入的小说纯文本
    :param source_file: 源文件名（用于记录来源）
    :return: 最终通过验证的 QA 对列表
    """
    # 初始化 LLM 客户端
    api_client = LLMClient(Config.MODEL_NAME)

    try:
        # 步骤1：文本预处理（分割为 64k-128k token 片段）
        print("="*50)
        print("步骤1：开始文本预处理")
        segments = preprocess_novel_text(raw_novel_text)
        if not segments:
            raise ValueError("预处理后无有效片段，无法继续")

        # 步骤2：处理每个片段（关键词提取→跳链生成→QA生成→验证）
        all_final_qa = []
        for seg_idx, segment in enumerate(segments, 1):
            print("\n" + "="*50)
            print(f"处理片段 {seg_idx}/{len(segments)}（token 数：{Config.count_tokens(segment)}）")

            # 子步骤2.1：提取关键词
            print("子步骤2.1：提取关键词")
            keywords = extract_keywords(segment, api_client)
            if not keywords["entity_keywords"] or not keywords["event_keywords"]:
                print("关键词提取失败，跳过当前片段")
                continue

            # 子步骤2.2：生成跳分离链
            print("子步骤2.2：生成跳分离链")
            hop_chains = generate_hop_chains(segment, keywords, Config.HOP_DEPTHS, api_client)
            if not hop_chains:
                print("跳链生成失败，跳过当前片段")
                continue

            # 子步骤2.3：生成 QA 对
            print("子步骤2.3：生成 QA 对")
            qa_pairs = generate_qa_pairs(hop_chains, segment, api_client, source_file)
            if not qa_pairs:
                print("QA 生成失败，跳过当前片段")
                continue

            # 子步骤2.4：质量验证（使用项目级validator）
            print("子步骤2.4：质量验证（小说QA验证器-宽松模式）")
            
            project_validator = create_loose_validator()
            final_qa = project_validator.validate_all_qa(qa_pairs, segment)
            all_final_qa.extend(final_qa)

        # 步骤3：保存结果（JSONL格式）
        if all_final_qa:
            with open(Config.OUTPUT_PATH, "w", encoding="utf-8") as f:
                for qa in all_final_qa:
                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")
            print(f"\n" + "="*50)
            print(f"所有流程完成！最终 QA 对已保存至：{Config.OUTPUT_PATH}")
            print(f"总 QA 数量：{len(all_final_qa)}（含 1-4 跳）")
        else:
            print("\n所有片段处理完成，但未生成有效 QA 对")

        return all_final_qa

    except Exception as e:
        print(f"\n流程执行失败：{str(e)}")
        return []


# -------------------------- 9. 文件输入功能 --------------------------
def read_novel_from_file(file_path: str) -> str:
    """
    从txt文件读取小说文本
    :param file_path: txt文件路径
    :return: 文件内容字符串
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"成功读取文件：{file_path}")
        print(f"文件内容长度：{len(content)} 字符")
        return content
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return ""
    except UnicodeDecodeError:
        print(f"错误：文件 {file_path} 编码格式不支持，请确保文件为UTF-8编码")
        return ""
    except Exception as e:
        print(f"错误：读取文件 {file_path} 时发生异常：{str(e)}")
        return ""


# -------------------------- 10. 主函数：支持文件输入 --------------------------
if __name__ == "__main__":
    import argparse

    # 设置命令行参数
    parser = argparse.ArgumentParser(description="NovelHopQA 问题构造工具")
    parser.add_argument("--input_file", "-i", type=str, required=True,
                       help="输入的小说txt文件路径")
    parser.add_argument("--output_file", "-o", type=str,
                       default="novelhopqa_qa_pairs_1.json",
                       help="输出的QA对文件路径（默认：novelhopqa_qa_pairs.json）")

    args = parser.parse_args()

    # 更新输出路径
    Config.OUTPUT_PATH = args.output_file

    # 从文件读取小说文本
    novel_text = read_novel_from_file(args.input_file)
    if not novel_text:
        print("无法读取文件内容，程序退出")
        exit(1)

    # 提取文件名（不包含路径）
    import os
    source_filename = os.path.basename(args.input_file)

    # 运行 NovelHopQA 构造流程
    print(f"开始处理文件：{args.input_file}")
    print(f"输出文件：{args.output_file}")
    print(f"源文件名：{source_filename}")
    novelhopqa_constructor(novel_text, source_filename)