import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, BitsAndBytesConfig
from data_manager import DataManager
from vocab_indexer import VocabIndexer
from state_machine import GenerationStateMachine
from logits_processor import ConstraintLogitsProcessor
import json

def build_prompt_messages(task_type: str, cipai: str, theme: str, requirement: str = "", cipai_data_path: str = "PoeTone-main/data/cipai_data.json", poem_path: str = "Meter/songci.json", use_thinking: bool = True):
    """
    根据给定的任务类型，构造对应的大模型 Prompt 消 Messages 列表（支持 zero-shot, one-shot, completion, instruction）
    """
    messages = []
    
    # 构造可选的详细写作要求
    req_text = f"\n详细写作要求：{requirement}\n" if requirement else ""

    if task_type == "zero-shot":
        messages = [
            {"role": "system", "content": "你是一位宋代词人。请按照用户提供的词牌、主题和详细要求创作一首词。\n\n你的输出必须遵循以下格式：\n1. 首先输出你对主题和创作思路的简要分析。\n2. 接着输出标题，格式为：[title]词牌·标题（例如：[title]浣溪沙·续写）\n [content]正文。\n 正文中绝对不得包含大纲、段落标记等废话。"},
            {"role": "user", "content": f"请以《{cipai}》为词牌，以“{theme}”为主题，创作一首宋词。{req_text}"}
        ]
        
    elif task_type in ["one-shot", "completion", "instruction"]:
        # 需要加载外部数据
        if task_type in ["one-shot", "completion"]:
            try:
                with open(cipai_data_path, 'r', encoding='utf-8') as f:
                    cipai_data = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"{cipai_data_path} 不存在，必须要有此数据文件才能运行 {task_type}。")
            
        if task_type == "one-shot":
            example = cipai_data["one_shot_examples"].get(cipai, "")
            messages = [
                {"role": "system", "content": "你是一位宋代词人，擅长模仿范例进行创作。\n\n你的输出必须遵循以下格式：\n1. 首先输出你对范例风格的分析及你的创作思路。\n2. 接着输出标题，格式为：[title]词牌·标题（例如：[title]浣溪沙·续写）\n [content]正文。，然后紧接着输出正文。"},
                {"role": "user", "content": f"这是一首以《{cipai}》为词牌的范例：\n\n{example}\n\n现在，请模仿这首词的风格和格律，以“{theme}”为主题，创作一首全新的词。{req_text}"}
            ]
        elif task_type == "completion":
            first_half = cipai_data["completion_data"].get(cipai, {}).get("first_half", "")
            messages = [
                {"role": "system", "content": "你是一位宋代词人，擅长续写词作。\n\n你的输出必须遵循以下格式：\n1. 首先输出你对上阕意境的分析及你对下阕的构思。\n2. 接着输出标题，格式为：[title]词牌·标题（例如：[title]浣溪沙·续写）\n [content]正文。内容完全原创，不得与原词下阕雷同。"},
                {"role": "user", "content": f"这是著名词牌《{cipai}》的上阕：\n\n{first_half}\n\n请你以此为开篇，围绕“{theme}”这一主题，创作一个全新的下阕。{req_text}"}
            ]
        elif task_type == "instruction":
            # 动态从本地的格律文件生成详细格律规则
            with open(poem_path, 'r', encoding='utf-8') as sf:
                poem_data = json.load(sf)
            if cipai not in poem_data:
                raise ValueError(f"词牌 {cipai} 未在 {poem_path} 中找到。")
            
            c_dict = poem_data[cipai]
            rules = f"【{cipai}】格律要求：\n要求押{c_dict.get('rhyme_type', '韵')}。\n"
            rules += "注：格律中的“/”表示词句内部的节奏停顿（你无需输出标点，只需体会节奏），“、”表示此处必须输出顿号作为明确的句读。\n"
            for i in range(c_dict.get('number_of_stanzas', 2)):
                stanza = c_dict.get(f"stanza{i+1}", {})
                lines = stanza.get("lines", [])
                
                rhyme_marks = {}
                for k, v in stanza.items():
                    if k.startswith("rhyme_") and k.endswith("_positions"):
                        num = k.split("_")[1]
                        for pos in v:
                            rhyme_marks[pos] = f"（此句末尾需押第{num}部韵）"

                rules += f"第{i+1}阕：\n"
                for j, line_pattern in enumerate(lines):
                    rhyme_mark = rhyme_marks.get(j + 1, "")
                    # 修复：计算字数时不包含 /
                    pure_pattern = line_pattern.replace("/", "")
                    rules += f" - 第{j+1}句 ({len(pure_pattern)}字)：{line_pattern} {rhyme_mark}\n"
                    
            messages = [
                {"role": "system", "content": "你是一位宋代词人。请根据用户提供的词牌、主题以及格律要求创作一首词。\n\n你必须严格遵循以下输出格式，绝对不能遗漏任何标记：\n首先，写出你对主题的理解及布局分析。\n接着，必须换行并输出标题，格式为：\n[title]词牌·标题\n最后，必须换行并严格输出 [content] 标记，紧接着输出正文：\n[content]正文。\n\n**关键要求**：\n1. `[title]` 和 `[content]` 标记是程序解析的依赖，绝对不可以省略、修改或替换！\n2. 正文中不得包含段落标记（如“第一片”）、注脚或额外废话！不能存在“平仄中”的格律文本。"},
                {"role": "user", "content": f"请为我创作一首词。\n主题：“{theme}”\n词牌：《{cipai}》\n{req_text}\n必须遵守以下格律：\n{rules}\n请先输出分析，然后必须输出 `[title]词牌·标题`，最后必须输出 `[content]正文`。不要遗漏 `[content]` 标记！\n请开始创作："}
            ]
            
    if not use_thinking:
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = "/no_think " + msg["content"]
            
    return messages

def build_tangpoem_prompt_messages(task_type: str, cipai: str, theme: str, requirement: str = "", poem_path: str = "Meter/TongPoem.json", use_thinking: bool = True):
    """
    构造唐诗专用的大模型 Prompt Messages 列表
    """
    messages = []
    
    req_text = f"\n详细写作要求：{requirement}\n" if requirement else ""

    if task_type == "instruction":
        # 动态从本地的唐诗 JSON 生成详细格律规则
        with open(poem_path, 'r', encoding='utf-8') as sf:
            poem_data = json.load(sf)
        if cipai not in poem_data:
            raise ValueError(f"诗体 {cipai} 未在 {poem_path} 中找到。")
        
        c_dict = poem_data[cipai]
        rules = f"【{cipai}】格律要求：\n要求押{c_dict.get('rhyme_type', '韵')}。\n"
        rules += "注：格律中的“/”表示句内部的节奏停顿（你无需输出标点，只需体会节奏）。奇数句将自动生成逗号，偶数句将自动生成句号。\n"
        for i in range(c_dict.get('number_of_stanzas', 1)):
            stanza = c_dict.get(f"stanza{i+1}", {})
            lines = stanza.get("lines", [])
            
            rhyme_marks = {}
            for k, v in stanza.items():
                if k.startswith("rhyme_") and k.endswith("_positions"):
                    num = k.split("_")[1] if len(k.split("_")) >= 3 and k.split("_")[1].isdigit() else "1"
                    for pos in v:
                        rhyme_marks[pos] = f"（此句末尾需押第{num}部韵）"

            rules += f"第{i+1}段：\n"
            for j, line_pattern in enumerate(lines):
                rhyme_mark = rhyme_marks.get(j + 1, "")
                pure_pattern = line_pattern.replace("/", "")
                rules += f" - 第{j+1}句 ({len(pure_pattern)}字)：{line_pattern} {rhyme_mark}\n"
                
        messages = [
            {"role": "system", "content": "你是一位唐代诗人。请根据用户提供的诗体、主题以及格律要求创作一首唐诗。\n\n你必须严格遵循以下输出格式，绝对不能遗漏任何标记：\n首先，写出你对主题的理解及布局分析。\n接着，必须换行并输出标题，格式为：\n[title]诗体·标题\n最后，必须换行并严格输出 [content] 标记，紧接着输出正文：\n[content]正文。\n\n**关键要求**：\n1. `[title]` 和 `[content]` 标记是程序解析的依赖，绝对不可以省略、修改或替换！\n2. 正文中不得包含段落标记、注脚或额外废话！不能存在“平仄中”的格律文本。"},
            {"role": "user", "content": f"请为我创作一首唐诗。\n主题：“{theme}”\n体裁：《{cipai}》\n{req_text}\n必须遵守以下格律：\n{rules}\n请先输出分析，然后必须输出 `[title]诗体·标题`，最后必须输出 `[content]正文`。不要遗漏 `[content]` 标记！\n请开始创作："}
        ]
        
    if not use_thinking:
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = "/no_think " + msg["content"]
            
    return messages

def main():
    # ================= 集中配置区域 =================
    # 1. 模型配置
    model_name = r"C:\Users\26051\.cache\modelscope\hub\models\Qwen\Qwen3-4B"
    # model_name = r"C:\Users\26051\.cache\modelscope\hub\models\LLM-Research\Llama-3.2-3B-Instruct"
    # model_name = r"C:\Users\26051\.cache\modelscope\hub\models\deepseek-ai\DeepSeek-R1-Distill-Qwen-1.5B"
    use_bitsandbytes = True

    # 2. 任务与生成配置
    meter_type = "唐诗"  # 可选："宋词", "唐诗"
    rhyme_dict_name = "Xinyun"  # 可选："Cilin" (词林正韵), "Pinshui" (平水韵), "Xinyun" (中华新韵)
    
    task_type = "instruction"
    theme = "祝福智科专业学习学妹们前程似锦，学业有成，未来可期"
    cipai_name = "五律仄起"
    detailed_requirement = """
    祝福智科专业学习学妹们前程似锦，学业有成，未来可期
    """
    
    use_constraints = True  # 设置为 False 即可进行无约束对比实验
    use_thinking = False    # DeepSeek R1 必须设为 True 以保留 <think> 思考过程
    num_generations = 3     # 多次输出模式下生成的数量（设置为 1 即单次）
    save_output = False     # True 是否将结果保存到 output 目录
    # ===============================================

    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model {model_name} (use_bitsandbytes={use_bitsandbytes})...")
    # 根据是否启用 bitsandbytes 构建 from_pretrained 参数
    if use_bitsandbytes:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    # 1. 基础数据准备
    rhyme_dict_path = f"Rhyme/{rhyme_dict_name}.json"
    if meter_type == "唐诗":
        poem_path = "Meter/TongPoem.json"
        is_tangpoem = True
    else:
        poem_path = "Meter/songci.json"
        is_tangpoem = False

    data_manager = DataManager(rhyme_dict_path=rhyme_dict_path, poem_path=poem_path)

    # 2.词表索引构建 (离线运行一次)
    vocab_indexer = VocabIndexer(tokenizer, data_manager)

    # 3. 构建 Prompt (运用模仿 PoeTone 的四种任务策略)
    print(f"\nBuilding prompt for task: {task_type} (Theme: {theme})")
    if is_tangpoem:
        messages = build_tangpoem_prompt_messages(task_type, cipai_name, theme, requirement=detailed_requirement, poem_path=poem_path, use_thinking=use_thinking)
    else:
        messages = build_prompt_messages(task_type, cipai_name, theme, requirement=detailed_requirement, poem_path=poem_path, use_thinking=use_thinking)

    # 使用 Chat Template
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    input_prompt_len = inputs.input_ids.shape[1]

    if save_output:
        output_dir = os.path.join("output", cipai_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{cipai_name}.txt")
        # 如果需要每次运行清空旧文件可以追加此句：
        # open(output_file, 'w', encoding='utf-8').close()

    print(f"Debug: use_thinking = {use_thinking}")
    print(f"Debug: Last prompt message = {messages[-1]['content']}")

    if use_constraints:
        print(f"\nStarting generation ({num_generations} times) with constrained decoding...")
    else:
        print(f"\nStarting generation ({num_generations} times) without constraints (free decoding)...")

    for i in range(num_generations):
        print(f"\n=== [Generation {i+1}/{num_generations}] ===")

        # 4. 初始化状态机和干预器（每次生成必须重新初始化，因为状态机内部包含断点、押韵等历史状态）
        processors = None
        if use_constraints:
            state_machine = GenerationStateMachine(cipai_name, data_manager)
            logits_processor = ConstraintLogitsProcessor(
                vocab_indexer=vocab_indexer,
                state_machine=state_machine,
                tokenizer=tokenizer,
                input_prompt_len=input_prompt_len
            )
            logits_processor.is_tangpoem_flag = is_tangpoem
            processors = LogitsProcessorList([logits_processor])

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                logits_processor=processors,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.8
            )

        outputs = tokenizer.decode(output_ids[0][input_prompt_len:], skip_special_tokens=True)

        print("\n[生成结果]")
        print(outputs)

        if save_output:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"=== 作品 {i+1} ===\n")
                f.write(outputs.strip())
                f.write("\n\n")
            print(f"已将作品 {i+1} 存入 {output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[启动失败] {exc}")
        raise
