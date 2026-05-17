#!/usr/bin/env python3
"""
批量诗词生成实验脚本 — 读取 batch_config.json，遍历词牌/诗体 × 主题 × 多次生成，
输出到 experiments/output/ 目录。

文件命名: {模型}-{词牌/诗体}-{主题}.txt
"""

import torch
import os
import sys
import json
import traceback
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, BitsAndBytesConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from data_manager import DataManager
from vocab_indexer import VocabIndexer
from state_machine import GenerationStateMachine, TangPoemStateMachine
from logits_processor import ConstraintLogitsProcessor, TangPoemLogitsProcessor
from main import parse_tang_format, build_prompt_messages, build_tangpoem_prompt_messages


def sanitize_filename(s):
    return s.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_one_batch(model, tokenizer, inputs, input_prompt_len, processors,
                       gen_params, num_generations, tokenizer_pad_id):
    """运行 num_generations 次生成，返回 decoded 文本列表。"""
    results = []
    for i in range(num_generations):
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_params["max_new_tokens"],
                    logits_processor=processors,
                    pad_token_id=tokenizer_pad_id,
                    do_sample=gen_params["do_sample"],
                    top_p=gen_params["top_p"],
                    temperature=gen_params["temperature"]
                )
            except RuntimeError:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                raise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        output_text = tokenizer.decode(output_ids[0][input_prompt_len:], skip_special_tokens=True)
        results.append(output_text)
    return results


def run_songci_experiments(config, model, tokenizer, vocab_indexer, data_manager_songci, gen_params, output_dir):
    """执行所有宋词实验。"""
    sc = config["songci"]
    model_cfg = config["models"][0]  # 当前仅支持单模型
    model_name = model_cfg["name"]
    use_thinking = model_cfg["use_thinking"]
    rhyme_dict_name = sc["rhyme_dict_name"]
    task_type = sc["task_type"]
    num_generations = sc["num_generations"]
    tokenizer_pad_id = tokenizer.eos_token_id

    total = len(sc["cipai_list"]) * len(sc["themes"])
    pbar = tqdm(total=total, desc="宋词批量生成", unit="组")

    for cipai in sc["cipai_list"]:
        for entry in sc["themes"]:
            theme = entry["theme"]
            detailed_req = entry.get("detailed_requirement", "")
            safe_theme = sanitize_filename(theme)
            out_file = os.path.join(output_dir, f"{model_name}-{cipai}-{safe_theme}.txt")

            pbar.set_postfix_str(f"{cipai}·{theme}")

            try:
                messages = build_prompt_messages(
                    task_type, cipai, theme,
                    requirement=detailed_req,
                    poem_path=os.path.join(PROJECT_ROOT, "Songci_Meter"),
                    use_thinking=use_thinking,
                    rhyme_dict_name=rhyme_dict_name
                )
            except Exception as e:
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(f"[错误] 构建 Prompt 失败: {e}\n")
                pbar.update(1)
                continue

            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(f"# 模型: {model_name}\n")
                f.write(f"# 词牌: {cipai}\n")
                f.write(f"# 主题: {theme}\n")
                f.write(f"# 韵书: {rhyme_dict_name}\n")
                f.write(f"# task_type: {task_type}\n")
                f.write(f"# 每词牌生成数: {num_generations}\n")
                f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# ========================================\n\n")

            for gen_idx in range(num_generations):
                inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
                input_prompt_len = inputs.input_ids.shape[1]

                state_machine = GenerationStateMachine(cipai, data_manager_songci)
                logits_processor = ConstraintLogitsProcessor(
                    vocab_indexer=vocab_indexer,
                    state_machine=state_machine,
                    tokenizer=tokenizer,
                    input_prompt_len=input_prompt_len
                )
                processors = LogitsProcessorList([logits_processor])

                try:
                    results = generate_one_batch(
                        model, tokenizer, inputs, input_prompt_len,
                        processors, gen_params, 1, tokenizer_pad_id
                    )
                    output_text = results[0]
                except Exception as e:
                    output_text = f"[生成错误] {e}"

                with open(out_file, 'a', encoding='utf-8') as f:
                    f.write(f"=== 作品 {gen_idx + 1} ===\n")
                    f.write(output_text.strip())
                    f.write("\n\n")

            pbar.update(1)

    pbar.close()


def run_tangpoem_experiments(config, model, tokenizer, vocab_indexer, data_manager, gen_params, output_dir):
    """执行所有唐诗实验。"""
    tp = config["tangpoem"]
    model_cfg = config["models"][0]
    model_name = model_cfg["name"]
    use_thinking = model_cfg["use_thinking"]
    rhyme_dict_name = tp["rhyme_dict_name"]
    task_type = tp["task_type"]
    num_generations = tp["num_generations"]
    tokenizer_pad_id = tokenizer.eos_token_id

    total = len(tp["forms"]) * len(tp["themes"])
    pbar = tqdm(total=total, desc="唐诗批量生成", unit="组")

    for form in tp["forms"]:
        form_name = form["name"]
        line_length = form["line_length"]
        num_lines = form["num_lines"]

        for entry in tp["themes"]:
            theme = entry["theme"]
            detailed_req = entry.get("detailed_requirement", "")
            safe_theme = sanitize_filename(theme)
            out_file = os.path.join(output_dir, f"{model_name}-{form_name}-{safe_theme}.txt")

            pbar.set_postfix_str(f"{form_name}·{theme}")

            try:
                messages = build_tangpoem_prompt_messages(
                    task_type, form_name, theme,
                    requirement=detailed_req,
                    use_thinking=use_thinking,
                    line_length=line_length,
                    num_lines=num_lines,
                    rhyme_dict_name=rhyme_dict_name
                )
            except Exception as e:
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(f"[错误] 构建 Prompt 失败: {e}\n")
                pbar.update(1)
                continue

            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            _, _, rhyme_type = parse_tang_format(form_name)

            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(f"# 模型: {model_name}\n")
                f.write(f"# 诗体: {form_name}\n")
                f.write(f"# 主题: {theme}\n")
                f.write(f"# 韵书: {rhyme_dict_name}\n")
                f.write(f"# task_type: {task_type}\n")
                f.write(f"# 每体裁生成数: {num_generations}\n")
                f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# ========================================\n\n")

            for gen_idx in range(num_generations):
                inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
                input_prompt_len = inputs.input_ids.shape[1]

                state_machine = TangPoemStateMachine(
                    line_length=line_length,
                    num_lines=num_lines,
                    rhyme_type=rhyme_type,
                    data_manager=data_manager
                )
                logits_processor = TangPoemLogitsProcessor(
                    vocab_indexer=vocab_indexer,
                    state_machine=state_machine,
                    tokenizer=tokenizer,
                    input_prompt_len=input_prompt_len
                )
                processors = LogitsProcessorList([logits_processor])

                try:
                    results = generate_one_batch(
                        model, tokenizer, inputs, input_prompt_len,
                        processors, gen_params, 1, tokenizer_pad_id
                    )
                    output_text = results[0]
                except Exception as e:
                    output_text = f"[生成错误] {e}"

                with open(out_file, 'a', encoding='utf-8') as f:
                    f.write(f"=== 作品 {gen_idx + 1} ===\n")
                    f.write(output_text.strip())
                    f.write("\n\n")

            pbar.update(1)

    pbar.close()


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_config.json")
    config = load_config(config_path)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    model_cfg = config["models"][0]
    model_name = model_cfg["name"]
    model_path = model_cfg["path"]
    use_bitsandbytes = model_cfg.get("use_bitsandbytes", True)
    gen_params = config["generation_params"]

    print(f"=" * 60)
    print(f"  批量诗词生成实验")
    print(f"  模型: {model_name}")
    print(f"  输出目录: {output_dir}")
    print(f"  宋词: {'启用' if config['songci']['enabled'] else '禁用'}")
    print(f"  唐诗: {'启用' if config['tangpoem']['enabled'] else '禁用'}")
    print(f"=" * 60)

    # 加载模型
    print(f"\n[1/3] 加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[2/3] 加载模型 (use_bitsandbytes={use_bitsandbytes})...")
    if use_bitsandbytes:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    # 构建词表索引（宋词和唐诗共用）
    print("[3/3] 构建词表索引...")
    rhyme_dict_path = os.path.join(PROJECT_ROOT, "Rhyme", f"{config['songci']['rhyme_dict_name']}.json")
    poem_path = os.path.join(PROJECT_ROOT, "Songci_Meter")
    data_manager = DataManager(rhyme_dict_path=rhyme_dict_path, poem_path=poem_path)
    vocab_indexer = VocabIndexer(tokenizer, data_manager)

    # 执行宋词实验
    if config["songci"]["enabled"]:
        print(f"\n{'=' * 60}")
        print(f"  开始宋词批量生成")
        print(f"  词牌数: {len(config['songci']['cipai_list'])}")
        print(f"  主题数: {len(config['songci']['themes'])}")
        print(f"  每组合生成数: {config['songci']['num_generations']}")
        print(f"  总文件数: {len(config['songci']['cipai_list']) * len(config['songci']['themes'])}")
        print(f"  总作品数: {len(config['songci']['cipai_list']) * len(config['songci']['themes']) * config['songci']['num_generations']}")
        print(f"{'=' * 60}")

        run_songci_experiments(
            config, model, tokenizer, vocab_indexer, data_manager,
            gen_params, output_dir
        )

    # 执行唐诗实验
    if config["tangpoem"]["enabled"]:
        print(f"\n{'=' * 60}")
        print(f"  开始唐诗批量生成")
        print(f"  诗体数: {len(config['tangpoem']['forms'])}")
        print(f"  主题数: {len(config['tangpoem']['themes'])}")
        print(f"  每组合生成数: {config['tangpoem']['num_generations']}")
        print(f"  总文件数: {len(config['tangpoem']['forms']) * len(config['tangpoem']['themes'])}")
        print(f"  总作品数: {len(config['tangpoem']['forms']) * len(config['tangpoem']['themes']) * config['tangpoem']['num_generations']}")
        print(f"{'=' * 60}")

        run_tangpoem_experiments(
            config, model, tokenizer, vocab_indexer, data_manager,
            gen_params, output_dir
        )

    print(f"\n{'=' * 60}")
    print(f"  全部实验完成！")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[批量生成失败] {exc}")
        traceback.print_exc()
        raise
