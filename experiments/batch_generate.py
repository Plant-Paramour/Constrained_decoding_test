#!/usr/bin/env python3
"""
批量诗词生成实验脚本 — 读取 batch_config.json，遍历词牌/诗体 × 主题 × 多次生成，
输出到 experiments/output/ 目录。

支持：
  - 自动检测 GPU 显存 / 系统内存
  - 多 GPU 并行加速（spawn 独立子进程，每 GPU 一个模型副本）
  - 单 GPU / CPU 串行兜底
  - 灵活量化配置：none / fp16 / 8bit / 4bit

文件命名: {模型}-({task_type})-{词牌/诗体}-{主题}.txt
"""

import torch
import os
import sys
import json
import traceback
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, BitsAndBytesConfig

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from data_manager import DataManager
from vocab_indexer import VocabIndexer
from state_machine import GenerationStateMachine, TangPoemStateMachine
from logits_processor import ConstraintLogitsProcessor, TangPoemLogitsProcessor
from main import parse_tang_format, build_prompt_messages, build_tangpoem_prompt_messages


# ============================================================
#  量化参数工具
# ============================================================

def _get_quantization_kwargs(quantization: str):
    """
    解析量化参数，返回 AutoModelForCausalLM.from_pretrained() 的关键字参数字典。

    支持的值（大小写不敏感）：
      - "none" / "fp16" / "float16"  → FP16 半精度，不量化
      - "8bit" / "int8"              → 8-bit 量化（BitsAndBytes）
      - "4bit" / "int4" / "nf4"      → 4-bit NF4 量化（BitsAndBytes，双重量化）
    """
    q = quantization.lower().strip()
    if q in ("none", "fp16", "float16", ""):
        return {"torch_dtype": torch.float16}
    elif q in ("8bit", "int8", "8"):
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    elif q in ("4bit", "int4", "nf4", "4"):
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        }
    else:
        raise ValueError(
            f"不支持的量化参数: '{quantization}'，可选值: none, fp16, 8bit, 4bit"
        )


def _quantization_label(q: str) -> str:
    """返回量化的可读标签。"""
    q = q.lower().strip()
    mapping = {"none": "FP16", "fp16": "FP16", "float16": "FP16",
               "8bit": "8-bit", "int8": "8-bit",
               "4bit": "4-bit NF4", "int4": "4-bit NF4", "nf4": "4-bit NF4"}
    return mapping.get(q, q)


# ============================================================
#  硬件检测模块
# ============================================================

class HardwareDetector:
    """检测当前设备的 CPU 内存、GPU 显存，给出并行度建议。"""

    @staticmethod
    def get_system_ram_info():
        """返回系统内存 (total_gb, available_gb, used_gb, percent_used)。"""
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            return {
                'total_gb': round(mem.total / (1024 ** 3), 1),
                'available_gb': round(mem.available / (1024 ** 3), 1),
                'used_gb': round(mem.used / (1024 ** 3), 1),
                'percent_used': mem.percent,
            }
        return {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'percent_used': 0}

    @staticmethod
    def get_gpu_info():
        """返回每张 GPU 的详细信息列表。"""
        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_vram = round(props.total_memory / (1024 ** 3), 1)
            try:
                free_mem, _ = torch.cuda.mem_get_info(i)
                free_vram = round(free_mem / (1024 ** 3), 1)
            except Exception:
                free_vram = None
            allocated = round(torch.cuda.memory_allocated(i) / (1024 ** 3), 2)
            gpus.append({
                'index': i,
                'name': props.name,
                'total_vram_gb': total_vram,
                'free_vram_gb': free_vram,
                'allocated_gb': allocated,
                'compute_capability': f'{props.major}.{props.minor}',
                'multi_processor_count': props.multi_processor_count,
            })
        return gpus

    @staticmethod
    def estimate_model_vram_gb(model_path, quantization="8bit"):
        """根据模型参数量 + 量化方案估算单副本显存占用（GB）。"""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(config, 'num_parameters'):
                param_count = config.num_parameters
            elif hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
                h, L, V = config.hidden_size, config.num_hidden_layers, config.vocab_size
                param_count = 12 * h * h * L + V * h
            else:
                return None
        except Exception:
            return None

        q = quantization.lower().strip()
        if q in ("4bit", "int4", "nf4", "4"):
            bytes_per_param = 0.5
        elif q in ("8bit", "int8", "8"):
            bytes_per_param = 1.0
        else:  # none / fp16
            bytes_per_param = 2.0

        overhead = 1.15  # optimizer buffers / KV cache 等
        return round(param_count * bytes_per_param * overhead / (1024 ** 3), 2)

    @staticmethod
    def recommend_parallelism(model_vram_estimate_gb=None, safety_margin=0.15):
        """返回并行建议：最大 worker 数、策略类型、GPU 分配方案。"""
        gpus = HardwareDetector.get_gpu_info()

        if not gpus:
            cpu_count = os.cpu_count() or 4
            return {
                'max_parallel_workers': 1,
                'strategy': 'cpu_only',
                'gpu_assignments': [],
                'reason': '未检测到 GPU，回退至串行模式',
            }

        if len(gpus) == 1:
            gpu = gpus[0]
            free_vram = gpu['free_vram_gb'] or gpu['total_vram_gb']
            if model_vram_estimate_gb and free_vram > model_vram_estimate_gb * (1 + safety_margin) * 2:
                return {
                    'max_parallel_workers': 1,
                    'strategy': 'single_gpu',
                    'gpu_assignments': [0],
                    'reason': f'单 GPU 空闲显存 {free_vram} GB，模型预估 {model_vram_estimate_gb} GB，串行模式',
                }
            return {
                'max_parallel_workers': 1,
                'strategy': 'single_gpu',
                'gpu_assignments': [0],
                'reason': f'单 GPU (空闲 {free_vram} GB)，串行模式',
            }

        workers = []
        for gpu in gpus:
            free_vram = gpu['free_vram_gb'] or gpu['total_vram_gb']
            if model_vram_estimate_gb is None or free_vram > model_vram_estimate_gb * (1 + safety_margin):
                workers.append(gpu['index'])

        if len(workers) >= 2:
            return {
                'max_parallel_workers': len(workers),
                'strategy': 'multi_gpu',
                'gpu_assignments': workers,
                'reason': f'{len(gpus)} 张 GPU 可用，{len(workers)} 张空闲显存充足，启用多 GPU 并行',
            }
        return {
            'max_parallel_workers': 1,
            'strategy': 'single_gpu',
            'gpu_assignments': [workers[0]] if workers else [0],
            'reason': '多 GPU 但仅有 1 张空闲显存充足，回退串行',
        }

    @staticmethod
    def print_summary(model_path=None, quantization="8bit"):
        """打印硬件资源摘要并返回并行建议。"""
        q_label = _quantization_label(quantization)
        print(f"\n{'=' * 60}")
        print(f"  🔍 硬件资源检测")
        print(f"{'=' * 60}")

        ram = HardwareDetector.get_system_ram_info()
        if HAS_PSUTIL:
            print(f"  系统内存 : 总计 {ram['total_gb']} GB |"
                  f" 可用 {ram['available_gb']} GB | 已用 {ram['percent_used']}%")
        else:
            print(f"  系统内存 : (psutil 未安装，无法检测)")

        gpus = HardwareDetector.get_gpu_info()
        if not gpus:
            print(f"  GPU      : 无 — 将以 CPU 模式运行")
        else:
            print(f"  GPU 数量 : {len(gpus)}")
            for gpu in gpus:
                free_str = f"空闲 {gpu['free_vram_gb']} GB" if gpu['free_vram_gb'] is not None else "空闲 N/A"
                print(f"    GPU {gpu['index']} : {gpu['name']}"
                      f" | 总计 {gpu['total_vram_gb']} GB | {free_str}"
                      f" | CC {gpu['compute_capability']}")

        model_est = None
        if model_path:
            model_est = HardwareDetector.estimate_model_vram_gb(model_path, quantization)
            if model_est:
                print(f"\n  模型显存预估: ~{model_est} GB/副本 ({q_label})")

        rec = HardwareDetector.recommend_parallelism(model_est)
        print(f"\n  量化方案 : {q_label}")
        print(f"  并行策略 : {rec['strategy']}")
        print(f"  推荐并行度: {rec['max_parallel_workers']}")
        print(f"  原因      : {rec['reason']}")
        if rec['gpu_assignments']:
            print(f"  GPU 分配  : {rec['gpu_assignments']}")
        print(f"{'=' * 60}\n")
        return rec


# ============================================================
#  工具函数
# ============================================================

def sanitize_filename(s):
    return s.replace('/', '_').replace('\\', '_').replace(':', '_')\
            .replace('*', '_').replace('?', '_').replace('"', '_')\
            .replace('<', '_').replace('>', '_').replace('|', '_')


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_one_batch(model, tokenizer, inputs, input_prompt_len, processors,
                       gen_params, num_generations, tokenizer_pad_id):
    """运行 num_generations 次生成，返回 decoded 文本列表。"""
    results = []
    for _ in range(num_generations):
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_params["max_new_tokens"],
                    logits_processor=processors,
                    pad_token_id=tokenizer_pad_id,
                    do_sample=gen_params["do_sample"],
                    temperature=gen_params["temperature"],
                    top_p=gen_params["top_p"],
                    top_k=gen_params.get("top_k", 0),
                    min_p=gen_params.get("min_p", 0.0),
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


# ============================================================
#  配置分割 — 将全量任务切分为每个 worker 的子配置
# ============================================================

def _split_list_evenly(lst, num_chunks):
    """将列表均匀分割为 num_chunks 份。"""
    chunk_size = (len(lst) + num_chunks - 1) // num_chunks
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _build_worker_configs(config, num_workers):
    """根据并行 worker 数量，生成每个 worker 的子配置列表。"""
    worker_configs = []

    sc_enabled = config.get('songci', {}).get('enabled', False)
    tp_enabled = config.get('tangpoem', {}).get('enabled', False)

    songci_chunks = []
    if sc_enabled:
        sc = config['songci']
        cipai_list = sc.get('cipai_list', [])
        themes = sc.get('themes', [])
        tasks = []
        for cipai in cipai_list:
            for theme_entry in themes:
                tasks.append({
                    'cipai': cipai,
                    'theme': theme_entry['theme'],
                    'detailed_requirement': theme_entry.get('detailed_requirement', ''),
                })
        songci_chunks = _split_list_evenly(tasks, num_workers)

    tangpoem_chunks = []
    if tp_enabled:
        tp = config['tangpoem']
        forms = tp.get('forms', [])
        themes = tp.get('themes', [])
        tasks = []
        for form in forms:
            for theme_entry in themes:
                tasks.append({
                    'form': form,
                    'theme': theme_entry['theme'],
                    'detailed_requirement': theme_entry.get('detailed_requirement', ''),
                })
        tangpoem_chunks = _split_list_evenly(tasks, num_workers)

    for i in range(num_workers):
        wc = {
            'worker_index': i,
            'songci_tasks': songci_chunks[i] if i < len(songci_chunks) else [],
            'tangpoem_tasks': tangpoem_chunks[i] if i < len(tangpoem_chunks) else [],
        }
        worker_configs.append(wc)

    return worker_configs


# ============================================================
#  多 GPU 并行 Worker（子进程入口）
# ============================================================

def _parallel_worker(gpu_id, worker_tasks, config, gen_params, output_dir, project_root):
    """
    子进程入口：在指定 GPU 上加载模型，处理分配给本 worker 的所有任务。
    使用 spawn 上下文启动，每次启动重新初始化 CUDA。
    """
    worker_idx = worker_tasks['worker_index']

    # ---- 限制 CUDA 可见设备 ----
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # spawn 子进程中 torch 等模块会被重新导入，此处重新 import 以确保使用受限设备
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModel, \
        AutoTokenizer as _AutoTokenizer, BitsAndBytesConfig as _Bits, LogitsProcessorList as _LPL

    # 量化工具（子进程内联版本，避免跨进程依赖）
    def _wk_quant_kwargs(q: str):
        q = q.lower().strip()
        if q in ("none", "fp16", "float16", ""):
            return {"torch_dtype": _torch.float16}
        elif q in ("8bit", "int8", "8"):
            return {"quantization_config": _Bits(load_in_8bit=True)}
        elif q in ("4bit", "int4", "nf4", "4"):
            return {
                "quantization_config": _Bits(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=_torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            }
        else:
            raise ValueError(f"不支持的量化参数: '{q}'")

    # 重新设置工作目录和 sys.path
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from data_manager import DataManager as _DM
    from vocab_indexer import VocabIndexer as _VI
    from state_machine import GenerationStateMachine as _GSM, TangPoemStateMachine as _TPSM
    from logits_processor import ConstraintLogitsProcessor as _CLP, TangPoemLogitsProcessor as _TPLP
    from main import parse_tang_format as _ptf, build_prompt_messages as _bpm, \
        build_tangpoem_prompt_messages as _btpm

    model_cfg = config['models'][0]
    model_path = model_cfg['path']
    model_name = model_cfg['name']
    quantization = model_cfg.get('quantization', '8bit')
    use_thinking = model_cfg.get('use_thinking', False)

    sc_cfg = config.get('songci', {})
    tp_cfg = config.get('tangpoem', {})
    sc_rhyme = sc_cfg.get('rhyme_dict_name', 'Xinyun')
    tp_rhyme = tp_cfg.get('rhyme_dict_name', 'Xinyun')
    sc_task_type = sc_cfg.get('task_type', 'instruction')
    tp_task_type = tp_cfg.get('task_type', 'instruction')
    sc_num_gen = sc_cfg.get('num_generations', 1)
    tp_num_gen = tp_cfg.get('num_generations', 1)
    compare_mode = config.get('compare_experiment', False)
    pad_token_id = None

    q_label = _quantization_label(quantization) if '_quantization_label' in dir() else quantization

    def _log(msg):
        print(f"[GPU:{gpu_id}|W{worker_idx}] {msg}")

    # ---- 加载模型 ----
    _log(f"加载 tokenizer: {model_path}")
    tokenizer = _AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pad_token_id = tokenizer.eos_token_id

    _log(f"加载模型 (量化={q_label})...")
    model_kwargs = _wk_quant_kwargs(quantization)
    model = _AutoModel.from_pretrained(
        model_path,
        device_map='auto',
        trust_remote_code=True,
        **model_kwargs,
    ).eval()

    _log(f"模型加载完成，设备: {model.device}")

    # ---- 加载韵书 & 构建词表索引 ----
    rhyme_dict_path = os.path.join(project_root, 'Rhyme', f'{sc_rhyme}.json')
    poem_path = os.path.join(project_root, 'Songci_Meter')
    data_manager = _DM(rhyme_dict_path=rhyme_dict_path, poem_path=poem_path)
    vocab_indexer = _VI(tokenizer, data_manager)

    # ---- 宋词任务 ----
    songci_tasks = worker_tasks.get('songci_tasks', [])
    tangpoem_tasks = worker_tasks.get('tangpoem_tasks', [])

    def _run_one_songci_task(task, run_constrained):
        cipai = task['cipai']
        theme = task['theme']
        detailed_req = task.get('detailed_requirement', '')
        safe_theme = sanitize_filename(theme)
        subdir = 'constrained_decoding' if run_constrained else 'free_decoding'
        tag = f'[{cipai}·{theme}]'

        if compare_mode:
            out_file = os.path.join(output_dir, subdir,
                                    f'{model_name}-({sc_task_type})-{cipai}-{safe_theme}.txt')
        else:
            out_file = os.path.join(output_dir,
                                    f'{model_name}-({sc_task_type})-{cipai}-{safe_theme}.txt')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        try:
            messages = _bpm(
                sc_task_type, cipai, theme,
                requirement=detailed_req,
                poem_path=os.path.join(project_root, 'Songci_Meter'),
                use_thinking=use_thinking,
                rhyme_dict_name=sc_rhyme,
            )
        except Exception as e:
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(f'[错误] 构建 Prompt 失败: {e}\n')
            return

        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f'# 模型: {model_name}\n')
            f.write(f'# 词牌: {cipai}\n')
            f.write(f'# 主题: {theme}\n')
            f.write(f'# 韵书: {sc_rhyme}\n')
            f.write(f'# task_type: {sc_task_type}\n')
            f.write(f'# 量化方案: {q_label}\n')
            f.write(f'# 约束解码: {"启用" if run_constrained else "禁用"}\n')
            f.write(f'# 每词牌生成数: {sc_num_gen}\n')
            f.write(f'# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'# ========================================\n\n')

        for gen_idx in range(sc_num_gen):
            inputs = tokenizer(chat_prompt, return_tensors='pt').to(model.device)
            input_prompt_len = inputs.input_ids.shape[1]

            if run_constrained:
                state_machine = _GSM(cipai, data_manager)
                logits_processor = _CLP(
                    vocab_indexer=vocab_indexer,
                    state_machine=state_machine,
                    tokenizer=tokenizer,
                    input_prompt_len=input_prompt_len,
                )
                processors = _LPL([logits_processor])
            else:
                processors = None

            try:
                results = generate_one_batch(
                    model, tokenizer, inputs, input_prompt_len,
                    processors, gen_params, 1, pad_token_id,
                )
                output_text = results[0]
            except Exception as e:
                output_text = f'[生成错误] {e}'

            with open(out_file, 'a', encoding='utf-8') as f:
                f.write(f'=== 作品 {gen_idx + 1} ===\n')
                f.write(output_text.strip())
                f.write('\n\n')

        _log(f'{tag} {"约束" if run_constrained else "自由"} 生成完成 → {os.path.basename(out_file)}')

    def _run_one_tangpoem_task(task, run_constrained):
        form = task['form']
        form_name = form['name']
        line_length = form['line_length']
        num_lines = form['num_lines']
        theme = task['theme']
        detailed_req = task.get('detailed_requirement', '')
        safe_theme = sanitize_filename(theme)
        subdir = 'constrained_decoding' if run_constrained else 'free_decoding'
        tag = f'[{form_name}·{theme}]'

        if compare_mode:
            out_file = os.path.join(output_dir, subdir,
                                    f'{model_name}-({tp_task_type})-{form_name}-{safe_theme}.txt')
        else:
            out_file = os.path.join(output_dir,
                                    f'{model_name}-({tp_task_type})-{form_name}-{safe_theme}.txt')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        try:
            messages = _btpm(
                tp_task_type, form_name, theme,
                requirement=detailed_req,
                use_thinking=use_thinking,
                line_length=line_length,
                num_lines=num_lines,
                rhyme_dict_name=tp_rhyme,
            )
        except Exception as e:
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(f'[错误] 构建 Prompt 失败: {e}\n')
            return

        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, _, rhyme_type = _ptf(form_name)

        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f'# 模型: {model_name}\n')
            f.write(f'# 诗体: {form_name}\n')
            f.write(f'# 主题: {theme}\n')
            f.write(f'# 韵书: {tp_rhyme}\n')
            f.write(f'# task_type: {tp_task_type}\n')
            f.write(f'# 量化方案: {q_label}\n')
            f.write(f'# 约束解码: {"启用" if run_constrained else "禁用"}\n')
            f.write(f'# 每体裁生成数: {tp_num_gen}\n')
            f.write(f'# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'# ========================================\n\n')

        for gen_idx in range(tp_num_gen):
            inputs = tokenizer(chat_prompt, return_tensors='pt').to(model.device)
            input_prompt_len = inputs.input_ids.shape[1]

            if run_constrained:
                state_machine = _TPSM(
                    line_length=line_length,
                    num_lines=num_lines,
                    rhyme_type=rhyme_type,
                    data_manager=data_manager,
                )
                logits_processor = _TPLP(
                    vocab_indexer=vocab_indexer,
                    state_machine=state_machine,
                    tokenizer=tokenizer,
                    input_prompt_len=input_prompt_len,
                )
                processors = _LPL([logits_processor])
            else:
                processors = None

            try:
                results = generate_one_batch(
                    model, tokenizer, inputs, input_prompt_len,
                    processors, gen_params, 1, pad_token_id,
                )
                output_text = results[0]
            except Exception as e:
                output_text = f'[生成错误] {e}'

            with open(out_file, 'a', encoding='utf-8') as f:
                f.write(f'=== 作品 {gen_idx + 1} ===\n')
                f.write(output_text.strip())
                f.write('\n\n')

        _log(f'{tag} {"约束" if run_constrained else "自由"} 生成完成 → {os.path.basename(out_file)}')

    # ---- 执行任务 ----
    total_tasks = len(songci_tasks) + len(tangpoem_tasks)
    stages = 2 if compare_mode else 1
    _log(f'共 {total_tasks} 个任务组，对比模式: {compare_mode} (×{stages})')

    if compare_mode:
        for task in songci_tasks:
            _run_one_songci_task(task, run_constrained=False)
        for task in tangpoem_tasks:
            _run_one_tangpoem_task(task, run_constrained=False)
        for task in songci_tasks:
            _run_one_songci_task(task, run_constrained=True)
        for task in tangpoem_tasks:
            _run_one_tangpoem_task(task, run_constrained=True)
    else:
        for task in songci_tasks:
            _run_one_songci_task(task, run_constrained=True)
        for task in tangpoem_tasks:
            _run_one_tangpoem_task(task, run_constrained=True)

    _log('所有任务完成！')


# ============================================================
#  并行调度器
# ============================================================

def run_parallel(config, gen_params, output_dir):
    """多 GPU 并行入口：分割任务 → spawn 子进程 → 等待全部完成。"""
    model_cfg = config['models'][0]
    quantization = model_cfg.get('quantization', '8bit')

    rec = HardwareDetector.recommend_parallelism(
        HardwareDetector.estimate_model_vram_gb(
            model_cfg['path'], quantization
        )
    )
    num_workers = rec['max_parallel_workers']
    gpu_assignments = rec['gpu_assignments']

    if num_workers < 2:
        print("[并行调度] 并行度不足，回退至主进程串行模式")
        return False

    worker_configs = _build_worker_configs(config, num_workers)

    print(f"\n{'=' * 60}")
    print(f"  ⚡ 多 GPU 并行模式: {num_workers} 个 Worker")
    print(f"{'=' * 60}")
    for i, wc in enumerate(worker_configs):
        sc_n = len(wc['songci_tasks'])
        tp_n = len(wc['tangpoem_tasks'])
        print(f"  Worker {i} → GPU {gpu_assignments[i]} :"
              f" 宋词 {sc_n} 组, 唐诗 {tp_n} 组")
    print(f"{'=' * 60}\n")

    ctx = mp.get_context('spawn')
    processes = []

    for i in range(num_workers):
        gpu_id = gpu_assignments[i]
        worker_tasks = worker_configs[i]
        p = ctx.Process(
            target=_parallel_worker,
            args=(gpu_id, worker_tasks, config, gen_params, output_dir, PROJECT_ROOT),
            name=f'Worker-{i}-GPU-{gpu_id}',
        )
        p.start()
        processes.append(p)
        print(f"[主进程] 启动 Worker {i} (GPU {gpu_id}), PID={p.pid}")

    for i, p in enumerate(processes):
        p.join()
        print(f"[主进程] Worker {i} (PID={p.pid}) 已完成, exitcode={p.exitcode}")

    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        print(f"[主进程] ⚠️ 以下 Worker 异常退出: {failed}")

    return True


# ============================================================
#  串行实验执行器（保持原有逻辑，供单 GPU / CPU 回退使用）
# ============================================================

def run_songci_experiments(config, model, tokenizer, vocab_indexer, data_manager_songci,
                           gen_params, output_dir, use_constraints=True, q_label=""):
    """执行所有宋词实验（串行）。"""
    sc = config["songci"]
    model_cfg = config["models"][0]
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
            out_file = os.path.join(output_dir, f"{model_name}-({task_type})-{cipai}-{safe_theme}.txt")

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
                if q_label:
                    f.write(f"# 量化方案: {q_label}\n")
                f.write(f"# 约束解码: {'启用' if use_constraints else '禁用'}\n")
                f.write(f"# 每词牌生成数: {num_generations}\n")
                f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# ========================================\n\n")

            for gen_idx in range(num_generations):
                inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
                input_prompt_len = inputs.input_ids.shape[1]

                if use_constraints:
                    state_machine = GenerationStateMachine(cipai, data_manager_songci)
                    logits_processor = ConstraintLogitsProcessor(
                        vocab_indexer=vocab_indexer,
                        state_machine=state_machine,
                        tokenizer=tokenizer,
                        input_prompt_len=input_prompt_len
                    )
                    processors = LogitsProcessorList([logits_processor])
                else:
                    processors = None

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


def run_tangpoem_experiments(config, model, tokenizer, vocab_indexer, data_manager,
                             gen_params, output_dir, use_constraints=True, q_label=""):
    """执行所有唐诗实验（串行）。"""
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
            out_file = os.path.join(output_dir, f"{model_name}-({task_type})-{form_name}-{safe_theme}.txt")

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
                if q_label:
                    f.write(f"# 量化方案: {q_label}\n")
                f.write(f"# 约束解码: {'启用' if use_constraints else '禁用'}\n")
                f.write(f"# 每体裁生成数: {num_generations}\n")
                f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# ========================================\n\n")

            for gen_idx in range(num_generations):
                inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
                input_prompt_len = inputs.input_ids.shape[1]

                if use_constraints:
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
                else:
                    processors = None

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


# ============================================================
#  主入口
# ============================================================

def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_config.json")
    config = load_config(config_path)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    model_cfg = config["models"][0]
    model_name = model_cfg["name"]
    model_path = model_cfg["path"]
    quantization = model_cfg.get("quantization", "8bit")  # 兼容旧配置: 无此字段默认 8bit
    gen_params = config["generation_params"]
    q_label = _quantization_label(quantization)

    songci_cfg = config.get('songci', {})
    tangpoem_cfg = config.get('tangpoem', {})
    songci_enabled = songci_cfg.get('enabled', False)
    tangpoem_enabled = tangpoem_cfg.get('enabled', False)

    print(f"{'=' * 60}")
    print(f"  批量诗词生成实验")
    print(f"  模型: {model_name}")
    print(f"  量化方案: {q_label}")
    print(f"  输出目录: {output_dir}")
    print(f"  宋词: {'启用' if songci_enabled else '禁用'}")
    print(f"  唐诗: {'启用' if tangpoem_enabled else '禁用'}")
    print(f"  对比实验: {'启用' if config.get('compare_experiment', False) else '禁用'}")
    print(f"{'=' * 60}")

    # ---- 硬件检测 ----
    rec = HardwareDetector.print_summary(model_path, quantization)

    # ---- 判断是否走并行 ----
    can_parallel = rec['strategy'] == 'multi_gpu' and rec['max_parallel_workers'] >= 2

    if can_parallel and not config.get('force_sequential', False):
        parallel_ok = run_parallel(config, gen_params, output_dir)
        if parallel_ok:
            print(f"\n{'=' * 60}")
            print(f"  全部实验完成！（多 GPU 并行模式）")
            print(f"  输出目录: {output_dir}")
            print(f"{'=' * 60}")
            return

    # ---- 串行模式（单 GPU / CPU 兜底） ----
    print(f"\n[1/3] 加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[2/3] 加载模型 (量化={q_label})...")
    model_kwargs = _get_quantization_kwargs(quantization)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        **model_kwargs,
    ).eval()

    print("[3/3] 构建词表索引...")
    rhyme_dict_name = (
        songci_cfg.get('rhyme_dict_name') or
        tangpoem_cfg.get('rhyme_dict_name') or
        "Xinyun"
    )
    rhyme_dict_path = os.path.join(PROJECT_ROOT, "Rhyme", f"{rhyme_dict_name}.json")
    poem_path = os.path.join(PROJECT_ROOT, "Songci_Meter")
    data_manager = DataManager(rhyme_dict_path=rhyme_dict_path, poem_path=poem_path)
    vocab_indexer = VocabIndexer(tokenizer, data_manager)

    compare = config.get("compare_experiment", False)

    if compare:
        total_steps = (1 if songci_enabled else 0) + (1 if tangpoem_enabled else 0)
        total_steps *= 2  # free + constrained
        step = 0

        print(f"\n{'=' * 60}")
        print(f"  🔬 对比实验模式：将依次运行无约束 → 约束两轮生成")
        print(f"{'=' * 60}")

        base_output = output_dir

        free_output_dir = os.path.join(base_output, "free_decoding")
        os.makedirs(free_output_dir, exist_ok=True)

        if songci_enabled:
            step += 1
            print(f"\n{'=' * 60}")
            print(f"  [对比实验 {step}/{total_steps}] 宋词 — 无约束自由生成")
            print(f"  输出目录: {free_output_dir}")
            print(f"{'=' * 60}")
            run_songci_experiments(
                config, model, tokenizer, vocab_indexer, data_manager,
                gen_params, free_output_dir, use_constraints=False, q_label=q_label
            )

        if tangpoem_enabled:
            step += 1
            print(f"\n{'=' * 60}")
            print(f"  [对比实验 {step}/{total_steps}] 唐诗 — 无约束自由生成")
            print(f"  输出目录: {free_output_dir}")
            print(f"{'=' * 60}")
            run_tangpoem_experiments(
                config, model, tokenizer, vocab_indexer, data_manager,
                gen_params, free_output_dir, use_constraints=False, q_label=q_label
            )

        constrained_output_dir = os.path.join(base_output, "constrained_decoding")
        os.makedirs(constrained_output_dir, exist_ok=True)

        if songci_enabled:
            step += 1
            print(f"\n{'=' * 60}")
            print(f"  [对比实验 {step}/{total_steps}] 宋词 — 约束解码生成")
            print(f"  输出目录: {constrained_output_dir}")
            print(f"{'=' * 60}")
            run_songci_experiments(
                config, model, tokenizer, vocab_indexer, data_manager,
                gen_params, constrained_output_dir, use_constraints=True, q_label=q_label
            )

        if tangpoem_enabled:
            step += 1
            print(f"\n{'=' * 60}")
            print(f"  [对比实验 {step}/{total_steps}] 唐诗 — 约束解码生成")
            print(f"  输出目录: {constrained_output_dir}")
            print(f"{'=' * 60}")
            run_tangpoem_experiments(
                config, model, tokenizer, vocab_indexer, data_manager,
                gen_params, constrained_output_dir, use_constraints=True, q_label=q_label
            )
    else:
        if songci_enabled:
            sc = songci_cfg
            n = len(sc['cipai_list']) * len(sc['themes'])
            print(f"\n{'=' * 60}")
            print(f"  开始宋词批量生成")
            print(f"  词牌数: {len(sc['cipai_list'])}")
            print(f"  主题数: {len(sc['themes'])}")
            print(f"  每组合生成数: {sc['num_generations']}")
            print(f"  总文件数: {n}")
            print(f"  总作品数: {n * sc['num_generations']}")
            print(f"{'=' * 60}")
            run_songci_experiments(
                config, model, tokenizer, vocab_indexer, data_manager,
                gen_params, output_dir, q_label=q_label
            )

        if tangpoem_enabled:
            tp = tangpoem_cfg
            n = len(tp['forms']) * len(tp['themes'])
            print(f"\n{'=' * 60}")
            print(f"  开始唐诗批量生成")
            print(f"  诗体数: {len(tp['forms'])}")
            print(f"  主题数: {len(tp['themes'])}")
            print(f"  每组合生成数: {tp['num_generations']}")
            print(f"  总文件数: {n}")
            print(f"  总作品数: {n * tp['num_generations']}")
            print(f"{'=' * 60}")
            run_tangpoem_experiments(
                config, model, tokenizer, vocab_indexer, data_manager,
                gen_params, output_dir, q_label=q_label
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
