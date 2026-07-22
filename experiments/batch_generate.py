#!/usr/bin/env python3
"""
批量诗词生成实验脚本 v2.0 — 智能资源调度版

核心特性：
  - 多模型排队：配置 3~4 个模型参数，调度器自动排序依次执行
  - 资源感知调度：检测 GPU 显存，大模型优先，剩余空间塞小模型
  - 同模型自并行：同一模型可多 GPU 分担任务（max_parallel 控制）
  - 动态 GPU 复用：任一模型完成后立即释放资源 → 找下一个能跑的模型
  - 末模型全开：仅剩最后一个模型时，所有空闲 GPU 一拥而上并行处理
  - 共享任务队列：同模型多 Worker 通过 multiprocessing.Queue 动态抢任务

架构：
  Master 进程 ── 硬件检测 → 模型排序 → GPU 贪心分配 → 启动初始 Worker
     └── 监控循环：轮询 Worker 存活 → GPU 释放 → 动态调度下一个模型/Worker

用法：
  python experiments/batch_generate.py
  （所有配置在 experiments/batch_config.json 中）
"""

import torch
import os
import sys
import json
import time
import traceback
import multiprocessing as mp
from datetime import datetime
from typing import List, Dict, Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoConfig


# ============================================================
#  量化参数工具
# ============================================================

def _get_quantization_kwargs(quantization: str) -> dict:
    """解析量化参数，返回 AutoModelForCausalLM.from_pretrained() 的关键字参数字典。"""
    from transformers import BitsAndBytesConfig
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
        raise ValueError(f"不支持的量化参数: '{quantization}'，可选值: none, fp16, 8bit, 4bit")


def _quantization_label(q: str) -> str:
    """返回量化的可读标签。"""
    mapping = {
        "none": "FP16", "fp16": "FP16", "float16": "FP16",
        "8bit": "8-bit", "int8": "8-bit",
        "4bit": "4-bit NF4", "int4": "4-bit NF4", "nf4": "4-bit NF4",
    }
    return mapping.get(q.lower().strip(), q)


# ============================================================
#  硬件检测模块
# ============================================================

class HardwareDetector:
    """检测 CPU 内存、GPU 显存，估算模型占用，给出并行度建议。"""

    @staticmethod
    def get_system_ram_info() -> dict:
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
    def get_gpu_info() -> List[dict]:
        """返回每张 GPU 的详细信息列表（含实时空闲显存）。"""
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
                free_vram = total_vram
            gpus.append({
                'index': i,
                'name': props.name,
                'total_vram_gb': total_vram,
                'free_vram_gb': free_vram,
                'compute_capability': f'{props.major}.{props.minor}',
            })
        return gpus

    @staticmethod
    def estimate_model_vram_gb(model_path: str, quantization: str = "8bit",
                                model_name: str = "") -> Optional[float]:
        """根据模型架构参数 + 量化方案估算单副本显存占用（GB）。

        精确计算 GQA / SwiGLU / tie_word_embeddings / RMS Norm 等现代架构特性。
        支持从模型名称回退估算（如 "Qwen3-4B" → 4B params）。
        """
        param_count = None
        detail_parts = []  # 调试信息

        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            # 方法1：配置文件直接声明了参数量
            if hasattr(config, 'num_parameters'):
                try:
                    pc = int(config.num_parameters)
                    if pc > 0:
                        param_count = pc
                        detail_parts.append(f"config.num_parameters={pc:,}")
                except (TypeError, ValueError):
                    pass

            # 方法2：从架构参数精细计算
            if param_count is None:
                h = getattr(config, 'hidden_size', None)
                L = getattr(config, 'num_hidden_layers', None)
                if h is not None and L is not None:
                    V = getattr(config, 'vocab_size', 0)
                    n_heads = getattr(config, 'num_attention_heads', 1)
                    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
                    # head_dim 可能存储在不同属性名下（Qwen3 用 head_dim，部分旧模型
                    # 用 hidden_size//num_heads 隐含推导）
                    d_head = getattr(config, 'head_dim', None)
                    if d_head is None:
                        d_head = getattr(config, 'hidden_size', h) // n_heads
                    # intermediate_size 检测（SwiGLU MLP 中间维度）
                    intermediate = getattr(config, 'intermediate_size', None)
                    if intermediate is None:
                        intermediate = getattr(config, 'ffn_dim', None)
                    if intermediate is None:
                        intermediate = getattr(config, 'moe_intermediate_size', None)
                    if intermediate is None:
                        intermediate = h * 8 // 3  # Llama 系列默认比例
                    tie_emb = getattr(config, 'tie_word_embeddings', False)

                    h_attn = n_heads * d_head
                    h_kv_attn = n_kv_heads * d_head

                    # 每层参数：Q + K + V + O + gate + up + down + 2×RMS Norm
                    attn_params = 2 * h * h_attn + 2 * h * h_kv_attn
                    mlp_params = 3 * h * intermediate
                    norm_params = 2 * h + 2 * h  # attention norm + mlp norm（RMS 各 h 维）
                    per_layer = attn_params + mlp_params + norm_params

                    total = V * h + L * per_layer
                    if not tie_emb:
                        total += V * h  # 未绑定的 LM head
                    # 最终 LayerNorm
                    total += h

                    param_count = total
                    detail_parts.append(
                        f"h={h}, L={L}, V={V}, n_heads={n_heads}, n_kv={n_kv_heads}, "
                        f"d_head={d_head}, intermediate={intermediate}, tie_emb={tie_emb}"
                    )
                    detail_parts.append(f"computed_params={param_count:,}")
        except Exception as e:
            detail_parts.append(f"config_load_failed: {e}")

        # 方法3：从模型名称回退估算（如 "Qwen3-4B" → 4B）
        if param_count is None and model_name:
            import re
            m = re.search(r'(\d+\.?\d*)\s*[Bb]', model_name)
            if m:
                fallback_b = float(m.group(1))
                param_count = int(fallback_b * 1e9)
                detail_parts.append(f"name_fallback: {model_name} → {param_count:,}")
            else:
                detail_parts.append("estimate_failed: no architecture config and no size hint in name")

        if param_count is None:
            if detail_parts:
                print(f"  [预估] {model_name or model_path}: {'; '.join(detail_parts)}")
            return None

        q = quantization.lower().strip()
        if q in ("4bit", "int4", "nf4", "4"):
            bytes_per_param = 0.5
        elif q in ("8bit", "int8", "8"):
            bytes_per_param = 1.0
        else:
            bytes_per_param = 2.0  # FP16 / BF16

        overhead = 1.20
        vram = round(param_count * bytes_per_param * overhead / (1024 ** 3), 2)
        detail_parts.append(f"quant={q}, bytes={bytes_per_param}, overhead={overhead}, vram={vram}GB")
        print(f"  [预估] {model_name or model_path}: {'; '.join(detail_parts)}")
        return vram

    @staticmethod
    def print_summary(models_info: List[dict]) -> dict:
        """打印硬件资源摘要，返回 GPU 信息列表。"""
        print(f"\n{'=' * 60}")
        print(f"  硬件资源检测")
        print(f"{'=' * 60}")

        ram = HardwareDetector.get_system_ram_info()
        if HAS_PSUTIL:
            print(f"  系统内存 : 总计 {ram['total_gb']} GB |"
                  f" 可用 {ram['available_gb']} GB | 已用 {ram['percent_used']}%")
        else:
            print(f"  系统内存 : (psutil 未安装)")

        gpus = HardwareDetector.get_gpu_info()
        if not gpus:
            print(f"  GPU      : 无 — CPU 模式")
        else:
            print(f"  GPU 数量 : {len(gpus)}")
            for gpu in gpus:
                print(f"    GPU {gpu['index']} : {gpu['name']}"
                      f" | 总计 {gpu['total_vram_gb']} GB | 空闲 {gpu['free_vram_gb']} GB"
                      f" | CC {gpu['compute_capability']}")

        if models_info:
            print(f"\n  模型显存预估:")
            for m in models_info:
                est = m.get('estimated_vram_gb')
                if est:
                    print(f"    {m['name']} ({m['quantization']}): ~{est} GB/副本"
                          f" | max_parallel={m['max_parallel']}")
                else:
                    print(f"    {m['name']} ({m['quantization']}): 无法估算")

        print(f"{'=' * 60}\n")
        return gpus


# ============================================================
#  工具函数
# ============================================================

def sanitize_filename(s: str) -> str:
    return s.replace('/', '_').replace('\\', '_').replace(':', '_')\
            .replace('*', '_').replace('?', '_').replace('"', '_')\
            .replace('<', '_').replace('>', '_').replace('|', '_')


# ============================================================
#  中断续写（Checkpoint）管理
# ============================================================

class CheckpointManager:
    """基于 .done 标记文件的中断续写管理器。

    每个完成的任务在 output_dir/.checkpoints/ 下写入一个标记文件，
    重启时读取所有标记文件，在 build_all_tasks 阶段过滤已完成任务。
    多 Worker 并发安全（每个任务写独立文件）。
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = os.path.join(checkpoint_dir, '.checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._completed: set = set()
        self._load()

    def _load(self):
        """读取所有 .done 标记文件，恢复已完成任务集合。"""
        if not os.path.isdir(self.checkpoint_dir):
            return
        for fname in os.listdir(self.checkpoint_dir):
            if fname.endswith('.done'):
                # 文件名即 task_hash
                self._completed.add(fname[:-5])  # 去掉 .done 后缀

    def mark_done(self, task_hash: str):
        """标记一个任务已完成（写 .done 空文件）。"""
        self._completed.add(task_hash)
        marker = os.path.join(self.checkpoint_dir, f'{task_hash}.done')
        try:
            with open(marker, 'w', encoding='utf-8') as f:
                f.write('')
        except Exception:
            pass  # 标记写入失败不应中断主流程

    def is_done(self, task_hash: str) -> bool:
        """检查任务是否已完成。"""
        return task_hash in self._completed

    @staticmethod
    def make_task_hash(task: dict) -> str:
        """为任务生成唯一标识（短哈希，用作文件名）。

        标识包含: 模型名 + 体裁 + 词牌/诗体 + 主题 + 约束模式
        """
        import hashlib
        meter = task['meter_type']
        name = task.get('cipai', task.get('form', {}).get('name', '?'))
        theme = task['theme']
        constraints = 'C' if task.get('use_constraints', True) else 'F'
        model = task.get('model_name', '?')
        task_type = task.get('task_type', '?')
        raw = f"{model}|{meter}|{name}|{theme}|{constraints}|{task_type}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]

    def summary(self) -> str:
        return f"已恢复 {len(self._completed)} 个已完成任务"


# ============================================================
#  实验运行日志
# ============================================================

class TeeLogger:
    """同时输出到 stdout 和日志文件的 Tee 日志器。

    在 main() 开始时初始化，将所有 print 输出同步写入日志文件。
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_path = os.path.join(log_dir, f'experiment_{timestamp}.log')
        self._file = open(self.log_path, 'w', encoding='utf-8')
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def write(self, message):
        self._original_stdout.write(message)
        self._file.write(message)

    def flush(self):
        self._original_stdout.flush()
        self._file.flush()

    def start(self):
        """开始将 stdout/stderr tee 到日志文件。"""
        sys.stdout = self
        sys.stderr = self

    def stop(self):
        """恢复原始 stdout/stderr 并关闭日志文件。"""
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._file.close()

    def get_path(self) -> str:
        return self.log_path


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_one_batch(model, tokenizer, inputs, input_prompt_len, processors,
                       gen_params: dict, num_generations: int, tokenizer_pad_id: int) -> List[str]:
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
#  任务构建
# ============================================================

def build_all_tasks(config: dict) -> List[dict]:
    """
    遍历所有启用的模型 × (宋词/唐诗) × 词牌/诗体 × 主题，
    生成全量任务列表。每个任务是一个独立的工作单元（一个文件）。
    compare_experiment 模式下每个组合拆成 constrained + free 两个任务。
    返回: [ { model_idx, model_name, model_path, quantization, use_thinking,
              meter_type, cipai/form, theme, ... }, ... ]
    """
    tasks = []
    enabled_models = [
        (i, m) for i, m in enumerate(config.get('models', []))
        if m.get('enabled', True)
    ]
    if not enabled_models:
        print("[警告] 没有启用的模型！请检查 batch_config.json → models")
        return tasks

    compare_mode = config.get('compare_experiment', False)
    sc_cfg = config.get('songci', {})
    tp_cfg = config.get('tangpoem', {})

    for model_idx, model_cfg in enabled_models:
        model_name = model_cfg['name']
        model_path = model_cfg['path']
        quantization = model_cfg.get('quantization', '8bit')
        use_thinking = model_cfg.get('use_thinking', False)
        q_label = _quantization_label(quantization)

        # ---- 宋词任务 ----
        if sc_cfg.get('enabled', False):
            sc_rhyme = sc_cfg.get('rhyme_dict_name', 'Xinyun')
            sc_task_type = sc_cfg.get('task_type', 'instruction')
            sc_num_gen = sc_cfg.get('num_generations', 1)
            for cipai in sc_cfg.get('cipai_list', []):
                for theme_entry in sc_cfg.get('themes', []):
                    base = {
                        'model_idx': model_idx,
                        'model_name': model_name,
                        'model_path': model_path,
                        'quantization': quantization,
                        'q_label': q_label,
                        'use_thinking': use_thinking,
                        'meter_type': '宋词',
                        'cipai': cipai,
                        'theme': theme_entry['theme'],
                        'detailed_requirement': theme_entry.get('detailed_requirement', ''),
                        'task_type': sc_task_type,
                        'rhyme_dict_name': sc_rhyme,
                        'num_generations': sc_num_gen,
                    }
                    if compare_mode:
                        tasks.append({**base, 'use_constraints': False, 'compare_mode_implied': True})
                        tasks.append({**base, 'use_constraints': True, 'compare_mode_implied': True})
                    else:
                        tasks.append({**base, 'use_constraints': True})

        # ---- 唐诗任务 ----
        if tp_cfg.get('enabled', False):
            tp_rhyme = tp_cfg.get('rhyme_dict_name', 'Xinyun')
            tp_task_type = tp_cfg.get('task_type', 'instruction')
            tp_num_gen = tp_cfg.get('num_generations', 1)
            for form in tp_cfg.get('forms', []):
                for theme_entry in tp_cfg.get('themes', []):
                    base = {
                        'model_idx': model_idx,
                        'model_name': model_name,
                        'model_path': model_path,
                        'quantization': quantization,
                        'q_label': q_label,
                        'use_thinking': use_thinking,
                        'meter_type': '唐诗',
                        'form': form,
                        'theme': theme_entry['theme'],
                        'detailed_requirement': theme_entry.get('detailed_requirement', ''),
                        'task_type': tp_task_type,
                        'rhyme_dict_name': tp_rhyme,
                        'num_generations': tp_num_gen,
                    }
                    if compare_mode:
                        tasks.append({**base, 'use_constraints': False, 'compare_mode_implied': True})
                        tasks.append({**base, 'use_constraints': True, 'compare_mode_implied': True})
                    else:
                        tasks.append({**base, 'use_constraints': True})

    return tasks


# ============================================================
#  Worker 进程（在指定 GPU 上加载模型，从共享队列消费任务）
# ============================================================

def _worker_process(
    gpu_id: int,
    model_cfg: dict,
    task_queue,            # multiprocessing.Queue — 共享任务队列
    progress_counter,      # multiprocessing.Value('i') — 已完成任务计数
    gen_params: dict,
    output_dir: str,
    project_root: str,
    checkpoint_dir: str = "",  # 断点续写 .done 标记目录
):
    """
    子进程入口（spawn 上下文）。
    在指定 GPU 上加载模型，循环从队列取任务执行，直到队列空。
    多个 Worker 可共享同一个 task_queue（同模型并行）。
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # spawn 后重新 import
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModel, \
        AutoTokenizer as _AutoTokenizer, BitsAndBytesConfig as _Bits, \
        LogitsProcessorList as _LPL

    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from data_manager import DataManager as _DM
    from vocab_indexer import VocabIndexer as _VI
    from state_machine import GenerationStateMachine as _GSM, TangPoemStateMachine as _TPSM
    from logits_processor import ConstraintLogitsProcessor as _CLP, TangPoemLogitsProcessor as _TPLP
    from main import parse_tang_format as _ptf, build_prompt_messages as _bpm, \
        build_tangpoem_prompt_messages as _btpm

    model_path = model_cfg['path']
    model_name = model_cfg['name']
    quantization = model_cfg.get('quantization', '8bit')
    use_thinking = model_cfg.get('use_thinking', False)
    q_label = _quantization_label(quantization)

    def _log(msg: str):
        print(f"[GPU:{gpu_id}|{model_name}] {msg}", flush=True)

    # ---- 加载模型 ----
    _log(f"加载 tokenizer: {model_path}")
    tokenizer = _AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pad_token_id = tokenizer.eos_token_id

    _log(f"加载模型 (量化={q_label})...")
    q = quantization.lower().strip()
    if q in ("none", "fp16", "float16", ""):
        model_kwargs = {"torch_dtype": _torch.float16}
    elif q in ("8bit", "int8", "8"):
        model_kwargs = {"quantization_config": _Bits(load_in_8bit=True)}
    elif q in ("4bit", "int4", "nf4", "4"):
        model_kwargs = {
            "quantization_config": _Bits(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        }
    else:
        raise ValueError(f"不支持的量化参数: '{quantization}'")

    model = _AutoModel.from_pretrained(
        model_path,
        device_map='auto',
        trust_remote_code=True,
        **model_kwargs,
    ).eval()

    _log(f"模型加载完成，设备: {model.device}")

    # ---- 加载韵书 & 词表索引 ----
    rhyme_dict_name = model_cfg.get('_default_rhyme', 'Xinyun')

    rhyme_dict_path = os.path.join(project_root, 'Rhyme', f'{rhyme_dict_name}.json')
    poem_path = os.path.join(project_root, 'Songci_Meter')
    data_manager = _DM(rhyme_dict_path=rhyme_dict_path, poem_path=poem_path)
    vocab_indexer = _VI(tokenizer, data_manager)

    # Cache DataManager per rhyme_dict_name for tasks with different rhymes
    _dm_cache = {rhyme_dict_name: data_manager}
    _vi_cache = {rhyme_dict_name: vocab_indexer}

    def _get_dm_vi(rhyme_name: str):
        if rhyme_name not in _dm_cache:
            _rhyme_path = os.path.join(project_root, 'Rhyme', f'{rhyme_name}.json')
            _dm = _DM(rhyme_dict_path=_rhyme_path, poem_path=poem_path)
            _vi = _VI(tokenizer, _dm)
            _dm_cache[rhyme_name] = _dm
            _vi_cache[rhyme_name] = _vi
        return _dm_cache[rhyme_name], _vi_cache[rhyme_name]

    # ---- 任务处理循环 ----
    processed = 0
    while True:
        try:
            task = task_queue.get(timeout=3)
        except Exception:
            # 超时 3 秒无任务 → 队列已空，退出
            break

        if task is None:
            # 哨兵信号
            break

        try:
            _process_one_task(
                task=task,
                model=model,
                tokenizer=tokenizer,
                pad_token_id=pad_token_id,
                model_name=model_name,
                q_label=q_label,
                use_thinking=use_thinking,
                gen_params=gen_params,
                output_dir=output_dir,
                project_root=project_root,
                get_dm_vi=_get_dm_vi,
                log_fn=_log,
            )
            _mark_task_done = True
        except Exception as e:
            _log(f"[错误] 任务执行失败: {task.get('cipai', task.get('form', {}).get('name', '?'))}"
                 f" / {task.get('theme', '?')} — {e}")
            traceback.print_exc()
            _mark_task_done = False

        processed += 1
        # Manager ValueProxy 在 Python 3.12 没有 get_lock()，直接赋值即可
        #（Manager 服务端本身串行化所有操作，无需客户端锁）
        progress_counter.value += 1

        # 断点续写：成功后写入 .done 标记
        if _mark_task_done and checkpoint_dir:
            try:
                import hashlib as _hl
                meter = task['meter_type']
                name = task.get('cipai', task.get('form', {}).get('name', '?'))
                theme = task['theme']
                constraints = 'C' if task.get('use_constraints', True) else 'F'
                model_nm = task.get('model_name', '?')
                task_type = task.get('task_type', '?')
                raw = f"{model_nm}|{meter}|{name}|{theme}|{constraints}|{task_type}"
                task_hash = _hl.md5(raw.encode('utf-8')).hexdigest()[:12]
                chk_dir = os.path.join(checkpoint_dir, '.checkpoints')
                os.makedirs(chk_dir, exist_ok=True)
                marker = os.path.join(chk_dir, f'{task_hash}.done')
                with open(marker, 'w', encoding='utf-8') as _mf:
                    _mf.write('')
            except Exception:
                pass  # 标记写入失败不应中断主流程

    _log(f"完成，共处理 {processed} 个任务")


def _process_one_task(
    task: dict,
    model,
    tokenizer,
    pad_token_id: int,
    model_name: str,
    q_label: str,
    use_thinking: bool,
    gen_params: dict,
    output_dir: str,
    project_root: str,
    get_dm_vi,
    log_fn,
):
    """处理单个任务（一个文件）。"""
    from main import parse_tang_format, build_prompt_messages, build_tangpoem_prompt_messages
    from state_machine import GenerationStateMachine, TangPoemStateMachine
    from logits_processor import ConstraintLogitsProcessor, TangPoemLogitsProcessor
    from transformers import LogitsProcessorList

    meter_type = task['meter_type']
    use_constraints = task['use_constraints']
    rhyme_dict_name = task['rhyme_dict_name']
    prompt_task_type = task['task_type']
    num_generations = task['num_generations']
    theme = task['theme']
    detailed_req = task.get('detailed_requirement', '')
    safe_theme = sanitize_filename(theme)

    data_manager, vocab_indexer = get_dm_vi(rhyme_dict_name)

    # 确定输出文件路径
    if task.get('compare_mode_implied'):
        subdir = 'constrained_decoding' if use_constraints else 'free_decoding'
    else:
        subdir = ''

    if meter_type == '宋词':
        cipai = task['cipai']
        tag = f'[{cipai}·{theme}]'
        if subdir:
            out_file = os.path.join(output_dir, subdir,
                                    f'{model_name}-({prompt_task_type})-{cipai}-{safe_theme}.txt')
        else:
            out_file = os.path.join(output_dir,
                                    f'{model_name}-({prompt_task_type})-{cipai}-{safe_theme}.txt')

        try:
            messages = build_prompt_messages(
                prompt_task_type, cipai, theme,
                requirement=detailed_req,
                poem_path=os.path.join(project_root, 'Songci_Meter'),
                use_thinking=use_thinking,
                rhyme_dict_name=rhyme_dict_name,
            )
        except Exception as e:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(f'[错误] 构建 Prompt 失败: {e}\n')
            return

        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f'# 模型: {model_name}\n')
            f.write(f'# 词牌: {cipai}\n')
            f.write(f'# 主题: {theme}\n')
            f.write(f'# 韵书: {rhyme_dict_name}\n')
            f.write(f'# task_type: {prompt_task_type}\n')
            f.write(f'# 量化方案: {q_label}\n')
            f.write(f'# 约束解码: {"启用" if use_constraints else "禁用"}\n')
            f.write(f'# 每词牌生成数: {num_generations}\n')
            f.write(f'# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'# ========================================\n\n')

        for gen_idx in range(num_generations):
            inputs = tokenizer(chat_prompt, return_tensors='pt').to(model.device)
            input_prompt_len = inputs.input_ids.shape[1]

            if use_constraints:
                state_machine = GenerationStateMachine(cipai, data_manager)
                logits_processor = ConstraintLogitsProcessor(
                    vocab_indexer=vocab_indexer,
                    state_machine=state_machine,
                    tokenizer=tokenizer,
                    input_prompt_len=input_prompt_len,
                )
                processors = LogitsProcessorList([logits_processor])
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

    else:  # 唐诗
        form = task['form']
        form_name = form['name']
        line_length = form['line_length']
        num_lines = form['num_lines']
        tag = f'[{form_name}·{theme}]'

        if subdir:
            out_file = os.path.join(output_dir, subdir,
                                    f'{model_name}-({prompt_task_type})-{form_name}-{safe_theme}.txt')
        else:
            out_file = os.path.join(output_dir,
                                    f'{model_name}-({prompt_task_type})-{form_name}-{safe_theme}.txt')

        try:
            messages = build_tangpoem_prompt_messages(
                prompt_task_type, form_name, theme,
                requirement=detailed_req,
                use_thinking=use_thinking,
                line_length=line_length,
                num_lines=num_lines,
                rhyme_dict_name=rhyme_dict_name,
            )
        except Exception as e:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(f'[错误] 构建 Prompt 失败: {e}\n')
            return

        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, _, rhyme_type = parse_tang_format(form_name)

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f'# 模型: {model_name}\n')
            f.write(f'# 诗体: {form_name}\n')
            f.write(f'# 主题: {theme}\n')
            f.write(f'# 韵书: {rhyme_dict_name}\n')
            f.write(f'# task_type: {prompt_task_type}\n')
            f.write(f'# 量化方案: {q_label}\n')
            f.write(f'# 约束解码: {"启用" if use_constraints else "禁用"}\n')
            f.write(f'# 每体裁生成数: {num_generations}\n')
            f.write(f'# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'# ========================================\n\n')

        for gen_idx in range(num_generations):
            inputs = tokenizer(chat_prompt, return_tensors='pt').to(model.device)
            input_prompt_len = inputs.input_ids.shape[1]

            if use_constraints:
                state_machine = TangPoemStateMachine(
                    line_length=line_length,
                    num_lines=num_lines,
                    rhyme_type=rhyme_type,
                    data_manager=data_manager,
                )
                logits_processor = TangPoemLogitsProcessor(
                    vocab_indexer=vocab_indexer,
                    state_machine=state_machine,
                    tokenizer=tokenizer,
                    input_prompt_len=input_prompt_len,
                )
                processors = LogitsProcessorList([logits_processor])
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

    log_fn(f'{tag} {"约束" if use_constraints else "自由"} 生成完成 → {os.path.basename(out_file)}')


# ============================================================
#  智能调度器 — 核心算法
# ============================================================

class SmartScheduler:
    """
    资源感知的智能调度器。

    算法流程：
      1. 按估算显存从大到小排序模型队列
      2. 初始放置：贪心分配 GPU，大模型优先，剩余空间塞小模型
      3. 为每个 (模型, GPU) 分配创建共享任务队列 + Worker 进程
      4. 监控循环：轮询 Worker 存活状态
         - Worker 完成 → 释放 GPU 资源 → 尝试放置下一个待运行模型
         - 若无待运行模型且只剩一个模型在跑 → 为其增加 Worker（末模型并行）
      5. 全部完成，打印统计
    """

    def __init__(self, config: dict, gen_params: dict, output_dir: str):
        self.config = config
        self.gen_params = gen_params
        self.output_dir = output_dir
        self.project_root = PROJECT_ROOT

        self.safety_margin = config.get('safety_vram_margin', 0.20)
        self.poll_interval = config.get('poll_interval_sec', 2.0)
        self.global_max_parallel = config.get('global_max_parallel', -1)

        # 模型信息列表（按显存降序排列）
        self.models: List[dict] = []
        # GPU 状态: { 'index': int, 'total_vram': float, 'free_vram': float, 'name': str }
        self.gpus: List[dict] = []
        # 全量任务列表
        self.all_tasks: List[dict] = []

        # 运行时状态
        self._manager: Optional[mp.Manager] = None
        # model_key -> { 'queue': Queue, 'counter': Value, 'total': int, 'cfg': dict }
        self._model_queues: dict = {}
        # list of { 'process': Process, 'gpu_id': int, 'model_key': str, 'model_name': str }
        self._running_workers: List[dict] = []
        # set of model_key that are fully done
        self._finished_models: set = set()
        # 显式模型状态机: model_key -> 'pending' | 'running' | 'finished'
        # 解决旧逻辑中通过 running_workers + finished_models 推导状态的不确定性
        self._model_status: Dict[str, str] = {}
        # 连续失败计数器（model_key -> int），用于检测 OOM/Crash 死循环
        self._consecutive_failures: Dict[str, int] = {}

    # ------------------------------------------------------------------
    #  初始化 & 排序
    # ------------------------------------------------------------------

    def _build_model_list(self):
        """构建启用的模型列表，估算显存，按降序排列。"""
        enabled = []
        for i, m in enumerate(self.config.get('models', [])):
            if not m.get('enabled', True):
                print(f"  [跳过] 模型 '{m.get('name', f'#{i}')}' 已禁用")
                continue
            est = HardwareDetector.estimate_model_vram_gb(
                m['path'], m.get('quantization', '8bit'),
                model_name=m.get('name', ''),
            )
            enabled.append({
                'idx': i,
                'key': f"model_{i}",
                'name': m['name'],
                'path': m['path'],
                'quantization': m.get('quantization', '8bit'),
                'use_thinking': m.get('use_thinking', False),
                'max_parallel': m.get('max_parallel', 1),
                'estimated_vram_gb': est,
                'cfg': m,
            })

        # 按估算显存降序排列（大模型优先）
        enabled.sort(key=lambda x: x.get('estimated_vram_gb') or 0, reverse=True)
        self.models = enabled

        if not enabled:
            print("[错误] 没有启用的模型！")
            return

        print(f"\n  模型调度顺序（大模型优先）:")
        for rank, m in enumerate(self.models):
            est_str = f"~{m['estimated_vram_gb']} GB" if m['estimated_vram_gb'] else "未知"
            print(f"    {rank + 1}. {m['name']} ({m['quantization']}) — 预估 {est_str}"
                  f" | max_parallel={m['max_parallel']}")

    def _detect_gpus(self):
        """检测 GPU 并初始化可用显存追踪。"""
        raw = HardwareDetector.get_gpu_info()
        if not raw:
            print("[错误] 未检测到 GPU，无法运行批量实验。")
            self.gpus = []
            return

        self.gpus = []
        for g in raw:
            self.gpus.append({
                'index': g['index'],
                'name': g['name'],
                'total_vram': g['total_vram_gb'],
                'free_vram': g['free_vram_gb'] * 0.90,  # 保留 10% 给系统开销
                'allocated_models': [],  # [(model_key, allocated_vram), ...]
            })

    # ------------------------------------------------------------------
    #  放置算法
    # ------------------------------------------------------------------

    def _can_place_on_gpu(self, gpu: dict, model: dict) -> bool:
        """检查模型是否能放入指定 GPU。"""
        if model['estimated_vram_gb'] is None:
            return True  # 无法估算时乐观允许
        required = model['estimated_vram_gb'] * (1 + self.safety_margin)
        return gpu['free_vram'] >= required

    def _allocate_gpu(self, gpu: dict, model: dict):
        """在 GPU 上分配显存给模型。"""
        if model['estimated_vram_gb'] is not None:
            required = model['estimated_vram_gb'] * (1 + self.safety_margin)
            gpu['free_vram'] -= required
            gpu['allocated_models'].append((model['key'], required))

    def _release_gpu(self, gpu: dict, model_key: str):
        """释放 GPU 上指定模型占用的显存。"""
        for i, (mk, allocated) in enumerate(gpu['allocated_models']):
            if mk == model_key:
                gpu['free_vram'] += allocated
                gpu['allocated_models'].pop(i)
                return

    def _initial_placement(self) -> List[Tuple[int, dict]]:
        """
        初始 GPU 分配（贪心算法）。
        大模型优先，按顺序尝试放入空闲显存最大的 GPU。
        若一个模型允许多 GPU（max_parallel > 1），则尝试放到多张卡上。
        返回: [(gpu_id, model_dict), ...] — 初始启动的 (GPU, 模型) 配对列表
        """
        assignments: List[Tuple[int, dict]] = []
        model_gpu_counts: dict = {}  # model_key → 已分配 GPU 数

        for model in self.models:
            model_gpu_counts[model['key']] = 0
            max_p = model['max_parallel']

            # 按空闲显存降序排列 GPU
            sorted_gpus = sorted(self.gpus, key=lambda g: g['free_vram'], reverse=True)

            for gpu in sorted_gpus:
                if model_gpu_counts[model['key']] >= max_p:
                    break
                if self._can_place_on_gpu(gpu, model):
                    self._allocate_gpu(gpu, model)
                    assignments.append((gpu['index'], model))
                    model_gpu_counts[model['key']] += 1

            if model_gpu_counts[model['key']] == 0:
                print(f"  [警告] 模型 '{model['name']}' 无法放入任何 GPU！"
                      f"（需要 ~{model['estimated_vram_gb']} GB，"
                      f"最大 GPU 空闲 {max(g['free_vram'] for g in self.gpus):.1f} GB）")

        return assignments

    # ------------------------------------------------------------------
    #  监控 & 动态调度
    # ------------------------------------------------------------------

    def _start_worker(self, gpu_id: int, model: dict,
                       checkpoint_dir: str = "") -> mp.Process:
        """在指定 GPU 上启动指定模型的 Worker 进程。"""
        mq = self._model_queues[model['key']]
        ctx = mp.get_context('spawn')
        # 断点续写目录
        chk_dir = checkpoint_dir or self.output_dir
        p = ctx.Process(
            target=_worker_process,
            args=(
                gpu_id,
                model['cfg'],
                mq['queue'],
                mq['counter'],
                self.gen_params,
                self.output_dir,
                self.project_root,
                chk_dir,
            ),
            name=f'Worker-{model["name"]}-GPU{gpu_id}',
        )
        p.start()
        self._running_workers.append({
            'process': p,
            'gpu_id': gpu_id,
            'model_key': model['key'],
            'model_name': model['name'],
        })
        self._model_status[model['key']] = 'running'
        return p

    def _get_running_model_keys(self) -> set:
        """返回当前正在运行（有活跃 Worker）的模型 key 集合。"""
        alive = set(w['model_key'] for w in self._running_workers if w['process'].is_alive())
        # 同步 _model_status：还在运行的标记为 running
        for mk in alive:
            if self._model_status.get(mk) != 'finished':
                self._model_status[mk] = 'running'
        return alive

    def _get_pending_models(self) -> List[dict]:
        """返回尚未启动且未完成的模型列表（按显存降序）。

        使用显式 _model_status 而非从 running_workers 推导，杜绝状态不一致。
        """
        return [
            m for m in self.models
            if self._model_status.get(m['key'], 'pending') == 'pending'
        ]

    def _model_has_remaining_tasks(self, model_key: str) -> bool:
        """检查模型是否还有未完成的任务（Worker 正常退出后调用）。"""
        mq = self._model_queues.get(model_key)
        if not mq:
            return False
        return mq['counter'].value < mq['total']

    def _get_idle_gpus(self) -> List[int]:
        """返回所有没有活跃 Worker 的 GPU 索引列表。"""
        busy_gpus = set()
        for w in self._running_workers:
            if w['process'].is_alive():
                busy_gpus.add(w['gpu_id'])
        return [g['index'] for g in self.gpus if g['index'] not in busy_gpus]

    def _try_schedule_on_gpu(self, gpu_id: int, max_consecutive_failures: int) -> bool:
        """尝试在指定 GPU 上调度一个合适的模型。

        优先级：
          1. 启动尚未运行的 pending 模型（大模型优先）
          2. 末模型并行加速（仅剩一个模型有任务时为其增加 Worker）
          3. 已 running 但无活跃 Worker 的模型（Worker 异常全灭后的恢复）

        返回 True 表示成功调度了一个 Worker。
        """
        gpu = self.gpus[gpu_id]
        pending = self._get_pending_models()

        # 过滤掉连续崩溃过多的模型
        schedulable_pending = [
            m for m in pending
            if self._consecutive_failures.get(m['key'], 0) < max_consecutive_failures
        ]
        skipped = [m for m in pending
                    if self._consecutive_failures.get(m['key'], 0) >= max_consecutive_failures]
        if skipped:
            names = ', '.join(m['name'] for m in skipped)
            print(f"  [跳过] 以下模型连续失败 {max_consecutive_failures} 次，暂停调度: {names}")

        # 优先级 1：启动 pending 模型
        for model in schedulable_pending:
            if self._can_place_on_gpu(gpu, model):
                running_count = sum(
                    1 for w in self._running_workers
                    if w['model_key'] == model['key'] and w['process'].is_alive()
                )
                if running_count < model['max_parallel']:
                    self._allocate_gpu(gpu, model)
                    p = self._start_worker(gpu_id, model)
                    print(f"  [调度] GPU {gpu_id} → {model['name']} (PID={p.pid})"
                          f" [新模型启动, 状态: pending→running]")
                    return True

        # 优先级 1b：重新调度"已 running 但无活跃 Worker"的模型
        # （Worker 异常全灭 → 状态被 _update_model_status 退回 pending，由优先级 1 处理；
        #   但若状态尚未来得及更新就进入本函数，这里做兜底检查）
        for model in self.models:
            mk = model['key']
            if mk in self._finished_models:
                continue
            mq = self._model_queues.get(mk)
            if not mq:
                continue
            has_alive = any(
                w['model_key'] == mk and w['process'].is_alive()
                for w in self._running_workers
            )
            remaining = mq['total'] - mq['counter'].value
            if remaining > 0 and not has_alive and self._model_status.get(mk) != 'pending':
                # 模型有剩余任务但无 Worker → 状态不一致，回退到 pending
                self._model_status[mk] = 'pending'
                if self._can_place_on_gpu(gpu, model):
                    running_count = sum(
                        1 for w in self._running_workers
                        if w['model_key'] == mk and w['process'].is_alive()
                    )
                    if running_count < model['max_parallel']:
                        self._allocate_gpu(gpu, model)
                        p = self._start_worker(gpu_id, model)
                        print(f"  [调度] GPU {gpu_id} → {model['name']} (PID={p.pid})"
                              f" [异常恢复, 剩余 {remaining} 任务]")
                        return True

        # 优先级 2：末模型并行加速
        running_keys = self._get_running_model_keys()
        running_models = [m for m in self.models if m['key'] in running_keys]
        still_pending = self._get_pending_models()

        if len(running_models) == 1 and not still_pending:
            only_model = running_models[0]
            running_count = sum(
                1 for w in self._running_workers
                if w['model_key'] == only_model['key'] and w['process'].is_alive()
            )
            mq = self._model_queues[only_model['key']]
            remaining = mq['total'] - mq['counter'].value
            if (running_count < only_model['max_parallel']
                    and remaining > 0
                    and self._can_place_on_gpu(gpu, only_model)):
                self._allocate_gpu(gpu, only_model)
                p = self._start_worker(gpu_id, only_model)
                print(f"  [调度] GPU {gpu_id} → {only_model['name']} (PID={p.pid})"
                      f" [末模型并行, 剩余 {remaining} 任务]")
                return True

        # 无法调度：输出诊断
        reason_parts = []
        if still_pending:
            reason_parts.append(
                f"{len(still_pending)} 个模型待运行但显存不足"
                f" (需 >{gpu['free_vram']:.1f} GB)"
            )
        if not reason_parts:
            reason_parts.append('所有模型已完成或运行中')
        print(f"  [空闲] GPU {gpu_id} 暂无合适模型可调度"
              f" ({'; '.join(reason_parts)})"
              f" | 空闲显存 {gpu['free_vram']:.1f} GB")
        return False

    def _update_model_status(self, model_key: str):
        """根据队列计数器 + 活跃 Worker 数同步模型状态。"""
        mq = self._model_queues.get(model_key)
        all_done = mq and mq['counter'].value >= mq['total']
        has_alive_workers = any(
            w['model_key'] == model_key and w['process'].is_alive()
            for w in self._running_workers
        )
        if all_done:
            self._model_status[model_key] = 'finished'
            self._finished_models.add(model_key)
        elif has_alive_workers:
            self._model_status[model_key] = 'running'
        else:
            # 无活跃 Worker 但任务未完成 → 回退到 pending 等待重新调度
            if model_key not in self._finished_models:
                self._model_status[model_key] = 'pending'

    def _print_status(self):
        """打印当前调度状态（含显式状态机信息）。"""
        running_info = []
        for w in self._running_workers:
            if not w['process'].is_alive():
                continue
            mq = self._model_queues.get(w['model_key'])
            if mq:
                done = mq['counter'].value
                total = mq['total']
                running_info.append(f"{w['model_name']}@GPU{w['gpu_id']}({done}/{total})")
            else:
                running_info.append(f"{w['model_name']}@GPU{w['gpu_id']}")

        pending = self._get_pending_models()
        pending_names = [m['name'] for m in pending]

        gpu_status = ', '.join(
            f"GPU{g['index']}:{g['free_vram']:.1f}GB空闲"
            for g in self.gpus
        )

        # 显式状态一览
        status_summary = []
        for m in self.models:
            st = self._model_status.get(m['key'], '?')
            mq = self._model_queues.get(m['key'])
            prog = f"({mq['counter'].value}/{mq['total']})" if mq else ''
            status_summary.append(f"{m['name']}[{st}]{prog}")

        print(f"\n  [调度状态] {datetime.now().strftime('%H:%M:%S')}")
        print(f"    GPU显存: {gpu_status}")
        print(f"    模型状态: {' | '.join(status_summary)}")
        print(f"    运行中:   {', '.join(running_info) if running_info else '(无)'}")
        print(f"    待运行:   {', '.join(pending_names) if pending_names else '(无)'}")
        print(f"    已完成:   {len(self._finished_models)}/{len(self.models)} 个模型")

    def run(self):
        """主调度入口。"""
        print(f"\n{'=' * 60}")
        print(f"  智能调度器启动")
        print(f"{'=' * 60}")

        # Step 1: 初始化
        self._build_model_list()
        if not self.models:
            print("[错误] 无可用模型，退出。")
            return

        self._detect_gpus()
        if not self.gpus:
            print("[错误] 无可用 GPU，退出。")
            return

        # Step 2: 构建全量任务并按模型分组
        self.all_tasks = build_all_tasks(self.config)
        if not self.all_tasks:
            print("[错误] 无任务可执行，请检查配置。")
            return

        tasks_by_model: Dict[str, List[dict]] = {}
        for task in self.all_tasks:
            mk = f"model_{task['model_idx']}"
            tasks_by_model.setdefault(mk, []).append(task)

        total_tasks = len(self.all_tasks)
        print(f"\n  全量任务: {total_tasks} 个（跨 {len(tasks_by_model)} 个模型）")
        for mk, tasks in tasks_by_model.items():
            model_name = next((m['name'] for m in self.models if m['key'] == mk), mk)
            print(f"    {model_name}: {len(tasks)} 个任务")

        # Step 2.5: 初始化中断续写管理器
        # CheckpointManager 内部自动追加 .checkpoints 子目录
        checkpoint_mgr = CheckpointManager(self.output_dir)
        print(f"\n  [断点续写] {checkpoint_mgr.summary()}")

        # Step 3: 为每个模型创建共享任务队列，初始化显式状态机
        self._manager = mp.Manager()
        total_skipped = 0
        for mk, tasks in tasks_by_model.items():
            # 过滤已完成任务
            filtered = [t for t in tasks
                        if not checkpoint_mgr.is_done(CheckpointManager.make_task_hash(t))]
            skipped = len(tasks) - len(filtered)
            total_skipped += skipped
            model_name = next((m['name'] for m in self.models if m['key'] == mk), mk)
            if skipped:
                print(f"  [续写] {model_name}: 跳过 {skipped} 个已完成任务，"
                      f"剩余 {len(filtered)}/{len(tasks)}")

            if not filtered:
                # 所有任务均已完成 → 标记为 finished
                self._model_status[mk] = 'finished'
                self._finished_models.add(mk)
                print(f"  [续写] {model_name}: 全部完成，跳过")
                self._model_queues[mk] = {
                    'queue': None,
                    'counter': self._manager.Value('i', len(tasks)),
                    'total': len(tasks),
                    'cfg': next((m['cfg'] for m in self.models if m['key'] == mk), {}),
                }
                continue

            # 推断此模型的默认韵书（用于 Worker 初始化 DataManager）
            default_rhyme = filtered[0].get('rhyme_dict_name', 'Xinyun') if filtered else 'Xinyun'
            model_cfg = next((m['cfg'] for m in self.models if m['key'] == mk), {})
            model_cfg['_default_rhyme'] = default_rhyme

            queue = self._manager.Queue()
            for t in filtered:
                queue.put(t)
            self._model_queues[mk] = {
                'queue': queue,
                'counter': self._manager.Value('i', 0),
                'total': len(filtered),  # 使用过滤后的任务数
                'cfg': model_cfg,
            }
            # 初始化显式状态：所有模型从 pending 开始
            self._model_status[mk] = 'pending'
            self._consecutive_failures[mk] = 0

        if total_skipped:
            print(f"  [续写] 总计跳过 {total_skipped} 个已完成任务")

        # 修正 total_tasks 为过滤后的数量
        total_tasks = sum(
            mq['total'] for mq in self._model_queues.values()
        )

        # Step 4: 初始 GPU 放置
        print(f"\n  [初始放置] 贪心分配 GPU...")
        initial_assignments = self._initial_placement()

        if not initial_assignments:
            print("[错误] 初始放置失败：没有模型能放入任何 GPU。")
            self._manager.shutdown()
            return

        print(f"\n  初始放置结果:")
        for gpu_id, model in initial_assignments:
            est = f"~{model['estimated_vram_gb']}GB" if model['estimated_vram_gb'] else "?"
            gpu = self.gpus[gpu_id]
            print(f"    GPU {gpu_id} ({gpu['name']}, 剩余 {gpu['free_vram']:.1f}GB)"
                  f" ← {model['name']} ({est})")

        # 检查是否有模型没分配到 GPU
        unplaced = [m for m in self.models
                    if not any(a[1]['key'] == m['key'] for a in initial_assignments)]
        if unplaced:
            print(f"\n  [未放置] 以下模型暂无可用的 GPU：")
            for m in unplaced:
                print(f"    {m['name']} — 等待 GPU 释放...")

        # Step 5: 启动初始 Worker
        print(f"\n  [启动 Worker]")
        for gpu_id, model in initial_assignments:
            p = self._start_worker(gpu_id, model)
            print(f"    GPU {gpu_id} → {model['name']} (PID={p.pid})")

        # Step 6: 监控循环 — 动态调度核心（显式状态机驱动）
        print(f"\n{'=' * 60}")
        print(f"  开始监控调度循环（轮询间隔 {self.poll_interval}s）")
        print(f"{'=' * 60}")

        start_time = time.time()
        last_status_time = start_time
        MAX_CONSECUTIVE_FAILURES = 3  # 同一模型连续异常退出上限

        while True:
            # ---- 6a. 检查已完成的 Worker，同步状态 ----
            newly_freed_gpus = []
            still_running = []

            for w_info in self._running_workers:
                p = w_info['process']
                if not p.is_alive():
                    p.join()  # 回收僵尸进程
                    exitcode = p.exitcode
                    gpu_id = w_info['gpu_id']
                    model_key = w_info['model_key']
                    model_name = w_info['model_name']

                    if exitcode != 0:
                        self._consecutive_failures[model_key] = \
                            self._consecutive_failures.get(model_key, 0) + 1
                        print(f"\n  [警告] Worker {model_name}@GPU{gpu_id} 异常退出"
                              f" (exitcode={exitcode}, "
                              f"连续失败 {self._consecutive_failures[model_key]}/{MAX_CONSECUTIVE_FAILURES})")
                    else:
                        # 正常退出 → 重置失败计数
                        self._consecutive_failures[model_key] = 0

                    # 同步模型状态（基于计数器 + 剩余活跃 Worker）
                    self._update_model_status(model_key)
                    new_status = self._model_status.get(model_key, '?')

                    if new_status == 'finished':
                        mq = self._model_queues.get(model_key)
                        total = mq['total'] if mq else '?'
                        done = mq['counter'].value if mq else '?'
                        print(f"\n  [完成] 模型 '{model_name}' 全部任务完成！"
                              f" ({done}/{total}), 状态 → finished")

                    # 释放 GPU 资源
                    self._release_gpu(self.gpus[gpu_id], model_key)
                    newly_freed_gpus.append(gpu_id)
                    print(f"  [释放] GPU {gpu_id} 资源已回收"
                          f" (空闲 {self.gpus[gpu_id]['free_vram']:.1f} GB)"
                          f" | 模型 {model_name} 状态: {new_status}")
                else:
                    still_running.append(w_info)

            self._running_workers = still_running

            # ---- 6b. 对每个刚释放的 GPU 执行调度决策 ----
            scheduled_this_round: set = set()  # 本轮已调度的 GPU，避免重复
            for gpu_id in newly_freed_gpus:
                if self._try_schedule_on_gpu(gpu_id, MAX_CONSECUTIVE_FAILURES):
                    scheduled_this_round.add(gpu_id)

            # ---- 6b2. 重新检查所有空闲 GPU（含历史遗留的空闲 GPU） ----
            # 场景：GPU 在上一轮被释放但无法容纳 pending 模型（显存不足），
            #       之后同卡上另一个模型也释放了资源 → 现在可能够用了。
            #       旧逻辑仅在 newly_freed_gpus 上触发调度，导致该 GPU 永久空闲。
            idle_gpus = self._get_idle_gpus()
            for gpu_id in idle_gpus:
                if gpu_id not in scheduled_this_round and gpu_id not in newly_freed_gpus:
                    pending = self._get_pending_models()
                    if pending:
                        if self._try_schedule_on_gpu(gpu_id, MAX_CONSECUTIVE_FAILURES):
                            scheduled_this_round.add(gpu_id)

            # ---- 6c. 终止条件检查 ----
            # 将连续失败的模型标记为"不可恢复"
            for m in self.models:
                mk = m['key']
                if self._consecutive_failures.get(mk, 0) >= MAX_CONSECUTIVE_FAILURES:
                    if self._model_status.get(mk) not in ('finished',):
                        self._model_status[mk] = 'finished'
                        self._finished_models.add(mk)
                        mq = self._model_queues.get(mk)
                        done = mq['counter'].value if mq else '?'
                        total = mq['total'] if mq else '?'
                        print(f"  [放弃] 模型 '{m['name']}' 连续失败 {MAX_CONSECUTIVE_FAILURES} 次，"
                              f"标记为不可恢复 ({done}/{total})")

            all_finished = len(self._finished_models) >= len(self.models)
            if all_finished and not self._running_workers:
                break

            if all_finished:
                # 所有模型标记完成，等待剩余 Worker 自然退出（最多 5s）
                time.sleep(1)
                if not any(w['process'].is_alive() for w in self._running_workers):
                    break

            # 安全网：检查是否有模型处于 pending 但无 GPU 可用（死锁检测）
            pending_models = self._get_pending_models()
            # 放宽死锁检测：只要有 pending 模型且有空闲 GPU 就尝试强制放置
            idle_gpus = self._get_idle_gpus()
            if pending_models and idle_gpus:
                # 排查：是否所有 pending 模型的预估显存都超过空闲 GPU
                max_free = max(self.gpus[g]['free_vram'] for g in idle_gpus)
                all_too_big = all(
                    (m.get('estimated_vram_gb') or 999) * (1 + self.safety_margin) > max_free
                    for m in pending_models
                )
                if all_too_big:
                    print(f"\n  [死锁] {len(pending_models)} 个模型待运行但显存不足！"
                          f" GPU 最大空闲 {max_free:.1f} GB")
                    # 将最小显存需求的模型强制放置（降级运行）
                    smallest = min(pending_models,
                                   key=lambda m: m.get('estimated_vram_gb') or 999)
                    est = smallest.get('estimated_vram_gb') or 999
                    best_gpu = max(idle_gpus, key=lambda g: self.gpus[g]['free_vram'])
                    if self.gpus[best_gpu]['free_vram'] >= self.gpus[best_gpu]['total_vram'] * 0.3:
                        self._allocate_gpu(self.gpus[best_gpu], smallest)
                        p = self._start_worker(best_gpu, smallest)
                        print(f"  [死锁恢复] GPU {best_gpu} → {smallest['name']} (PID={p.pid})"
                              f" [降级放置, 预估 {est}GB, 可用 {max_free:.1f}GB]")
                    else:
                        for m in pending_models:
                            print(f"  [死锁] {m['name']} 无法放置，标记为不可恢复")
                            self._model_status[m['key']] = 'finished'
                            self._finished_models.add(m['key'])
                else:
                    # 有可放置的模型 → 强制调度
                    for gpu_id in idle_gpus:
                        if self._try_schedule_on_gpu(gpu_id, MAX_CONSECUTIVE_FAILURES + 999):
                            print(f"  [死锁恢复] GPU {gpu_id} 强制调度成功")

            # ---- 6d. 定期打印状态 ----
            now = time.time()
            if now - last_status_time >= 30:
                self._print_status()
                last_status_time = now

            time.sleep(self.poll_interval)

        # Step 7: 完成
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print(f"\n{'=' * 60}")
        print(f"  全部实验完成！")
        print(f"  总耗时: {hours}h {minutes}m {seconds}s")
        print(f"  完成模型: {len(self._finished_models)}/{len(self.models)}")
        total_done = sum(
            mq['counter'].value for mq in self._model_queues.values()
        )
        print(f"  完成任务: {total_done}/{total_tasks}")
        print(f"  输出目录: {self.output_dir}")
        print(f"{'=' * 60}")

        if self._manager:
            self._manager.shutdown()


# ============================================================
#  主入口
# ============================================================

def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_config.json")
    config = load_config(config_path)

    # ---- 初始化运行日志（最先执行，确保所有输出被记录） ----
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    tee_logger = TeeLogger(log_dir)
    tee_logger.start()
    try:
        _main_impl(config, config_path)
    finally:
        tee_logger.stop()
        log_path = tee_logger.get_path()
        # 用原始 stdout 输出日志路径（因为 tee 已停止）
        print(f"\n[日志] 运行日志已保存至: {log_path}")


def _main_impl(config: dict, config_path: str):
    """main() 的实际实现，由 main() 包裹 TeeLogger 后调用。"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    gen_params = config["generation_params"]

    # 检查是否启用
    songci_enabled = config.get('songci', {}).get('enabled', False)
    tangpoem_enabled = config.get('tangpoem', {}).get('enabled', False)
    if not songci_enabled and not tangpoem_enabled:
        print("[错误] 宋词和唐诗均未启用，请检查 batch_config.json")
        return

    enabled_models = [m for m in config.get('models', []) if m.get('enabled', True)]
    if not enabled_models:
        print("[错误] 没有启用的模型，请检查 batch_config.json → models")
        return

    print(f"{'=' * 60}")
    print(f"  批量诗词生成实验 v2.0 — 智能资源调度")
    print(f"  模型数: {len(enabled_models)}")
    for m in enabled_models:
        print(f"    - {m['name']} ({_quantization_label(m.get('quantization', '8bit'))})"
              f" max_parallel={m.get('max_parallel', 1)}")
    print(f"  宋词: {'启用' if songci_enabled else '禁用'}")
    print(f"  唐诗: {'启用' if tangpoem_enabled else '禁用'}")
    print(f"  对比实验: {'启用' if config.get('compare_experiment', False) else '禁用'}")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 60}")

    # ---- 强制串行模式 ----
    if config.get('force_sequential', False):
        print("\n  [force_sequential=True] 强制串行模式，使用传统执行路径。")
        _run_sequential_legacy(config, gen_params, output_dir)
        return

    # ---- 硬件检测 ----
    models_info = []
    for m in enabled_models:
        est = HardwareDetector.estimate_model_vram_gb(
            m['path'], m.get('quantization', '8bit'),
            model_name=m.get('name', ''),
        )
        models_info.append({
            'name': m['name'],
            'quantization': _quantization_label(m.get('quantization', '8bit')),
            'estimated_vram_gb': est,
            'max_parallel': m.get('max_parallel', 1),
        })

    gpus = HardwareDetector.print_summary(models_info)

    if not gpus:
        print("[回退] 无 GPU，使用传统串行模式。")
        _run_sequential_legacy(config, gen_params, output_dir)
        return

    # ---- 智能调度 ----
    scheduler = SmartScheduler(config, gen_params, output_dir)
    scheduler.run()


def _run_sequential_legacy(config: dict, gen_params: dict, output_dir: str):
    """
    传统串行模式（force_sequential=True 或 CPU 兜底时使用）。
    所有模型依次执行，每个模型内部按原有逻辑串行。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    from data_manager import DataManager
    from vocab_indexer import VocabIndexer
    from state_machine import GenerationStateMachine, TangPoemStateMachine
    from logits_processor import ConstraintLogitsProcessor, TangPoemLogitsProcessor
    from main import parse_tang_format, build_prompt_messages, build_tangpoem_prompt_messages

    compare_mode = config.get('compare_experiment', False)
    sc_cfg = config.get('songci', {})
    tp_cfg = config.get('tangpoem', {})
    sc_enabled = sc_cfg.get('enabled', False)
    tp_enabled = tp_cfg.get('enabled', False)

    enabled_models = [m for m in config.get('models', []) if m.get('enabled', True)]

    for model_cfg in enabled_models:
        model_name = model_cfg['name']
        model_path = model_cfg['path']
        quantization = model_cfg.get('quantization', '8bit')
        use_thinking = model_cfg.get('use_thinking', False)
        q_label = _quantization_label(quantization)

        print(f"\n{'=' * 60}")
        print(f"  模型: {model_name} ({q_label})")
        print(f"{'=' * 60}")

        print(f"[1/3] 加载 tokenizer: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print(f"[2/3] 加载模型 (量化={q_label})...")
        model_kwargs = _get_quantization_kwargs(quantization)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, **model_kwargs,
        ).eval()

        rhyme_dict_name = sc_cfg.get('rhyme_dict_name') or tp_cfg.get('rhyme_dict_name') or 'Xinyun'
        print(f"[3/3] 构建词表索引 (韵书: {rhyme_dict_name})...")
        rhyme_dict_path = os.path.join(PROJECT_ROOT, 'Rhyme', f'{rhyme_dict_name}.json')
        poem_path = os.path.join(PROJECT_ROOT, 'Songci_Meter')
        data_manager = DataManager(rhyme_dict_path=rhyme_dict_path, poem_path=poem_path)
        vocab_indexer = VocabIndexer(tokenizer, data_manager)
        pad_token_id = tokenizer.eos_token_id

        def _run_tasks(meter_type, use_constraints, subdir=None):
            if meter_type == '宋词':
                cipai_list = sc_cfg.get('cipai_list', [])
                themes = sc_cfg.get('themes', [])
                rhyme = sc_cfg.get('rhyme_dict_name', 'Xinyun')
                task_type = sc_cfg.get('task_type', 'instruction')
                num_gen = sc_cfg.get('num_generations', 1)
                for cipai in cipai_list:
                    for entry in themes:
                        theme = entry['theme']
                        detailed_req = entry.get('detailed_requirement', '')
                        safe_theme = sanitize_filename(theme)

                        if subdir:
                            out_file = os.path.join(output_dir, subdir,
                                                    f'{model_name}-({task_type})-{cipai}-{safe_theme}.txt')
                        else:
                            out_file = os.path.join(output_dir,
                                                    f'{model_name}-({task_type})-{cipai}-{safe_theme}.txt')
                        os.makedirs(os.path.dirname(out_file), exist_ok=True)

                        try:
                            messages = build_prompt_messages(
                                task_type, cipai, theme, requirement=detailed_req,
                                poem_path=poem_path, use_thinking=use_thinking,
                                rhyme_dict_name=rhyme,
                            )
                        except Exception as e:
                            with open(out_file, 'w', encoding='utf-8') as f:
                                f.write(f'[错误] 构建 Prompt 失败: {e}\n')
                            continue

                        chat_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )

                        with open(out_file, 'w', encoding='utf-8') as f:
                            f.write(f'# 模型: {model_name}\n')
                            f.write(f'# 词牌: {cipai}\n')
                            f.write(f'# 主题: {theme}\n')
                            f.write(f'# 约束解码: {"启用" if use_constraints else "禁用"}\n')
                            f.write(f'# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                            f.write(f'# ========================================\n\n')

                        for gen_idx in range(num_gen):
                            inputs = tokenizer(chat_prompt, return_tensors='pt').to(model.device)
                            ipl = inputs.input_ids.shape[1]

                            if use_constraints:
                                sm = GenerationStateMachine(cipai, data_manager)
                                lp = ConstraintLogitsProcessor(
                                    vocab_indexer=vocab_indexer, state_machine=sm,
                                    tokenizer=tokenizer, input_prompt_len=ipl,
                                )
                                processors = LogitsProcessorList([lp])
                            else:
                                processors = None

                            try:
                                results = generate_one_batch(
                                    model, tokenizer, inputs, ipl, processors,
                                    gen_params, 1, pad_token_id,
                                )
                                output_text = results[0]
                            except Exception as e:
                                output_text = f'[生成错误] {e}'

                            with open(out_file, 'a', encoding='utf-8') as f:
                                f.write(f'=== 作品 {gen_idx + 1} ===\n')
                                f.write(output_text.strip())
                                f.write('\n\n')

                        print(f"  [{model_name}] {cipai}·{theme} 完成")

            else:  # 唐诗
                forms = tp_cfg.get('forms', [])
                themes = tp_cfg.get('themes', [])
                rhyme = tp_cfg.get('rhyme_dict_name', 'Xinyun')
                task_type = tp_cfg.get('task_type', 'instruction')
                num_gen = tp_cfg.get('num_generations', 1)
                for form in forms:
                    form_name = form['name']
                    line_length = form['line_length']
                    num_lines = form['num_lines']
                    _, _, rhyme_type = parse_tang_format(form_name)

                    for entry in themes:
                        theme = entry['theme']
                        detailed_req = entry.get('detailed_requirement', '')
                        safe_theme = sanitize_filename(theme)

                        if subdir:
                            out_file = os.path.join(output_dir, subdir,
                                                    f'{model_name}-({task_type})-{form_name}-{safe_theme}.txt')
                        else:
                            out_file = os.path.join(output_dir,
                                                    f'{model_name}-({task_type})-{form_name}-{safe_theme}.txt')
                        os.makedirs(os.path.dirname(out_file), exist_ok=True)

                        try:
                            messages = build_tangpoem_prompt_messages(
                                task_type, form_name, theme, requirement=detailed_req,
                                use_thinking=use_thinking, line_length=line_length,
                                num_lines=num_lines, rhyme_dict_name=rhyme,
                            )
                        except Exception as e:
                            with open(out_file, 'w', encoding='utf-8') as f:
                                f.write(f'[错误] 构建 Prompt 失败: {e}\n')
                            continue

                        chat_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )

                        with open(out_file, 'w', encoding='utf-8') as f:
                            f.write(f'# 模型: {model_name}\n')
                            f.write(f'# 诗体: {form_name}\n')
                            f.write(f'# 主题: {theme}\n')
                            f.write(f'# 约束解码: {"启用" if use_constraints else "禁用"}\n')
                            f.write(f'# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                            f.write(f'# ========================================\n\n')

                        for gen_idx in range(num_gen):
                            inputs = tokenizer(chat_prompt, return_tensors='pt').to(model.device)
                            ipl = inputs.input_ids.shape[1]

                            if use_constraints:
                                sm = TangPoemStateMachine(
                                    line_length=line_length, num_lines=num_lines,
                                    rhyme_type=rhyme_type, data_manager=data_manager,
                                )
                                lp = TangPoemLogitsProcessor(
                                    vocab_indexer=vocab_indexer, state_machine=sm,
                                    tokenizer=tokenizer, input_prompt_len=ipl,
                                )
                                processors = LogitsProcessorList([lp])
                            else:
                                processors = None

                            try:
                                results = generate_one_batch(
                                    model, tokenizer, inputs, ipl, processors,
                                    gen_params, 1, pad_token_id,
                                )
                                output_text = results[0]
                            except Exception as e:
                                output_text = f'[生成错误] {e}'

                            with open(out_file, 'a', encoding='utf-8') as f:
                                f.write(f'=== 作品 {gen_idx + 1} ===\n')
                                f.write(output_text.strip())
                                f.write('\n\n')

                        print(f"  [{model_name}] {form_name}·{theme} 完成")

        if compare_mode:
            if sc_enabled:
                print(f"\n  [{model_name}] 宋词 — 无约束自由生成")
                _run_tasks('宋词', use_constraints=False, subdir='free_decoding')
            if tp_enabled:
                print(f"\n  [{model_name}] 唐诗 — 无约束自由生成")
                _run_tasks('唐诗', use_constraints=False, subdir='free_decoding')
            if sc_enabled:
                print(f"\n  [{model_name}] 宋词 — 约束解码生成")
                _run_tasks('宋词', use_constraints=True, subdir='constrained_decoding')
            if tp_enabled:
                print(f"\n  [{model_name}] 唐诗 — 约束解码生成")
                _run_tasks('唐诗', use_constraints=True, subdir='constrained_decoding')
        else:
            if sc_enabled:
                _run_tasks('宋词', use_constraints=True)
            if tp_enabled:
                _run_tasks('唐诗', use_constraints=True)

        # 每个模型跑完释放显存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"  全部实验完成！（串行模式）")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[批量生成失败] {exc}")
        traceback.print_exc()
        raise
