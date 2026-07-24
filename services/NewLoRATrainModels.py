# -*-coding : utf-8 -*-
# @Author   : zhangtao
# @FileName : NewLoRATrainModels.py
# @Desc     : LoRA 微调模型
# @Time     : 2025/12/30
# @Software : PyCharm

import os

# 必须在导入Torch/Paddle相关模块前设置，避免多进程训练触发OpenBLAS线程告警。
os.environ['OMP_NUM_THREADS'] = os.environ.get('AI_OMP_NUM_THREADS', '1')
os.environ['MKL_NUM_THREADS'] = os.environ.get('AI_OMP_NUM_THREADS', '1')

import torch
import torch_npu
import json
import ast
import gc
import hashlib
from typing import Callable, Optional
import torch.optim as optim
from tqdm import tqdm

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_NPU_ENABLE_CPU_FALLBACK'] = '1'
os.environ['TORCH_NPU_FALLBACK_DEBUG'] = '0'
os.environ['TORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.7'
os.environ['ACCELERATE_DISABLE_ALL'] = '1'
os.environ['TRANSFORMERS_NO_ACCELERATE'] = '1'
os.environ['TORCH_DISABLE_BFLOAT16'] = '1'
os.environ['TRANSFORMERS_DISABLE_BF16'] = '1'
os.environ['TORCH_NPU_ENABLE_FUSION'] = '0'
os.environ['ACL_DISABLE_OP_FUSION'] = '1'
from core.LoggerDetector import logger

class LoRATrainModelNPU:
    DEFAULT_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    REQUIRED_OUTPUT_FIELDS = {"审核结果", "审核依据"}
    _SYSTEM_PROMPT = (
        "你是专业的中文档案审核助手。请分析输入的档案信息，并严格按照 JSON 格式返回审核结果。"
    )

    def __init__(
            self,
            base_model_path: str,
            lora_r: int = 8,
            lora_alpha: int = 16,
            npu_device_id: int = 0,
            seed: int = 42,
            world_size: int = 4,
            rank: int = 0,
            master_port: Optional[str] = None
    ):
        self.base_model_path = base_model_path
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.seed = seed
        self.world_size = world_size
        self.rank = rank
        self.npu_device_id = npu_device_id
        self.master_port = str(master_port or os.environ.get('MASTER_PORT', '29500'))
        self.per_gpu_samples = None
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.master_port

        self.device = torch.device(f"npu:{self.npu_device_id}")
        torch.npu.set_device(self.device)
        torch.npu.set_compile_mode(jit_compile=False)
        logger.info(f"已绑定进程到NPU:{self.npu_device_id}")
        torch.manual_seed(seed)
        if torch_npu.npu.is_available():
            torch_npu.npu.manual_seed(seed)
        if self.rank == 0 and not os.path.exists(base_model_path):
            raise FileNotFoundError(f"基础模型路径不存在: {base_model_path}")
        self.model = None
        self.peft_model = None
        self.tokenizer = None
        self.dist_sampler = None
        self.model_dtype = torch.float16
        self.index_dtype = torch.int64
        self.optimizer = None
        self.scheduler = None
        self.train_model = None

    def setup_environment(self):
        logger.info(f"初始化NPU:{self.npu_device_id}环境，总卡数：{self.world_size}，MASTER_PORT={self.master_port}")

        if not torch_npu.npu.is_available():
            raise RuntimeError("NPU不可用，请检查驱动安装")
        if self.world_size > 1:
            try:
                backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "hccl")
                dist.init_process_group(
                    backend=backend,
                    world_size=self.world_size,
                    rank=self.rank,
                    init_method='env://'
                )
            except Exception as e:
                raise RuntimeError(f"分布式进程初始化失败：{e}")

        torch.npu.empty_cache()
        gc.collect()

        if self.rank == 0:
            logger.info(f"{self.world_size}卡分布式环境初始化完成")

    def _report_progress(
            self,
            progress_callback: Optional[Callable[[float], None]],
            percentage: float
    ) -> None:
        if self.rank != 0 or progress_callback is None:
            return
        try:
            progress_callback(round(max(0.0, min(100.0, percentage)), 1))
        except Exception as error:
            # 进度写入失败不能中断模型训练。
            logger.warning(f"训练进度上报失败: {error}")

    def load_model_npu(self, continue_training: bool = False, previous_model_path: Optional[str] = None):
        logger.info(f"加载模型（CPU中转）: {self.base_model_path}（NPU:{self.npu_device_id}）")

        if continue_training and not previous_model_path:
            raise ValueError("增量训练需要提供previous_model_path")

        # 1. 加载分词器
        logger.info(f"加载分词器...（NPU:{self.npu_device_id}）")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                previous_model_path if (
                            continue_training and os.path.exists(previous_model_path)) else self.base_model_path,
                padding_side="right",
                use_fast=True,
                trust_remote_code=False,
                low_cpu_mem_usage=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.rank == 0:
                logger.info(f"分词器加载完成，词汇表大小: {len(self.tokenizer)}")

        except Exception as e:
            logger.error(f"加载分词器失败: {e}")
            raise

        logger.info(f"NPU:{self.npu_device_id}）")
        try:
            config = AutoConfig.from_pretrained(
                self.base_model_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True
            )
            config.use_flash_attention = False
            config.bf16 = False
            config.fp16 = True
            config.torch_dtype = torch.float32
            config.gradient_checkpointing = True
            config.use_cache = False

            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                config=config,
                low_cpu_mem_usage=True,
                device_map=None,
                dtype=torch.float32,
                trust_remote_code=False,
                ignore_mismatched_sizes=True,
                load_in_8bit=False,
                load_in_4bit=False
            )

        except Exception as e:
            logger.error(f"CPU加载模型失败: {e}")
            raise
        try:
            self.model = self.model.to(dtype=self.model_dtype)
            if hasattr(self.model, 'embed_tokens') and self.model.embed_tokens is not None:
                self.model.embed_tokens.weight.data = self.model.embed_tokens.weight.data.to(self.model_dtype)

        except Exception as e:
            logger.error(f"转换失败: {e}")
            raise
        if continue_training and os.path.exists(previous_model_path):
            logger.info(f"加载LoRA权重...（NPU:{self.npu_device_id}）")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    previous_model_path,
                    device_map=None,
                    dtype=self.model_dtype,
                    ignore_mismatched_sizes=True
                )
            except Exception as e:
                logger.error(f"加载LoRA权重失败: {e}")
                raise

        logger.info(f"迁移模型到NPU:{self.npu_device_id}（Float16）")
        try:
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"模型迁移到NPU失败: {e}")
            raise

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False

        # 仅设置config.gradient_checkpointing不会注册实际的检查点逻辑，
        # 必须显式启用，才能在反向传播时以计算换显存。
        checkpoint_model = (
            self.model.get_base_model()
            if isinstance(self.model, PeftModel)
            else self.model
        )
        try:
            checkpoint_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # 兼容不支持gradient_checkpointing_kwargs的Transformers版本。
            checkpoint_model.gradient_checkpointing_enable()
        logger.info(f"梯度检查点已启用（NPU:{self.npu_device_id}）")

        total_params = sum(p.numel() for p in self.model.parameters())
        if self.rank == 0:
            logger.info(f"模型加载完成，总参数量: {total_params / 1e9:.2f}B")

        return self.model

    def apply_lora_npu(self, target_modules: Optional[list] = None, continue_training: bool = False):
        if continue_training and isinstance(self.model, PeftModel):
            logger.info(f"启用LoRA微调（NPU:{self.npu_device_id}）")
            self.peft_model = self.model
            self.peft_model.enable_input_require_grads()
            self.peft_model.train()

            for name, param in self.peft_model.named_parameters():
                if "lora" in name.lower() or "adapter" in name.lower():
                    param.requires_grad = True
                    param.data = param.data.to(self.device)
                else:
                    param.requires_grad = False
            return self.peft_model

        logger.info(f"应用LoRA配置（NPU:{self.npu_device_id}）")

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules or self.DEFAULT_TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        try:
            self.peft_model = get_peft_model(self.model, lora_config)
            self.peft_model.enable_input_require_grads()
            self.peft_model.train()
            for name, param in self.peft_model.named_parameters():
                if "lora" in name.lower() or "adapter" in name.lower():
                    param.requires_grad = True
                    param.data = param.data.to(self.device)
                else:
                    param.requires_grad = False
        except Exception as e:
            logger.error(f"应用LoRA失败: {e}")
            raise

        if self.rank == 0:
            total_trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            logger.info(f"可训练参数：{total_trainable_params / 1e6:.2f}M")

        return self.peft_model

    def load_and_prepare_data(self, dataset_path: str, max_samples: Optional[int] = None):
        logger.info(f"加载数据集: {dataset_path}（NPU:{self.npu_device_id}）")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

        if os.path.getsize(dataset_path) == 0:
            raise ValueError(f"数据集文件为空: {dataset_path}")

        if not dataset_path.endswith('.jsonl'):
            raise ValueError(f"训练数据集必须是jsonl文件: {dataset_path}")

        all_data = []
        invalid_lines = []
        with open(dataset_path, 'r', encoding='utf-8') as handle:
            for line_num, line in enumerate(handle, 1):
                if max_samples and len(all_data) >= max_samples:
                    break
                if not line.strip():
                    continue
                try:
                    data_item = json.loads(line)
                except json.JSONDecodeError as error:
                    invalid_lines.append(f"第{line_num}行JSON错误: {error}")
                    continue
                error = self._normalize_dataset_item(data_item)
                if error:
                    invalid_lines.append(f"第{line_num}行{error}")
                    continue
                all_data.append(data_item)

        if invalid_lines:
            preview = "; ".join(invalid_lines[:5])
            raise ValueError(f"训练数据存在{len(invalid_lines)}条无效记录: {preview}")
        if not all_data:
            raise ValueError("数据集没有instruction/input/output有效样本")
        if len(all_data) < self.world_size:
            raise ValueError(
                f"有效训练样本数{len(all_data)}小于训练卡数{self.world_size}，请降低worldSize"
            )

        self.per_gpu_samples = (len(all_data) + self.world_size - 1) // self.world_size
        logger.info(
            f"数据集加载完成，总样本数={len(all_data)}，由DistributedSampler按rank切分"
        )

        return Dataset.from_list(all_data)

    @classmethod
    def _normalize_dataset_item(cls, item) -> Optional[str]:
        """校验训练样本，并将两字段专家标注规范化为四字段模型输出。"""
        if not isinstance(item, dict):
            return "不是JSON对象"
        if not all(item.get(key) for key in ("instruction", "input", "output")):
            return "缺少instruction/input/output"
        try:
            output = json.loads(item["output"])
        except (TypeError, json.JSONDecodeError):
            try:
                output = ast.literal_eval(item["output"])
            except Exception:
                return "output不是有效JSON"
        if not isinstance(output, dict):
            return "output不是JSON对象"
        missing = cls.REQUIRED_OUTPUT_FIELDS - set(output)
        if missing:
            return f"output缺少字段: {','.join(sorted(missing))}"
        if not str(output.get("审核结果") or "").strip():
            return "审核结果为空"
        basis = str(output.get("审核依据") or "").strip()
        if not basis:
            return "审核依据为空"
        output["审核结果"] = str(output["审核结果"]).strip()
        output["审核依据"] = basis
        output["置信度"] = cls._normalize_confidence(output.get("置信度"))
        thinking = str(output.get("思考过程") or "").strip()
        output["思考过程"] = thinking or basis
        item["output"] = json.dumps(output, ensure_ascii=False)
        return None

    @staticmethod
    def _normalize_confidence(value) -> float:
        try:
            confidence = float(str(value).replace("%", ""))
            if confidence > 10:
                confidence /= 10
            return round(max(0.0, min(10.0, confidence)), 1)
        except (TypeError, ValueError):
            return 8.0

    def _build_prompt(self, instruction: str, input_text: str) -> str:
        user_content = (
            f"{instruction.strip()}\n\n"
            f"{input_text.strip()}\n\n"
            "请只输出一行JSON，不要输出解释性前缀。\n"
            "输出："
        )
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return (
                f"{self._SYSTEM_PROMPT}\n\n"
                f"{user_content}"
            )

    def _normalize_output(self, output_text: str) -> str:
        output_text = output_text.strip()
        try:
            json.loads(output_text)
        except Exception:
            try:
                output_text = json.dumps(ast.literal_eval(output_text), ensure_ascii=False)
            except Exception:
                pass

        eos = self.tokenizer.eos_token or ""
        if output_text.endswith(eos):
            return output_text
        return output_text + eos

    def tokenize_dataset_npu(self, dataset, max_length: int = 256, create_sampler: bool = True):
        logger.info(
            f"数据集分词（NPU:{self.npu_device_id}），已启用 Prompt Loss 掩码，"
            f"按实际长度保存，最大长度上限: {max_length}"
        )

        def tokenize_function(examples):
            batch_len = len(next(iter(examples.values()))) if examples else 0
            instructions = examples.get("instruction", ["无指令"] * batch_len)
            inputs = examples.get("input", ["无输入"] * batch_len)
            outputs = examples.get("output", ["无输出"] * batch_len)

            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []

            for inst, inp, out in zip(instructions, inputs, outputs):
                inst = inst if isinstance(inst, str) else "无指令"
                inp = inp if isinstance(inp, str) else "无输入"
                out = out if isinstance(out, str) else "无输出"

                prompt_text = self._build_prompt(inst, inp)
                response_text = self._normalize_output(out)

                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

                if len(response_ids) >= max_length:
                    response_ids = response_ids[:max_length - 1]
                    response_ids.append(self.tokenizer.eos_token_id)

                max_prompt_len = max_length - len(response_ids)
                if max_prompt_len <= 0:
                    raise ValueError("max_length过短，无法保留答案token")
                if len(prompt_ids) > max_prompt_len:
                    prefix_len = max(1, max_prompt_len // 3)
                    suffix_len = max_prompt_len - prefix_len
                    prompt_ids = prompt_ids[:prefix_len] + prompt_ids[-suffix_len:]

                input_ids = prompt_ids + response_ids
                labels = [-100] * len(prompt_ids) + response_ids
                # 不在数据集预处理阶段固定补齐到max_length。保留每条样本的实际长度，
                # 由_collate_batch在取出批次时动态补齐到该批次的最长样本。
                attention_mask = [1] * len(input_ids)

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)

            return {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "labels": batch_labels
            }

        try:
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=4,
                remove_columns=dataset.column_names
            )
            logger.info(f"分词及掩码处理成功，有效样本数: {len(tokenized_dataset)}（NPU:{self.npu_device_id}）")
        except Exception as e:
            raise RuntimeError(f"训练数据分词或标签掩码失败: {e}") from e

        if len(tokenized_dataset) == 0:
            raise RuntimeError("分词后没有有效数据")
        if create_sampler:
            self.dist_sampler = DistributedSampler(
                tokenized_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False
            )

        if self.rank == 0:
            label_count = sum(1 for x in tokenized_dataset[0]['labels'] if x != -100)
            token_lengths = [len(item['input_ids']) for item in tokenized_dataset]
            logger.info(f"数据集分词完成，总有效样本数: {len(tokenized_dataset)}")
            logger.info(
                "动态长度统计："
                f"最短={min(token_lengths)}，最长={max(token_lengths)}，"
                f"平均={sum(token_lengths) / len(token_lengths):.1f} token，"
                f"上限={max_length}"
            )
            logger.info(f"样例检测：第0条可学习答案token数={label_count}，前30个Label={tokenized_dataset[0]['labels'][:30]}")

        return tokenized_dataset

    @staticmethod
    def _split_train_validation(dataset, validation_ratio: float = 0.15, min_samples: int = 10):
        """按题名/输入指纹分组拆分，降低近重复数据泄漏风险。"""
        min_samples = max(2, int(min_samples))
        if len(dataset) < min_samples:
            raise ValueError(
                f"有效样本仅{len(dataset)}条，至少需要{min_samples}条，无法建立可靠训练/验证集"
            )
        train_indices, validation_indices = [], []
        for index, item in enumerate(dataset):
            input_text = str(item.get("input") or "")
            try:
                payload = json.loads(input_text)
                group_key = str(payload.get("题名") or input_text)
            except Exception:
                group_key = input_text
            bucket = int(hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:8], 16) % 10000
            if bucket < int(validation_ratio * 10000):
                validation_indices.append(index)
            else:
                train_indices.append(index)
        if not validation_indices:
            validation_indices.append(train_indices.pop())
        if not train_indices:
            train_indices.append(validation_indices.pop())
        return dataset.select(train_indices), dataset.select(validation_indices)

    def _evaluate_validation(self, tokenized_validation, batch_size: int) -> float:
        sampler = DistributedSampler(
            tokenized_validation,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
        )

        loader = DataLoader(
            tokenized_validation,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._collate_batch,
        )
        self.train_model.eval()
        loss_sum = torch.tensor(0.0, device=self.device)
        batch_count = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                with torch.autocast(device_type='npu', dtype=self.model_dtype, enabled=True):
                    loss = self.train_model(**batch).loss
                loss_sum += loss.detach().float()
                batch_count += 1
        if dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        self.train_model.train()
        return (loss_sum / batch_count.clamp_min(1)).item()

    def _collate_batch(self, batch):
        if not batch:
            raise ValueError("训练批次为空")

        batch_max_length = max(len(item["input_ids"]) for item in batch)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id 未设置，无法动态补齐训练批次")

        input_ids = []
        attention_masks = []
        labels = []
        for item in batch:
            item_length = len(item["input_ids"])
            padding_length = batch_max_length - item_length
            input_ids.append(item["input_ids"] + [pad_token_id] * padding_length)
            attention_masks.append(item["attention_mask"] + [0] * padding_length)
            labels.append(item["labels"] + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=self.index_dtype),
            "attention_mask": torch.tensor(attention_masks, dtype=self.index_dtype),
            "labels": torch.tensor(labels, dtype=self.index_dtype),
        }

    def setup_optimizer_npu(self, total_training_steps: int, learning_rate: float = 1e-4):
        logger.info(f"设置优化器（NPU:{self.npu_device_id}) ")

        if not self.peft_model:
            raise RuntimeError("Peft模型未初始化，无法设置优化器")

        trainable_params = [
            p for p in self.peft_model.parameters()
            if p.requires_grad and p.numel() > 0
        ]
        if len(trainable_params) == 0:
            trainable_params = [
                p for name, p in self.peft_model.named_parameters()
                if "lora" in name.lower()
            ]
        if len(trainable_params) == 0:
            raise RuntimeError(f"可训练参数为空（NPU:{self.npu_device_id}）")

        try:
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=max(1, int(total_training_steps * 0.03)),
                num_training_steps=total_training_steps
            )
            logger.info(f"优化器初始化成功，待优化参数数：{len(trainable_params)}，learning_rate={learning_rate}")
        except Exception as e:
            logger.error(f"优化器初始化失败：{e}")
            raise

        return self.optimizer

    def train_manual_npu(self,
                         dataset_path: str,
                         output_dir: str,
                         num_epochs: int = 4,
                         max_samples: Optional[int] = None,
                         continue_training: bool = False,
                         previous_model_path: Optional[str] = None,
                         max_length: int = 2048,
                         batch_size: int = 1,
                         learning_rate: float = 1e-4,
                         gradient_accumulation_steps: int = 2,
                         validation_ratio: float = 0.15,
                         min_samples: int = 10,
                         progress_callback: Optional[Callable[[float], None]] = None):

        self.setup_environment()
        self._report_progress(progress_callback, 2.0)

        # 数据量校验和训练/验证拆分必须早于7B模型加载，失败时避免占用NPU和加载权重。
        dataset = self.load_and_prepare_data(dataset_path, max_samples=max_samples)
        train_dataset, validation_dataset = self._split_train_validation(
            dataset,
            validation_ratio=validation_ratio,
            min_samples=min_samples,
        )
        self._report_progress(progress_callback, 4.0)

        self.load_model_npu(continue_training=continue_training, previous_model_path=previous_model_path)
        self._report_progress(progress_callback, 7.0)

        self.apply_lora_npu(continue_training=continue_training)
        self._report_progress(progress_callback, 10.0)

        self._report_progress(progress_callback, 12.0)
        tokenized_dataset = self.tokenize_dataset_npu(
            train_dataset, max_length=max_length, create_sampler=True
        )
        tokenized_validation = self.tokenize_dataset_npu(
            validation_dataset, max_length=max_length, create_sampler=False
        )
        self._report_progress(progress_callback, 16.0)

        train_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            sampler=self.dist_sampler,
            collate_fn=self._collate_batch,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=None
        )

        if self.world_size > 1:
            self.train_model = DDP(
                self.peft_model,
                device_ids=[self.npu_device_id],
                output_device=self.npu_device_id,
                find_unused_parameters=False
            )
        else:
            self.train_model = self.peft_model

        self.train_model.train()

        gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        total_training_steps = max(
            1,
            ((len(train_dataloader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps) * num_epochs
        )
        self.setup_optimizer_npu(total_training_steps=total_training_steps, learning_rate=learning_rate)
        self._report_progress(progress_callback, 18.0)

        torch.npu.empty_cache()
        gc.collect()

        if self.rank == 0:
            logger.info(f"开始{self.world_size}卡分布式训练")
            logger.info(
                f"配置：epochs={num_epochs}, batch_size={batch_size}, "
                f"gradient_accumulation_steps={gradient_accumulation_steps}, "
                f"learning_rate={learning_rate}, 训练样本={len(tokenized_dataset)}, "
                f"验证样本={len(tokenized_validation)}，每卡约{self.per_gpu_samples}样本"
            )

        best_validation_loss = float("inf")
        best_epoch = 0
        training_history = []
        epoch_progress_span = (94.0 - 18.0) / max(1, num_epochs)
        for epoch in range(num_epochs):
            if self.dist_sampler:
                self.dist_sampler.set_epoch(epoch)

            if self.rank == 0:
                logger.info(f"\n===== 第{epoch + 1}/{num_epochs}轮训练 =====")
                pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}", smoothing=0.1)

            epoch_loss = 0.0
            step = 0
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    torch.npu.reset_peak_memory_stats(self.device)
                except TypeError:
                    # 兼容不接收device参数的Torch-NPU版本。
                    try:
                        torch.npu.reset_peak_memory_stats()
                    except Exception as error:
                        logger.warning(f"重置NPU显存峰值失败，继续训练: {error}")
                except Exception as error:
                    # 显存监控属于辅助能力，不能因为接口差异中断训练。
                    logger.warning(f"重置NPU显存峰值失败，继续训练: {error}")

                batch_device = {}
                for k, v in batch.items():
                    if k in ["input_ids", "labels", "attention_mask"]:
                        batch_device[k] = v.to(self.device, non_blocking=True)
                    else:
                        batch_device[k] = v.to(self.device, non_blocking=True, dtype=self.model_dtype)

                with torch.autocast(device_type='npu', dtype=self.model_dtype, enabled=True):
                    outputs = self.train_model(**batch_device)
                    raw_loss = outputs.loss
                    loss = raw_loss / gradient_accumulation_steps

                loss.backward()

                if self.rank == 0:
                    try:
                        allocated_gb = torch.npu.memory_allocated(self.device) / 1024 ** 3
                        reserved_gb = torch.npu.memory_reserved(self.device) / 1024 ** 3
                        peak_gb = torch.npu.max_memory_allocated(self.device) / 1024 ** 3
                        logger.info(
                            f"训练批次显存：epoch={epoch + 1}, batch={batch_idx + 1}, "
                            f"batch_size={batch_device['input_ids'].shape[0]}, "
                            f"实际长度={batch_device['input_ids'].shape[1]} token, "
                            f"已分配={allocated_gb:.2f}GB, 保留={reserved_gb:.2f}GB, "
                            f"峰值={peak_gb:.2f}GB"
                        )
                    except Exception as error:
                        logger.warning(f"读取NPU显存统计失败，继续训练: {error}")

                should_step = (
                        (batch_idx + 1) % gradient_accumulation_steps == 0
                        or (batch_idx + 1) == len(train_dataloader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(self.peft_model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += raw_loss.item()
                step += 1

                if self.rank == 0:
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{epoch_loss / step:.4f}"})
                    epoch_start = 18.0 + epoch * epoch_progress_span
                    train_percentage = epoch_start + (
                        epoch_progress_span * 0.85 * (batch_idx + 1) / max(1, len(train_dataloader))
                    )
                    self._report_progress(progress_callback, train_percentage)

            gc.collect()
            torch.npu.empty_cache()

            if self.rank == 0:
                pbar.close()
                avg_loss = epoch_loss / step
                logger.info(f"第{epoch + 1}轮训练完成，平均损失: {avg_loss:.4f}")

            validation_loss = self._evaluate_validation(tokenized_validation, batch_size=batch_size)
            if self.rank == 0:
                avg_loss = epoch_loss / max(1, step)
                training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": round(avg_loss, 6),
                    "validation_loss": round(validation_loss, 6),
                })
                logger.info(f"第{epoch + 1}轮验证损失: {validation_loss:.4f}")
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_epoch = epoch + 1
                    os.makedirs(output_dir, exist_ok=True)
                    self.peft_model.save_pretrained(output_dir, safe_serialization=True)
                    self.tokenizer.save_pretrained(output_dir)
                    logger.info(f"已保存最佳Adapter: epoch={best_epoch}, val_loss={best_validation_loss:.4f}")
                self._report_progress(
                    progress_callback,
                    18.0 + (epoch + 1) * epoch_progress_span,
                )
            if dist.is_initialized():
                dist.barrier()

        if self.rank == 0:
            logger.info("\n训练完成，保存模型...")
            self._report_progress(progress_callback, 97.0)
            os.makedirs(output_dir, exist_ok=True)

            for name, param in self.peft_model.named_parameters():
                if "lora" in name.lower():
                    param.data = param.data.to(self.model_dtype)

            metrics = {
                "best_epoch": best_epoch,
                "best_validation_loss": round(best_validation_loss, 6),
                "train_sample_count": len(tokenized_dataset),
                "validation_sample_count": len(tokenized_validation),
                "history": training_history,
            }
            with open(os.path.join(output_dir, "training_metrics.json"), "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, ensure_ascii=False, indent=2)
            logger.info(f"最佳模型已保存到: {output_dir}; metrics={metrics}")
            self._report_progress(progress_callback, 99.0)

        if dist.is_initialized():
            dist.barrier()

        if self.rank == 0:
            return metrics
        return None

    def cleanup(self):
        logger.info(f"清理资源（NPU:{self.npu_device_id}）")

        self.model = None
        self.peft_model = None
        self.train_model = None
        self.tokenizer = None
        self.dist_sampler = None
        self.optimizer = None
        self.scheduler = None

        gc.collect()
        torch.npu.empty_cache()

        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.warning(f"销毁分布式进程组失败：{e}")

        if self.rank == 0:
            logger.info(f"{self.world_size}卡资源清理完成")
