# -*-coding : utf-8 -*-
# @Author   : zhangtao
# @FileName : ArchiveReviewer.py
# @Desc     : NPU推理
# @Time     : 2025/12/30
# @Software : PyCharm

import gc
import os
import re
import time
import json
import warnings
from peft import PeftModel
from typing import Optional, List

try:
    import torch_npu
    _NPU_AVAILABLE = True
except ImportError:
    _NPU_AVAILABLE = False
    warnings.warn(
        "torch_npu 未安装，NPU 不可用。"
    )

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.LoggerDetector import logger
from services.ReviewKnowledgeService import ReviewRuleEngine

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM']            = 'true'
os.environ['CUDA_VISIBLE_DEVICES']              = ''

_MAX_INPUT_LEN  = int(os.environ.get("REVIEW_MAX_INPUT_LEN", "4096"))
_MAX_CONTENT_CHARS = int(os.environ.get("REVIEW_MAX_CONTENT_CHARS", "4000"))
_MAX_NEW_TOKENS = int(os.environ.get("REVIEW_MAX_NEW_TOKENS", "512"))
_DTYPE          = torch.float16

_STOP_STRINGS = [
    '<｜end▁of▁sentence｜>',
    '<|end_of_sentence|>',
    '<|endoftext|>',
    '</s>',
    '<|im_end|>',
]

SPECIAL_TOKENS = _STOP_STRINGS + ['<s>', '<pad>', '<unk>']

def normalize_keyword_review_signal(archive_item: dict, model_type: str) -> tuple[list[str], str]:
    """规范化划控关键字及预审结果；鉴定审核不使用该信号。"""
    if model_type != "hk":
        return [], ""

    raw_keywords = archive_item.get("keywords")
    if raw_keywords is None:
        raw_keywords = []
    if not isinstance(raw_keywords, list):
        raise ValueError("划控审核字段keywords必须是字符串数组")

    keywords = []
    seen = set()
    for value in raw_keywords:
        if not isinstance(value, str):
            raise ValueError("划控审核字段keywords中的元素必须是字符串")
        keyword = value.strip()
        if keyword and keyword not in seen:
            seen.add(keyword)
            keywords.append(keyword)

    # 没有有效关键字时，预审结果必须清空，防止复用上一次规则结果。
    if not keywords:
        return [], ""

    keyword_result = str(archive_item.get("audit_result") or "").strip()
    if keyword_result not in ("开放", "控制"):
        raise ValueError("划控审核存在keywords时，audit_result必须为开放或控制")
    return keywords, keyword_result

warnings.filterwarnings("ignore")

def _make_tensor_move_hook(target_device: torch.device):
    def _move(obj):
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.to(target_device)
        if isinstance(obj, (tuple, list)):
            moved = [_move(x) for x in obj]
            return type(obj)(moved)
        return obj

    def _hook(module, inputs, outputs):
        return _move(outputs)

    return _hook


def dispatch_model_to_multi_npu(
    model: torch.nn.Module,
    npu_device_ids: List[int],
) -> torch.nn.Module:
    n = len(npu_device_ids)
    first_dev = torch.device(f'npu:{npu_device_ids[0]}')
    last_dev  = torch.device(f'npu:{npu_device_ids[-1]}')

    inner = None
    for attr in ('model', 'transformer'):
        if hasattr(model, attr):
            inner = getattr(model, attr)
            break
    if inner is None:
        logger.warning("未找到内层模型，回退到单卡")
        return model.to(first_dev)

    layers = None
    for attr in ('layers', 'h', 'blocks'):
        if hasattr(inner, attr):
            layers = getattr(inner, attr)
            break
    if layers is None or len(layers) == 0:
        logger.warning("未找到 decoder layers，回退到单卡")
        return model.to(first_dev)

    n_layers          = len(layers)
    layers_per_device = (n_layers + n - 1) // n

    logger.info(f"多卡分层: {n_layers} 层 / {n} 卡 = 约 {layers_per_device} 层/卡")

    for attr in ('embed_tokens', 'wte', 'word_embeddings',
                 'embed_positions', 'rotary_emb'):
        if hasattr(inner, attr):
            getattr(inner, attr).to(first_dev)

    for attr in ('norm', 'ln_f', 'final_layer_norm'):
        if hasattr(inner, attr):
            getattr(inner, attr).to(last_dev)
    if hasattr(model, 'lm_head'):
        model.lm_head.to(last_dev)

    _hooks = []
    for idx, layer in enumerate(layers):
        card_idx   = min(idx // layers_per_device, n - 1)
        cur_device = torch.device(f'npu:{npu_device_ids[card_idx]}')
        layer.to(cur_device)

        next_card_idx = min((idx + 1) // layers_per_device, n - 1)
        next_device   = torch.device(f'npu:{npu_device_ids[next_card_idx]}')
        if cur_device != next_device:
            h = layer.register_forward_hook(_make_tensor_move_hook(next_device))
            _hooks.append(h)
            logger.info(f"  layer[{idx:3d}] → {cur_device}  [hook → {next_device}]")
        else:
            logger.debug(f"  layer[{idx:3d}] → {cur_device}")

    if hasattr(model, 'lm_head') and last_dev != first_dev:
        h = model.lm_head.register_forward_hook(_make_tensor_move_hook(first_dev))
        _hooks.append(h)
        logger.info(f"  lm_head     → {last_dev}  [hook → {first_dev}]  (logits回迁)")

    model._npu_pipeline_hooks = _hooks   # 保持引用防止被 GC
    logger.info(f"多卡分层完成 | 共注册 {len(_hooks)} 个 hook")
    return model


# ─────────────────────────────────────────────────────────────────────────────
class ArchiveReviewer:
    def __init__(
        self,
        base_model_path: str,
        lora_model_path: str,
        batch_size: int = 8,
        model_type: str = None,
        npu_device_ids: Optional[List[int]] = None,
    ):
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.batch_size      = batch_size
        self.tokenizer       = None
        self.model           = None
        self.model_type      = model_type

        self.npu_device_ids = self._resolve_device_ids(npu_device_ids)
        self._multi_npu     = len(self.npu_device_ids) > 1
        self.device         = self._get_primary_device()

        if _NPU_AVAILABLE:
            torch_npu.npu.set_device(self.npu_device_ids[0])
            torch.npu.set_compile_mode(jit_compile=False)   # type: ignore
            logger.info(
                f"NPU 设备: {self.npu_device_ids} | "
                f"主设备: {self.device} | "
                f"多卡模式: {self._multi_npu}"
            )
        else:
            logger.warning("torch_npu 不可用，回退到 CPU")

        if self.model_type == "hk":
            self.required_basis   = "依据《连云港市档案馆延期开放档案标准及范围（试用）》"
            self.valid_results    = ["开放", "控制"]
            self.default_result   = "开放"
            self.control_keywords = [
                "控制", "延期", "敏感", "涉密", "秘密", "机密", "内部",
                "人事", "工资", "个人信息", "隐私", "不宜公开", "限制"
            ]
            self.open_keywords = [
                "开放", "公开", "通知", "公告", "总结", "报告", "批复",
                "规划", "符合开放", "可以公开", "无涉密"
            ]
        else:
            self.required_basis = "依据档案保管规范"
            self.valid_results  = ["10年", "15年", "30年", "60年", "永久"]
            self.default_result = "30年"
            self.period_keywords = {
                "永久": ["永久", "重要政策", "重大规划", "历史意义", "永久保存"],
                "60年": ["60年", "人事档案", "土地", "产权", "不动产"],
                "30年": ["30年", "一般行政", "常规", "普通文件", "行政管理"],
                "15年": ["15年", "临时工作", "阶段性", "短期项目"],
                "10年": ["10年", "日常事务", "一般事务", "例行"],
            }

        self.rule_engine = ReviewRuleEngine(self.model_type)


    @staticmethod
    def _resolve_device_ids(ids: Optional[List[int]]) -> List[int]:
        if ids is not None and len(ids) > 0:
            return ids
        env_val = os.environ.get('NPU_DEVICE_IDS', '').strip()
        if env_val:
            try:
                parsed = [int(x.strip()) for x in env_val.split(',') if x.strip()]
                if parsed:
                    return parsed
            except ValueError:
                logger.warning(f"NPU_DEVICE_IDS 格式有误: '{env_val}'，使用默认 [0]")
        return [int(os.environ.get('NPU_DEVICE_ID', '0'))]

    def _get_primary_device(self) -> torch.device:
        if _NPU_AVAILABLE:
            return torch.device(f'npu:{self.npu_device_ids[0]}')
        return torch.device('cpu')

    def _npu_empty_cache(self):
        if _NPU_AVAILABLE:
            torch_npu.npu.empty_cache()

    def _npu_memory_info(self) -> str:
        if not _NPU_AVAILABLE:
            return "N/A（CPU 模式）"
        lines = []
        for did in self.npu_device_ids:
            dev  = torch.device(f'npu:{did}')
            used = torch_npu.npu.memory_allocated(dev) / 1024 ** 3
            rsv  = torch_npu.npu.memory_reserved(dev)  / 1024 ** 3
            lines.append(f"npu:{did} 已用={used:.2f}GB 保留={rsv:.2f}GB")
        return " | ".join(lines)

    def _build_eos_ids(self) -> List[int]:
        eos_ids = set()

        for attr in ('eos_token_id', 'pad_token_id'):
            val = getattr(self.tokenizer, attr, None)
            if val is not None and val >= 0:
                eos_ids.add(val)

        added = getattr(self.tokenizer, 'added_tokens_encoder', {})
        for token_str, token_id in added.items():
            for stop in _STOP_STRINGS:
                if stop in token_str or token_str in stop:
                    eos_ids.add(token_id)
                    logger.debug(f"added_tokens_encoder 命中: {token_str!r} → {token_id}")

        for s in _STOP_STRINGS:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(s)
                unk = getattr(self.tokenizer, 'unk_token_id', None)
                if tid is not None and tid != unk and tid >= 0:
                    eos_ids.add(tid)
            except Exception:
                pass

        for s in _STOP_STRINGS:
            try:
                ids = self.tokenizer.encode(s, add_special_tokens=True)
                if len(ids) == 1:
                    eos_ids.add(ids[0])
            except Exception:
                pass

        result = sorted(eos_ids)
        logger.info(f"EOS token ids: {result}")
        return result

    def load_model(self):
        logger.info("=" * 60)
        logger.info(
            f"初始化 NPU 推理 | 设备: {self.npu_device_ids} "
            f"| dtype: {_DTYPE} | 批次: {self.batch_size}"
        )
        start_time = time.time()

        try:
            # Step1: 分词器
            logger.info("Step1: 加载分词器...")
            t0 = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side="left",
                local_files_only=True,
                use_fast=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self._eos_token_ids = self._build_eos_ids()
            eos_tokens_str = [self.tokenizer.convert_ids_to_tokens(i) for i in self._eos_token_ids]
            logger.info(
                f"分词器加载完成，耗时: {time.time() - t0:.2f}s | "
                f"EOS ids: {self._eos_token_ids} | EOS tokens: {eos_tokens_str}"
            )

            logger.info("Step2: 加载基础模型（CPU）...")
            t0 = time.time()
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=_DTYPE,
                low_cpu_mem_usage=True,
                device_map='cpu',
                use_cache=True,
            )
            logger.info(f"基础模型加载完成，耗时: {time.time() - t0:.2f}s")

            logger.info("Step3: 加载 & 合并 LoRA 权重...")
            t0 = time.time()
            peft_model   = PeftModel.from_pretrained(
                base_model, self.lora_model_path, local_files_only=True
            )
            merged_model = peft_model.merge_and_unload()
            logger.info(f"LoRA 合并完成，耗时: {time.time() - t0:.2f}s")
            del peft_model, base_model
            gc.collect()

            logger.info(f"Step4: 迁移模型到 NPU（设备: {self.npu_device_ids}）...")
            t0 = time.time()

            if not self._multi_npu:
                self.model = merged_model.to(self.device)
                logger.info(f"单卡迁移完成 → {self.device}，耗时: {time.time() - t0:.2f}s")
            else:
                self.model = dispatch_model_to_multi_npu(
                    merged_model, self.npu_device_ids
                )
                logger.info(f"多卡分层完成，耗时: {time.time() - t0:.2f}s")

            del merged_model
            gc.collect()
            self._npu_empty_cache()

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            if not self._multi_npu:
                try:
                    ver = tuple(
                        int(x) for x in torch.__version__.split(".")[:2]
                        if x.isdigit()
                    )
                    if ver >= (2, 0):
                        backend = "npu" if _NPU_AVAILABLE else "inductor"
                        self.model = torch.compile(
                            self.model,
                            mode="reduce-overhead",
                            fullgraph=False,
                            backend=backend,
                        )
                        logger.info(f"torch.compile 已启用（backend={backend}）")
                except Exception as e:
                    logger.warning(f"torch.compile 失败: {e}，跳过")
            else:
                logger.info("多卡模式下跳过 torch.compile")

            logger.info("Step6: 模型预热...")
            t0 = time.time()
            dummy = self._tokenize_batch(['请思考'], max_length=16)
            with torch.no_grad():
                _ = self.model.generate(
                    **dummy,
                    max_new_tokens=16,
                    eos_token_id=self._eos_token_ids,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            del dummy
            gc.collect()
            self._npu_empty_cache()
            logger.info(f"预热完成，耗时: {time.time() - t0:.2f}s")

            logger.info(f"显存状态: {self._npu_memory_info()}")
            logger.info(f"模型加载总耗时: {time.time() - start_time:.2f}s")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"模型加载失败: {e}")


    _SYSTEM_PROMPT = (
        "你是专业的中文档案审核助手。请只输出合法 JSON，不要输出解释性前缀、Markdown 或英文。"
    )

    def _build_user_content(self, archive_item: dict) -> str:
        title = archive_item.get('title', '无题名')
        date  = archive_item.get('date_time', '')[:10]
        content = str(archive_item.get('content') or '').strip()
        metadata_fields = [
            ("门类", archive_item.get("archive_category")),
            ("机构或问题", archive_item.get("organization_problem")),
            ("全宗号", archive_item.get("fonds_no")),
            ("文号", archive_item.get("document_no")),
            ("责任者", archive_item.get("responsible_org")),
            ("原保管期限", archive_item.get("retention_period")),
            ("归档年度", archive_item.get("archive_year")),
        ]
        metadata_context = "\n".join(
            f"{label}：{str(value).strip()}"
            for label, value in metadata_fields
            if str(value or "").strip()
        ) or "（未提供其他档案元数据）"
        if len(content) > _MAX_CONTENT_CHARS:
            head_size = _MAX_CONTENT_CHARS * 2 // 3
            tail_size = _MAX_CONTENT_CHARS - head_size
            content = (
                content[:head_size]
                + "\n【正文过长，中间部分已省略】\n"
                + content[-tail_size:]
            )
        archive_text = content if content else "（未提供档案正文）"
        retrieved_rules = archive_item.get("retrieved_rules") or []
        if retrieved_rules:
            rule_context = "\n\n".join(
                f"[{rule['rule_id']}] 来源：{rule['source']}\n{rule['text']}"
                for rule in retrieved_rules
            )
        else:
            rule_context = "（未检索到候选规范条款）"

        lang_prefix = "【语言要求】所有输出内容必须使用简体中文，禁止出现英文。\n\n"

        if self.model_type == "hk":
            keywords, keyword_result = normalize_keyword_review_signal(
                archive_item,
                self.model_type,
            )
            if keywords:
                keyword_context = (
                    f"【关键字规则预审】命中关键字：{'、'.join(keywords)}\n"
                    f"关键字预审结果：{keyword_result}\n"
                    f"审核结果已经由关键字规则确定为【{keyword_result}】，不得改成其他结果。"
                    "请结合命中关键字、档案正文和候选规范条款，生成审核依据、"
                    "思考过程和置信度，并在JSON的审核结果字段中原样返回该结论。\n\n"
                )
            else:
                keyword_context = (
                    "【关键字规则预审】未提供或未命中关键字，"
                    "请结合档案内容和候选规范条款独立判断开放或控制。\n\n"
                )
            return (
                lang_prefix +
                "你是档案划控专家。对下面的档案进行开放/控制审核。\n\n"
                "【审核标准】依据《连云港市档案馆延期开放档案标准及范围（试用）》\n"
                "- 控制：含机密/秘密/内部/人事/工资/涉密/敏感/个人信息等字样\n"
                "- 开放：公开/通知/公告/总结/报告/批复/规划等常规公文\n\n"
                + keyword_context +
                "【输出要求】只输出一行 JSON，字段顺序为：审核结果、审核依据、置信度、思考过程。\n"
                '{"审核结果":"开放或控制",'
                '"审核依据":"依据《连云港市档案馆延期开放档案标准及范围（试用）》，<具体理由>",'
                '"置信度":<6.0到10.0的数字>,"思考过程":"<判定推理>"}\n\n'
                f"题名为：{title}, 日期为：{date}\n"
                f"【档案元数据】\n{metadata_context}\n"
                f"【档案正文】\n{archive_text}\n"
                f"【RAG检索到的候选规范条款】\n{rule_context}\n"
                "审核依据必须引用真实候选条款编号；不能编造条款。\n"
                "请结合题名、日期和档案正文进行审核。\n输出："
            )
        else:
            return (
                lang_prefix +
                "你是档案鉴定专家。对下面的档案确定保管期限。\n\n"
                "【期限标准】\n"
                "- 永久：重要政策/重大规划/历史性文件\n"
                "- 60年：人事档案/土地产权/不动产相关\n"
                "- 30年：一般行政管理/常规工作文件\n"
                "- 15年：阶段性/临时工作文件\n"
                "- 10年：日常事务/例行事务\n\n"
                "【输出要求】只输出一行 JSON，字段顺序为：审核结果、审核依据、置信度、思考过程。\n"
                '{"审核结果":"10年/15年/30年/60年/永久之一",'
                '"审核依据":"依据档案保管规范，<具体理由>","置信度":<6.0到10.0的数字>,'
                '"思考过程":"<判定推理>"}\n\n'
                f"题名为：{title}, 日期为：{date}\n"
                f"【档案元数据】\n{metadata_context}\n"
                f"【档案正文】\n{archive_text}\n"
                f"【RAG检索到的候选规范条款】\n{rule_context}\n"
                "只能从适用规范允许的保管期限中选择，审核依据必须引用真实候选条款编号。\n"
                "请结合题名、日期和档案正文进行鉴定。\n输出："
            )

    def _apply_chat_template(self, user_content: str) -> str:
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"apply_chat_template 失败: {e}，手动拼接回退")
            merged = f"{self._SYSTEM_PROMPT}\n\n{user_content}"
            return (
                f"<|im_start|>user\n{merged}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    def _generate_prompt(self, archive_item: dict) -> str:
        return self._apply_chat_template(self._build_user_content(archive_item))

    @staticmethod
    def _apply_missing_content_guard(result: dict, archive_item: dict) -> dict:
        """无正文时降低结论强度，防止仅凭题名产生过高置信度。"""
        if (
            str(archive_item.get("content") or "").strip()
            or archive_item.get("keywords")
        ):
            return result

        guarded = dict(result)
        try:
            confidence = float(guarded.get("置信度", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        guarded["置信度"] = round(max(0.0, min(6.0, confidence)), 1)

        limitation = "未提供档案正文，本次仅依据题名、日期及档案元数据辅助判断，建议人工复核。"
        basis = str(guarded.get("审核依据") or "").strip()
        if limitation not in basis:
            guarded["审核依据"] = f"{basis}；{limitation}" if basis else limitation
        thinking = str(guarded.get("思考过程") or "").strip()
        if limitation not in thinking:
            guarded["思考过程"] = f"{thinking} {limitation}".strip()
        return guarded

    def _apply_keyword_decision(self, result: dict, archive_item: dict) -> dict:
        """关键字规则命中时锁定划控结论，模型仅补充可解释字段。"""
        keywords, keyword_result = normalize_keyword_review_signal(
            archive_item,
            self.model_type,
        )
        if not keywords:
            return result

        forced = dict(result)
        model_result = str(forced.get("审核结果") or "").strip()
        if model_result != keyword_result:
            logger.warning(
                f"模型结论与关键字预审不一致，使用关键字结果: "
                f"model={model_result}, keyword={keyword_result}"
            )
        forced["审核结果"] = keyword_result

        signal = f"命中关键字【{'、'.join(keywords)}】，关键字预审结果为【{keyword_result}】"
        basis = str(forced.get("审核依据") or "").strip()
        if signal not in basis:
            forced["审核依据"] = f"{signal}；{basis}" if basis else signal
        thinking = str(forced.get("思考过程") or "").strip()
        if signal not in thinking:
            forced["思考过程"] = f"{signal}。{thinking}" if thinking else signal
        return forced

    def _tokenize_batch(self, texts: List[str], max_length: int = _MAX_INPUT_LEN) -> dict:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding='longest',
            return_attention_mask=True,
        )
        return {k: v.contiguous().to(self.device) for k, v in inputs.items()}


    @staticmethod
    def _split_think_and_body(raw_text: str) -> tuple:
        think_parts = re.findall(r'<think>(.*?)</think>', raw_text, re.DOTALL)
        thinking = ' '.join(
            re.sub(r'\s+', ' ', p).strip()
            for p in think_parts if p.strip()
        ) if think_parts else ""
        body = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        return thinking, body

    def _extract_thinking(self, raw_text: str, fallback_body: str = "") -> str:
        think_parts = re.findall(r'<think>(.*?)</think>', raw_text, re.DOTALL)
        if think_parts:
            t = ' '.join(
                re.sub(r'\s+', ' ', p).strip() for p in think_parts if p.strip()
            )
            if t:
                return t[:1000]
        body = fallback_body or raw_text
        m = re.search(r'["\']?思考过程["\']?\s*[:：]\s*["\']([^"\'}\n]{5,})["\']', body)
        if m:
            return m.group(1).strip()[:1000]
        json_pos = body.find('{')
        if json_pos > 30:
            pre = re.sub(r'\s+', ' ', body[:json_pos]).strip()
            if 15 < len(pre) <= 1000:
                return pre
        return "模型未输出明确的推理过程"

    @staticmethod
    def _clean_text(text: str) -> str:

        for tok in SPECIAL_TOKENS:
            text = text.replace(tok, '')
        text = re.sub(r'\s{3,}', ' ', text)

        # 这里先对截图里高频出现的繁体进行强制纠正
        font_map = {"該": "该", "為": "为", "開": "开", "放": "放", "權": "权", "處": "处", "審": "审", "發": "发"}
        for k, v in font_map.items():
            text = text.replace(k, v)

        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped and not re.search(r'[\u4e00-\u9fff]', stripped):
                ascii_ratio = sum(1 for c in stripped if ord(c) < 128) / len(stripped)
                is_json_line = bool(re.search(r'[{}\[\]"\']\s*[:，,]|[{}\[\]]', stripped))
                if ascii_ratio > 0.8 and not is_json_line:
                    logger.debug(f"丢弃英文行: {stripped[:80]}")
                    continue

            # 这里针对非JSON键名的英文单词进行清洗
            if "{" not in line and "}" not in line:
                # 匹配连续的英文字母单词并替换为空（保留中文）
                line = re.sub(r'\b[a-zA-Z]{2,}\b', '', line)

            cleaned.append(line)
        return '\n'.join(cleaned).strip()

    def _infer_result_from_thinking(self, thinking: str) -> str:
        if not thinking or thinking == "模型未输出明确的推理过程":
            return self.default_result
        if self.model_type == "hk":
            ctrl  = sum(thinking.count(k) for k in self.control_keywords)
            open_ = sum(thinking.count(k) for k in self.open_keywords)
            logger.info(f"思考推断 → 控制:{ctrl} 开放:{open_}")
            return "控制" if ctrl > open_ else "开放"
        else:
            for period, kws in self.period_keywords.items():
                if any(k in thinking for k in kws):
                    logger.info(f"思考推断 → 期限: {period}")
                    return period
            return self.default_result

    def _build_basis_from_thinking(self, thinking: str, result: str) -> str:
        key = thinking[:100].rstrip('，。,.') if thinking else ""
        if key and key != "模型未输出明确的推理过程":
            return f"{self.required_basis}，{key}"
        return f"{self.required_basis}，根据档案标题和内容特征，综合判定为【{result}】"

    def _is_valid_result_value(self, value: str) -> bool:
        value = str(value).strip()
        if self.model_type == "jd":
            return any(v in value or value in v for v in self.valid_results)
        return any(v in value for v in self.valid_results)

    def _fill_missing_fields(self, parsed: dict, thinking: str) -> dict:
        result_val = str(parsed.get("审核结果", "")).strip()
        if len(str(parsed.get("审核依据", "")).strip()) < 5:
            parsed["审核依据"] = self._build_basis_from_thinking(thinking, result_val)
        try:
            float(parsed.get("置信度", ""))
        except (TypeError, ValueError):
            parsed["置信度"] = 7.5
        if not parsed.get("思考过程") or len(str(parsed.get("思考过程", ""))) < 5:
            parsed["思考过程"] = thinking
        return parsed

    def _parse_structured_text(self, text: str, thinking: str) -> Optional[dict]:
        result_m = re.search(
            r'(?:审核结果[：:]|【审核结果】)\s*[\n\r]*\s*["\']?([^\n\r"\'【]{1,20})["\']?', text
        )
        if not result_m:
            return None
        result_val = result_m.group(1).strip().strip('"\'')
        basis_val  = ""
        basis_m = re.search(
            r'(?:审核依据[：:]|【审核依据】)\s*[\n\r]*(.*?)(?=\n[^\n]{0,15}[：:]|\n【|\Z)',
            text, re.DOTALL
        )
        if basis_m:
            basis_val = re.sub(r'\s+', ' ', basis_m.group(1)).strip().rstrip('。.,;；')
        if not basis_val:
            m2 = re.search(r'(?:审核依据[：:]|【审核依据】)\s*(.{5,200})', text)
            if m2:
                basis_val = m2.group(1).strip()
        conf_val = 7.5
        cm = re.search(r'置信度[：:]\s*([0-9]{1,2}(?:\.[0-9])?)', text)
        if cm:
            try:
                conf_val = float(cm.group(1))
            except Exception:
                pass
        parsed = {"思考过程": thinking, "审核结果": result_val,
                  "审核依据": basis_val or "", "置信度": conf_val}
        parsed = self._fill_missing_fields(parsed, thinking)
        if self._validate_structure(parsed):
            logger.info("结构化文本解析成功（策略4）")
            return parsed
        return None

    def _extract_json(self, body: str, thinking: str) -> Optional[dict]:

        def _try_parse(s: str) -> Optional[dict]:
            for c in [s, s.replace("'", '"'),
                      s.replace("：", ":").replace("，", ","),
                      re.sub(r'(?<=[{,])\s*([^\s\'"{}:,]+)\s*(?=:)', r'"\1"', s)]:
                try:
                    return json.loads(c)
                except Exception:
                    pass
            return None

        def _valid(d) -> bool:
            return isinstance(d, dict) and self._is_valid_result_value(
                str(d.get("审核结果", ""))
            )

        for block in re.findall(r'```(?:json)?\s*(.*?)\s*```', body, re.DOTALL):
            p = _try_parse(block.strip())
            if p and _valid(p):
                p = self._fill_missing_fields(p, thinking)
                if self._validate_structure(p):
                    logger.info("markdown代码块解析成功（策略0）")
                    return p

        body = re.sub(r'```(?:json)?\s*|\s*```', '', body)

        for pat in [
            r'\{[^{}]*?["\']?思考过程["\']?\s*[:：][^{}]*?["\']?审核结果["\']?\s*[:：][^{}]*?["\']?审核依据["\']?\s*[:：][^{}]*?["\']?置信度["\']?\s*[:：][^{}]*?\}',
            r'\{[^{}]*?["\']?审核结果["\']?\s*[:：][^{}]*?["\']?审核依据["\']?\s*[:：][^{}]*?["\']?置信度["\']?\s*[:：][^{}]*?["\']?思考过程["\']?\s*[:：][^{}]*?\}',
        ]:
            for m in re.finditer(pat, body, re.DOTALL):
                p = _try_parse(m.group())
                if p and _valid(p):
                    p = self._fill_missing_fields(p, thinking)
                    if self._validate_structure(p):
                        logger.info("4字段JSON解析成功（策略1）")
                        return p

        for pat in [
            r'\{[^{}]*?["\']?审核结果["\']?\s*[:：][^{}]*?["\']?审核依据["\']?\s*[:：][^{}]*?["\']?置信度["\']?\s*[:：][^{}]*?\}',
            r'\{[^{}]{15,800}?\}',
        ]:
            for m in re.finditer(pat, body, re.DOTALL):
                p = _try_parse(m.group())
                if p and _valid(p):
                    p = self._fill_missing_fields(p, thinking)
                    if self._validate_structure(p):
                        logger.info("宽松JSON解析成功（策略2）")
                        return p

        try:
            rm = re.search(r'["\']?审核结果["\']?\s*[:：]\s*["\']?([^"\'}\n,]{1,20})["\']?', body)
            bm = re.search(r'["\']?审核依据["\']?\s*[:：]\s*["\']([^"\'}\n]{5,200})["\']', body)
            cm = re.search(r'["\']?置信度["\']?\s*[:：]\s*([0-9]{1,2}(?:\.[0-9])?)', body)
            if rm:
                p = {
                    "思考过程": thinking,
                    "审核结果": rm.group(1).strip().strip('"\''),
                    "审核依据": bm.group(1).strip() if bm else "",
                    "置信度":   float(cm.group(1)) if cm else 8.0,
                }
                p = self._fill_missing_fields(p, thinking)
                if self._validate_structure(p):
                    logger.info("逐字段提取成功（策略3）")
                    return p
        except Exception as e:
            logger.debug(f"策略3异常: {e}")

        if re.search(r'审核结果[：:]|【审核结果】', body):
            p = self._parse_structured_text(body, thinking)
            if p:
                return p

        return None

    def _validate_structure(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        for f in ["审核结果", "审核依据"]:
            if f not in data:
                return False
        if not self._is_valid_result_value(str(data.get("审核结果", ""))):
            return False
        if len(str(data.get("审核依据", "")).strip()) < 5:
            return False
        try:
            c = float(data.get("置信度", 8.0))
            if not (0 <= c <= 10):
                return False
        except Exception:
            pass
        return True

    def _postprocess(self, raw_text: str) -> dict:
        logger.debug(f"模型原始输出: {raw_text}")

        thinking, body = self._split_think_and_body(raw_text)
        if not thinking:
            thinking = self._extract_thinking(raw_text, fallback_body=body)
        logger.info(
            f"思考过程({len(thinking)}字): "
            f"{thinking[:120]}{'...' if len(thinking) > 120 else ''}"
        )

        body_clean = self._clean_text(body)
        parsed = self._extract_json(body_clean, thinking)
        if parsed is None:
            logger.warning("JSON提取失败，启用关键词推断兜底")
            inferred = self._infer_result_from_thinking(thinking)
            parsed = {
                "思考过程": thinking,
                "审核结果": inferred,
                "审核依据": self._build_basis_from_thinking(thinking, inferred),
                "置信度": 7.0,
            }
        else:
            logger.info("JSON提取成功")

        bad_patterns = ["<与think内容", "推理摘要", "模型未输出明确", "在此处结合"]
        current_thinking = parsed.get("思考过程", "").strip()

        if not current_thinking or any(p in current_thinking for p in bad_patterns):
            if thinking and not any(p in thinking for p in bad_patterns):
                parsed["思考过程"] = thinking
            else:
                parsed["思考过程"] = "已对档案标题及基本信息完成合规性要素审查。"
        else:
            thinking = parsed["思考过程"]

        logger.info(
            f"思考过程({len(parsed['思考过程'])}字): "
            f"{parsed['思考过程'][:120]}..."
        )

        result_raw = str(parsed.get("审核结果", "")).strip()
        matched    = self.default_result
        for v in self.valid_results:
            if self.model_type == "jd":
                if v == result_raw or v in result_raw or result_raw in v:
                    matched = v
                    break
            else:
                if v in result_raw:
                    matched = v
                    break
        parsed["审核结果"] = matched

        basis = str(parsed.get("审核依据", "")).strip().rstrip('。.,;；')
        if len(basis) < 10 or self.required_basis not in basis:
            basis = (
                f"{self.required_basis}，{basis}"
                if len(basis) >= 5
                else f"{self.required_basis}，根据档案内容综合判定为【{matched}】"
            )
        parsed["审核依据"] = basis

        try:
            conf = round(max(6.0, min(10.0, float(parsed.get("置信度", 8.0)))), 1)
        except Exception:
            conf = 8.0
        parsed["置信度"] = conf

        logger.info(f"最终结果: {parsed['审核结果']} | 置信度: {parsed['置信度']}")
        return parsed


    def batch_review(self, archive_data: list) -> list:
        if not archive_data:
            raise ValueError("archive_data 为空")
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model()")

        logger.info(f"开始批量审核 | 总数据: {len(archive_data)} | 批次: {self.batch_size}")
        total_start   = time.time()
        all_results   = []
        total_batches = (len(archive_data) + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for batch_idx in range(total_batches):
                batch_start = time.time()
                start_idx   = batch_idx * self.batch_size
                end_idx     = min(start_idx + self.batch_size, len(archive_data))
                batch       = archive_data[start_idx:end_idx]
                logger.info(f"\n批次 {batch_idx + 1}/{total_batches} | 处理 {len(batch)} 条")

                # 训练集只用于LoRA训练与追溯。审核时所有档案（包括无正文档案）
                # 都进入RAG + LoRA推理，禁止按题名和日期直接复制训练答案。
                prompts = [self._generate_prompt(item) for item in batch]
                inputs = self._tokenize_batch(prompts)

                actual_len = inputs['input_ids'].shape[1]
                logger.info(f"实际输入token数: {actual_len} / {_MAX_INPUT_LEN}")
                if actual_len >= _MAX_INPUT_LEN:
                    logger.warning("输入已达上限，可能截断！")

                logger.info("NPU推理中...")
                gen_start = time.time()
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self._eos_token_ids,
                    use_cache=True,
                    num_beams=1,
                    repetition_penalty=1.05,
                )
                logger.info(f"推理完成，耗时: {time.time() - gen_start:.2f}s")

                input_len = inputs['input_ids'].shape[1]
                outputs_new_cpu = outputs[:, input_len:].cpu()
                generated_texts = self.tokenizer.batch_decode(
                    outputs_new_cpu,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                batch_results = [self._postprocess(raw_text) for raw_text in generated_texts]

                del inputs, outputs, outputs_new_cpu, generated_texts
                gc.collect()
                self._npu_empty_cache()

                for item, result in zip(batch, batch_results):
                    result = self.rule_engine.validate(
                        result,
                        item.get("retrieved_rules") or [],
                    )
                    # 规则引擎和模型都不能覆盖客户端关键字预审给出的确定结论。
                    result = self._apply_keyword_decision(result, item)
                    result = self._apply_missing_content_guard(result, item)
                    logger.info(f"\n--- 档案 ID: {item.get('arid', 'N/A')} ---")
                    if not isinstance(result['思考过程'], str):
                        thinking = ' '.join(result['思考过程'])
                    else:
                        thinking = result['思考过程']

                    all_results.append({
                        "arid":     item.get("arid", ""),
                        "jg":       result["审核结果"],
                        "yj":       result["审核依据"],
                        "zxd":      result["置信度"],
                        "thinking": thinking,
                    })

                logger.info(f"批次耗时: {time.time() - batch_start:.2f}s")

        total_time = time.time() - total_start
        avg_time   = total_time / len(all_results) if all_results else 0
        logger.info("\n" + "=" * 60)
        logger.info(f"审核完成 | 总耗时: {total_time:.2f}s | 单条均值: {avg_time:.2f}s")
        logger.info(f"显存状态: {self._npu_memory_info()}")
        logger.info("=" * 60)
        return all_results

    def cleanup(self):
        if self.model is not None:
            hooks = getattr(self.model, "_npu_pipeline_hooks", [])
            for hook in hooks:
                try:
                    hook.remove()
                except Exception:
                    pass
        self.model = None
        self.tokenizer = None
        gc.collect()
        self._npu_empty_cache()
        logger.info("ArchiveReviewer资源已释放")


# ── 入口测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_MODEL_PATH    = "/root/autodl-tmp/ai_analyse/base_model/DeepSeek-R1-Distill-Qwen-7B"
    LORA_MODEL_PATH_HK = "/root/autodl-tmp/ai_analyse/models/deepseek-r1-HK"
    LORA_MODEL_PATH_JD = "/root/autodl-tmp/ai_analyse/models/deepseek-r1-JD"

    test_data_hk = [
        {"arid": "hk_1", "title": "连云港市人民政府关于2024年度工作总结的报告",   "date_time": "2024-12-31"},
        {"arid": "hk_2", "title": "市政府关于批准灌云县土地征收成片开发方案的批复", "date_time": "2024-01-15"},
        {"arid": "hk_3", "title": "关于干部任免的通知（内部）",                     "date_time": "2024-06-20"},
    ]

    test_data_jd = [
        {"arid": "jd_1", "title": "连云港市人民政府关于2025年度工作总结的报告",        "date_time": "2025-12-31"},
        {"arid": "jd_2", "title": "市政府关于同意连云港内河港总体规划（2035年）的批复", "date_time": "2025-03-21"},
        {"arid": "jd_3", "title": "市政府关于批准灌云县土地征收成片开发方案的批复",     "date_time": "2024-05-06"},
    ]

    def _run_test(model_type: str, lora_path: str, data: list,
                  npu_ids: Optional[List[int]] = None):
        tag = "档案划控 (HK)" if model_type == "hk" else "档案鉴定 (JD)"
        print(f"\n{'=' * 80}\n测试：{tag} | NPU设备: {npu_ids or [0]}\n{'=' * 80}")
        reviewer = ArchiveReviewer(
            BASE_MODEL_PATH, lora_path,
            batch_size=2,
            model_type=model_type,
            npu_device_ids=npu_ids,
        )
        try:
            reviewer.load_model()
            results = reviewer.batch_review(data)
            print(f"\n{'档案划控' if model_type == 'hk' else '档案鉴定'}结果：")
            print("-" * 80)
            for i, res in enumerate(results, 1):
                print(f"\n{i}. 档案ID:  {res['arid']}")
                print(f"   审核结果: {res['jg']}")
                print(f"   审核依据: {res['yj']}")
                print(f"   置信度:   {res['zxd']}")
                print(f"   思考过程: {res['thinking'][:200]}")
                print("-" * 80)
        except Exception as e:
            import traceback
            print(f"运行失败: {e}")
            traceback.print_exc()

    # 单卡 NPU 0
    _run_test("hk", LORA_MODEL_PATH_HK, test_data_hk, npu_ids=[0])

    # 多卡（4卡，手动分层，无需 accelerate）
    # _run_test("hk", LORA_MODEL_PATH_HK, test_data_hk, npu_ids=[0, 1, 2, 3])

    # 鉴定任务
    # _run_test("jd", LORA_MODEL_PATH_JD, test_data_jd, npu_ids=[0])
