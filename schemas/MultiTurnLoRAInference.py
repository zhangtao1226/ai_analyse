# -*-coding : utf-8 -*-
# @Author   : zhangtao
# @FileName : MultiTurnLoRAInference.py
# @Desc     : 
# @Time     : 2025/11/14 16:24
# @Software : PyCharm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
from typing import List, Dict, Optional


class MultiTurnLoRAInference:
    """支持多轮对话的 LoRA 微调模型推理类"""

    def __init__(self, base_model_name: str, lora_model_path: str, use_quantization: bool = True):
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        self.conversation_history = []

    def load_model(self):
        """加载模型和分词器"""

        print("开始加载多轮对话模型...")

        # 基础配置
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.use_quantization and torch.cuda.is_available():
            # 使用量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            # 不使用量化
            model_kwargs["torch_dtype"] = torch.float16
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = None

        try:
            # 加载基础模型
            print("加载基础模型中...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                **model_kwargs
            )

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )

            # 确保分词器设置正确
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载 LoRA 适配器
            print("加载 LoRA 适配器中...")
            if self.use_quantization:
                self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)
            else:
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.lora_model_path,
                    torch_dtype=torch.float16
                )

            # 确保模型在评估模式
            self.model.eval()

            print("✅ 多轮对话模型加载成功!")

            # 打印设备信息
            if hasattr(self.model, 'hf_device_map'):
                print(f"设备映射: {self.model.hf_device_map}")
            else:
                device = next(self.model.parameters()).device
                print(f"模型设备: {device}")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def _build_multiturn_prompt(self, instruction: str, input_text: str = "", history: List[Dict] = None) -> str:
        """构建多轮对话提示"""

        # 如果有对话历史，先构建历史部分
        history_text = ""
        if history:
            for turn in history:
                user_msg = turn.get("user", "")
                assistant_msg = turn.get("assistant", "")

                if user_msg:
                    history_text += f"### User:\n{user_msg}\n\n"
                if assistant_msg:
                    history_text += f"### Assistant:\n{assistant_msg}\n\n"

        # 构建当前轮次的提示
        if input_text:
            current_turn = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            current_turn = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # 组合历史和新提示
        full_prompt = history_text + current_turn

        return full_prompt

    def chat(self,
             instruction: str,
             input_text: str = "",
             max_new_tokens: int = 512,
             temperature: float = 0.7,
             do_sample: bool = True,
             keep_history: bool = True) -> str:
        """多轮对话接口"""

        if self.model is None or self.tokenizer is None:
            raise ValueError("请先加载模型")

        # 构建多轮对话提示
        prompt = self._build_multiturn_prompt(instruction, input_text, self.conversation_history)

        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,  # 增加最大长度以容纳历史
                padding=False
            )

            # 手动将输入移动到模型所在设备
            if hasattr(self.model, 'hf_device_map'):
                device = next(iter(self.model.hf_device_map.values()))
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # 生成参数
            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if do_sample:
                generate_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "repetition_penalty": 1.1,
                })
            else:
                generate_kwargs["do_sample"] = False

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(**generate_kwargs)

            # 解码
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # 如果选择保持历史，将当前对话添加到历史中
            if keep_history:
                self.conversation_history.append({
                    "user": f"{instruction}\n\n{input_text}".strip() if input_text else instruction,
                    "assistant": response
                })

                # 限制历史长度，避免超过模型最大长度
                if len(self.conversation_history) > 10:  # 最多保留10轮对话
                    self.conversation_history = self.conversation_history[-10:]

            return response

        except Exception as e:
            print(f"生成失败: {e}")
            return f"生成错误: {str(e)}"

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("对话历史已清空")

    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.conversation_history

    def save_conversation(self, file_path: str):
        """保存对话历史到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"对话历史已保存到: {file_path}")
        except Exception as e:
            print(f"保存对话历史失败: {e}")

    def load_conversation(self, file_path: str):
        """从文件加载对话历史"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"对话历史已从 {file_path} 加载")
        except Exception as e:
            print(f"加载对话历史失败: {e}")

    def interactive_chat(self):
        """交互式多轮对话"""
        if self.model is None:
            print("请先加载模型!")
            return

        print("\n" + "=" * 60)
        print("多轮对话模式已启动")
        print("输入 'quit' 退出，'clear' 清空历史，'history' 查看历史")
        print("=" * 60)

        while True:
            try:
                # 获取用户输入
                instruction = input("\n💬 请输入指令: ").strip()

                if instruction.lower() == 'quit':
                    break
                elif instruction.lower() == 'clear':
                    self.clear_history()
                    continue
                elif instruction.lower() == 'history':
                    print("\n📜 对话历史:")
                    for i, turn in enumerate(self.conversation_history, 1):
                        print(f"第{i}轮:")
                        print(f"  用户: {turn['user']}")
                        print(f"  助手: {turn['assistant']}")
                    continue

                input_text = input("📝 请输入补充信息 (可选，直接回车跳过): ").strip()

                # 生成回复
                print("🤔 思考中...")
                response = self.chat(
                    instruction=instruction,
                    input_text=input_text if input_text else "",
                    max_new_tokens=600,
                    temperature=0.3,
                    keep_history=True
                )

                print(f"\n🤖 助手回复:\n{response}")

            except KeyboardInterrupt:
                print("\n\n对话结束")
                break
            except Exception as e:
                print(f"对话过程中出错: {e}")