# -*-coding : utf-8 -*-
# @Author   : zhangtao
# @FileName : ArchiveExpertMultiTurn.py
# @Desc     : 
# @Time     : 2025/11/14 16:21
# @Software : PyCharm
import json

from schemas.MultiTurnLoRAInference import MultiTurnLoRAInference

class ArchiveExpertMultiTurn:
    """档案专家多轮对话系统"""
    def __init__(self, base_model_name: str, lora_model_path: str):
        self.inference = MultiTurnLoRAInference(base_model_name, lora_model_path)
        self.current_archive_info = None

    def load_model(self):
        """加载模型"""
        self.inference.load_model()

    def analyze_archive(self, title: str, date_time: str, date_now: str):
        """执行档案分析的两轮对话"""

        # 保存当前档案信息
        self.current_archive_info = {
            "title": title,
            "date_time": date_time,
            "date_now": date_now
        }

        print("=" * 60)
        print("📋 档案鉴定分析开始")
        print("=" * 60)
        print(f"档案题名: {title}")
        print(f"档案日期: {date_time}")
        print(f"当前日期: {date_now}")
        print("-" * 60)

        # 第一轮：任务理解和初步分析
        print("\n🔄 第一轮：专业分析框架构建")
        print("-" * 40)

        first_prompt = f"""您是一名专业的档案管理专家，专门从事档案鉴定与分析工作。

                        ## 核心任务说明
                        请基于您掌握的三合一制度知识，执行以下专业分析：
                        
                        1. **深度理解档案规范**：系统解析档案体系结构、归档范围和保管期限标准
                        2. **精准匹配题名内容**：将档案题名与三合一制度中的相关规定进行精确匹配
                        3. **专业分析框架**：运用专业判断标准进行档案状态评估
                        
                        ## 档案信息
                        - 档案题名：{title}
                        - 档案形成日期：{date_time}
                        - 当前参考日期：{date_now}
                        
                        请先进行初步分析，说明您将如何开展这项档案鉴定工作，包括：
                        - 您将参考哪些具体的三合一制度条款
                        - 您判断档案状态的主要依据和标准  
                        - 您对这份档案的初步专业判断方向
                        
                        请用专业的档案管理术语进行回复，体现您的专业分析思路。
                    """

        first_response = self.inference.chat(
            instruction=first_prompt,
            max_new_tokens=800,
            temperature=0.3,
            keep_history=True
        )

        print(f"💡 专家分析思路:\n{first_response}")

        # 第二轮：具体分析和格式化输出
        print("\n🔄 第二轮：标准化鉴定结果输出")
        print("-" * 40)

        second_prompt = """基于上一轮的分析框架，现在请您完成具体的档案鉴定工作并输出标准化结果。

                        ## 专业分析要求
                        请严格按照以下专业标准执行：
                        
                        1. **过期状态判定公式**：
                           (当前年份 - 档案形成年份) ≥ 规定的保管期限 → 判定为过期
                        
                        2. **审核结果判定标准**：
                           - "开放"：档案内容符合公开条件，无保密限制
                           - "控制"：档案涉及敏感信息，需限制访问
                        
                        3. **置信度评估维度**：
                           - 题名与制度条款的匹配精确度
                           - 保管期限规定的明确程度  
                           - 档案内容的典型性判断
                        
                        ## 标准化输出格式
                        请严格按照以下JSON格式输出专业鉴定结果, 不要输出思考过程：
                        
                        {
                            "审核结果": "开放/控制",
                            "审核依据": "具体引用的三合一制度条款内容",
                            "置信度": 0.0-10.0的浮点数值
                        }
                        
                        要求：
                        - 审核依据必须具体引用三合一制度的相关条款
                        - 置信度需基于专业判断给出精确数值
                        - 确保JSON格式完全正确，可直接解析
                        - 使用简体中文表述
                    """

        second_response = self.inference.chat(
            instruction=second_prompt,
            max_new_tokens=600,
            temperature=0.1,  # 降低温度以获得更确定的输出
            keep_history=True
        )

        print(f"📊 标准化鉴定结果:\n{second_response}")

        # 尝试解析JSON结果
        try:
            # 从响应中提取JSON部分
            json_start = second_response.find('{')
            json_end = second_response.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = second_response[json_start:json_end]
                result = json.loads(json_str)
                print(f"\n✅ JSON解析成功:")
                print(f"   审核结果: {result.get('审核结果', 'N/A')}")
                print(f"   审核依据: {result.get('审核依据', 'N/A')}")
                print(f"   置信度: {result.get('置信度', 'N/A')}")
                return result
        except Exception as e:
            print(f"⚠️ JSON解析失败: {e}")
            print("请手动检查输出格式")

        return second_response

    def batch_analyze_archives(self, archive_list: list[dict[str, str]]):
        """批量分析多个档案"""
        results = []

        for i, archive in enumerate(archive_list, 1):
            print(f"\n📁 正在分析第 {i}/{len(archive_list)} 个档案...")

            # 清空历史，确保每个档案独立分析
            self.inference.clear_history()

            result = self.analyze_archive(
                title=archive.get("title", ""),
                date_time=archive.get("date_time", ""),
                date_now=archive.get("date_now", "")
            )

            results.append({
                "archive_info": archive,
                "analysis_result": result
            })

            # 添加间隔
            if i < len(archive_list):
                print("\n" + "=" * 60)

        return results

    def interactive_analysis(self):
        """交互式档案分析"""
        print("\n" + "=" * 60)
        print("🗂️ 交互式档案鉴定分析系统")
        print("=" * 60)

        while True:
            try:
                print("\n请提供档案信息：")
                title = input("📝 档案题名: ").strip()
                if title.lower() == 'quit':
                    break

                date_time = input("📅 档案日期 (格式: YYYYMMDD): ").strip()
                date_now = input("⏰ 当前日期 (格式: YYYYMMDD): ").strip()

                if not all([title, date_time, date_now]):
                    print("❌ 请完整填写所有档案信息")
                    continue

                # 执行分析
                self.analyze_archive(title, date_time, date_now)

                # 询问是否继续
                continue_analysis = input("\n是否继续分析其他档案? (y/n): ").strip().lower()
                if continue_analysis != 'y':
                    break

            except KeyboardInterrupt:
                print("\n\n分析结束")
                break
            except Exception as e:
                print(f"分析过程中出错: {e}")


if __name__ == "__main__":
    # 配置模型路径
    base_model_name = r"D:\ZTprojects\ai_analyse\base_model\DeepSeek-R1-Distill-Qwen-14B"
    lora_model_path = r"D:\ZTprojects\ai_analyse\test\models\continued_training"

    # 创建档案专家实例
    expert = ArchiveExpertMultiTurn(base_model_name, lora_model_path)

    # 加载模型
    print("加载专业档案分析模型中...")
    expert.load_model()

    # 示例档案数据
    sample_archives = [
        {
            "title": "关于印发《玄武区社区专职工作者管理办法》的通知",
            "date_time": "20130415",
            "date_now": "20231103"
        },
        {
            "title": "栖霞区住宅楼顶公用部位管理办法实施细则",
            "date_time": "20190415",
            "date_now": "20231103"
        },
        {
            "title": "玄武区残疾居民社会保险补贴暂行办法",
            "date_time": "20200415",
            "date_now": "20231103"
        }
    ]

    # 批量分析
    print("开始批量档案分析...")
    results = expert.batch_analyze_archives(sample_archives)

    # 显示汇总结果
    print("\n" + "=" * 60)
    print("📈 档案分析汇总报告")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        archive = result["archive_info"]
        analysis = result["analysis_result"]

        print(f"\n档案 {i}: {archive['title']}")
        print(f"形成日期: {archive['date_time']}")

        if isinstance(analysis, dict):
            print(f"审核结果: {analysis.get('审核结果', 'N/A')}")
            print(f"置信度: {analysis.get('置信度', 'N/A')}")
            print(f"审核依据: {analysis.get('审核依据', 'N/A')[:100]}...")
        else:
            print(f"分析结果: {analysis}")

        print("-" * 40)