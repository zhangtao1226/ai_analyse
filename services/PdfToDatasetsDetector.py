# -*-coding  : utf-8 -*-
# @Author    : zhangtao
# @File      : PdfToDatasetsDetector.py
# @Desc      : 将PDF文档转为训练模型的数据集(JSONL格式)
# @Time      : 2025/11/4 16:26
# @Software  : PyCharm
import json
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.LoggerDetector import logger

class PdfToDatasetsDetector:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.chunk_size = 1000
        self.chunk_overlap = 100

    def extract_pdf_content(self) -> str:
        """
        提取PDF内容
        :return:
        """
        logger.info(f"准备提取PDF文件内容: {self.input_path}")
        content = ""
        try:
            with pdfplumber.open(self.input_path) as pdf:
                for page in pdf.pages:
                    # 提取文本
                    text = page.extract_text()
                    if text:
                        content += text + "\n"

                    # 提取表格
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                content += " ".join([str(cell) if cell else "" for cell in row]) + "\n"
        except Exception as e:
            logger.error(f"PDF提取错误: {str(e)}")

        return content

    def split_text(self, text):
        """
        将长文本分块（适配模型上下文窗口）
        :param text:
        :return:
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,  # 每块最大字符数（根据模型窗口调整，如DeepSeek-R1通常支持4k/8k tokens）
            chunk_overlap=self.chunk_overlap,  # 块之间的重叠字符数（保留上下文关联）
            separators=["\n\n", "\n", "。", "，", "？", "《", "》", "（", "）", "“", "；", "：", "‘", "！"]  # 按自然分隔符拆分
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def format_for_finetuning(self, chunks):
        """
        将分块整理为微调格式（指令-输入-输出三元组）
        :param chunks:
        :return:
        """
        instruction = "请学习以下内容并掌握相关知识:"
        formatted_data = []
        for chunk in chunks:
            # 过滤过短的块（避免无效数据）
            if len(chunk.strip()) < 100:
                continue
            formatted_data.append({
                "instruction": instruction,     # 统一指令
                "input": chunk.strip(),         # 分块后的文本作为输入
                "output": chunk.strip()         # 输出(暂定)
            })
        return formatted_data

    def save_to_jsonl(self, data):
        """
        保存为JSONL文件（每行一个JSON对象）
        :param data:
        :return:
        """
        with open(self.output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def detector(self):
        # 1. 提取PDF文本
        pdf_text = self.extract_pdf_content()
        logger.info(f"提取到文本长度：{len(pdf_text)}字符")

        # 2. 分块（根据模型上下文窗口调整chunk_size，1 token≈1.5-2字符）
        chunks = self.split_text(pdf_text)  # 适配8k token窗口
        logger.info(f"分块后共{len(chunks)}个片段")

        # 3. 格式化为微调数据
        finetune_data = self.format_for_finetuning(chunks)

        # 4. 保存为JSONL
        self.save_to_jsonl(finetune_data)
        logger.info(f"已保存微调数据至：{self.output_path}")
        return self.output_path

if __name__ == '__main__':
    input_path = r"C:\Users\Administrator\Desktop\测试数据\规范（内部）.pdf"
    output_path = r"D:\ZTprojects\ai_analyse\core\datasets\hk_rules.jsonl"
    pdf_detector = PdfToDatasetsDetector(input_path, output_path)
    pdf_detector.detector()
