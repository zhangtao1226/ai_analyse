# -*-coding  : utf-8 -*-
# @Author    : zhangtao
# @File      : PaddleOCRServices.py
# @Desc      :
# @Time      : 2026/1/6 10:57
# @Software  : PyCharm

import os
import warnings

os.environ.setdefault('OMP_NUM_THREADS', os.getenv('OCR_CPU_THREADS', '4'))
os.environ.setdefault('MKL_NUM_THREADS', os.getenv('OCR_CPU_THREADS', '4'))
os.environ.setdefault('PADDLEOCR_DOWNLOAD_MODEL', 'False')
os.environ.setdefault('DISABLE_REMOTE_MODEL_SYNC', 'True')
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')


try:
    from paddleocr.utils.download import download_model
    def no_download(*args, **kwargs):
        return None
    download_model.__code__ = no_download.__code__
except ImportError:
    pass

from paddleocr import PaddleOCR

from core.LoggerDetector import logger

warnings.filterwarnings('ignore')

class PaddleOCRServices:
    def __init__(self, det_path=None, rec_path=None, device="npu:0", fallback_device="cpu"):
        self.det_path = det_path or "/home/erren/ai_analyse/ocrModel/ch_PP-OCRv4_det_infer"
        self.rec_path = rec_path or "/home/erren/ai_analyse/ocrModel/ch_PP-OCRv4_rec_infer"
        self.preferred_device = str(device or "npu:0").strip()
        self.fallback_device = str(fallback_device or "cpu").strip()
        self.active_device = None
        self.ocr = None
        self._initialize_with_fallback()

    def _create_ocr(self, device):
        return PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_angle_cls=False,
            lang='ch',
            det_model_dir=self.det_path,
            rec_model_dir=self.rec_path,
            # ========== 新增：彻底禁用cls模型 ==========
            cls_model_name=None,  # 关键！清空cls模型名，不匹配任何下载地址
            # 核心：禁用模型校验和下载的关键参数
            check_version=False,
            download_param=False,
            det_model_name=None,
            rec_model_name=None,
            cls_model_dir=None,
            device=device,
            show_log=False,
            # NPU不使用MKLDNN；CPU侧保持兼容优先，可在验证后单独开启高性能推理。
            enable_mkldnn=False
        )

    def _activate_device(self, device):
        self.ocr = self._create_ocr(device)
        self.active_device = device
        logger.info(
            f"PaddleOCR初始化完成，preferred={self.preferred_device}, active={self.active_device}"
        )

    def _initialize_with_fallback(self):
        candidates = [self.preferred_device]
        if self.fallback_device and self.fallback_device not in candidates:
            candidates.append(self.fallback_device)

        errors = []
        for index, candidate in enumerate(candidates):
            try:
                self._activate_device(candidate)
                return
            except Exception as error:
                errors.append(f"{candidate}: {error}")
                if index == 0 and len(candidates) > 1:
                    logger.warning(
                        f"PaddleOCR首选设备初始化失败，切换到{self.fallback_device}: {error}"
                    )
                else:
                    logger.error(f"PaddleOCR设备初始化失败，device={candidate}: {error}")
        raise RuntimeError(f"PaddleOCR所有设备初始化失败: {'; '.join(errors)}")

    def _switch_to_cpu_after_inference_error(self, error):
        if not self.fallback_device or self.active_device == self.fallback_device:
            return False
        logger.warning(
            f"PaddleOCR在{self.active_device}推理失败，切换到{self.fallback_device}并重试: {error}"
        )
        try:
            self._activate_device(self.fallback_device)
            return True
        except Exception as fallback_error:
            logger.error(f"PaddleOCR回退设备初始化失败，device={self.fallback_device}: {fallback_error}")
            return False

    def _predict_once(self, image_path):
        result = self.ocr.ocr(image_path, cls=False)
        ocr_text = []
        if result and result[0]:
            for line in result[0]:
                ocr_text.append(line[1][0])
        return ocr_text

    def ocr_predict(self, image_path):
        try:
            ocr_text = self._predict_once(image_path)
        except Exception as error:
            if not self._switch_to_cpu_after_inference_error(error):
                logger.error(f"OCR识别失败，device={self.active_device}: {error}")
                return []
            try:
                ocr_text = self._predict_once(image_path)
            except Exception as fallback_error:
                logger.error(
                    f"OCR回退后识别仍失败，device={self.active_device}: {fallback_error}"
                )
                return []

        logger.info(f"OCR识别成功，device={self.active_device}: {ocr_text}")
        return ocr_text

if __name__ == '__main__':
    ocr_detector = PaddleOCRServices()
    image_path = r"/home/erren/ai_analyse/test/images/1.jpg"
    if not os.path.exists(image_path):
        print(f"错误：图片路径不存在 {image_path}")
    else:
        text = ocr_detector.ocr_predict(image_path)
        print("\n=== 最终识别结果 ===")
        print(text)
