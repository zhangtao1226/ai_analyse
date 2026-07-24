# -*-coding : utf-8 -*-
# @Author   : zhangtao
# @FileName : OcrDetector.py
# @Desc     : 
# @Time     : 2025/11/19 09:27
# @Software : PyCharm


class OcrDetector:
    def __init__(self):
        pass

    def detect(self, image):
        pipeline = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_seal_recognition=False,
            use_table_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            # 布局检测

            layout_detection_model_name="PP-DocLayout_plus-L",
            layout_detection_model_dir=r"D:\ZTprojects\ocrApi\models\PP-DocLayout_plus-L_infer",
            chart_recognition_model_dir=r"D:\ZTprojects\ocrApi\models\PP-Chart2Table",
            text_detection_model_dir=r"D:\ZTprojects\ocrApi\models\PP-OCRv5_server_det_infer",
            text_recognition_model_dir=r"D:\ZTprojects\ocrApi\models\PP-OCRv5_server_rec_infer",

            device="cpu",
        )

        return pipeline.predict(image)