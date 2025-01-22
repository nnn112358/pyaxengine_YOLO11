import onnx
import os

def extract_yolo_backbone_detector(input_path, output_path):
    """YOLOモデルの特徴量抽出部分（バックボーン）を抽出する関数"""
    input_names = ["images"]
    output_names = [
        "/model.23/Concat_output_0",
        "/model.23/Concat_1_output_0", 
        "/model.23/Concat_2_output_0"
    ]
    onnx.utils.extract_model(input_path, output_path, input_names, output_names)

def extract_yolo_postprocessor(input_path, output_path):
    """YOLOモデルの後処理部分（検出結果の出力）を抽出する関数"""
    input_names = [       
        "/model.23/Concat_output_0",
        "/model.23/Concat_1_output_0", 
        "/model.23/Concat_2_output_0"
    ]
    output_names = ["output0"]
    onnx.utils.extract_model(input_path, output_path, input_names, output_names)

# Usage
extract_yolo_backbone_detector("yolo11n_320x320.onnx", "yolo11n_320x320_base.onnx")
extract_yolo_backbone_detector("yolo11n_640x640.onnx", "yolo11n_640x640_base.onnx")


