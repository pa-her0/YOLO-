import sys
import pathlib

# 指定 ultralytics-main 的路径
ultralytics_main_path = pathlib.Path("~/local/AICUP2024/yolov8/ALL/prune/finetune/ultralytics-main").expanduser()
sys.path.insert(0, str(ultralytics_main_path))

import argparse
from ultralytics import YOLO
import os
import shutil

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Finetuning")
    parser.add_argument('--model', type=str, required=True, help='Path to pruned model')
    parser.add_argument('--output', type=str, required=True, help='Path to save finetuned model')
    args = parser.parse_args()

    # 使用本地 ultralytics-main 的 yolov8.yaml
    yaml_path = str(ultralytics_main_path / "ultralytics/cfg/models/v8/yolov8.yaml")
    model = YOLO(yaml_path).load(args.model)
    model.train(
        data="/mnt/home/lyj/AICUP2024/yolov8/ALL/data.yaml",  # 确保路径正确
        epochs=10,
        batch=4,
        device=0,
        workers=0
    )

    default_model_path = os.path.join('runs', 'detect', 'train3', 'weights', 'best.pt')
    if os.path.exists(default_model_path):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        shutil.copy(default_model_path, args.output)
    else:
        raise FileNotFoundError(f"Finetuned model not found at {default_model_path}")
    
    print("Finetuning completed")

if __name__ == '__main__':
    main()