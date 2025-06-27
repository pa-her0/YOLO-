import os
from ultralytics import YOLO
import torch
import ultralytics
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
 
 
def main():
    print(ultralytics.__file__)
    model = YOLO(r'ultralytics/cfg/models/v8/yolov8.yaml').load('./prune1.pt')
    model.train(data="/home/lyj/local/Project/pre_data/combined_data.yaml", amp=False, imgsz=640, epochs=50, batch=20, device=0, workers=0)
 
 
if __name__ == '__main__':
    main()