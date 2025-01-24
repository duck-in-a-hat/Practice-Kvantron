from ultralytics import YOLO 



if __name__ == '__main__':
    
    model = YOLO(f'yolov8n.pt')
    model.train(data='C:/Users/vlad_/python prodgect/seller whith phone.v3i.yolov8', epochs=100, imgsz=640, batch=72, workers=2,
                cache=True, single_cls=True, device=0)