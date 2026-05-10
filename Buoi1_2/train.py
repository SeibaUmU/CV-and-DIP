from ultralytics import YOLO

DATA_YAML = r'D:\\XLA\\yolov11.v3i.yolov11\\data.yaml'

MODEL_NAME = 'yolo11s.pt'   

def main():
    model = YOLO(MODEL_NAME)

    model.train(
        data = DATA_YAML,
        epochs = 300,
        patience = 50,
        imgsz = 640,
        batch = 16,
        device = 0,
        workers = 2,
        cache = True,
        amp = True,
        optimizer = 'AdamW',
        lr0 = 0.001,
        weight_decay = 0.0005,
        cos_lr = True,
        freeze = 10,
        mosaic = 1.0,
        mixup = 0.0,
        degrees = 10.0,
        hsv_v = 0.4,
        translate = 0.1,
        scale = 0.5,
        plots = False,
        name = 'traffic_sign_result_fast'
    )

if __name__ == '__main__':
    main() 