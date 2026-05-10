from ultralytics import YOLO

# Load a model model = YOLO("yolo11s.yaml")  # build a new model from YAML
#model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11s.yaml").load("yolo11s.pt")  # build from YAML and transfer weights

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11s.pt") # Nên dùng bản .pt để có sẵn trọng số tốt nhất

    # Train the model
    results = model.train(data=r"D:\TaiLieu\XLA\XLA_TH\CV-and-DIP\Buoi1_2\YOLO\dataset.yaml", 
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