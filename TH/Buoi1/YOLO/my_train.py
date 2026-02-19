from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.yaml")  # build a new model from YAML
#model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.pt") # Nên dùng bản .pt để có sẵn trọng số tốt nhất

    # Train the model
    results = model.train(data="dataset.yaml", epochs=65, imgsz=640, batch=8, workers=0, fliplr=0.0, optimizer='Adam')