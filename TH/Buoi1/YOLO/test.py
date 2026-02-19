from ultralytics import YOLO
model = YOLO('yolov8n.pt')
if __name__ == "__main__":
    model.train(data='dataset.yaml', epochs=15,imgsz=640,batch=16,optimizer='Adam')
    metrics = model.val()