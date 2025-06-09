from ultralytics import YOLO

def train():
    model = YOLO("yolov8s.pt")

    model.train(
        data="D:/projects/defect_recog/data.yaml",
        epochs=100,
        imgsz=1280,
        batch=8,
        name="defect_recog_yolov8s_smallds",
        pretrained=True,
        workers=8,               # 加快数据加载速度
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=0.5,
        mixup=0.2,
    )

if __name__ == "__main__":
    train()
