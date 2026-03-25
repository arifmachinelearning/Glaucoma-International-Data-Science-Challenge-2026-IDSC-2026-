from ultralytics import YOLO

model = YOLO(r"C:\Users\User\PycharmProjects\hillel-yaffe-glaucoma\venv\Lib\site-packages\ultralytics\cfg\models\11\yolo11-cls.yaml")
model.train(data=r"C:\Users\User\PycharmProjects\hillel-yaffe-glaucoma\dataset_new", epochs=100, imgsz=224)

## YOLO11 = 0.939, 1.5 ms, 1.528, 3.2 GFLOPsy
## YOLO11 (improved) = 0.980, 1.9 ms, 0.824, 1.5 Glops
## YOLO26 = 0.736, 2.3 ms

##FPS

#1.6 GLOPs, 0.82 Parameter