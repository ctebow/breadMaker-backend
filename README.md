# breadMaker

- Uses YOLOv8 model to recognize drawn circuit components
- Runs circuit logic, creates LTSpice diagram, and gives step-by-step breadboarding instructions

## TO USE YOLO DETECTION WITH CURRENT WEIGHTS:
- yolo task=detect mode=predict model=path/to/best.pt source=path/to/image.jpg

### CURRENT MODEL:
- mAP@50: 0.79
- Box loss: 1.01
- Class loss: 0.49
- Looking to improve with more training, use tighter/better bounding boxes

### Link for yolov8 python usage
- https://docs.ultralytics.com/usage/python/
