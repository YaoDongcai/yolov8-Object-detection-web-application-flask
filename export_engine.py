from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model_object_detection = YOLO("yolov8n.pt")

# Export the model
# model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
# trt_model = YOLO("yolov8n.engine")

# Run inference
# results = model("https://ultralytics.com/images/bus.jpg")
results = model_object_detection.predict('test.jpg', conf=75/100)
print(results)
_, labels = results[0].plot()
print(labels)

