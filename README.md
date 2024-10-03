# YOLOv11-ONNX-TensorRT

## Installation

Ensure you have Python 3.6 or higher. Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python onnx_image.py --model yolo11n.onnx --image input_image.jpg --output output.jpg --conf-thres 0.5 --iou-thres 0.5
```
```bash
python onnx_video.py --model yolo11n.onnx --video video.avi --conf_thres 0.5 --iou_thres 0.5 --output output.avi
```
```bash
python onnx_webcam.py --model yolo11n.onnx --conf_thres 0.5 --iou_thres 0.5 --output output.avi
```

