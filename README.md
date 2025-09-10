
# YOLOv3 TensorFlow 2 (TF2) - Extended & Modified Version

This repository provides an extended and modified implementation of YOLOv3 in TensorFlow 2, based on [zzh8829/yolov3-tf2](https://github.com/zzh8829/yolov3-tf2). It supports training, evaluation, and inference for custom datasets and includes improvements for usability and compatibility.

## Table of Contents
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Pre-trained Weights Conversion](#pre-trained-weights-conversion)
- [Detection](#detection)
- [Training](#training)
- [Evaluation](#evaluation)
- [Dataset Preparation](#dataset-preparation)
- [References](#references)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/mos-ks/yolo-V3.git
cd yolo-V3
pip install -r requirements-gpu.txt
```

## Pre-trained Weights Conversion

Download official Darknet weights and convert them for TensorFlow usage:

```bash
# Download YOLOv3 weights
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf

# Download YOLOv3-tiny weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
python convert.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny
```

## Detection

Run detection on images, videos, or webcam input. Example commands:

```bash
# Detect objects in an image using YOLOv3
python detect.py --image ./data/meme.jpg

# Detect objects in an image using YOLOv3-tiny
python detect.py --weights ./checkpoints/yolov3-tiny.tf --tiny --image ./data/street.jpg

# Detect objects from webcam
python detect_video.py --video 0

# Detect objects in a video file
python detect_video.py --video path_to_file.mp4 --weights ./checkpoints/yolov3-tiny.tf --tiny

# Save output video
python detect_video.py --video path_to_file.mp4 --output ./output.avi
```

**Arguments:**
- `--weights`: Path to TensorFlow checkpoint.
- `--tiny`: Use YOLOv3-tiny model.
- `--image`: Path to input image.
- `--video`: Path to video file or webcam index.
- `--output`: Path to save output video.

## Training

To train on a custom dataset, you need to generate TFRecord files compatible with the TensorFlow Object Detection API.

**Dataset Preparation:**
- Use [Microsoft VoTT](https://github.com/Microsoft/VoTT) or similar tools to annotate images and export to TFRecord.
- For Pascal VOC, use [create_pascal_tf_record.py](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py).

**Training Command Example:**
```bash
python train.py \
    --dataset ./data/voc_train.tfrecord \
    --val_dataset ./data/voc_val.tfrecord \
    --classes ./data/voc2012.names \
    --model_name VOC_V1 \
    --num_classes 20 \
    --batch_size 16 \
    --epochs 3 \
    --weights ./checkpoints/yolov3.tf \
    --weights_num_classes 80 \
    --transfer no_output
```

**Arguments:**
- `--dataset`: Path to training TFRecord file.
- `--val_dataset`: Path to validation TFRecord file.
- `--classes`: Path to class names file.
- `--model_name`: Name for saving model checkpoints.
- `--num_classes`: Number of classes in your dataset.
- `--batch_size`: Training batch size.
- `--epochs`: Number of training epochs.
- `--weights`: Path to pre-trained weights.
- `--weights_num_classes`: Number of classes in pre-trained weights (default: 80 for COCO).
- `--transfer`: Transfer learning mode (e.g., `no_output`).

## Evaluation

Evaluate trained models using COCO metrics or custom evaluation scripts:

```bash
python eval.py \
    --val_image_dir "./data/val_images" \
    --val_json "./data/output_coco.json" \
    --weights "VOC_V1/yolov3_train_3.tf" \
    --classes_file ./data/voc2012.names \
    --num_classes 20 \
    --img_size 1024
```

**Arguments:**
- `--val_image_dir`: Directory with validation images.
- `--val_json`: COCO-format JSON file with ground truth.
- `--weights`: Path to trained model weights.
- `--classes_file`: Path to class names file.
- `--num_classes`: Number of classes.
- `--img_size`: Input image size for evaluation.

## Dataset Preparation

1. **Annotation:** Annotate your images using VoTT or other tools.
2. **TFRecord Conversion:** Convert annotations to TFRecord format.
3. **Class Names:** Create a `.names` file listing all class names, one per line.
4. **Organize Data:** Place images and TFRecord files in the `data/` directory.

## Troubleshooting & Tips
- Ensure all paths are correct and files exist.
- For GPU support, verify CUDA and cuDNN installation.
- Use compatible TensorFlow and Python versions as specified in `requirements-gpu.txt` and `conda-gpu.yml`.
- If you encounter errors, check log outputs and refer to the referenced repositories for solutions.

## References

Implementation is based on multiple sources due to the complexity of YOLOv3:
- [zzh8829/yolov3-tf2](https://github.com/zzh8829/yolov3-tf2) (parent repo)
- [pjreddie/darknet](https://github.com/pjreddie/darknet) (official YOLOv3)
- [AlexeyAB](https://github.com/AlexeyAB) (parameter explanations)
- [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3) (models, loss functions)
- [YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3) (data transformations, loss functions)
- [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) (models)
- [broadinstitute/keras-resnet](https://github.com/broadinstitute/keras-resnet) (batch normalization fix)
