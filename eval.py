import json
import os
import pdb

# import cv2
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import draw_outputs, load_darknet_weights

# or model.load_weights(...) if you have a .tf checkpoint


FLAGS = flags.FLAGS
flags.DEFINE_string("val_image_dir", None, "Path to the directory of images.")
flags.DEFINE_string("val_json", None, "Path to the COCO-format JSON.")
flags.DEFINE_integer("img_size", 416, "Inference image size.")
flags.DEFINE_integer("max_images", 0, "Max images to process (0 = all).")
flags.DEFINE_string("weights", None, "Path to YOLO weights.")
flags.DEFINE_bool("tiny", False, "Whether to use YOLO tiny.")
flags.DEFINE_integer("num_classes", 10, "Number of classes.")
flags.DEFINE_string("classes_file", None, "Path to .names file.")


def load_class_names(class_file):
    """
    Loads class names (like 'person', 'car', ...) from a .names file or
    any text file with one class name per line.
    """
    with open(class_file, "r") as f:
        lines = f.read().strip().splitlines()
    return lines


def main(_argv):
    logging.info("Loading YOLO model...")

    # Build YOLO (pseudocode - adjust to your actual YOLO implementation)
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info(f"Model loaded from {FLAGS.weights}")

    # Load class names (optional)
    class_names = load_class_names(FLAGS.classes_file)  # if needed

    # -----------------------------------------------------------------------
    # COCO GT
    # -----------------------------------------------------------------------
    coco_gt = COCO(FLAGS.val_json)
    img_ids = coco_gt.getImgIds()
    if FLAGS.max_images > 0:
        img_ids = img_ids[: FLAGS.max_images]

    # This returns a list of dicts, e.g. [{'id': 123, 'file_name': 'xxx.jpg', 'width':..., 'height':...}, ...]
    img_infos = coco_gt.loadImgs(img_ids)

    def _gen():
        for info in img_infos:
            file_name = info["file_name"]
            img_id = info["id"]
            full_path = os.path.join(FLAGS.val_image_dir, file_name)

            # Read image
            raw = tf.io.read_file(full_path)
            image = tf.io.decode_jpeg(raw, channels=3)

            # Save original shape
            orig_height = tf.shape(image)[0]
            orig_width = tf.shape(image)[1]

            yield (image, img_id, orig_width, orig_height)

    dataset = tf.data.Dataset.from_generator(
        _gen,
        output_types=(tf.uint8, tf.int32, tf.int32, tf.int32),
        output_shapes=((None, None, 3), (), (), ()),
    )

    # 4) Single-image inference loop
    #    (no batching => we do expand_dims below).
    coco_detections = []
    # for image_tensor, img_id_tensor, orig_w_tensor, orig_h_tensor in tqdm(
    #     dataset, total=len(img_infos)
    # ):
    for image_tensor, img_id_tensor, orig_w_tensor, orig_h_tensor in dataset:
        # A) Convert to float32, resize to (FLAGS.img_size, FLAGS.img_size), normalize
        resized_img = tf.image.resize(
            tf.cast(image_tensor, tf.float32), (FLAGS.img_size, FLAGS.img_size)
        )
        resized_img = resized_img / 255.0
        batch_input = tf.expand_dims(resized_img, 0)  # shape [1, 416, 416, 3]

        # B) Run YOLO once (single image in a batch)
        boxes, scores, classes, nums = yolo(batch_input, training=False)

        # logging.info("detections:")
        # for i in range(nums[0]):
        #     logging.info(
        #         "\t{}, {}, {}".format(
        #             class_names[int(classes[0][i])],
        #             np.array(scores[0][i]),
        #             np.array(boxes[0][i]),
        #         )
        #     )

        # img = cv2.cvtColor(image_tensor.numpy(), cv2.COLOR_RGB2BGR)
        # # pdb.set_trace()
        # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        # cv2.imwrite("./output_eval.jpg", img)
        # pdb.set_trace()
        # C) Convert boxes from YOLO coords -> original coords
        img_id_int = int(img_id_tensor.numpy())
        orig_w = float(orig_w_tensor.numpy())
        orig_h = float(orig_h_tensor.numpy())

        valid_dets = nums[0].numpy()
        for det_idx in range(valid_dets):
            # YOLO box (normalized in [0,1], or sometimes different—adjust if needed)
            x1_norm, y1_norm, x2_norm, y2_norm = boxes[0][det_idx]

            # Scale up to 416×416 (if YOLO output is normalized)
            x1_resized = x1_norm * FLAGS.img_size
            y1_resized = y1_norm * FLAGS.img_size
            x2_resized = x2_norm * FLAGS.img_size
            y2_resized = y2_norm * FLAGS.img_size

            # Now map from 416×416 back to the original shape
            x1_orig = x1_resized * (orig_w / FLAGS.img_size)
            y1_orig = y1_resized * (orig_h / FLAGS.img_size)
            x2_orig = x2_resized * (orig_w / FLAGS.img_size)
            y2_orig = y2_resized * (orig_h / FLAGS.img_size)

            # Clamp
            x1_orig = max(0.0, min(orig_w, x1_orig))
            x2_orig = max(0.0, min(orig_w, x2_orig))
            y1_orig = max(0.0, min(orig_h, y1_orig))
            y2_orig = max(0.0, min(orig_h, y2_orig))

            width = x2_orig - x1_orig
            height = y2_orig - y1_orig

            # Build final detection
            score_val = float(scores[0][det_idx])
            cls_idx = int(classes[0][det_idx])
            detection = {
                "image_id": img_id_int,
                "category_id": cls_idx + 1,
                "bbox": [x1_orig, y1_orig, width, height],
                "score": score_val,
            }
            coco_detections.append(detection)
        # pdb.set_trace()
    # -----------------------------------------------------------------------
    # Evaluate with pycocotools
    # -----------------------------------------------------------------------
    # pdb.set_trace()
    coco_dt = coco_gt.loadRes(coco_detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    precision = coco_eval.eval["precision"][:, :, :, 0, -1]
    ap_per_class = []
    for c in range(precision.shape[-1]):  # iterate over all classes
        precision_c = precision[:, :, c]
        # Only consider values if > -1.
        precision_c = precision_c[precision_c > -1]
        ap_c = np.mean(precision_c)*100.0 if precision_c.size else -1.0
        ap_per_class.append(ap_c)

    for i, ap_value in enumerate(ap_per_class):
        class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
        logging.info(f"Class: {class_name}, AP: {ap_value:.4f}")


if __name__ == "__main__":
    flags.mark_flag_as_required("val_image_dir")
    flags.mark_flag_as_required("val_json")
    flags.mark_flag_as_required("weights")
    flags.mark_flag_as_required("classes_file")
    app.run(main)
