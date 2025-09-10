#!/usr/bin/env python3
import argparse
import json

import numpy as np


def load_bdd_boxes(ann_files, target_img_size=None):
    """
    ann_files: list of paths to BDD JSON annotation files
    target_img_size: (width, height) to scale boxes to.
                     If None, no scaling is applied.

    Returns: boxes (N x 2) array of (width, height)
    """
    all_boxes = []

    for ann_file in ann_files:
        with open(ann_file, "r") as f:
            data = json.load(f)

        # 'data' should be a list of image annotations
        # Each item might look like:
        # {
        #   "name": "xxxx.jpg",
        #   "width": 1280,
        #   "height": 720,
        #   "labels": [
        #       {
        #           "category": "car",
        #           "box2d": { "x1": 100, "y1": 200, "x2": 300, "y2": 400 }
        #       },
        #       ...
        #   ]
        # }
        for img_anno in data:
            if "width" not in img_anno or "height" not in img_anno:
                # You may need to hardcode BDD's original size (1280x720)
                # if it's not in the JSON
                original_width, original_height = 1280, 720
            else:
                original_width = img_anno["width"]
                original_height = img_anno["height"]

            if "labels" not in img_anno:
                continue

            for label in img_anno["labels"]:
                box = label.get("box2d", None)
                if not box:
                    # Some labels might not have box2d, e.g., "poly2d" for lanes.
                    continue

                x1, y1 = box["x1"], box["y1"]
                x2, y2 = box["x2"], box["y2"]
                w = x2 - x1
                h = y2 - y1

                # If scaling to a target size (W,H)
                if target_img_size is not None:
                    target_w, target_h = target_img_size
                    scale_w = target_w / float(original_width)
                    scale_h = target_h / float(original_height)
                    w *= scale_w
                    h *= scale_h

                # Filter out invalid / zero-size
                if w <= 0 or h <= 0:
                    continue

                all_boxes.append([w, h])

    return np.array(all_boxes, dtype=np.float32)


def iou(box, clusters):
    """
    box: shape (2,) => (w, h)
    clusters: shape (k, 2) => each row is (w, h)
    Returns: IoU of 'box' with each cluster as a 1-D array of length k.
    """
    w_box, h_box = box[0], box[1]
    w_clusters = clusters[:, 0]
    h_clusters = clusters[:, 1]

    # Intersection
    intersect_w = np.minimum(w_box, w_clusters)
    intersect_h = np.minimum(h_box, h_clusters)
    intersection = intersect_w * intersect_h

    # Union = area(box) + area(cluster) - intersection
    box_area = w_box * h_box
    cluster_areas = w_clusters * h_clusters
    union = box_area + cluster_areas - intersection + 1e-9  # small epsilon
    return intersection / union


def run_kmeans(boxes, k=9, seed=42, max_iter=1000):
    """
    boxes: (N x 2) array of (w, h)
    k: number of clusters (anchors) to find
    seed: random seed
    max_iter: maximum iterations
    Returns: k cluster centers (anchor boxes) as (k x 2) array.
    """
    np.random.seed(seed)

    # 1) Randomly choose k boxes as initial cluster centers
    indices = np.random.choice(len(boxes), k, replace=False)
    clusters = boxes[indices].copy()

    box_cluster_ids = np.zeros(boxes.shape[0], dtype=np.int32)

    for _ in range(max_iter):
        # 2) Assign each box to the cluster with highest IoU
        iou_values = np.zeros((boxes.shape[0], k), dtype=np.float32)
        for c in range(k):
            iou_values[:, c] = iou(clusters[c], boxes)

        new_box_cluster_ids = np.argmax(iou_values, axis=1)

        # 3) If no change, stop
        if np.all(box_cluster_ids == new_box_cluster_ids):
            break
        box_cluster_ids = new_box_cluster_ids

        # 4) Update clusters by taking mean w/h of members
        for c in range(k):
            if np.any(box_cluster_ids == c):
                clusters[c] = np.mean(boxes[box_cluster_ids == c], axis=0)

    return clusters


def avg_iou(boxes, clusters):
    """Compute the average IoU of 'boxes' to their nearest cluster in 'clusters'."""
    iou_values = np.zeros((boxes.shape[0], clusters.shape[0]), dtype=np.float32)
    for c in range(clusters.shape[0]):
        iou_values[:, c] = iou(clusters[c], boxes)
    best_ious = np.max(iou_values, axis=1)
    return np.mean(best_ious)


def main():
    parser = argparse.ArgumentParser(
        description="Find best YOLO anchors using BDD JSON annotations."
    )
    parser.add_argument(
        "--ann_files",
        nargs="+",
        required=True,
        help="Paths to one or more BDD JSON annotation files.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=None,
        help="(width height) to scale boxes to. E.g. 640 640. If omitted, no scaling.",
    )
    parser.add_argument(
        "--clusters", type=int, default=9, help="Number of anchor clusters (default 9)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default 42).",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for k-means (default 1000).",
    )
    args = parser.parse_args()

    # 1) Load bounding boxes from BDD
    boxes = load_bdd_boxes(args.ann_files, target_img_size=args.target_size)
    if boxes.size == 0:
        print("No bounding boxes found! Check your annotation format.")
        return

    print(f"Loaded {len(boxes)} boxes from {len(args.ann_files)} annotation file(s).")

    # 2) Run k-means to find anchor boxes
    anchors = run_kmeans(boxes, k=args.clusters, seed=args.seed, max_iter=args.max_iter)
    mean_iou = avg_iou(boxes, anchors)
    print(f"Found anchors (unsorted). Average IoU: {mean_iou:.4f}")

    # 3) Sort anchors by area (w*h) ascending
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]

    # Recompute average IoU just to confirm (should be same or very close)
    mean_iou_sorted = avg_iou(boxes, anchors)

    print(f"Final sorted anchors (width, height):")
    for w, h in anchors:
        print(f"({w:.2f}, {h:.2f})")
    print(f"Average IoU to anchors (sorted): {mean_iou_sorted:.4f}")

    # If you want them as integer pairs:
    int_anchors = [(int(round(w)), int(round(h))) for w, h in anchors]
    print("Integer anchors (for config):", int_anchors)


if __name__ == "__main__":
    main()


# python anchors_gen.py \
#     --ann_files /lsdf/kit/scc/projects/kktmt/app/efficientdet_uncertainty/datasets/BDD100K/bdd100k/labels/bdd100k_labels_images_val.json /lsdf/kit/scc/projects/kktmt/app/efficientdet_uncertainty/datasets/BDD100K/bdd100k/labels/bdd100k_labels_images_train.json \
#     --target_size 1024 1024 \
#     --clusters 9 \
#     --seed 42 \
#     --max_iter 1000
