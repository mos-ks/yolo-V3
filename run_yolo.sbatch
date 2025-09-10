#!/bin/bash
#SBATCH --job-name=yolov3_training  # Job name
#SBATCH --output=yolov3_job_output_%j.txt    # Standard output (%j will be replaced by job ID)
#SBATCH --error=yolov3_job_error_%j.txt      # Standard error (%j will be replaced by job ID)
#SBATCH --partition=advanced                 # Partition (queue) to submit to (use the appropriate partition for GPUs)
#SBATCH --gres=gpu:full:1                    # Request 1 GPU
#SBATCH --mem=64G                         # Request of system memory (adjust based on your system's availability)
#SBATCH --cpus-per-task=32                # Request CPU cores (adjust based on your task's CPU requirements)
#SBATCH --time=1:00:00                    # Job time limit, 08:00:00
#SBATCH --mail-type=ALL                   # Send email on job start, end, and fail
#SBATCH --mail-user=--                    # Set your email address for notifications
#SBATCH --constraint=LSDF
#SBATCH --export=ALL

# Load any necessary modules (if required by your system)
module purge   
source /home/---/miniconda3/etc/profile.d/conda.sh
conda activate tf-yolo

# ------  disable XLA (preferred) ------
export TF_XLA_FLAGS=--tf_xla_auto_jit=off
export TF_XLA_ENABLE_XLA_DEVICES=0
export TF_ENABLE_XLA=0
# --------------------------------------

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Define or hardcode APP_DIR
APP_DIR="/---"
# Check directory
if [ ! -d "$APP_DIR" ]; then
    echo "Error: Directory $APP_DIR does not exist or is not accessible."
    exit 1
fi

DATA_DIR="$APP_DIR/object_detection/yolov3/AL"
MODEL_DIR="$APP_DIR/object_detection/yolov3/models"
VAL_RECORD="$APP_DIR/datasets/KITTI/tf/_val.tfrecord"
CLASSES_FILE="$APP_DIR/object_detection/yolov3/data/kitti.names"
CHECKPOINT="$APP_DIR/object_detection/yolov3/checkpoints/yolov3.tf"

START_INDEX=0
END_INDEX=50

EVAL_LOG="$APP_DIR/object_detection/yolov3/eval_kitti.log"
ALL_FILES=($(find "$DATA_DIR" -type f -regex '.*_train_[123]_.*\.tfrecord' | sort))
TOTAL=${#ALL_FILES[@]}

echo "Total TFRecord files found: $TOTAL"

for ((i=START_INDEX; i<END_INDEX && i<TOTAL; i++)); do
    tfrecord_path="${ALL_FILES[$i]}"
    subfolder=$(basename "$(dirname "$tfrecord_path")")
    filename=$(basename "$tfrecord_path")
    iter_num=$(echo "$filename" | grep -oP '_train_\K[123]')

    # Check if filename includes seed0
    if [[ "$filename" == *"seed0"* ]]; then
        model_name="KITTI_${subfolder}_iter${iter_num}_seed0_V0"
    else
        model_name="KITTI_${subfolder}_iter${iter_num}_V0"
    fi
    model_output="$MODEL_DIR/$model_name/yolov3_train_50.tf.index"

    if [ ! -f "$model_output" ]; then
        echo "Training: $model_name"
        mkdir -p "$MODEL_DIR/$model_name"
        python3 "$APP_DIR/object_detection/yolov3/train.py" \
            --val_dataset "$VAL_RECORD" \
            --dataset "$tfrecord_path" \
            --classes "$CLASSES_FILE" \
            --model_name "$model_name" \
            --num_classes 7 \
            --weights "$CHECKPOINT" \
            --weights_num_classes 80 \
            --transfer no_output \
            --batch_size 8 \
            --epochs 50 >> "$MODEL_DIR/$model_name/train.log" 2>&1 &
    else
        echo "Already trained: $model_name"
    
    echo "Evaluating: $model_name"
    python3 "$APP_DIR/object_detection/yolov3/eval.py" \
        --val_image_dir "$APP_DIR/datasets/KITTI/training/image_2" \
        --val_json "$APP_DIR/datasets/KITTI/output_coco.json" \
        --weights "$MODEL_DIR/$model_name/yolov3_train_50.tf" \
        --classes_file "$CLASSES_FILE" \
        --num_classes 7 \
        --img_size 1024 >> "$EVAL_LOG" 2>&1
    fi

done

wait
