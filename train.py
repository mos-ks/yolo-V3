import os
import pdb
import time

import cv2
import numpy as np
import tensorflow as tf
import yolov3_tf2.dataset as dataset
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        LearningRateScheduler, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from yolov3_tf2.models import (YoloLoss, YoloV3, YoloV3Tiny, yolo_anchor_masks,
                               yolo_anchors, yolo_tiny_anchor_masks,
                               yolo_tiny_anchors)
from yolov3_tf2.utils import freeze_all

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# pdb.set_trace()
flags.DEFINE_string("model_name", "bdd_orig", "path to dataset")
flags.DEFINE_string("dataset", "", "path to dataset")
flags.DEFINE_string("dataset_2", "", "path to dataset")
flags.DEFINE_boolean("rcf", False, "path to dataset")
flags.DEFINE_string("val_dataset", "", "path to validation dataset")
flags.DEFINE_boolean("tiny", False, "yolov3 or yolov3-tiny")
flags.DEFINE_string("weights", "./checkpoints/yolov3.tf", "path to weights file")
flags.DEFINE_string("classes", "./data/coco.names", "path to classes file")
flags.DEFINE_enum(
    "mode",
    "fit",
    ["fit", "eager_fit", "eager_tf"],
    "fit: model.fit, "
    "eager_fit: model.fit(run_eagerly=True), "
    "eager_tf: custom GradientTape",
)
flags.DEFINE_enum(
    "transfer",
    "none",
    ["none", "darknet", "no_output", "frozen", "fine_tune"],
    "none: Training from scratch, "
    "darknet: Transfer darknet, "
    "no_output: Transfer all but output, "
    "frozen: Transfer and freeze all, "
    "fine_tune: Transfer all and freeze darknet only",
)
flags.DEFINE_integer("size", 1024, "image size")
flags.DEFINE_integer("epochs", 2, "number of epochs")
flags.DEFINE_integer("batch_size", 8, "batch size")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_integer("num_classes", 80, "number of classes in the model")
flags.DEFINE_integer(
    "weights_num_classes",
    None,
    "specify num class for `weights` file if different, "
    "useful in transfer learning with different number of classes",
)
flags.DEFINE_boolean(
    "multi_gpu", False, "Use if wishing to train with more than 1 GPU."
)


class SaveEveryNEpochs(Callback):
    """Save the model (weights) every N epochs."""

    def __init__(self, n, save_path_template):
        super().__init__()
        self.n = n
        self.save_path_template = save_path_template

    def on_epoch_end(self, epoch, logs=None):
        # Epoch numbers are zero-based internally, so (epoch+1) is the "human" epoch count.
        if (epoch + 1) % self.n == 0:
            save_path = self.save_path_template.format(epoch=epoch + 1)
            self.model.save_weights(save_path)
            print(f"\nSaved checkpoint to {save_path}")

def cosine_warmup_scheduler(epoch, current_lr):
    warmup_epochs = 5
    base_lr = 1e-3
    min_lr = 1e-5
    total_epochs = 100
    
    if epoch < warmup_epochs:
        # Linear warmup from 0 to base_lr
        return base_lr * float(epoch + 1) / warmup_epochs
    else:
        # Cosine decay from epoch warmup_epochs -> total_epochs
        progress = float(epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi) * progress))
        decayed = (base_lr - min_lr) * cosine_decay + min_lr
        return decayed.numpy()


def setup_model():
    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # Configure the model for transfer learning
    if FLAGS.transfer == "none":
        pass  # Nothing to do
    elif FLAGS.transfer in ["darknet", "no_output"]:
        # Darknet transfer is a special case that works
        # with incompatible number of classes
        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size,
                training=True,
                classes=FLAGS.weights_num_classes or FLAGS.num_classes,
            )
        else:
            model_pretrained = YoloV3(
                FLAGS.size,
                training=True,
                classes=FLAGS.weights_num_classes or FLAGS.num_classes,
            )
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == "darknet":
            model.get_layer("yolo_darknet").set_weights(
                model_pretrained.get_layer("yolo_darknet").get_weights()
            )
            freeze_all(model.get_layer("yolo_darknet"))
        elif FLAGS.transfer == "no_output":
            for l in model.layers:
                if not l.name.startswith("yolo_output"):
                    l.set_weights(model_pretrained.get_layer(l.name).get_weights())
                    freeze_all(l)
    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == "fine_tune":
            # freeze darknet and fine tune other layers
            darknet = model.get_layer("yolo_darknet")
            freeze_all(darknet)
        elif FLAGS.transfer == "frozen":
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes) for mask in anchor_masks]

    model.compile(
        optimizer=optimizer, loss=loss, run_eagerly=(FLAGS.mode == "eager_fit")
    )

    return model, optimizer, loss, anchors, anchor_masks


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    # Setup
    if FLAGS.multi_gpu:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)

        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
        FLAGS.batch_size = BATCH_SIZE

        with strategy.scope():
            model, optimizer, loss, anchors, anchor_masks = setup_model()
    else:
        model, optimizer, loss, anchors, anchor_masks = setup_model()

    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size
        )
    else:
        train_dataset = dataset.load_fake_dataset()
    # for example in train_dataset.take(49): image, gt = example
    # print("Image Shape:", image.shape)
    # print("Ground Truth (GT):", gt)
    # import matplotlib.patches as patches
    # import matplotlib.pyplot as plt
    # cls_names = ["pedestrian","rider","car","truck","bus","train","motorcycle","bicycle","traffic light","traffic sign"]
    # image = image.numpy().astype(np.uint8)
    # gt = gt.numpy()
    # fig, ax = plt.subplots(1, figsize=(10, 10))
    # ax.imshow(image)
    # box = gt[5]
    # x_min, y_min, x_max, y_max, class_id = box
    # x_min, y_min, x_max, y_max = x_min * FLAGS.size, y_min * FLAGS.size, x_max * FLAGS.size, y_max * FLAGS.size
    # rect = patches.Rectangle((x_min, y_min),x_max - x_min, y_max - y_min, linewidth=2,edgecolor="red",facecolor="none",)
    # ax.add_patch(rect)
    # ax.text(x_min, y_min, f"Class: {cls_names[int(class_id)]}", color="yellow", fontsize=12)
    # plt.savefig("test_dataset.png")
    # plt.close()
    train_dataset = train_dataset.shuffle(buffer_size=512)
    if FLAGS.rcf:
        train_dataset = train_dataset.batch(FLAGS.batch_size-1)
        # Combine the datasets
        def combine_batches(batch1, batch2):            
            # Assuming each batch is a tuple (data, labels)
            data1, labels1 = batch1
            data2, labels2 = batch2

            # Combine data and labels separately
            combined_data = tf.concat([data1, data2], axis=0)
            combined_labels = tf.concat([labels1, labels2], axis=0)

            return combined_data, combined_labels


        train_dataset_2 = dataset.load_tfrecord_dataset(
            FLAGS.dataset_2, FLAGS.classes, FLAGS.size
        )
        train_dataset_2 = train_dataset_2.shuffle(buffer_size=512)
        train_dataset_2 = train_dataset_2.batch(1)
        # Use zip to pair batches and then map to combine them
        combined_dataset = tf.data.Dataset.zip((train_dataset, train_dataset_2))
        train_dataset = combined_dataset.map(lambda batch1, batch2: combine_batches(batch1, batch2))
    else:
        train_dataset = train_dataset.batch(FLAGS.batch_size)

    train_dataset = train_dataset.map(
        lambda x, y: (
            dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size),
        )
    )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size
        )
    else:
        val_dataset = dataset.load_fake_dataset()
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(
        lambda x, y: (
            dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size),
        )
    )

    if FLAGS.mode == "eager_tf":
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean("val_loss", dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                logging.info(
                    "{}_train_{}, {}, {}".format(
                        epoch,
                        batch,
                        total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss)),
                    )
                )
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info(
                    "{}_val_{}, {}, {}".format(
                        epoch,
                        batch,
                        total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss)),
                    )
                )
                avg_val_loss.update_state(total_loss)

            logging.info(
                "{}, train: {}, val: {}".format(
                    epoch, avg_loss.result().numpy(), avg_val_loss.result().numpy()
                )
            )

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights("checkpoints/yolov3_train_{}.tf".format(epoch))
    else:

        lr_callback = LearningRateScheduler(cosine_warmup_scheduler)
        callbacks = [
            ReduceLROnPlateau(verbose=1),
            # lr_callback,
            # EarlyStopping(patience=3, verbose=1),
            # ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
            #                 verbose=1, save_weights_only=True),
            TensorBoard(
                log_dir=os.path.dirname(os.path.abspath(__file__))
                + f"/models/{FLAGS.model_name}/logs"
            ),
            SaveEveryNEpochs(
                n=10,
                save_path_template=os.path.dirname(os.path.abspath(__file__))
                + f"/models/{FLAGS.model_name}"
                + "/yolov3_train_{epoch}.tf",
            ),
        ]

        start_time = time.time()
        history = model.fit(
            train_dataset,
            epochs=FLAGS.epochs,
            callbacks=callbacks,
            validation_data=val_dataset,
        )
        end_time = time.time() - start_time
        print(f"Total Training Time: {end_time}")


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
