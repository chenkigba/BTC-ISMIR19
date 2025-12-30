# 注意：警告配置在 utils.warnings_config 中统一管理
# 如果此模块被单独导入，需要先导入 warnings_config
import numpy as np
import tensorflow as tf


class TF_Logger(object):
    """
    TensorBoard logger for a PyTorch training loop.

    Note: The original upstream code used TensorFlow 1.x `FileWriter`/`Summary`.
    This implementation uses TensorFlow 2.x `tf.summary.*` APIs.
    """

    def __init__(self, log_dir: str):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step: int):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step: int):
        """Log a list of images (numpy arrays)."""
        if images is None:
            return

        images_list = list(images)
        if len(images_list) == 0:
            return

        stacked = []
        for img in images_list:
            arr = np.asarray(img)
            if arr.ndim == 2:
                # HxW -> HxWx1
                arr = arr[..., None]
            if arr.ndim != 3:
                raise ValueError(f"Expected image with 2 or 3 dims, got shape={arr.shape}")
            if arr.shape[2] not in (1, 3, 4):
                raise ValueError(f"Expected channel dim to be 1/3/4, got shape={arr.shape}")

            if arr.dtype != np.uint8:
                # Normalize to 0-255
                arr = arr.astype(np.float32)
                amin = float(np.min(arr))
                amax = float(np.max(arr))
                if amax - amin < 1e-8:
                    arr = np.zeros_like(arr, dtype=np.uint8)
                else:
                    arr = ((arr - amin) / (amax - amin) * 255.0).clip(0, 255).astype(np.uint8)

            stacked.append(arr)

        batch = np.stack(stacked, axis=0)  # NxHxWxC

        with self.writer.as_default():
            tf.summary.image(tag, batch, step=step, max_outputs=len(images_list))
            self.writer.flush()

    def histo_summary(self, tag, values, step: int, bins: int = 1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step, buckets=bins)
            self.writer.flush()