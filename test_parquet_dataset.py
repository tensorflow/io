import tensorflow as tf
import tensorflow_io.arrow as arrow_io

dataset = arrow_io.ArrowParquetDataset(
    file_paths = ['/home/yye/training-platform/training-platform/bento/apps/demos/chicago_taxi/data/test.parquet'],
    column_names=('tips'),
    columns=(),
    output_types=(tf.float32),
    output_shapes=([]),
    batch_size=4,
    batch_mode='keep_remainder')

# This will iterate over each row of each file provided
for row in dataset:
    print(row)
