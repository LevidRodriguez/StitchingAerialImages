import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow_hub as hub

def image_input_fn(image_files):
    filename_queue = tf.train.string_input_producer(
        image_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    return tf.image.convert_image_dtype(image_tf, tf.float32)

def detector(IMAGE_1_JPG):
  print("...")
  np.random.seed(10)
  tf.reset_default_graph()
  tf.logging.set_verbosity(tf.logging.FATAL)
  m = hub.Module('https://tfhub.dev/google/delf/1')
  image_placeholder = tf.placeholder(
      tf.float32, shape=(None, None, 3), name='input_image')
  module_inputs = {
      'image': image_placeholder,
      'score_threshold': 100.0,
      'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
      'max_feature_num': 1000,
  }
  module_outputs = m(module_inputs, as_dict=True)
  image_tf = image_input_fn([IMAGE_1_JPG])
  with tf.train.MonitoredSession() as sess:
    results_dict = {}  # Stores the locations and their descriptors for each image
    for image_path in [IMAGE_1_JPG]:
        image = sess.run(image_tf)
        print('Extracting locations and descriptors from %s' % image_path)
        results_dict[image_path] = sess.run(
            [module_outputs['locations'], module_outputs['descriptors']],
            feed_dict={image_placeholder: image})
  
  locations_1, descriptors_1 = results_dict[IMAGE_1_JPG]
  print("Loaded image 1's %d features" % locations_1.shape[0])
  return locations_1, descriptors_1
  pass