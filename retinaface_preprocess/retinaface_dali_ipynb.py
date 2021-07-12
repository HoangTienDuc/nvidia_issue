from nvidia.dali.pipeline import Pipeline
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

image_dir = "data/images"
max_batch_size = 8

import math
@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    
    #decode image to cpu
    images = fn.decoders.image(jpegs, device='mixed', output_type = types.RGB)
    
    #resize image keep ratio
    images = fn.resize(images, resize_x=640, resize_y=0)
    
    #padding image
    images = fn.pad(images, axis_names="HW", shape=[480, 640])
    
    ##convert dimension WHC->CHW
    #images = fn.transpose(images,
    #                    perm=[2, 0, 1])

    return images, labels

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
%matplotlib inline

# to show image, using jupyter and convert image to cpu
def show_images(image_batch):
    columns = 4
    rows = (max_batch_size + 1) // (columns)
    fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.as_cpu().at(j))

pipe = simple_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
pipe.build()
pipe_out = pipe.run()
print(pipe_out)

images, labels = pipe_out
print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))

show_images(images)

type(images)
#convert to cpu .as_cpu()
image = images.as_cpu().at(0)

import numpy as np
print(np.array(image).shape)

