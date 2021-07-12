import nvidia.dali as dali
import nvidia.dali.types as types
import os
import argparse
from nvidia.dali.math import min

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="./1/model.dali")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    
    width = 640
    height = 480
    pipe = dali.pipeline.Pipeline(batch_size=256, num_threads=4, device_id=0)
    with pipe:
        origin_images = dali.fn.external_source(device="gpu", name="DALI_INPUT_0")
        # origin_images = dali.fn.color_space_conversion(origin_images, image_type=types.RGBA, output_type=types.RGB)
        # skip to decode image from file
        # images = dali.fn.image_decoder(images, device="cpu", output_type=types.RGB)
        # origin_images = dali.fn.cast(input_images, dtype=dali.types.FLOAT)  # shape is HWC
        shapes = dali.fn.shapes(origin_images, dtype=types.FLOAT)
        origin_w = dali.fn.slice(shapes, 1, 1, axes=[0])  # extract width
        origin_h = dali.fn.slice(shapes, 0, 1, axes=[0])  # extract height
        scale_factors = min(width / origin_w, height / origin_h)
        # scale_factors = dali.fn.cat(width / origin_w, height / origin_h, axis=0)
        scale_factors = dali.fn.expand_dims(scale_factors, axes=1)

        images = dali.fn.resize(origin_images, size=[height, width], mode='not_larger')
        # images = dali.fn.crop(images, crop_w=width, crop_h=height, out_of_bounds_policy="pad")
        images = dali.fn.pad(images, axis_names="HW", shape=dali.fn.cat(480, 640, axis=0))
        images = dali.fn.cast(images, dtype=types.FLOAT)
        images = dali.fn.transpose(images, perm=[2, 0, 1])
        pipe.set_outputs(images, scale_factors, origin_images)
        pipe.serialize(filename=args.save)

    print("Saved {}".format(args.save))
