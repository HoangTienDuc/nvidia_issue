import nvidia.dali as dali
import nvidia.dali.types as types
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="./models/arcface_preprocess/1/model.dali")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    
    pipe = dali.pipeline.Pipeline(batch_size=256, num_threads=4, device_id=0)
    with pipe:
        images = dali.fn.external_source(device="gpu", name="ARCFACE_INPUT_0")
        images = dali.fn.cast(images, dtype=types.FLOAT)
        images = dali.fn.transpose(images, perm=[2, 0, 1])
        pipe.set_outputs(images)
        pipe.serialize(filename=args.save)

    print("Saved {}".format(args.save))