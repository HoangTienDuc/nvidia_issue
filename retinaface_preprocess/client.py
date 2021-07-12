import os, sys
import numpy as np
import json
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import argparse
import time
import cv2


# if send image file without decode by opencv. decommand below and change dimension of config.pbtxt input shape to [-1]
# def load_image(img_path: str):
#     """
#     Loads an encoded image as an array of bytes.

#     This is a typical approach you'd like to use in DALI backend.
#     DALI performs image decoding, therefore this way the processing
#     can be fully offloaded to the GPU.
#     """
#     return np.fromfile(img_path, dtype='uint8')
# image_data = load_image(path)
# image_data = np.expand_dims(image_data, axis=0)


class Preprocess:
    def __init__(self, url, model_name):
        self.url = url
        self.model_name = model_name
        self.input_name = "DALI_INPUT_0"
        self.output_names = ["DALI_OUTPUT_0", "DALI_OUTPUT_1", "Origin_images"]
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)
        except Exception as e:
            print("channel creation failed: " + str(e))

    def excecute(self, batch):
        inputs = []
        outputs = []

        
        inputs.append(grpcclient.InferInput(self.input_name, image_data.shape, "UINT8"))
        
        for output_name in self.output_names:
            outputs.append(grpcclient.InferRequestedOutput(output_name))
        inputs[0].set_data_from_numpy(image_data)

        # 
        results = self.triton_client.infer(model_name,
                                inputs,
                                outputs=outputs)
        
        return results.as_numpy(self.output_names[0]), results.as_numpy(self.output_names[1]), results.as_numpy(self.output_names[2])


path = "2.jpg"
url = "localhost:8001"
model_name = "retinaface_preprocess"



image = cv2.imread(path)

tic = time.time()
image_data = np.expand_dims(image, axis=0)
preprocessor = Preprocess(url, model_name)
for i in range(1):
    preprocessed_image, scale_factor, origin_image = preprocessor.excecute(image_data)

toc = time.time()

print("scale_factor: ", scale_factor)

transposed_image = np.transpose(preprocessed_image[0], (1, 2, 0))
# print("results: ", scale_factor)
# print("preprocessed_image.shape: ", preprocessed_image.shape)
# print("results: ", preprocessed_image[0].dtype)
# print(transposed_image)
cv2.imwrite('test/processed.jpg', transposed_image)
