import os, sys
import numpy as np
import json
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import argparse
import time
import cv2


class Arcface_Preprocess:
    def __init__(self, url, model_name):
        self.url = url
        self.model_name = model_name
        self.input_name = "ARCFACE_INPUT_0"
        self.output_name = "ARCFACE_OUTPUT_0"
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)
        except Exception as e:
            print("channel creation failed: " + str(e))

    def excecute(self, batch):
        inputs = []
        outputs = []

        inputs.append(grpcclient.InferInput(self.input_name, batch.shape, "UINT8"))

        outputs.append(grpcclient.InferRequestedOutput(self.output_name))

        inputs[0].set_data_from_numpy(batch)

        results = self.triton_client.infer(model_name,
                                inputs,
                                request_id=str(1),
                                outputs=outputs)
        return results.as_numpy(self.output_name)


path = "/Storages/ducht/face/FR_demo/data/crop/duc/1616988229.132644.jpg"
url = "localhost:8001"
model_name = "arcface_preprocess"


image = cv2.imread(path)
image_data = np.expand_dims(image, axis=0)
tic = time.time()
preprocessor = Arcface_Preprocess(url, model_name)
preprocessed_image = preprocessor.excecute(image_data)
toc = time.time()
print("processing time: ", toc - tic)