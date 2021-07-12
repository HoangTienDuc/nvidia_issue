from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

model_name = "retinaface_postprocess"
shape = [4]

with httpclient.InferenceServerClient("localhost:8000") as client:

    net_0 = np.random.rand(1, 4, 15, 20).astype('float32')
    net_1 = np.random.rand(1, 8, 15, 20).astype('float32')
    net_2 = np.random.rand(1, 20, 15, 20).astype('float32')
    net_3 = np.random.rand(1, 6, 15, 20).astype('float32')
    net_4 = np.random.rand(1, 4, 30, 40).astype('float32')
    net_5 = np.random.rand(1, 8, 30, 40).astype('float32')
    net_6 = np.random.rand(1, 20, 30, 40).astype('float32')
    net_7 = np.random.rand(1, 6, 30, 40).astype('float32')
    net_8 = np.random.rand(1, 4, 60, 80).astype('float32')
    net_9 = np.random.rand(1, 8, 60, 80).astype('float32')
    net_10 = np.random.rand(1, 20, 60, 80).astype('float32')
    net_11 = np.random.rand(1, 6, 60, 80).astype('float32')

    inputs = [
        httpclient.InferInput("face_rpn_cls_prob_reshape_stride32", net_0.shape,
                              np_to_triton_dtype(net_0.dtype)),
        httpclient.InferInput("face_rpn_bbox_pred_stride32", net_1.shape,
                              np_to_triton_dtype(net_1.dtype)),
        httpclient.InferInput("face_rpn_landmark_pred_stride32", net_2.shape,
                              np_to_triton_dtype(net_2.dtype)),
        httpclient.InferInput("face_rpn_type_prob_reshape_stride32", net_3.shape,
                              np_to_triton_dtype(net_3.dtype)),
        httpclient.InferInput("face_rpn_cls_prob_reshape_stride16", net_4.shape,
                              np_to_triton_dtype(net_4.dtype)),
        httpclient.InferInput("face_rpn_bbox_pred_stride16", net_5.shape,
                              np_to_triton_dtype(net_5.dtype)),
        httpclient.InferInput("face_rpn_landmark_pred_stride16", net_6.shape,
                              np_to_triton_dtype(net_6.dtype)),
        httpclient.InferInput("face_rpn_type_prob_reshape_stride16", net_7.shape,
                              np_to_triton_dtype(net_7.dtype)),
        httpclient.InferInput("face_rpn_cls_prob_reshape_stride8", net_8.shape,
                              np_to_triton_dtype(net_8.dtype)),
        httpclient.InferInput("face_rpn_bbox_pred_stride8", net_9.shape,
                              np_to_triton_dtype(net_9.dtype)),
        httpclient.InferInput("face_rpn_landmark_pred_stride8", net_10.shape,
                              np_to_triton_dtype(net_10.dtype)),
        httpclient.InferInput("face_rpn_type_prob_reshape_stride8", net_11.shape,
                              np_to_triton_dtype(net_11.dtype))
    
    ]

    inputs[0].set_data_from_numpy(net_0)
    inputs[1].set_data_from_numpy(net_1)
    inputs[2].set_data_from_numpy(net_2)
    inputs[3].set_data_from_numpy(net_3)
    inputs[4].set_data_from_numpy(net_4)
    inputs[5].set_data_from_numpy(net_5)
    inputs[6].set_data_from_numpy(net_6)
    inputs[7].set_data_from_numpy(net_7)
    inputs[8].set_data_from_numpy(net_8)
    inputs[9].set_data_from_numpy(net_9)
    inputs[10].set_data_from_numpy(net_10)
    inputs[11].set_data_from_numpy(net_11)

    outputs = [
        httpclient.InferRequestedOutput("det"),
        httpclient.InferRequestedOutput("landmarks"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    print(" OUTPUT0 ({})".format(response.as_numpy("det")))
    print("OUTPUT1 ({})".format(response.as_numpy("landmarks")))
