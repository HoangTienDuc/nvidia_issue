# This branch is for Error in verifyHeader [issue](https://forums.developer.nvidia.com/t/serialization-error-in-verifyheader-0-magic-tag-does-not-match/182781)

## Reproduce
Step 1: Create dali and tensorrt models
- In this repo, there are several model were create and optimize by dali and tensorrt. Ex: retinaface preprocess model was created by retinaface_preprocess/serialize.py
- All models are successfully run on triton 21.04-py3

Step 2: Set up deepstream config file for detect object
This step aim to run detection by using "ensemble_retinaface" models.
- First check "Are preprocess model run correct?". So deepstream config file was modified as './dstest_ssd_nopostprocess.txt' file.
        These was an error in this step 
```
E0705 08:34:25.996654 18559 logging.cc:43] coreReadArchive.cpp (32) - Serialization Error in verifyHeader: 0 (Magic tag does not match)
E0705 08:34:25.996738 18559 logging.cc:43] INVALID_STATE: std::exception
E0705 08:34:25.996744 18559 logging.cc:43] INVALID_CONFIG: Deserialize the cuda engine failed.
W0705 08:34:25.996751 18559 autofill.cc:225] Autofiller failed to detect the platform for retinaface_preprocess (verify contents of model directory or use --log-verbose=1 for more details)
W0705 08:34:25.996756 18559 autofill.cc:248] Proceeding with simple config for now
I0705 08:34:25.997093 18559 model_repository_manager.cc:810] loading: retinaface_preprocess:1
E0705 08:34:26.007827 18559 model_repository_manager.cc:986] failed to load 'retinaface_preprocess' version 1: Not found: unable to load backend library: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: cannot allocate memory in static TLS block
ERROR: infer_trtis_server.cpp:1044 Triton: failed to load model retinaface_preprocess, triton_err_str:Invalid argument, err_msg:load failed for model 'retinaface_preprocess': version 1: Not found: unable to load backend library: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: cannot allocate memory in static TLS block;

ERROR: infer_trtis_backend.cpp:45 failed to load model: retinaface_preprocess, nvinfer error:NVDSINFER_TRTIS_ERROR
ERROR: infer_trtis_backend.cpp:184 failed to initialize backend while ensuring model:retinaface_preprocess ready, nvinfer error:NVDSINFER_TRTIS_ERROR
0:00:09.895479387 18559      0x408dd20 ERROR          nvinferserver gstnvinferserver.cpp:362:gst_nvinfer_server_logger:<primary-inference> nvinferserver[UID 5]: Error in createNNBackend() <infer_trtis_context.cpp:246> [UID = 5]: failed to initialize trtis backend for model:retinaface_preprocess, nvinfer error:NVDSINFER_TRTIS_ERROR
I0705 08:34:26.008037 18559 server.cc:280] Waiting for in-flight requests to complete.
I0705 08:34:26.008055 18559 server.cc:295] Timeout 30: Found 0 live models and 0 in-flight non-inference requests
0:00:09.895599365 18559      0x408dd20 ERROR          nvinferserver gstnvinferserver.cpp:362:gst_nvinfer_server_logger:<primary-inference> nvinferserver[UID 5]: Error in initialize() <infer_base_context.cpp:81> [UID = 5]: create nn-backend failed, check config file settings, nvinfer error:NVDSINFER_TRTIS_ERROR
0:00:09.895611735 18559      0x408dd20 WARN           nvinferserver gstnvinferserver_impl.cpp:439:start:<primary-inference> error: Failed to initialize InferTrtIsContext
0:00:09.895617385 18559      0x408dd20 WARN           nvinferserver gstnvinferserver_impl.cpp:439:start:<primary-inference> error: Config file path: /data/deepstream-retinaface/dstest_ssd_nopostprocess.txt
0:00:09.895959470 18559      0x408dd20 WARN           nvinferserver gstnvinferserver.cpp:460:gst_nvinfer_server_start:<primary-inference> error: gstnvinferserver_impl start failed
Error: gst-resource-error-quark: Failed to initialize InferTrtIsContext (1): gstnvinferserver_impl.cpp(439): start (): /GstPipeline:pipeline0/GstNvInferServer:primary-inference:
Config file path: /data/deepstream-retinaface/dstest_ssd_nopostprocess.txt
```