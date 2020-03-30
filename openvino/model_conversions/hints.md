# TensorFlow Model Conversion
How to convert a TensorFlow Model from the Object Detection Model Zoo into an Intermediate Representation using the Model Optimizer.

As noted in the related documentation, there is a difference in method when using a frozen graph vs. an unfrozen graph. Since freezing a graph is a TensorFlow-based function and not one specific to OpenVINO itself, in this exercise, you will only need to work with a frozen graph. However, I encourage you to try to freeze and load an unfrozen model on your own as well.

For this exercise, first download the SSD MobileNet V2 COCO model from here. Use the tar -xvf command with the downloaded file to unpack it.

From there, find the Convert a TensorFlow* Model header in the documentation, and feed in the downloaded SSD MobileNet V2 COCO model's .pb file.

If the conversion is successful, the terminal should let you know that it generated an IR model. The locations of the .xml and .bin files, as well as execution time of the Model Optimizer, will also be output.

Note: Converting the TF model will take a little over one minute in the workspace.

Hints & Troubleshooting
Make sure to pay attention to the note in this section regarding the --reverse_input_channels argument. If you are unsure about this argument, you can read more here.

There is additional documentation specific to converting models from TensorFlow's Object Detection Zoo here. You will likely need both the --tensorflow_use_custom_operations_config and --tensorflow_object_detection_api_pipeline_config arguments fed with their related files.


Note: We are using the 2019R3 version of the OpenVINO™ Toolkit in the classroom. In 2020R1 (and likely future updates), --tensorflow_use_custom_operations_config was re-named to --transformations_config. As such, if you are working locally with the latest version of the toolkit, you’ll need to make sure to use the updated argument naming.

Here's what I entered to convert the SSD MobileNet V2 model from TensorFlow:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
This is pretty long! I would suggest considering setting a path environment variable for the Model Optimizer if you are working locally on a Linux-based machine. You could do something like this:

export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer
And then when you need to use it, you can utilize it with $MOD_OPT/mo.py instead of entering the full long path each time. In this case, that would also help shorten the path to the ssd_v2_support.json file used.