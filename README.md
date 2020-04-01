# computer_vision

Applications utilizing the Open Source Computer Vision library (OpenCV) in combination with Intel's OpenVino toolkit to harness the power of artificial intelligence at the edge in IoT devices.  Apps include classification of vehicles from video, classification while streaming, and more.  

## Computer Vision Models

* Classification: Determines the class that an image or object in an image belong to.  Yes/no used to determined highest probability class.
* Detection: Determines objects within image, typically with bounding boxes and classification. User typically has ability to set confidence threshold for bounding boxes.
* Segmentation: classification by section - semantic (all instances of a class are considered one) and instance (separate instances of a class as separate objects)


## Steps for SSD for Vechicles

* Convert a bounding box model to an IR with the Model Optimizer.
* Pre-process the model as necessary.
* Use an async request to perform inference on each video frame.
* Extract the results from the inference request.
* Make the requests and feed back the results within the application.
* Perform any necessary post-processing steps to get the bounding boxes.
* Add a command line argument to allow for different confidence thresholds for the model.
* Add a command line argument to allow for different bounding box colors for the output.

python app.py -m frozen_inference_graph.xml -ct 0.6 -c BLUE

### Downlad and convert model

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

## Server Communication

app.py adds code for MQTT so that the node server receives the calculated stats, which includes: 
* Sets IP address and port
* Connects to the MQTT client
* Publishes the calculated statistics to the client
* Sends the output frame (not the input image, but the processed output) to the ffserver

Get the MQTT broker installed and running
cd webservice/server/node-server
node ./server.js
You should see a message that Mosca server started..
Get the UI Node Server running.
cd webservice/ui
npm run dev
After a few seconds, you should see webpack: Compiled successfully.
Start the ffserver
sudo ffserver -f ./ffmpeg/server.conf
Start the actual application.
* source the environment for OpenVINO in the new terminal:
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
-video_size 1280x720
-i - http://0.0.0.0:3004/fac.ffm


## Processing Model Outputs

python3 app.py -m model.xml

Scenario: you have a cat and two dogs at your house.

* If both dogs are in a room together, they are best buds, and everything is going well.
* If the cat and dog #1 are in a room together, they are also good friends, and everything is fine.
* However, if the cat and dog #2 are in a room together, they don't get along, and you may need to either pull them apart, or at least play a pre-recorded message from your smart speaker to tell them to cut it out.

In the input video, some combination or the cat and dogs may be in view. There is also an IR that is able to determine which of these, if any, are on screen.

While the best model for this is likely an object detection model that can identify different breeds, this model provides the following detections: one for one or less pets on screen, one for the bad combination of the cat and dog #2, and one for the fine combination of the cat and dog #1. This is model.xml in the same directory.

The code prints to the terminal anytime the bad combination of the cat and dog #2 are detected together.

