import argparse
import cv2
from inference import Network

INPUT_STREAM = "test_video.mp4"
# CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TO CHECK: Add additional arguments and descriptions for:
    conf_desc = "The confidence thresholds used to draw bounding boxes"
    color_desc = "The color of the bounding boxes: 'R', 'G', or 'B'"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-conf", help=conf_desc, default='0.6')
    optional.add_argument("-color", help=color_desc, default='G')
    args = parser.parse_args()

    return args

def get_color(args):
    if args == 'B':
        return (0, 0, 255)
    elif args == 'G':
        return (0, 255, 0)
    elif args == 'R':
        return (255, 0, 0)
    else: return(1)


def infer_on_video(args):
    ### Initialize the Inference Engine
    plugin = Network()

    ### Load the network model into the IE
    plugin.load_model(args.m, args.d)
    input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, `0x00000021` on Udacity IDE, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 'cv2.VideoWriter_fourcc("M","J","P","G")', 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### Pre-process the frame
        image = cv2.resize(frame, (input_shape[3], input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, 3, input_shape[2], input_shape[3])

        ### Perform inference on the frame
        plugin.async_inference(image)

        ### Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
        ### Update the frame to include detected bounding boxes
            for box in result[0][0]:
                if box[2] >= float(args.conf):
                    x_min = int(box[3] * width)
                    y_min = int(box[4] * height)
                    x_max = int(box[5] * width)
                    y_max = int(box[6] * height)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), get_color(args.color), 1)
                    # Write out the frame
                    out.write(frame)
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
