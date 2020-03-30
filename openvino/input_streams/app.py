import argparse
import cv2
import numpy as np

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Handle an input stream")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc)
    args = parser.parse_args()

    return args


def capture_stream(args):
    ### Handle image, video or webcam
    isImage = False
    if args.i == 'WEB-CAM':
        args.i = 0
    elif args.i.split('.')[1] == 'jpg' or args.i.split('.')[1] == 'bmp':
        isImage = True

    ### Get and open video capture
    video_capture = cv2.VideoCapture(args.i)
    video_capture.open(args.i)
    
    if not isImage: 
        output = cv2.VideoWriter('output.mp4', 0x00000021, 30, (100,100))
    else: 
        output = None
    ### Process frames until end
    while video_capture.isOpened():
        flag, frame = video_capture.read()
        if not flag:
            break
        key = cv2.waitKey(60)
        
        ### Re-size the frame to 100x100
        frame = cv2.resize(frame, (100, 100))
        
        ### Add Canny Edge Detection to the frame, 
        ### with min & max values of 100 and 200
        
        frame = cv2.Canny(frame, 100, 200)
        
        ### make a 3-channel image
        frame = np.dstack((frame, frame, frame))
        
        ### write out frame for video or image
        if isImage:
            cv2.imwrite('output.jpg', frame)
            
        else: 
            output.write(frame)
        if key == 27:
            break
    if not isImage:
        
        ### Close the stream and any windows at the end of the application
        output.release()
        video_capture.release
        cv2.destroyAllWindows

def main():
    args = get_args()
    capture_stream(args)


if __name__ == "__main__":
    main()
