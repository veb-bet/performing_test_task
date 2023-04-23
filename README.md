# performing_test_task

A program (console application) that allows you to search for certain objects on video and save videos with them in a separate file. 
We will use yolo (https://github.com/ultralytics/yolov5 ) for object search and OpenCV for video manipulation.

Requirements:

Input:
path to the source video file
name of the object class (one of the classes supported by YOLO or another selected model)

Output:
a video file collected from the frames on which the found object is cut.

Examples of input and output data (object class “cat"):
https://drive.google.com/file/d/1RiudS0mFTDlzfS8L-Qhl_j2eD_dpljM_/view
https://drive.google.com/file/d/1RMAhyMKcagNeN_p9O_-zAmvh4Ovcw0Vf/view
