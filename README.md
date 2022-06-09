# Feature Detection and Tracking in OpenCV

## Requirements
The whole project has been developed using `Python 3.10` and the following libraries that can be installed using the command `pip install *package_name*`:
* `opencv-python`
* `opencv-contrib-python`
* `numpy`

## Materials
The `material` folder contains some example videos together with the mandatory video provided for evaluation.

## Running the code
Run `python DetectAndTrack.py -h` to display the help message, showing the available options and flags to customize the run.
For completeness, the message is also reported here:

`usage: DetectAndTrack.py [-h] [-s VIDEO_SOURCE] [-d {ORB,SIFT,STAR,FAST}] [-i DETECT_INTERVAL] [-t TRACK_LEN] [--DEBUG] [--SAVE]`

Feature tracker based on Lukas-Kanade optical flow and different feature detectors

Options:
* `-h, --help`: show this help message and exit
* `-s VIDEO_SOURCE, --video_source VIDEO_SOURCE`: The path to the video to be processed. Defaults to `material/Contesto_industriale1.mp4`. Use `webcam` to use te webcam as source (Q to quit)
* `-d {ORB,SIFT,STAR,FAST}, --detector {ORB,SIFT,STAR,FAST}`: The type of feature detector to use. If no detector is passed, all the four detectors are used, one after the other
* `-i DETECT_INTERVAL, --detect_interval DETECT_INTERVAL`: The frequency of keypoint detection (in frames). Defaults to `30`
* `-t TRACK_LEN, --track_len TRACK_LEN`: The maximal length of each track. Defaults to `10`
* `--DEBUG`: Show debugging windows and info
* `--SAVE`: Save the output file in `filename/output_<DETECTOR_TYPE>.mp4`