import argparse
import time
import cv2
from Detector import Detector, AVAILABLE_DETECTORS
from Tracker import Tracker


def main(video_src, detector_type, detect_interval, track_len, DEBUG=False, SAVE=False):
    if DEBUG:
        print('Feature detector used:\t\t{}'.format(detector_type))
        start = time.time()
    detector = Detector(detector_type=detector_type, DEBUG=DEBUG)
    Tracker(detector=detector, video_src=video_src, detect_interval=detect_interval, track_len=track_len,
            DEBUG=DEBUG, SAVE=SAVE).run()
    if DEBUG:
        end = time.time()
        print('\tOverall execution time:\t\t\t{:.4f} sec.'.format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature tracker based on Lukas-Kanade optical flow and different"
                                                 " feature detectors")

    parser.add_argument('-s', '--video_source', type=str, help='The path to the video to be processed. '
                                                               'Defaults to `material/Contesto_industriale1.mp4`. '
                                                               'Use \'webcam\' to use te webcam as source (Q to quit)',
                        default='material/Contesto_industriale1.mp4')
    parser.add_argument('-d', '--detector', type=str, choices={'SIFT', 'FAST', 'ORB', 'STAR'},
                        help='The type of feature detector to use. If no detector is passed, all the four detectors '
                             'are used, one after the other')
    parser.add_argument('-i', '--detect_interval', type=int, help='The frequency of keypoint detection (in frames).'
                                                                  ' Defaults to 30',
                        default=30)
    parser.add_argument('-t', '--track_len', type=int, help='The maximal length of each track. Defaults to 10',
                        default=10)
    parser.add_argument('--DEBUG', action='store_true', help='Show debugging windows and info')
    parser.add_argument('--SAVE', action='store_true', help='Save the output file in '
                                                            '\'filename/output_<DETECTOR_TYPE>.mp4\'')
    args = parser.parse_args()

    if args.video_source == 'webcam':
        args.video_source = None

    if args.detector is None:
        for detector_type in AVAILABLE_DETECTORS.keys():
            main(video_src=args.video_source, detector_type=detector_type, detect_interval=args.detect_interval,
                 track_len=args.track_len, DEBUG=args.DEBUG, SAVE=args.SAVE)
            cv2.destroyAllWindows()
    else:
        main(video_src=args.video_source, detector_type=args.detector, detect_interval=args.detect_interval,
             track_len=args.track_len, DEBUG=args.DEBUG, SAVE=args.SAVE)
        cv2.destroyAllWindows()
