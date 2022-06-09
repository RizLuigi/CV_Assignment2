import os
import time
import cv2
import numpy as np

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class Tracker:
    def __init__(self, detector, detect_interval=30, track_len=10, video_src="material/Contesto_industriale1.mp4",
                 DEBUG=False, SAVE=False):
        self.prev_gray = None
        self.track_len = track_len
        self.detect_interval = detect_interval
        self.tracks = []
        self.tracks_colors = []
        self.cap = cv2.VideoCapture(video_src) if video_src is not None else cv2.VideoCapture(0)
        self.frame_idx = 0
        self.detector = detector
        self.DEBUG = DEBUG
        if SAVE:
            dirname = "{}_results".format(os.path.splitext(video_src)[0])
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            if DEBUG:
                filename = "{}/output_{}_DEBUG.mp4".format(dirname, detector.detector_type)
            else:
                filename = "{}/output_{}.mp4".format(dirname, detector.detector_type)
            ret, frame = self.cap.read()
            height, width, _ = frame.shape
            if height > 1000:
                scale_percent = int(100 / (int(height / 1000) + 1))
                frame = cv2.resize(frame, (int(width * scale_percent / 100), int(height * scale_percent / 100)),
                                   interpolation=cv2.INTER_AREA)
            height, width, _ = frame.shape
            self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        else:
            self.out = None

    def run(self):
        detect_num = 0
        detected_points = 0
        detection_time = 0.0
        track_num = 0
        detected_tracks = 0
        deleted_tracks = 0

        while True:
            ret, frame = self.cap.read()

            # If video end reached
            if not ret:
                break

            height, width, _ = frame.shape
            if height > 1000:
                scale_percent = int(100 / (int(height / 1000) + 1))
                frame = cv2.resize(frame, (int(width * scale_percent / 100), int(height * scale_percent / 100)),
                                   interpolation=cv2.INTER_AREA)

            # Copy frame to draw features on top of it and convert to grayscale
            frame_copy = frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                # Extract last point of each track
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # Evaluate optical flow forwards and backwards
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                # Evaluate distance between initial points and back-projected ones
                d = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
                # Generate mask for good points
                good = d < 1
                # Generate new tracks by extending previous tracks whose last point is a good point
                new_tracks = []
                index = 0
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        del self.tracks_colors[index]
                        deleted_tracks += 1
                        continue
                    tr.append((x, y))
                    # Delete oldest point of the track
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    color = (int(self.tracks_colors[index][0]),
                             int(self.tracks_colors[index][1]),
                             int(self.tracks_colors[index][2]))
                    cv2.circle(frame_copy, (int(x), int(y)), 2, color, -1)
                    cv2.polylines(frame_copy, [np.int32(tr)], False, color)
                    index += 1
                self.tracks = new_tracks
                track_num += 1
                detected_tracks += len(self.tracks)

            if self.frame_idx % self.detect_interval == 0:
                # Mask will be used to obscure areas where there are already some keypoints
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 20, 0, -1)
                # Apply detector
                start = time.time()
                p = self.detector.detect(frame_gray, mask=mask)
                end = time.time()
                detection_time += end - start
                detect_num += 1
                detected_points += len(p)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                        self.tracks_colors.append(np.random.choice(range(256), size=3))

            self.frame_idx += 1
            self.prev_gray = frame_gray

            text = 'Found {} tracks'.format(len(self.tracks))
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2
            if self.DEBUG:
                cv2.imshow('{} Tracks'.format(self.detector.detector_type),
                           cv2.putText(frame_copy, text, org, font, fontScale, color, thickness, cv2.LINE_AA))
                if self.out is not None:
                    self.out.write(cv2.putText(frame_copy, text, org, font, fontScale, color, thickness,
                                               cv2.LINE_AA).astype('uint8'))
            else:
                cv2.imshow('{} Tracks'.format(self.detector.detector_type), frame_copy)
                if self.out is not None:
                    self.out.write(frame_copy.astype('uint8'))

            if cv2.waitKey(1) == ord('q') or not ret:
                break

        if self.DEBUG:
            print('\tAverage number of detected keypoints:      \t {:.4f}'.format(detected_points / detect_num))
            print('\tAverage number of tracks per frame:        \t {:.4f}'.format(detected_tracks / track_num))
            print('\tAverage number of deleted tracks per frame:\t {:.4f}'.format(deleted_tracks / track_num))
            print('\tAverage time for a detection step:         \t {:.4f} sec.'.format(detection_time / detect_num))

        self.cap.release()
        if self.out is not None:
            self.out.release()
