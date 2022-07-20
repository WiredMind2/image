import cv2
import numpy as np
import time

def long_exposure(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    img, img_gray = None, None
    mask = None

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        cv2.imshow('Frame', frame)

        if img is None:
            img = frame
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(img.shape, np.uint8)
            continue
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.subtract(img, frame)

        search = np.where(mask==0)
        mask[search] = diff[search]

        cv2.imshow('Diff', diff)
        # cv2.imshow('Mask', mask)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    video_path = 'story.mp4'
    out = 'output.png'
    long_exposure(video_path, out)