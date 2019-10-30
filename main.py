import argparse
import logging
import os
import sys

import cv2
import numpy as np


def blend(img, overlay):
    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_img = 1.0 - alpha_overlay

    result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for c in range(0, 3):
        result[:, :, c] = (alpha_img * img[:, :, c] + alpha_overlay * overlay[:, :, c])

    return result


def correct(img, points):
    assert len(points) == 4

    h = img.shape[0]
    w = img.shape[1]
    print(h, w)

    src = np.float32(points)
    dst = np.float32([(0, 0), (w, 0), (w, h), (0, h)])

    matrix = cv2.getPerspectiveTransform(src, dst)

    wrapped = cv2.warpPerspective(img, matrix, (w, h))
    cv2.imwrite('wrapped.jpg', wrapped)

    lab = cv2.cvtColor(wrapped, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    cl = clahe.apply(l)

    tr = 255 - cv2.threshold(255 - cl, 70, 255, cv2.THRESH_TOZERO)[1]
    cl = cv2.addWeighted(cl, 0.75, tr, 0.25, 0)

    cl = ((cl / 255) ** 2) * 255  # enhance contrast
    return cl


index = 0


def mouse_callback(event, x, y, flags, param):
    del flags # unused

    if event == cv2.EVENT_LBUTTONDOWN:
        if not 'points' in param[index]:
            param[index]['points'] = []

        if len(param[index]['points']) >= 4:
            return

        param[index]['points'].append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if 'points' in param[index]:
            param[index]['points'] = []

def main():
    parser = argparse.ArgumentParser(
        description="Show image or images in a folder")
    parser.add_argument("files", nargs='+')
    args = parser.parse_args()

    images = []
    for path in args.files:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                images += [{'path': os.path.join(root, file)} for file in files]
        elif os.path.isfile(path):
            images[path] = {'path': path}
        else:
            logging.warning('Object %s is not a direcotry, nor a file!', path)

    if len(images) == 0:
        logging.error('No files specified!')
        sys.exit(1)

    cv2.namedWindow('App', flags=cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('App', mouse_callback, param=images)

    frame = None
    overlay = None

    global index

    # get window property is there to detect window close, once it is closed
    # in will return -1, it doesn't matter which flag we will use
    while cv2.getWindowProperty('App', 0) >= 0:
        print(index)
        if frame is None:
            frame_rgb = cv2.imread(images[index]['path'])
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)

        overlay = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
        if 'points' in images[index]:
            print(images[index]['points'])
            for point in images[index]['points']:
                cv2.circle(overlay, point, 50, (0, 255, 0, 255), 50)

        cv2.imshow('App', blend(frame, overlay))

        key = cv2.waitKeyEx(delay=40)
        if key == 27:
            break  # ESC
        elif key == 13:
            # enter
            # now we need to correct all images
            for (k, v) in enumerate(images):
                if 'points' in v and len(v['points']) == 4:
                    img = cv2.imread(v['path'])
                    corrected = correct(img, v['points'])
                    cv2.imwrite('.\\{:04d}.jpg'.format(k), corrected)
            break
        elif (key == 2424832) or (key & 0xff == ord('[')):
            # left arrow key or '[' pressed
            index = max(index - 1, 0)
            frame = None
        elif (key == 2555904) or (key & 0xff == ord(']')):
            # right arrow key or ']' pressed
            index = min(index + 1, len(images) - 1)
            frame = None

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
