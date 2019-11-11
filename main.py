""" Simple script for correcting images (mainly slides) and converting to pdf."""

import argparse
import logging
import os
import sys
import tempfile
import shutil

import cv2
import fpdf
import numpy as np


class Settings:
    """Class encapsulating all cli parameters and more."""

    def __init__(self):
        parser = argparse.ArgumentParser(description='Correct image or images and create pdf.')
        parser.add_argument('--output', default='.\\merged.pdf', help='output file location')
        parser.add_argument('--keep-images', action='store_true', help='if selected, intermediate images kept on disk')
        parser.add_argument('files', nargs='+', help='list of files and/or directories to process')
        args = parser.parse_args()

        self.output = args.output
        self.files = self._list_files(args.files)
        self.keep_images = args.keep_images
        self.image_dir = tempfile.mkdtemp('slides-to-pdf')
        # 4:3 aspect ratio with plenty of resolution
        self.image_width = 1632
        self.image_height = 1224

    def _list_files(self, paths):
        """Takes a list of paths and enumerate all files within them, recursively"""
        result = []
        for path in paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    result += [os.path.join(root, file) for file in files]
            elif os.path.isfile(path):
                result.append(path)
            else:
                logging.warning('Object %s is not a directory, nor a file!', path)
        return result


SETTINGS = Settings()


def blend(img, overlay):
    """Takes two OpenCV images and merge them together, it will honor alpha."""
    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_img = 1.0 - alpha_overlay

    result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for channel in range(3):
        result[:, :, channel] = alpha_img * img[:, :, channel] + alpha_overlay * overlay[:, :, channel]

    return result


def correct(img, points):
    """Takes image and four points, it will correct the perspective and enhance contrast."""
    assert len(points) == 4

    height = img.shape[0]
    width = img.shape[1]

    src = np.float32(points)
    dst = np.float32([(0, 0), (width, 0), (width, height), (0, height)])

    matrix = cv2.getPerspectiveTransform(src, dst)
    wrapped = cv2.warpPerspective(img, matrix, (width, height))

    lab = cv2.cvtColor(wrapped, cv2.COLOR_BGR2LAB)
    lightness, _, _ = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(lightness)

    thresholded = 255 - cv2.threshold(255 - enhanced, 70, 255, cv2.THRESH_TOZERO)[1]
    enhanced = cv2.addWeighted(enhanced, 0.75, thresholded, 0.25, 0)

    enhanced = ((enhanced / 255) ** 2) * 255  # enhance contrast
    return enhanced


def process_images(images):
    """Go through all images, correct them and save to pdf."""
    paths = []

    for (k, val) in enumerate(images):
        if 'points' in val and len(val['points']) == 4:
            img = cv2.imread(val['path'])
            corrected = correct(img, val['points'])
            paths.append(os.path.join(SETTINGS.image_dir, '{:04d}.jpg'.format(k)))
            cv2.imwrite(paths[-1], corrected)

    pdf = fpdf.FPDF('L', 'pt', (SETTINGS.image_height, SETTINGS.image_width))

    for path in paths:
        pdf.add_page()
        pdf.image(path, 0, 0, SETTINGS.image_width, SETTINGS.image_height)

    pdf.output(SETTINGS.output, "F")


def mouse_callback(event, x, y, flags, param):
    """Callback function for OpenCV window mouse event."""
    del flags  # unused

    index = param['index']
    images = param['images']

    if event == cv2.EVENT_LBUTTONDOWN:
        if not 'points' in images[index]:
            images[index]['points'] = []

        if len(images[index]['points']) >= 4:
            return

        images[index]['points'].append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if 'points' in images[index]:
            images[index]['points'] = []


def print_guide():
    """Print the usage of the gui"""
    print("""
Images are open in window, you can scroll through them by using arrow keys, or
alternatively, by using "[" for left and "]" for right. Use your mouse to
select four points where corners of the slides are. Start with top left and
follow in clockwise order. It is crucial to use this order, otherwise the
perspective correction will not work. If you made a mistake, just press right
mouse button, it will clear the selection and you can select the points one
more time. When you processed all images just hit Enter and the processing will
start.

If you wish to exit the application at any time you can press ESC.""")


def keep_running():
    """Return true if the application window is not closed. It is not very
    accurate, since we depend on something not completely related, but it is
    the best we have in OpenCV alone.
    """
    # get window property is there to detect window close, once it is closed
    # in will return -1, it doesn't matter which flag we will use
    return cv2.getWindowProperty('App', 0) >= 0


def main():
    """Main entry point. It will establish the gui and govern the process of images correction."""
    images = [{'path': path} for path in SETTINGS.files]

    if len(images) == 0:
        logging.error('No files specified!')
        sys.exit(1)

    data = {'images': images, 'index': 0}

    cv2.namedWindow('App', flags=cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('App', mouse_callback, param=data)

    frame = None
    overlay = None

    print_guide()

    while keep_running():
        index = data['index']
        images = data['images']

        if frame is None:
            frame_rgb = cv2.imread(images[index]['path'])
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)

        overlay = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
        if 'points' in images[index]:
            for point in images[index]['points']:
                cv2.circle(overlay, point, 50, (0, 255, 0, 255), 50)

        cv2.imshow('App', blend(frame, overlay))

        key = cv2.waitKeyEx(delay=40)
        if key == 27:
            break  # ESC
        elif key == 13:
            # enter, now we need to correct all images
            process_images(images)
            break
        elif (key == 2424832) or (key & 0xff == ord('[')):
            # left arrow key or '[' pressed
            data['index'] = max(data['index'] - 1, 0)
            frame = None
        elif (key == 2555904) or (key & 0xff == ord(']')):
            # right arrow key or ']' pressed
            data['index'] = min(data['index'] + 1, len(images) - 1)
            frame = None

    cv2.destroyAllWindows()

    if not SETTINGS.keep_images:
        shutil.rmtree(SETTINGS.image_dir)
    else:
        print(f'Images are available at { SETTINGS.image_dir }.')


if __name__ == '__main__':
    main()
