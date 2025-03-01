import os
import subprocess
from tkinter import filedialog

class ImageEditor:
    def __init__(self, path) -> None:
        self.media_path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        self.cmd_args = ['-hide_banner', '-loglevel', 'error', '-y']

    def convert_image(self, path=None, new_ext=None):
        # Convert 'self.media_path' img to jpg 'path' img
        if path is None:
            path, ext = os.path.splitext(self.media_path)

            if new_ext is None:
                new_ext = '.jpg'

            path = path + new_ext
        else:
            path = os.path.abspath(path)

        subprocess.call(['ffmpeg', *self.cmd_args, '-i', self.media_path, path])
        self.media_path = path
        return self.media_path

class VideoEditor:
    def __init__(self, path) -> None:
        self.media_path = path
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        self.cmd_args = ['-hide_banner', '-loglevel', 'error', '-y']

    def save_frame(self, timestamp=None, ext=None, path=None):
        if timestamp is None:
            timestamp = 0
        else:
            timestamp = int(timestamp)

        if path is None:
            path, old_ext = os.path.splitext(self.media_path)

            if ext is None:
                ext = '.jpg'
            elif not ext.startswith('.'):
                ext = '.' + ext

            path = path + ext

        subprocess.call(['ffmpeg', *self.cmd_args, '-ss', str(timestamp), '-i', self.media_path, '-vframes', '1', path])

if __name__ == '__main__':
    # path = input('File path: ')
    path = filedialog.askopenfilename(filetypes=[('Video', ('.mp4', '.mov', '.avi'))])
    editor = VideoEditor(path)
    editor.save_frame(2, ext='jpg')