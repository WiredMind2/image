import json
import os
import subprocess
import sys
from tkinter.filedialog import askdirectory

MAX_WIDTH = 1280
MAX_HEIGHT = 720

ALLOWED_CODEC = ['h264']

def get_video_info(path):
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
        '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,bit_rate,width,height', path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = json.loads(result.stdout)
    stream_info = info['streams'][0]
    codec_name = stream_info['codec_name']
    bit_rate = stream_info['bit_rate']
    width = stream_info['width']
    height = stream_info['height']

    return codec_name, bit_rate, width, height

def resize_video(path):
    if False:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        if err != b'':
            print(f'Error getting video dimensions for {path}: {err}')
            failed_path = os.path.join(os.path.dirname(path), 'corrupted')
            if not os.path.exists(failed_path):
                os.makedirs(failed_path)
            new = os.path.join(failed_path, os.path.basename(path))
            os.rename(path, new)
            return
        
        w, h = list(map(int, out.strip().split(b',')[:2]))

    codec, bitrate, w, h = get_video_info(path)


    swapped = False
    if h > w:
        w, h = h, w
        swapped = True
    
    size_ok = w <= MAX_WIDTH and h <= MAX_HEIGHT
    codec_ok = codec in ALLOWED_CODEC
    bitrate_ok = int(bitrate) <= 12e5
    
    edit = not (size_ok and codec_ok and bitrate_ok)

    if not edit:
        return

    filename, ext = path.rsplit('.', 1)
    tmp_path = filename + '_tmp.' + ext

    try:
        if not size_ok:
            os.rename(path, tmp_path)
            if w/MAX_WIDTH > h/MAX_HEIGHT:
                new_w = MAX_WIDTH
                # new_h = int(h * MAX_WIDTH / w)
                new_h = -2
            else:
                new_h = MAX_HEIGHT
                # new_w = int(w * MAX_HEIGHT / h)
                new_w = -2
            
            if swapped:
                new_w, new_h = new_h, new_w
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
            cmd = ['ffmpeg', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda', '-y', '-i', tmp_path, '-c:a', 'copy', '-vf', f'scale_cuda={new_w}:{new_h}', '-c:v', 'h264_nvenc', path]
            p = subprocess.Popen(cmd, stderr=subprocess.PIPE)
            out, err = p.communicate()
        
        if not codec_ok or not bitrate_ok:
            os.rename(path, tmp_path)
            cmd = ['ffmpeg', '-hwaccel', 'cuda', '-y', '-i', tmp_path, '-c:a', 'copy', '-c:v', 'h264_nvenc', '-preset', 'fast', '-b:v', '1M', path]
            p = subprocess.Popen(cmd, stderr=subprocess.PIPE)
            out, err = p.communicate()
    finally:
        os.remove(tmp_path)

    print(f'Resized {path}')

def resize_folder(path):
    log = False
    for file in os.listdir(path):
        if not os.path.isfile(os.path.join(path, file)):
            continue
        if not file.endswith('.mp4'):
            continue

        if not log:
            log = True
        try:
            resize_video(os.path.join(path, file))
        except Exception as e:
            print(f'Error with {file}: {e}')
            continue

    if log:
        print('Done resizing videos')

if __name__ == '__main__':
    if sys.argv[1:]:
        path = sys.argv[1]
    else:
        path = askdirectory()

    resize_folder(path)