import json
import base64
import os
import re
import cv2
import imageio.v3 as iio
import numpy as np
from tkinter.filedialog import askopenfilename, askdirectory

import requests

BASE_URL = 'http://127.0.0.1:7860/sdapi/v1/'


def submit_post(url: str, data: dict):
    """
    Submit a POST request to the given URL with the given data.
    """
    return requests.post(url, data=json.dumps(data))


def save_encoded_image(b64_image: str, output_path: str):
    """
    Save the given image to the given output path.
    """
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))


def load_decoded_image(path: str):
    """
    Save the given image to the given output path.
    """
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def video_to_images(path=None, output=None, ext='jpg', crop=None):
    if path is None:
        path = askopenfilename(title='Select a video')

        if path == '':
            return


    if output is None:
        output = os.path.join(os.path.dirname(__file__), 'processed_' + os.path.splitext(os.path.split(path)[1])[0])

    if not os.path.exists(output):
        os.makedirs(output)

    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        if crop:
            image = image[crop[1]:crop[3], crop[0]:crop[2]]
        cv2.imwrite(os.path.join(output, f"frame{str(count).zfill(6)}.{ext}"), image)
        success, image = vidcap.read()
        count += 1
    print(f'Read {count} frames in folder: {output}')


def images_to_video(path=None, output=None):
    if path is None:
        path = askdirectory(title="Select images directory")

        if path == '':
            return

    if output is None:
        output = os.path.join(path, 'output.mp4')

    imgs=[]
    for img in os.listdir(path):
        m = re.match(r'^frame(\d+)\.png', img)
        if not m:
            continue
        imgs.append((int(m.groups()[0]), cv2.imread(os.path.join(path, img))))

    imgs.sort(key=lambda e: e[0])
    
    height,width,layers=imgs[1][1].shape

    
    fourcc = cv2.VideoWriter_fourcc(*'mhm1')
    video=cv2.VideoWriter(output,-1,30,(width,height), )

    for i, img in imgs:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def images_to_gif(path, output):
    imgs=[]
    for img in os.listdir(path):
        m = re.match(r'^frame(\d+)\.png', img)
        if not m:
            continue
        imgs.append((int(m.groups()[0]), os.path.join(path, img)))

    imgs.sort(key=lambda e: e[0])

    iio.imwrite(output, list(map(lambda e: iio.imread(e[1]), imgs)), loop=0, duration=(1000/30) )



def txt2img(output_path, **kwargs):
    url = BASE_URL + 'txt2img'
    data = {
        'prompt': 'a dog wearing a hat',
        'negative_prompt': '',
        'denoising_strength': 0.75,
        'seed': -1,
        'batch_size': 1,
        'n_iter': 1,
        'steps': 50,
        # "sampler_name": "string",
        'cfg_scale': 7,
        'width': 512,
        'height': 512,
    }
    data |= kwargs
    response = submit_post(url, data)
    imgs = response.json()['images']
    if len(imgs) == 1:
        save_encoded_image(imgs[0], output_path)
    else:
        padding = len(str(len(imgs)))
        path, ext = os.path.splitext(output_path)
        for i, img in enumerate(imgs):
            save_encoded_image(img, path + '_' + str(i).zfill(padding) + ext)


def img2img(img_path, output_path, controlnet=None, **kwargs):
    url = BASE_URL + 'img2img'
    img = load_decoded_image(img_path)
    data = {
        'init_images': [img],
        'prompt': 'a dog wearing a blue hat',
        'negative_prompt': '',
        'denoising_strength': 0.75,
        'seed': -1,
        'batch_size': 1,
        'n_iter': 1,
        'steps': 50,
        # "sampler_name": "string",
        'cfg_scale': 7,
        'width': 512,
        'height': 512,
        'restore_faces': False,
    }
    if controlnet is not None:
        data['alwayson_script'] = {
            "controlnet": {
                "args": [controlnet]
            }
        }
    data |= kwargs
    response = submit_post(url, data)
    if response.status_code == 200:
        imgs = response.json()['images']
    else:
        data = response.json()
        print(data['error'], '-', data['detail'], '-', data['errors'])
        return 

    dirname = os.path.dirname(output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if len(imgs) == 1:
        save_encoded_image(imgs[0], output_path)
    else:
        padding = len(str(len(imgs)))
        path, ext = os.path.splitext(output_path)
        for i, img in enumerate(imgs):
            save_encoded_image(img, path + '_' + str(i+1).zfill(padding) + ext)


def batch_img2img(input_folder, output_folder, **kwargs):
    for file in os.listdir(input_folder):
        # TODO: Check if img?
        img2img(os.path.join(input_folder, file), output_folder, **kwargs)


if __name__ == '__main__':
    if False:
        txt2img('dog.png', prompt='a dog wearing a hat')

    if False:
        img2img('dog.png', 'dog2.png',
                prompt="a dog wearing a blue hat", n_iter=2, batch_size=2)

    if True:
        video_to_images(ext='png')
        # video_to_images(ext='png', crop=(127, 0, 127+336, 336))
        # video_to_images('video.mp4', 'output', ext='png', crop=(500, 355, 1079, 1605))

    if False:
        images_to_video()

    if False:
        images_to_gif('processed4', 'gif4.gif')

    if False:

        width, height = 600-250, 1200-400
        cut_s, cut_e = 155, 480
        mask = np.zeros((height, width, 3), np.uint8)
        # mask[0:height_cut, :] = (0, 0, 0)
        mask[cut_s:cut_e, :] = (255, 255, 255)
        img_mask = cv2.imencode('.png', mask)
        b64_mask = base64.b64encode(img_mask[1]).decode('utf-8')

        controlnet = {
            "module": "openpose_full",
            "model": "control_v11p_sd15_openpose [cab727d4]",
            "pixel_perfect": True
        }
        # batch_img2img('output', 'processed', prompt='nude girl, nipples', negative_prompt='shirt, shorts, leather', alwayson_scripts=args)
        img2img(
            'output/frame9.png', 'processed/frame9.png',
            prompt='nude girl, nipples, vagina',
            negative_prompt='shirt, shorts, leather',
            mask=b64_mask,
            inpaint_full_res=False,
            width=width,
            height=height,
            sampler_name="Euler a",
            cfg_scale= 7,
            denoising_strength= 0.75,
            controlnet=controlnet
        )

    if False:
        with open('mask.png', 'rb') as f:
            data = f.read()

        for file in os.listdir('output'):
            with open('masks/' + file, 'wb') as f:
                f.write(data)