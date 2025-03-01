import os
from tkinter.filedialog import askopenfilenames
from tkinter.simpledialog import askstring

IMAGES = ['.jpg', '.jpeg', '.png']

def clean_files(path):
    files=os.listdir(path)
    images = []
    for f in files:
        if os.path.splitext(f)[1].lower() in IMAGES:
            images.append(f)

    padding = len(str(len(images)))
    for i, f in enumerate(images):
        _, ext = os.path.splitext(f)
        os.rename(os.path.join(path, f), os.path.join(path, f'img_{str(i).zfill(padding)}{ext.lower()}'))

def repeat_prompt(path):
    while True:
        tag = askstring(title='Tag input', prompt='Please enter a tag')
        if tag is None or tag == '':
            break

        files = askopenfilenames(title=f'Select corresponding images for tag: {tag}', initialdir=path, filetypes=[('Images', ' '.join(map(lambda e: f'*{e}', IMAGES)))])
        if len(files) == 0:
            break

        for file in files:
            name, _ = os.path.splitext(file)
            tagfile = os.path.join(path, name + '.txt')
            if os.path.exists(tagfile):
                with open(tagfile, 'r') as f:
                    tags = list(map(lambda t: t.strip().lower(), f.read().split(',')))
            else:
                tags = []

            tags.append(tag)

            with open(tagfile, 'w') as f:
                f.write(', '.join(sorted(tags)))

path = 'C:/Users/William/Downloads/datasets/j_full/img'
clean_files(path)
repeat_prompt(path)