import os
from PIL import Image
from tkinter import filedialog

files = filedialog.askopenfilenames(filetypes=[('Images', ('.png', '.jpg', '.gif', '.bmp', '.webp'))])

format = input('Convert images to: ')
if format[0] != '.':
    format = '.' + format

for file in files:
    path = os.path.splitext(file)[0] + format
    print(f'Converting {file} to {path}')
    try:
        img = Image.open(file, 'r').convert('RGB')
        img.save(path)
    except Exception as e:
        print(f'Error on {file}: {str(e)}')

