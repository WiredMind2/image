import os
from PIL import Image
from tkinter import filedialog

files = filedialog.askopenfilenames(filetypes=[('Images', ('.png', '.jpg', '.gif', '.bmp', '.webp'))])

size_x, size_y = 246, 138
off = 5

resize = (1366,768)
ratio = min(resize[0]/size_x, resize[1]/size_y)
resize = (int(size_x*ratio), int(size_y*ratio))

for file in files:
    path, name = os.path.split(file)
    path = os.path.join(path, 'cuts', os.path.splitext(name)[0])
    
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        print('File already exists with same name as folder!')
        continue

    try:
        img = Image.open(file, 'r').convert('RGB')

        img.show(title='test')
        
        dim_x, dim_y = img.size
        grid_x, grid_y = (dim_x // (size_x+off)), (dim_y // (size_y+off))

        start_x, start_y = 0, dim_y - (size_y+off)*grid_y + off

        for x in range(grid_x):
            for y in range(grid_y):
                top, left = x*(size_x+off) + start_x, y*(size_y+off) + start_y
                
                box = (top, left, top + size_x, left + size_y)
                area = img.crop(box)
                
                if resize is not None:
                    area = area.resize(resize)#, resample=Image.Resampling.LANCZOS)
                
                area.save(os.path.join(path, f'img_{y*grid_x + x}.jpg'))

    except Exception as e:
        print(f'Error on {file}: {str(e)}')
        raise
    else:
        print(f'Done: {file}')

