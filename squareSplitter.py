import random, time
from tkinter import *
from PIL import Image, ImageDraw, ImageTk


def crossing(rect, axis, coord):
    if axis == "x":
        axis = 0
    elif axis == "y":
        axis = 1
    return rect[0][axis] <= coord and rect[1][axis] >= coord


def split_data(data, bbox):
    axis = "x" if bool(random.getrandbits(1)) else "y"
    if axis == "x":
        coord = random.randint(bbox[0], bbox[2])
    else:
        coord = random.randint(bbox[1], bbox[3])
    print(bbox, axis,  coord)
    # coord = random.randint(-size[0], size[1])
    offset = random.randint(-maxOff, maxOff)
    # coord = abs(coord)

    for rect in data[:]:
        if crossing(rect, axis, coord):
            if axis == "x":
                new_rect_a = (
                    (rect[0][0], rect[0][1] + offset),
                    (coord, rect[1][1] + offset)
                )
                new_rect_b = (
                    (coord, rect[0][1]),
                    (rect[1][0], rect[1][1])
                )
            else:
                new_rect_a = (
                    (rect[0][0] + offset, rect[0][1]),
                    (rect[1][0] + offset, coord)
                )
                new_rect_b = (
                    (rect[0][0], coord),
                    (rect[1][0], rect[1][1])
                )
            data.append(new_rect_a)
            data.append(new_rect_b)
            data.remove(rect)
        else:
            if axis == "x":
                if rect[0][0] < coord:
                    new_rect = (
                        (rect[0][0] + offset, rect[0][1]),
                        (rect[1][0] + offset, rect[1][1])
                    )
                    data.append(new_rect)
                    data.remove(rect)
            else:
                if rect[0][1] < coord:
                    new_rect = (
                        (rect[0][0], rect[0][1] + offset),
                        (rect[1][0], rect[1][1] + offset)
                    )
                    data.append(new_rect)
                    data.remove(rect)

    return axis, coord


def draw_anim():
    axis, coord = "x", -1
    for i in range(200):
        img_size = (1000, 1000)
        img = Image.new("RGB", img_size, color="#000000")  # TODO - Binary?
        d = ImageDraw.Draw(img)
        for r in data:
            d.rectangle(r, outline="#FFFFFF")
        bbox = img.getbbox()
        # d.rectangle((bbox[0] + 1, bbox[1] + 1, bbox[2] - 2, bbox[3] - 2), outline="#00FF00")  # Bbox outline 

        # d.rectangle(((0, 0), (img.size[0] - 1, img.size[1] - 1)), outline="#0000FF")
        if coord >= 0:
            if axis == "x":
                d.line(((coord, -img.size[1] * 100), (coord, img.size[1] * 100)), fill="#FF0000")
            else:
                d.line(((-img.size[0] * 100, coord), (img.size[0] * 100, coord)), fill="#FF0000")

        crp_img = img.crop(bbox)
        # crp_img = img
        tk_img = ImageTk.PhotoImage(crp_img)
        can.delete(ALL)
        can.create_image(size[0] // 2, size[1] // 2, image=tk_img, anchor="center")
        can.image = tk_img

        fen.update()
        time.sleep(1)

        axis, coord = split_data(data, bbox)


size = (1000, 1000)
maxOff = 150
rectSize = 0.1

img = Image.new("RGB", size, color="#000000")  # TODO - Binary?
d = ImageDraw.Draw(img)

data = [
    (
        (size[0] * (0.5 - rectSize), size[1] * (0.5 - rectSize)),
        (size[0] * (0.5 + rectSize), size[1] * (0.5 + rectSize))
    )
]

fen = Tk()
frame = Frame(fen, width=size[0], height=size[1])
frame.pack(expand=True, fill=BOTH)

can = Canvas(frame, bg="#000000", width=size[0], height=size[1])

can.pack(side=LEFT, expand=True, fill=BOTH)

fen.after(10, draw_anim)
fen.mainloop()
