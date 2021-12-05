import numpy as np
from mss import mss
import pygame, socket, cv2, pickle, struct

HOST = '192.168.0.1'
PORT = 1974

sct = mss()
monitor_number = 2
mon = sct.monitors[monitor_number]

width, height = 1366,768
bounding_box = {'top': mon["top"] + 0, 'left': mon["left"] + 0, 'width': width, 'height': height}
bounding_box["mon"] = monitor_number
scaleFactor = 1.5

#pygame.init()
#display = pygame.display.set_mode((bounding_box['width'],bounding_box['height']))
def is_cube(n):
    cbrt = np.cbrt(n)
    return cbrt ** 3 == n, int(cbrt)


def reduce_color_space(img, n_colors=64):
    n_valid, cbrt = is_cube(n_colors)

    if not n_valid:
        print("n_colors should be a perfect cube")
        return

    n_bits = int(np.log2(cbrt))

    if n_bits > 8:
        print("Can't generate more colors")
        return

    bitmask = int(f"{'1' * n_bits}{'0' * (8 - n_bits)}", 2)

    return img & bitmask

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            sct_img = np.array(sct.grab(bounding_box))[:, :, :3]
            sct_img = cv2.resize(sct_img, (int(width/scaleFactor), int(height/scaleFactor)))
            sct_img = reduce_color_space(sct_img)
            #sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGR2GRAY) #BGR2RGB for colors or BGR2GRAY for performances
            a = pickle.dumps(sct_img)
            message = struct.pack("Q",len(a))+a
            try:
                conn.sendall(message)
            except BaseException as e:
                print(e)
            #sct_img[:, :, [0, 2]] = sct_img[:, :, [2, 0]]
            
            #cv2.imshow('screen', sct_img)
            """surf = pygame.image.frombuffer(sct_img.tostring(), sct_img.shape[1::-1], "RGB")
            surf.convert()
            display.blit(surf,(0,0))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break"""

pygame.quit()
