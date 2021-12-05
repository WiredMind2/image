import socket
from time import sleep
import cv2, pickle, struct, pygame

host = socket.gethostname()  # get local machine name
port = 1974

running = True

client_socket = socket.socket()
data = b""
payload_size = struct.calcsize("Q")

client_socket = socket.socket()
try:
    client_socket.connect((host, port))
    print("Client connected!")
except ConnectionRefusedError:
    pass

pygame.init()
display = pygame.display.set_mode((1366,768))

with client_socket:
    while running:
        try:
            while len(data) < payload_size:
                packet = client_socket.recv(4*1024) # 4K
                if not packet:
                    running = False
                data+=packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q",packed_msg_size)[0]
            while len(data) < msg_size:
                data += client_socket.recv(4*1024)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame = pickle.loads(frame_data)

            """scale_percent = 200 # percent of original size
            dim = (int(frame.shape[1] * scale_percent / 100), int(frame.shape[0] * scale_percent / 100))

            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)"""
            
            #cv2.imshow("RECEIVING VIDEO",frame)

            frame[:, :, [0, 2]] = frame[:, :, [2, 0]]
            surf = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")
            surf = pygame.transform.scale(surf,(1366,768))
            surf.convert()
            display.blit(surf,(0,0))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        except:
            running = False

print('Done!')
