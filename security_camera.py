import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2

HOST = ('0.0.0.0', 7890)

WEB_PAGE_OLD = b"""
<html>
<body>
<img src="data:image/png; base64, {data}" alt="Couldn't load the image!" />
</body>
</html>
"""

WEB_PAGE = b"""
<html>
<body>
<img src="/image" />
</body>
</html>
"""


class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.mainpage()
        elif self.path == "/image":
            self.image()

    def mainpage(self):
        

        try:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            self.wfile.write(WEB_PAGE)
        except ConnectionAbortedError:
            pass

    def image(self):
        data = get_pic()
        try:
            self.send_response(200)
            self.send_header("Content-type", "image/png")
            self.end_headers()

            self.wfile.write(data)
        except ConnectionAbortedError:
            pass

def get_pic():

    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        retval, buffer = cv2.imencode('.png', frame)
        # jpg_as_text = base64.b64encode(buffer)
    else:
        buffer = ""

    vc.release()

    return buffer

if __name__ == "__main__":

    get_pic()

    webServer = HTTPServer(HOST, Server)
    print("Server started http://%s:%s" % HOST)

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")