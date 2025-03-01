import os
import requests
import subprocess
import threading

SD_PATH = "C:/Users/William/Documents/sd.webui/webui"

def start_API():
    def thread():
        cmd = f'cd .. && call environment.bat && cd webui && webui.bat --nowebui --xformers'
        print(cmd)
        subprocess.run(cmd, cwd=SD_PATH, shell=True)

    threading.Thread(target=thread).run()

if __name__ == '__main__':
    start_API()