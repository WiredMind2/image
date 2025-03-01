from collections import deque
import datetime
from functools import wraps
import time
import re
from tkinter.filedialog import askopenfilename
import subprocess as sp
import tkinter as tk
import os
import vlc

import tkvlc

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the path to the libvlc.dll file
vlc_path = os.path.join(script_dir, "libvlc.dll")

FFMPEG_BIN = "ffmpeg" # lol


def get_srt(path):
    command = [ FFMPEG_BIN, '-i', path, '-vn', '-an', '-codec:s:0', 'srt', '-f', 'srt', '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True)

    ffmpeg_output, _ = pipe.communicate()
    return ffmpeg_output

def iter_srt(path):
    data = get_srt(path).split('\n')
    buf = []
    found_start = False
    ind = 0
    for line in data:
        if not found_start:
            if line != '1':
                continue
            found_start = True

        if line.isdigit() and int(line) == ind+1:
            ind += 1
            if len(buf) >= 3:
                yield buf
            buf = []

        buf.append(line)

def parse_srt(path):
    for ind, timestamp, *data in iter_srt(path):
        timestamp = timestamp.strip().split(' --> ')
        start, stop = [time_to_sec(datetime.time.fromisoformat(t)) for t in timestamp]
        data = "\n".join(data)
        data = re.sub(r'(<.+?>)', '', data)
        yield ind, start, stop, data

def display_subs(path):
    started = False
    for ind, start, stop, data in parse_srt(path):
        if not started:
            print('Starting in 3 seconds')
            time.sleep(3)
            started = True
            print('Started!')
            begin = datetime.datetime.now()

        now = datetime.datetime.now()-begin
        ref = datetime.datetime.combine(start.date(), (datetime.datetime.min + now).time())
        if start > ref:
            time.sleep((start-ref).total_seconds())
        print(data)

def time_to_sec(time_obj):
    return time_obj.hour*3600 + time_obj.minute*60 + time_obj.second + time_obj.microsecond/1e6


class VLCPlayer(tkvlc.Player):
    def __init__(self, root, video, debug=False) -> None:
        super().__init__(root, video=video, debug=debug)

        self.events = {}

        # Play the video
        self.player.play()

    def add_callback(self, event, callback):
        self.events[event] = callback
        f = getattr(self, event, None)
        if f is not None:
            setattr(self, event, self.cb_wrapper(f, callback))

    def cb_wrapper(self, f, cb):
        # @wraps(f)
        def wrapper(*args, **kwargs):
            out = f(*args, **kwargs)
            cb(*args, **kwargs)
            return out
        return wrapper

class SubsReader():
    def __init__(self, path, media_player=True):
        self.media_player = media_player
        
        self.root = tk.Tk()
        self.root.title("Subs Reader")
        self.root.geometry("800x900")
        self.root.focus_force()

        self.root.grid_columnconfigure(0, weight=1)

        font = ("Arial", 20)

        labels_count = 10
        self.labels = []
        self.text_vars = []

        for i in range(labels_count):
            text_var = tk.StringVar()
            label = tk.Label(self.root, textvariable=text_var)
            label.text_var = text_var

            label.config(font=font)
            label.grid(column=0, row=i, sticky='ew')

            self.labels.append(label)
            self.text_vars.append(text_var)


        iter = parse_srt(path)
        self.data = deque(iter)


        self.label_textvar = tk.StringVar()
        self.label = tk.Label(self.root, textvariable=self.label_textvar)
        self.label.config(font=font)
        self.label.grid(column=1, row=0, sticky='ew')

        self.cursor_textvar = tk.StringVar()
        self.cursor = tk.Entry(self.root, textvariable=self.cursor_textvar)
        self.cursor.bind("<Return>", self.change_time)
        self.cursor.grid(column=1, row=1, sticky='ew')
        
        self.play_button_textvar = tk.StringVar()
        self.play_button_textvar.set("Play")
        self.play_button = tk.Button(self.root, textvariable=self.play_button_textvar, command=self.toggle_play)
        self.play_button.config(font=font)
        self.play_button.grid(column=1, row=2, sticky='ew')

        self.paused = True
        self.begin = 0

        self.start_delay = 5
        
        if self.media_player:
            self.start_player(path)

        self.update_time()

        self.root.mainloop()

    def toggle_play(self):
        if not self.paused:
            self.pause()
        else:
            self.play()

    def play(self, delay = True):
        self.play_button_textvar.set("Pause")

        def start():
            if delay is True:
                self.text_vars[0].set(f'Started!')

            self.paused = False
            self.begin = time.time() - self.begin
            self.update_display()

        if delay is True:
            for i in range(self.start_delay):
                self.root.after(i*1000, lambda i=i: self.text_vars[0].set(f'Starting in {self.start_delay-i} seconds'))

            self.root.after(self.start_delay*1000, start)
        else:
            start()

    def pause(self):
        self.paused = True
        self.play_button_textvar.set("Play")
        self.begin = time.time() - self.begin

    def update_display(self):
        if self.paused:
            return

        ind, start, stop, data = self.data[0]
        now = time.time() - self.begin

        while stop < now:
            self.data.rotate(-1)
            new_ind, start, stop, data = self.data[0]
            if new_ind < ind:
                # Looped back to beginning
                # No more captions
                return

        while start < now < stop:
            for line in data.split('\n'):
                line = line.strip()
                if line == "":
                    continue

                for i in range(len(self.text_vars)-1):
                    self.text_vars[i].set(self.text_vars[i+1].get())

                print(start, line)
                self.text_vars[-1].set(line)

            self.data.rotate(-1)
            ind, start, stop, data = self.data[0]

        self.root.after(int((start-now)*1000), self.update_display)

    def change_time(self, *_):
        val = self.cursor_textvar.get()
        if val:
            try:
                minute, second = map(int, val.split(':'))
            except ValueError:
                self.cursor_textvar.set('Invalid time')
                return
            else:
                self.cursor_textvar.set('')

                new_begin = 60*minute + second

                if self.paused:
                    begin = self.begin
                else:
                    begin = time.time() - self.begin

                if new_begin > begin:
                    # Skipped after
                    rot = -1
                else:
                    rot = 1

                ind, start, stop, data = self.data[0]
                
                while stop < new_begin:
                    self.data.rotate(-1)
                    new_ind, start, stop, data = self.data[0]
                    if new_ind < ind:
                        # Looped back to beginning
                        # No more captions
                        return

                if self.paused:
                    self.begin = new_begin
                else:
                    self.begin = time.time() - new_begin

    def update_time(self, *_):
        if self.paused:
            now = self.begin
        else:
            now = time.time() - self.begin

        if now > 0:
            now_obj = datetime.datetime.fromtimestamp(now)
        else:
            now_obj = datetime.datetime.fromtimestamp(0)
        now_obj -= datetime.timedelta(hours=4)
        self.label_textvar.set(now_obj.strftime('%H:%M:%S.%f'))
        
        self.root.after(100, self.update_time)

    # Media Player stuff

    def OnPlay(self, *_):
        self.play(delay=False)

    def OnPause(self, *_):
        self.pause()
    
    def OnTick(self, *_):
        t = max(0, self.media_player.player.get_time() // tkvlc._TICK_MS)
        if self.paused:
            self.begin = t
        else:
            self.begin = time.time() - t

    def start_player(self, path):
        root = tk.Toplevel(self.root)  # create a Tk.App()
        self.media_player = VLCPlayer(root, video=path)
        
        for event in ['OnPlay', 'OnPause', 'OnTick']:
            self.media_player.add_callback(event, getattr(self, event))
        
        if tkvlc._isWindows:  # see function _test() at the bottom of ...
            # <https://GitHub.com/python/cpython/blob/3.11/Lib/tkinter/__init__.py>
            root.iconify()
            root.update()
            root.deiconify()
            # root.mainloop()  # forever
            # root.destroy()  # this is necessary on Windows to avoid ...
            # # ... Fatal Python Error: PyEval_RestoreThread: NULL tstate
        # else:
        #     root.mainloop()  # forever

if __name__ == "__main__":
    file = askopenfilename(initialdir='E:\Animes')
    # display_subs(file)
    SubsReader(file, media_player=False)
