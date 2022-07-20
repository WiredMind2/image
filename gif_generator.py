from PIL import Image
import glob

def make_gif(frame_folder, output, **kwargs):
    frames = [Image.open(image) for image in reversed(glob.glob(f"{frame_folder}/*.png"))]
    frame_one = frames[-1]
    frame_one.save(output, format="GIF", append_images=frames,
               save_all=True, **kwargs)

if __name__ == "__main__":
    output = "logo.gif"
    folder = r"C:\Users\willi\Downloads\Titanium Logo"
    make_gif(folder, output, duration=100, loop=0)