print("Importing...")
import hotspin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

n = 100

frames = 200
updates_per_frame = 10

print("Initializing...\n")

fig, ax = plt.subplots(figsize=(16,16), layout='tight')

mm = hotspin.ASI.IP_PinwheelDiamond(1e-6, n, pattern="random")  # magnetic system
rgb_array = hotspin.plottools.get_rgb(mm)

image = ax.imshow(rgb_array)
plt.axis("off")

def animateImage(frame):
    print(f"Rendering frame {frame+1} of {frames}: {(frame+1)/frames*100:.2f}%")

    for _ in range(updates_per_frame):
        mm.update()

    image.set_array(hotspin.plottools.get_rgb(mm))

    return [image]  # iterable of artists

animation = FuncAnimation(fig, animateImage, frames=frames)
animation.save("new_test_animation2.mp4", writer="ffmpeg", fps=24)

print("\nDone")