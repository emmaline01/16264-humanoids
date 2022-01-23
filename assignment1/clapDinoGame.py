import pyaudio
import struct
import math
from tkinter import *
import time
#from PIL import ImageTk, Image  


# set up listening

# modified from https://realpython.com/playing-and-recording-sound-python/#pyaudio
# and https://stackoverflow.com/questions/4160175/detect-tap-with-pyaudio-from-live-mic/4160733
SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024  # Record in chunks of 1024 samples
sampleFormat = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
recordSecs = .05
framesPerBlock = int(fs * recordSecs)
clapThreshold = 0.05

tkAfter = None

p = pyaudio.PyAudio()  # Create an interface to PortAudio

stream = p.open(format=sampleFormat,
    channels=channels,
    rate=fs,
    frames_per_buffer=chunk,
    input=True)

# https://stackoverflow.com/questions/4160175/detect-tap-with-pyaudio-from-live-mic/4160733
def findRMSAmplitude(block):
    # RMS amplitude is defined as the square root of the 
    # mean over time of the square of the amplitude.
    # so we need to convert this string of bytes into 
    # a string of 16-bit samples...

    # we will get one short out for each 
    # two chars in the string.
    count = len(block)/2
    format = "%dh"%(count)
    shorts = struct.unpack( format, block)

    # iterate over the block.
    sumSquares = 0.0
    for sample in shorts:
        # sample is a signed short in +/- 32768. 
        # normalize it to 1.0
        n = sample * SHORT_NORMALIZE
        sumSquares += n * n

    return math.sqrt( sumSquares / count )

# listen for if a sound block recorded is as loud as/louder than a clap
def listenForClaps():
    block = stream.read(framesPerBlock)
    rmsAmp = findRMSAmplitude(block)
    if rmsAmp > clapThreshold:
        print("clap!")
        dinoJump()
    global tkAfter
    tkAfter = window.after(20, listenForClaps)

def dinoJump():
    vel = 3
    yDiff = vel
    gravity = -0.1
    while (yDiff > 0) :
        canvas.move(dinoCanvas, 0, -vel)
        canvas.update()
        time.sleep(0.01)
        vel += gravity
        yDiff += vel

def onTkClose():
    # close game/stop recording
    stream.stop_stream()
    stream.close()
    p.terminate()
    # remove the queued listenForClaps function call
    if tkAfter is not None:
        window.after_cancel(tkAfter)

    window.destroy()
    

#set up tkinter window
window = Tk(className = "Dino Game")
# window.geometry("200x200")
# window.configure(bg = "white")
canvas = Canvas(window, width=200, height=200)
canvas.pack()
#title = Label(window, text="Dino Game", bg = "white")
#title.pack(pady = 20)
window.protocol("WM_DELETE_WINDOW", onTkClose)

dino = PhotoImage(file="dino.png")
dino = dino.subsample(20)
dinoCanvas = canvas.create_image(50, 120, image=dino)
# dinoLabel = Label(window, image=dino)
# # dinoLabel.pack()
# dinoLabel.place(relx = 0.25, rely = 0.6, anchor = 'center')

# queue up the first listenForClaps function call
window.after(20, listenForClaps)
window.mainloop()