import pyaudio
import struct
import math
from tkinter import *
import time
from cactus import *
#from PIL import ImageTk, Image  


# set up listening through microphone

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

# gameOver = False
tkAfter = None
groundY = 150
jumping = False
dinoStartY = None
yDiff = None
vel = None

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
    global jumping, vel, yDiff
    if jumping:
        return
    block = stream.read(framesPerBlock)
    rmsAmp = findRMSAmplitude(block)
    if rmsAmp > clapThreshold:
        print("clap!")
        jumping = True
        vel = 6
        yDiff = vel

# update the dino jump if it's jumping
def dinoUpdate():
    global yDiff, vel, jumping
    gravity = -0.3

    if jumping:
        canvas.move(dinoCanvas, 0, -vel)
        vel += gravity
        yDiff += vel

        # hit the ground
        if (yDiff <= 0):
            # correct error by restoring dino to original y position
            _, currY, *_ = canvas.bbox(dinoCanvas)
            canvas.move(dinoCanvas, 0, dinoStartY - currY)
            jumping = False
            yDiff = 0

# update canvas by listening for claps, continuing dino jumps, and moving cacti
def updateCanvas():
    listenForClaps()
    dinoUpdate()
    cactus1.update()
    cactus2.update()
    canvas.update()
    time.sleep(0.01)

    global tkAfter
    if not (cactus1.getGameOver() or cactus2.getGameOver()):
        # only continue updating the game if the game isn't over
        tkAfter = window.after(20, updateCanvas)
    else:
        tkAfter = None

# close game/stop recording
def onTkClose():
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
title = Label(window, text="Dino Game - clap to jump")
title.place(relx = 0.5, rely = 0.1, anchor = 'center')
window.protocol("WM_DELETE_WINDOW", onTkClose)

# create game assets
dino = PhotoImage(file="dino.png")
dino = dino.subsample(15)
dinoCanvas = canvas.create_image(50, groundY, image=dino, anchor="s")
_, dinoStartY, *_ = canvas.bbox(dinoCanvas)

cactus = PhotoImage(file="cactus.png")
cactus = cactus.subsample(10)
cactus1 = Cactus(canvas, groundY, cactus, dinoCanvas, dino)
cactus2 = Cactus(canvas, groundY, cactus, dinoCanvas, dino)
# cactus1Canvas = canvas.create_image(220, groundY, image=cactus, anchor="s")
# cactus2Canvas = canvas.create_image(220, groundY, image=cactus, anchor="s")

# queue up the first listenForClaps function call
window.after(20, updateCanvas)
window.mainloop()