'''
Run the Arduino file Touch_control and this file to run our final project.
This file trains an ML model, then waits for button input from the Arduino.
Given that, it starts recording for 10 seconds. It then parses what was recorded
into one word per file, predicts what each word was, and sends its prediction to
the Arduino to for the robot to write.

'''


from SpeechClassification_hs import *
from Record import *
from RecordSentence import *
from Communications import *
from os.path import exists

# set up comms with Arduino
ser = initComms("COM3")


# create the ML model
print("Setting up ML model...")
model = M5(n_input=1, n_output=len(labels))
model.to(device)


# train and test the ML model

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  
# reduce the learning after 20 epochs by a factor of 10

log_interval = 20
n_epoch = 110

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []
# The transform needs to live on the same device as the model and the data
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval, optimizer, pbar, pbar_update)
        test(model, epoch, pbar, pbar_update)
        scheduler.step()


# record, predict what was said, and tell Arduino to write it
for i in range(2): # do this an arbitrary number of times
    print("\n\nWaiting to sense touch sensor...")

    # start recording when Arduino senses button press
    waitForSpecificResponse(ser, "button")
    recordAudio("main" + str(i) + "_.wav")
    separateWords("main" + str(i) + "_.wav", "main" + str(i) + "_.wav")


    # each numbered file is a separate word
    # go through each file and predict each word
    j = 0
    filename = "main" + str(i) + "_" + str(j)+ ".wav"
    stringToWrite = ""
    while exists(filename):
        waveform, sample_rate = torchaudio.load(filename)
        ipd.Audio(waveform.numpy(), rate=sample_rate)

        prediction = predict(model, waveform)
        stringToWrite += prediction + " "

        j += 1
        filename = "main" + str(i) + "_" + str(j)+ ".wav"
    stringToWrite = stringToWrite.strip()

    # send string to robot to write
    print(f"Predicted: {stringToWrite}")
    stringToWrite = encodeForArduino(stringToWrite)
    sendCommandToArduino(ser, stringToWrite)
    waitForSpecificResponse(ser, "starting to write")

closeComms(ser)