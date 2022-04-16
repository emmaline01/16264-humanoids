from SpeechClassification_hs import *
from Record import *
from RecordSentence import *
from os.path import exists


# create the ML model
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


i = 0
while (True):
    val = input("\nPress enter to record for 10sec:")
    recordAudio("main" + str(i) + "_.wav")
    separateWords("main" + str(i) + "_.wav", "main" + str(i) + "_.wav")

    j = 0
    filename = "main" + str(i) + "_" + str(j)+ ".wav"
    while exists(filename):
        waveform, sample_rate = torchaudio.load(filename)
        ipd.Audio(waveform.numpy(), rate=sample_rate)
        print(f"Predicted: {predict(model, waveform)}")

        j += 1
        filename = "main" + str(i) + "_" + str(j)+ ".wav"
