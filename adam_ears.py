from vosk import Model, KaldiRecognizer
import pyaudio
from settings import *

# model path
model = Model(model_path=voice_model)

recognizer = KaldiRecognizer(model, 16000)

microphone = pyaudio.PyAudio()

stream = microphone.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()
print("start speakingqqqww")
while True:
    data = stream.read(4096,exception_on_overflow=False)
    print("start speaking")
    if recognizer.AcceptWaveform(data):
        text = recognizer.Result()
