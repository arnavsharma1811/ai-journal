import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 0.9)

engine.say("Hello Arnav. I am zenith your voice agent, your jarvis. Ready to build.")
engine.runAndWait()
print("✅ TTS working")