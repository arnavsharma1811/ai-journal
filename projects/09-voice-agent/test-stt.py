import speech_recognition as sr

print("🎧 Testing Zenith's ears with Google Speech...")
print("="*50)

# Initialize recognizer
r = sr.Recognizer()

print("\n🎤 Speak now... (you have 5 seconds)")

try:
    with sr.Microphone() as source:
        # Adjust for ambient noise
        r.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")
        audio = r.listen(source, timeout=5, phrase_time_limit=5)
    
    print("Processing...")
    text = r.recognize_google(audio)
    print(f"\n📝 Zenith heard: \"{text}\"")
    
except sr.WaitTimeoutError:
    print("⏰ No speech detected")
except sr.UnknownValueError:
    print("❌ Google couldn't understand the audio")
except sr.RequestError as e:
    print(f"❌ Google API error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")