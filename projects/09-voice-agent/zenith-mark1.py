import speech_recognition as sr
import pyttsx3
import requests

print("🎙️ ZENITH - Your Voice Agent")
print("="*50)

# Initialize TTS (text-to-speech)
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 0.9)

def speak(text):
    print(f"🗣️ Zenith: {text[:100]}...")
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("\n🎤 Listening...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            text = r.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("⏰ No speech detected")
            return ""
        except sr.UnknownValueError:
            print("❌ Could not understand")
            return ""

def ask_llama(question):
    response = requests.post('http://localhost:11434/api/generate',
        json={"model": "llama3", "prompt": question, "stream": False})
    return response.json()['response']

# Main loop
speak("Zenith ready. Ask me anything.")

while True:
    question = listen()
    
    if not question:
        continue
    
    if "exit" in question.lower() or "quit" in question.lower():
        speak("Goodbye!")
        break
    
    print("🤔 Thinking...")
    answer = ask_llama(question)
    print(f"💡 Answer: {answer[:200]}...")
    speak(answer)