import json
import socket
import pyaudio
import vosk
import sys
import time

# Ubuntu (WSL) IP & port
WSL_IP = "172.27.242.132"  # Replace with your WSL IP
PORT = 12345
MODEL_PATH = "./models/vosk-model/vosk-model-en-us-0.22-lgraph"
MIC_INDEX = 1

# Wake/Sleep Words
WAKE_WORD = "hey zebra"
SLEEP_WORD = "sleep zebra"
EXIT_COMMAND = "exit this"

# Debugging & Logging
DEBUG = False
SHOW_VOSK_LOG_LEVEL = -1

# Load Vosk Offline Speech Model
vosk.SetLogLevel(SHOW_VOSK_LOG_LEVEL)
model = vosk.Model(MODEL_PATH)

# Initialize microphone input
recognizer = vosk.KaldiRecognizer(model, 16000)
audio = pyaudio.PyAudio()

if DEBUG:
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        print(f"Device {i}: {dev['name']}, Input channels: {dev['maxInputChannels']}, Output channels: {dev['maxOutputChannels']}")

stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096, input_device_index=MIC_INDEX)

# Create socket to send text to Ubuntu
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Wake/Sleep State
awake = True

def send_text_to_wsl(text):
    """Send recognized text to Ubuntu via UDP."""
    sock.sendto(text.encode(), (WSL_IP, PORT))

print("Listening for voice commands...")
send_text_to_wsl(WAKE_WORD)

while True:
    try:
        data = stream.read(4096, exception_on_overflow=False)
        
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()

            if not text or text == "the":
                continue  # Ignore empty input
            
            # Remove "the" if it appears at the start
            if text.startswith("the "):
                # Remove "the " and any extra spaces
                text = text[4:].strip()

            # Remove " the" if it appears at the end
            if text.endswith("the "):
                # Remove "the " and any extra spaces
                text = text.removesuffix(" the").strip()

            # Handle wake/sleep commands
            if text == WAKE_WORD:
                if not awake:
                    awake = True
                    print("Waking up...")
                    send_text_to_wsl(WAKE_WORD)
                continue
            elif text == SLEEP_WORD:
                if awake:
                    awake = False
                    print("Going to sleep...")
                    send_text_to_wsl(SLEEP_WORD)
                continue

            if not awake:
                print("Ignoring: {text}")
                continue  # Ignore commands if asleep

            # Handle exit command
            if text == EXIT_COMMAND:
                print("Exiting...")
                send_text_to_wsl(text)
                # Add short delay to allow UDP message to be sent
                time.sleep(0.2)
                break  # Exit the loop and close the program
            
            # Send recognized text to server
            if text != "the":
                print(f"Recognized: {text}")
                send_text_to_wsl(text)
    
    except KeyboardInterrupt:
        print("\nExiting...")
        break

# Cleanup
stream.stop_stream()
stream.close()
audio.terminate()
sock.close()
sys.exit(0)
