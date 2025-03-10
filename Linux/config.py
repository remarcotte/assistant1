# config.py
from langchain.memory import ChatMessageHistory

# Port to receive commands
PORT = 12345
BUFFER_SIZE = 1024

# Wake/Sleep Words
WAKE_WORD = "hey pinky"
SLEEP_WORD = "sleep pinky"
EXIT_COMMAND = "exit this"

MODEL_NAME = "assistant"
TTS_MODEL_PATH = "./models/voices/en_US-libritts_r-medium.onnx"

# Create a single chat history
message_history = ChatMessageHistory()

# Global state for command processor
global_state = {
    "awake": True
}
