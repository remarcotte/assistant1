# main.py
import threading
import os
import socket
import io
import wave
import subprocess
import queue
import requests
from piper.voice import PiperVoice
from langchain.llms import BaseLLM
from langchain.schema import LLMResult, Generation
# from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

import config

# Initialize PiperVoice
voice = PiperVoice.load(config.TTS_MODEL_PATH)

# Initialize the ChatOllama model
model = ChatOllama(model=config.MODEL_NAME)

# Define the structured prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # Auto-handles message history
    ("human", "{input}")  # User input
])

# Chain the components together
conversation_chain = RunnableWithMessageHistory(
    prompt | model,  # Combines prompt + model into one pipeline
    lambda session_id: config.message_history,  # Always return the same message history
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Global variables for TTS processing.
tts_queue = queue.Queue()
stop_event = threading.Event()
current_tts_process = None
tts_process_lock = threading.Lock()

def speak_response(response):
    """Synthesizes response using Piper and plays it via paplay."""
    global current_tts_process
    print(f"[TTS] Speaking: {response}")

    buffer = io.BytesIO()
    wave_writer = wave.open(buffer, 'wb')
    wave_writer.setnchannels(1)
    wave_writer.setsampwidth(2)
    wave_writer.setframerate(22050)

    voice.synthesize(response, wave_writer)
    wave_writer.close()
    buffer.seek(0)

    paplay_process = subprocess.Popen(["paplay", "/dev/stdin"], stdin=subprocess.PIPE)
    
    with tts_process_lock:
        current_tts_process = paplay_process
    
    try:
        paplay_process.stdin.write(buffer.getvalue())
        paplay_process.stdin.flush()
    except BrokenPipeError:
        pass

    try:
        paplay_process.stdin.close()
    except BrokenPipeError:
        pass

    paplay_process.wait()

    with tts_process_lock:
        current_tts_process = None

def tts_worker():
    """Worker thread that processes TTS messages from the queue."""
    while True:
        tts_text = tts_queue.get()
        if tts_text is None:
            break
        if stop_event.is_set():
            tts_queue.task_done()
            continue
        speak_response(tts_text)
        tts_queue.task_done()

def stop_tts():
    """Stops current TTS output and clears the queue."""
    global current_tts_process
    print("[TTS] Stop command received. Stopping current and queued TTS...")
    stop_event.set()
    
    with tts_process_lock:
        if current_tts_process is not None:
            try:
                current_tts_process.kill()
                print("[TTS] Killed current TTS process.")
            except Exception as e:
                print(f"[TTS] Error killing TTS process: {e}")
            current_tts_process = None

    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
            tts_queue.task_done()
        except queue.Empty:
            break

    stop_event.clear()
    print("[TTS] TTS queue cleared.")

# Listen for the commands
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", config.PORT))

print(f"Listening for commands on UDP {config.PORT}...")

def listen_for_commands(process_command_fn):
    """Listens for UDP commands."""
    while True:
        data, _ = sock.recvfrom(config.BUFFER_SIZE)
        command = data.decode().strip()
        if command:
            threading.Thread(target=process_command_fn, args=(command,), daemon=True).start()

def process_command(command):
    command_lower = command.strip().lower()

    if command_lower == config.EXIT_COMMAND:
        print("[CMD] Exiting program...")
        os._exit(0)

    if command_lower == config.WAKE_WORD:
        if not config.global_state["awake"]:
            config.global_state["awake"] = True
            print("[CMD] Wake word detected. System is now awake.")
            tts_queue.put("System awake. How can I help you?")
        return

    if command_lower == config.SLEEP_WORD:
        config.global_state["awake"] = False
        print("[CMD] Sleep command received. System going to sleep.")
        tts_queue.put(f"System going to sleep. Say '{config.WAKE_WORD}' to wake me up.")
        return

    if command_lower == "stop it":
        stop_tts()
        return

    if not config.global_state["awake"]:
        print("[CMD] System is asleep. Ignoring command.")
        return

    formatted_input = command

    print(f"[CMD] Processing command: {command}")
    result = conversation_chain.invoke(
        {"input": command},  # No need to manually format chat history
        config={"configurable": {"session_id": "global"}}
    )
    
    print(f"[DEBUG] Raw result from LLM: {result}")  # ✅ Debugging step

    # Ensure correct extraction of response text
    # if isinstance(result, dict) and "text" in result:
    response_text = result.content
    # elif isinstance(result, str):  # Sometimes result may already be a string
        # response_text = result
    # else:
        # response_text = str(result)  # Convert whatever it is to a string

    print(response_text)
    tts_queue.put(response_text)

# Start the assistant
def main():
    threading.Thread(target=tts_worker, daemon=True).start()
    listen_for_commands(process_command)

if __name__ == '__main__':
    main()
