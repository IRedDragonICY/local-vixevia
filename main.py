from pathlib import Path
import io
import cv2
import numpy as np
import ollama
import soundfile as sf
from faster_whisper import WhisperModel
import speech_recognition as sr
import time
from pydub import AudioSegment
from so_vits_svc_fork.inference.main import infer
from transformers import AutoProcessor, SeamlessM4Tv2Model
import sounddevice as sd

model_path, config_path = Path("models/audio/audio.pth"), Path("models/audio/audio.json")

# Load Whisper model for speech recognition
model = WhisperModel("models/faster-whisper-medium-id")
recognizer = sr.Recognizer()

# Load SeamlessM4T model for TTS
processor = AutoProcessor.from_pretrained("models/seamless-m4t-v2-large")
tts_model = SeamlessM4Tv2Model.from_pretrained("models/seamless-m4t-v2-large")


def process_frame(frame):
    img_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

    res = ollama.chat(
        model="moondream:latest",
        messages=[
            {
                'role': 'user',
                'content': 'Describe this image:',
                'images': [img_bytes]
            }
        ]
    )

    print(res['message']['content'])


def vision():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        process_frame(frame)
        cv2.imshow('AI Processed Frame', frame)


def text_to_speech(textinput, tgt_lang="ind"):
    return tts_model.generate(**processor(text=textinput, src_lang="ind", return_tensors="pt"), tgt_lang=tgt_lang)[
        0].numpy().squeeze()


def play_audio(audio):
    sd.play(audio, samplerate=44100), sd.wait()


def text():
    history = []
    with open('system_prompt', 'r') as file:
        system_prompt = file.read()

    history.append({'role': 'system', 'content': system_prompt})

    while True:
        history.append({'role': 'system', 'content': time.strftime("%Y-%m-%d %H:%M:%S")})
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            segments, info = model.transcribe(io.BytesIO(recognizer.listen(source).get_wav_data()))

        user_input = " ".join(segment.text for segment in segments).strip().replace("Fixi", "Vixi").replace("fixi",
                                                                                                            "Vixi")
        print("User:", user_input)

        # user_input = input("User: ")
        phrases_to_replace = [
            "Terima kasih kerana menonton!",
            "Terima kasih sudah menonton!",
            "Terima kasih sudah menonton.",
            " Terima kasih sudah menonton!",
            "Terima kasih sudah menonton!",
            "Terus sampai di video selanjutnya.",
            "Terus jumpa di video selanjutnya!",
            "Terus sampai jumpa di video selanjutnya",
            "Sampai jumpa di video selanjutnya!",
            "Terus sampai jumpa di video selanjutnya!",
            "Sampai jumpa di next video!",
        ]

        for phrase in phrases_to_replace:
            user_input = user_input.replace(phrase, "")

        history.append({'role': 'user', 'content': user_input})
        stream = ollama.chat(
            model='llama3.1:8b-instruct-q2_K',
            messages=history,
            stream=True,
        )

        response_content = ""
        for chunk in stream:
            response_content += chunk['message']['content']
            print(chunk['message']['content'], end='', flush=True)
        print()

        history.append({'role': 'assistant', 'content': response_content})

        speech = text_to_speech(response_content, tgt_lang="ind")
        sf.write("temp/response.wav", speech, 16000)

        infer(
            input_path=Path("temp/response.wav"),
            output_path=Path("temp/response.mp3"),
            model_path=model_path,
            config_path=config_path,
            max_chunk_seconds=35,
            device="cuda",
            speaker="",
            transpose=7,
        )
        speech = AudioSegment.from_file("temp/response.mp3")
        speech = np.array(speech.get_array_of_samples())
        play_audio(speech)


if __name__ == "__main__":
    # vision()
    text()
