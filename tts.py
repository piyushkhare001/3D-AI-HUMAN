from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from io import BytesIO
from gtts import gTTS
from pydub import AudioSegment
from chatbot import generate_answer_as_malla_reddy

app = FastAPI()


def speed_change(sound, speed=1.5):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)


def generate_and_modify_voice(text, lang='te', pitch_shift_semitones=-15, speed=1.5):
    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    new_sample_rate = int(audio.frame_rate * (2.0 ** (pitch_shift_semitones / 12.0)))
    pitched_audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    pitched_audio = pitched_audio.set_frame_rate(audio.frame_rate)

    sped_up_audio = speed_change(pitched_audio, speed)

    out_fp = BytesIO()
    sped_up_audio.export(out_fp, format="mp3")
    out_fp.seek(0)
    return out_fp


@app.post("/tts")
async def tts_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    audio_fp = generate_and_modify_voice(text)
    return StreamingResponse(audio_fp, media_type="audio/mpeg")


@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}


@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("text", "")
    answer = generate_answer_as_malla_reddy(question)
    return {"answer": answer}
