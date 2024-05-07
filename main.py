import uvicorn
from fastapi import FastAPI
import io
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


app = FastAPI()

@app.get("/")
async def root():
    r = requests.get('https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg', stream=True)
    aux_im = Image.open(io.BytesIO(r.content))


    model_id = "vikhyatk/moondream2"
    revision = "2024-04-02"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    # image = Image.open('<IMAGE_PATH>')
    enc_image = model.encode_image(aux_im)
    print(model.answer_question(enc_image, "Describe this image.", tokenizer))

    # return {"message": "Hello from FastAPI!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)