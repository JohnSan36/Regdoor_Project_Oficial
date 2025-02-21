from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from dotenv import load_dotenv, find_dotenv
import uvicorn
import os


load_dotenv(find_dotenv())
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")

app = FastAPI()

class WhatsappMessage(BaseModel):
    message: str
    type_message: str
    sender: str

@app.post("/webhook")
async def recieve_message(msg: WhatsappMessage):
    print(f"Nova mensagem de {msg.sender}; Conteúdo: {msg.message}; Type: {msg.type_message}")
    return {"Status": "Mensagem recebida com sucesso!"}

def send_message(to, message):
    url = "https://evolution.cybermindsolutions.com.br/api/v1/message/sendText"
    headers = {"Authorization": f"Bearer {EVOLUTION_API_KEY}"}
    payload = {"to": to, "message": message}

    response = requests.post(url, headers=headers, json=payload)
    return response.json()


import asyncio

if __name__ == "__main__":
    if asyncio.get_event_loop().is_running():
        print("O servidor já está rodando dentro de um loop assíncrono. Execute diretamente com 'uvicorn app:app --reload'")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)

