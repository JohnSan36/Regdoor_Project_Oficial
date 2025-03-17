from flask import Flask, request, jsonify
import os
import json
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def receive_message():
    try:
        body_corp = request.data  
        body_str = body_corp.decode("utf-8")
        print("Mensagem recebida:", body_str)  
        return jsonify({"Status": "Mensagem recebida com sucesso!"})
    
    except Exception as e:
        return jsonify({"Erro": f"Falha ao processar JSON: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
