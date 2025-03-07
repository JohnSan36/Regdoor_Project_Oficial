from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentExecutor
from fastapi import FastAPI, Request, HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_redis import RedisChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory   
from datetime import datetime
import os

import redis
import json
from datetime import datetime

load_dotenv(find_dotenv())

app = FastAPI()
api_key = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
REDIS_URL = "redis://default:A1ZDEbkF87w7TR0MPTBREnTFOnBgfBw9@redis-14693.c253.us-central1-1.gce.redns.redis-cloud.com:14693/0"

def obter_hora_e_data_atual():
    """Retorna a hora atual e a data de hoje."""
    agora = datetime.now()
    return agora.strftime("%Y-%m-%d - T%H:%M:%S")
data_atual = obter_hora_e_data_atual()


class ExtraiInformacoes(BaseModel):
    """Extrair informações de complience"""
    data: str = Field(description="data em que o o evento ocorreu (se a pessoa disser algo como ontem ou algo do tipo, perguntar a data especifica)")
    contatos: str = Field(
        description="nome das pessoas contidas no texto", 
        examples=[
            ("Me chamo Rafael e falei com o Junior sobre assunto XYZ.", "Rafael, Junior"), 
            ("Me chamo Alfredo e falei com o Severino sobre assunto a posse dos Eua.", "Alfredo, Severino")
        ])
    meio: str = Field(description="meio de contato dos contatos mencionados. Deve perguntar se foi google meet, presencial, ou qual meio foi utilizado.")
    cargo: str = Field(description="cargo dos contatos mencionados")
    organizacoes: str = Field(description="organizaçao dos contatos mencionados")
    jurisdicoes: str = Field(
        description="jurisdições mencionadas",
        examples=[
            ("Jane Doe is a Senior Regulatory Advisor at the Financial Conduct Authority (FCA) in the UK. I don't have her email ou phone number at the moment. ", "UK")
        ])
    representantes: str = Field(description="representantes dos contatos mencionados")
    assunto: str = Field(description="assunto do texto")
    resumo: str = Field(description="resumo do texto.")
    acoes_acompanhamento: str = Field(description="acoes de acompanhamento do texto.")
    sentimento: str = Field(description="sentimento expresso pelo individuo, deve ser 'positivo', 'negativo' ou 'neutro'.")


@tool(args_schema=ExtraiInformacoes)
def extrutura_informacao(
        data: str, 
        contatos: str, 
        meio: str,
        cargo: str, 
        organizacoes: str, 
        jurisdicoes: str, 
        representantes: str, 
        assunto: str, 
        resumo: str, 
        acoes_acompanhamento: str, 
        sentimento: str):
    
    """Extrutura as informações do texto"""
    return data, contatos, meio, cargo, organizacoes, jurisdicoes, representantes, assunto, resumo, acoes_acompanhamento, sentimento


toolls = [extrutura_informacao]
toolls_json = [convert_to_openai_function(tooll) for tooll in toolls]


prompt = ChatPromptTemplate.from_messages([
    ("system", f"Você é um assistente juridico que extrai informações do texto fornecido apenas quando todas as informações_necessarias estiverem presentes, e caso alguma delas não esteja, pergunte ao usuario antes de acionar a tool 'extrutura_informacao'. Pergunte uma coisa de cada vez até que todas as informações estejam presentes e você possa acionar o tool, fornecendo uma lista com todas informações contidas ao final. Para referencia a data atual é {data_atual}. Não utilize formatação markdown. Caso precise, sigo os exemplos em exemplos. Não use asteriscos '*' em suas mensagens. Proibido usar asteriscos '*' em suas mensagens. Proibido usar formatação markdown. Quando for listar algo, use '-' ao invés de '.' como nos exemplos exemplos_listas. REGRA: Para listar itens use o exemplo de exemplos_listas."),
    MessagesPlaceholder(variable_name="memory"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


def get_memory_for_user(whatsapp):
    memory = RedisChatMessageHistory(
        session_id=whatsapp, 
        redis_url=REDIS_URL)
    
    return ConversationBufferMemory(
        return_messages=True, 
        memory_key="memory", 
        chat_memory=memory)


pass_through = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_function_messages(x["intermediate_steps"])
)

@app.post("/webhook")
async def receive_message(request: Request):
    try:

        chain = pass_through | prompt | chat.bind(functions=toolls_json) | OpenAIFunctionsAgentOutputParser()
        
        body = await request.json()
        response = body.get("n8n_message", "")
        whatsapp_id = body.get("whatsapp_id", "")
        print("Mensagem recebida:", body)
        print(f"\n----------####### {whatsapp_id} #######------------")
        print(f"----------####### {response} #######------------\n")

        memoria = get_memory_for_user(whatsapp_id)
        print(f"Memória carregada para {whatsapp_id}: {memoria}")

        agent_executor = AgentExecutor(
            agent=chain,
            memory=memoria,
            tools=toolls,
            verbose=True,
            return_intermediate_steps=True
        )

        resposta = agent_executor.invoke({"input": response})
        resposta_final = resposta["output"]

        return {"status": resposta_final}



    except Exception as e:
        import traceback
        print("Erro no Webhook:", traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

   

    

   
        
    