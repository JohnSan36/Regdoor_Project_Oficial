from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema.agent import AgentFinish
#from langchain_core.tools import StructuredTool
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from pydantic import BaseModel, Field
#from langchain.tools import Tool
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
load_dotenv(find_dotenv())


chat = ChatOpenAI(model="gpt-4o-mini")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")


def obter_hora_e_data_atual():
    """Retorna a hora atual e a data de hoje."""
    agora = datetime.now()
    return agora.strftime("%d-%m-%Y - %H:%M:%S")


data_atual = obter_hora_e_data_atual()
app = Flask(__name__)


memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)


@app.route("/webhook", methods=["POST"])
def receive_message():
    
    try:
        body_corp = request.data  
        body_str = body_corp.decode("utf-8")
        print("Mensagem recebida:", body_str)  

        class ExtraiInformacoes(BaseModel):
            """Extrair informações dos textos fornecidos"""
            data: str = Field (description="data em que o o evento ocorreu")
            contatos : str = Field(
                description="nome das pessoas contidas no texto", 
                examples=[
                    ("Me chamo Rafael e falei com o Junior sobre assunto XYZ.", "Rafael, Junior"), 
                    ("Me chamo Alfredo e falei com o Severino sobre assunto a posse dos Eua.", "Alfredo, Severino")
                ])
            cargo: str = Field(description="cargo dos contatos mencionados")
            organizacoes : str = Field(description="organizaçao dos contatos mencionados")
            jurisdicoes : str = Field(
                description="jurisdições mencionadas",
                examples=[
                    ("Jane Doe is a Senior Regulatory Advisor at the Financial Conduct Authority (FCA) in the UK. I don't have her email or phone number at the moment. ", "UK")
                ])
            representantes : str = Field(description="representantes dos contatos mencionados")
            assunto : str = Field(description="assunto do texto, deve ser 'politica', 'economia' ou 'justica'.")
            resumo : str = Field(description="resumo do texto, deve ser uma breve descrição do evento, com no máximo 100 caracteres.")
            acoes_acompanhamento : str = Field(description="acoes de acompanhamento do texto.")
            sentimento : str = Field(description="sentimento expresso pelo individuo, deve ser 'positivo', 'negativo' ou 'neutro'.")


        @tool(args_schema=ExtraiInformacoes)
        def extrutura_informacao(
                data: str, 
                contatos : str, 
                cargo: str, 
                organizacoes: str, 
                jurisdicoes: str, 
                representantes: str, 
                assunto: str, 
                resumo: str, 
                acoes_acompanhamento: str, 
                sentimento: str ):
            
            """Extrutura as informações do texto"""
            return data, contatos, cargo, organizacoes, jurisdicoes, representantes, assunto, resumo, acoes_acompanhamento, sentimento


        toolls = [extrutura_informacao]
        toolls_json = [convert_to_openai_function(tooll) for tooll in toolls]


        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Você é um assistente juridico que extrai informações do texto fornecido. Para referencia a data atual é {data_atual}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        print(memory)


        pass_through = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_function_messages(x["intermediate_steps"])
        )
        agent_chain = pass_through | prompt | chat.bind(functions=toolls_json) | OpenAIFunctionsAgentOutputParser()


        def run_agent(input):
            passos_intermediarios = []
            while True:
                resposta = agent_chain.invoke({
                    "input": input,
                    "agent_scratchpad": format_to_openai_function_messages(passos_intermediarios)
                })
                if isinstance(resposta, AgentFinish):
                    return resposta
                observacao = toolls[resposta.tool].run(resposta.tool_input)
                passos_intermediarios.append((resposta, observacao))


        agent_executor = AgentExecutor(
            agent=agent_chain,
            memory=memory,
            tools=toolls,
            verbose=True,
            return_intermediate_steps=True
        )

        #print("\n\nConteúdo da memória antes da execução:", memory.load_memory_variables({}))

        resposta = agent_executor.invoke({"input": body_str})
        resposta_final = resposta["output"]
        
        #print(resposta)
        #print("\n\nConteúdo da memória após a execução:", memory.load_memory_variables({}))

        return jsonify({"Status": resposta_final})

    except Exception as e:
        return jsonify({"Erro": f"Falha ao processar JSON: {str(e)}"}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)