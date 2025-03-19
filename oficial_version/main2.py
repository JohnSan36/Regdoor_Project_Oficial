from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentExecutor
from fastapi import FastAPI, Request, HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.tools import BaseTool
from langchain_redis import RedisChatMessageHistory
from pydantic import BaseModel, Field
from datetime import datetime
import requests
import os
load_dotenv(find_dotenv())


informações_necessarias = """
# Date
- It must be automatically captured based on the day the interaction occurred, not when it was recorded. If the user provides relative terms like "yesterday" or "tomorrow," you should request a specific date to avoid ambiguity.
# Contacts
- People present at the meeting.
# Medium
- If the user does not mention the medium, ask which medium was used (e.g., Google Meet, in-person, etc.).
# Organizations
- Which organization the members present at the meeting belonged to. If the user mentions only the jurisdiction or a partial name of a regulatory body, you must confirm the full name of the organization and ensure the jurisdiction is specified.
# Jurisdiction
- If it is missing and cannot be identified through the organization, you must confirm the country or region related to the body or regulatory entity.
# Company Representatives
- If the user does not explicitly mention company representatives, you must ask them to confirm whether someone from their company participated.
# Follow-up Actions
- If follow-ups are mentioned, you must extract deadlines, responsible parties, and specific next steps.  
- If the user does not mention follow-up actions, ask if there are any tasks to be completed.
"""


informações_necessarias_2 = """
# Cargo
- Se a função de um contato for mencionada de forma incompleta (por exemplo, "consultor", "gerente"), o Agente de IA deve pedir o título completo (por exemplo, "Consultor Chefe de Políticas" em vez de apenas "Consultor").
- Se a função não estiver clara ou estiver faltando, o Agente de IA deve solicitar ao usuário que especifique a posição da pessoa dentro de sua organização.
- Se o usuário fornecer apenas uma designação geral como "executivo" ou "oficial", o Agente de IA deve esclarecer a função do indivíduo (por exemplo, regulatória, conformidade, política, jurídica).
- Se o contato for de um órgão regulador, confirme se ele está envolvido na formulação de políticas, supervisão ou aplicação da lei.
- Se o contato for de uma empresa, confirme se sua função é em conformidade, política, jurídica ou relações governamentais.
- Se o contato for de uma associação, confirme se ele é um representante da indústria, formulador de políticas ou especialista em defesa.
"""


exemplos = """
<exemplo 1>
# Correção de Entrada Parcial:
- Usuário: "Fiz uma ligação com um regulador do Reino Unido."
- IA: "Você poderia especificar qual órgão regulador e o nome do indivíduo?"
# Nome da Organização Ausente:
- Usuário: "Encontrei-me com John Doe."
- IA: "Você poderia confirmar qual organização John Doe representa?"
# Confirmando Acompanhamentos:
- Usuário: "Discutimos atualizações de políticas."
- IA: "Alguma ação de acompanhamento ou prazo específico foi mencionado?"
</exemplo 1>

<exemplo 2>
1- Data: Você mencionou "ontem". Poderia me informar a data específica em que a conversa ocorreu?
2- Contatos: Você se referiu a um representante do Banco Central do Brasil. Qual é o nome completo dele?
3- Jurisdição: Presumo que estamos falando sobre o Brasil, está correto?
4- Representantes da Empresa: Algum representante da sua empresa participou dessa conversa?
5- Ações de Acompanhamento: Você mencionou que precisamos preparar documentação adicional. Há prazos ou responsáveis para essa tarefa?
</exemplo 2>
"""


exemplos_listas = """
# Usando '-' ao invés de '.'.
1-Fulano de ciclano pelinous;
2-Lorem Ipsum é simplesmente uma simulação de texto da indústria tipográfica e de impressos, e vem sendo utilizado desde o século XVI, quando um impressor desconhecido pegou uma bandeja de tipos e os embaralhou para fazer um livro de modelos de tipos;
3-Se popularizou na década de 60, quando a Letraset lançou decalques contendo passagens de Lorem Ipsum;
4-Existem muitas variações disponíveis de passagens de Lorem Ipsum, mas a maioria sofreu algum tipo de alteração, seja por inserção de passagens com humor, ou palavras aleatórias que não parecem nem um pouco convincentes;
"""


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
    assunto: str = Field(description="assunto do texto o qual você deve identificar através do historico de conversas.")
    resumo: str = Field(description="resumo do texto o qual você deve criar através do historico de conversas..")
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


class BuscarPessoasSchema(BaseModel):
    contato: str = Field(description="Nome ou parte do nome do contato.")
    organization: str = Field(description="Nome da organização a ser buscada.")


@tool(args_schema=BuscarPessoasSchema)
def buscar_pessoas_tool(contato: str, organization: str):
    """Busca contatos e organizações utilizando a API do Regdoor."""

    url_contact = f"https://dev-api.regdoor.com/api/ai/contacts?query={contato}"
    url_organization = f"https://dev-api.regdoor.com/api/ai/organizations?query={organization}"

    try:
        return_contacts = requests.get(url_contact)
        return_contacts.raise_for_status()
        return_organization = requests.get(url_organization)
        return_organization.raise_for_status()

    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    
    contacts_list = return_contacts.json()['items']
    organizations_list = return_organization.json()['items']

    target_uuid_set = {org['uuid'] for org in organizations_list if org['name'].lower() == organization.lower()}
    
    filtered_contacts = [
        contact for contact in contacts_list 
        if contato.lower() in contact['name'].lower()
        and any(org['uuid'] in target_uuid_set for org in contact['organizations'])
    ]

    return {
        "contacts": filtered_contacts,
        "organizations": [org for org in organizations_list if org['name'].lower() == organization.lower()],
    }

toolls = [extrutura_informacao, buscar_pessoas_tool]
toolls_json = [convert_to_openai_function(tooll) for tooll in toolls]


prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
        - You are a legal assistant that extracts information from the provided texts only when all the {informações_necessarias} is present, after verifying if the supplied texts contain all the necessary information, activate the 'extract_information' tool; if any of it is missing, ask the user before activating the tool.
        - Whenever you have the name of an individual and the organization they work for, activate the 'search_people_tool' and return only their full name and the organization they work for, and if the exact name of the person is not found in that organization, return similar names within the same organization.
        - Ask one thing at a time until all the information is present and you can activate the appropriate tools, providing a list with all the information obtained with the 'extract_information' tool at the end of the conversation.
        - Whenever there is a person's name and their organization, search the database using the 'search_people_tool'.
        - For reference, the current date is {data_atual}. Do not use markdown formatting. If needed, follow the examples in {exemplos}. Do not use asterisks '*' in your messages. It is forbidden to use asterisks '*' in your messages. It is forbidden to use markdown formatting. When listing something, use '-' instead of '.' as in the examples {exemplos_listas}. RULE: For listing items, follow the example of {exemplos_listas}.
        """),
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
        response = body["n8n_message"]
        whatsapp_id = body['whatsapp_id']
        
        print("Mensagem recebida:", body)
        print(f"\n----------####### {whatsapp_id} #######------------")
        print(f"----------####### {response} #######------------\n")

        memoria = get_memory_for_user(whatsapp_id)
        print("-----------------------", memoria, "-----------------------\n")

        agent_executor = AgentExecutor(
            agent=chain,
            memory=memoria,
            tools=toolls,
            verbose=True,
            return_intermediate_steps=True
        )

        resposta = agent_executor.invoke({"input": response})
        resposta_final = resposta["output"]

        return {"Status": resposta_final}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao processar JSON: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)