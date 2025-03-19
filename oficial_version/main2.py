from langchain.agents.format_scratchpad import format_to_openai_function_messages  # Remova: Não usado diretamente para Gemini
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser  # Remova: Específico para OpenAI
from langchain_core.utils.function_calling import convert_to_openai_function #Remova: Específico para OpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentExecutor
from fastapi import FastAPI, Request, HTTPException
#from langchain_openai import ChatOpenAI # Remova: Não é mais usado
from langchain_core.tools import tool
from langchain_core.tools import BaseTool
from langchain_redis import RedisChatMessageHistory
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
import redis
import requests
import os
import json # Importe json para manipulação de payloads
load_dotenv(find_dotenv())


informações_necessarias = """
# Data 
- Deve ser capturada automaticamente com base no dia em que a interação ocorreu, e não quando foi registrada, e caso o usuário fornecer termos relativos como "ontem" ou "amanhã", o você deve pedir uma data específica para evitar ambiguidade.
# Contatos 
- Pessoas presentes na reunião.
# Meio
- Caso o usuario não mencione o meio, pergunte qual foi o meio utilizado (ex: Google Meet, presencial, etc...).
# Organizações
- A qual organização pertenciam os membros presentes na reunião. Se o usuário mencionar apenas a jurisdição, ou um nome parcial de um órgão regulador, você deve confirmar o nome completo da organização e garantir a especificidade da jurisdição.
# Jurisdição
- Deve ser identificada através das informações obtidas através da organização do contato, mas jamais pergunte isso diretamente (exemplo: BCRA, já esta implicito que se trata de Argentina).
# Representantes da Empresa
- Se o usuário não mencionar explicitamente representantes da empresa, você deve solicitar que confirmem se alguém de sua empresa participou.
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


exemplo_confirmacao = """
Ótimo, obrigado pela informação. Com base no que você compartilhou, aqui estão as informações estruturadas da reunião:

- Data: 2025-03-17
- Contatos: Tamas Simonyi
- Meio: Google Meet
- Cargo: Não mencionado no texto original
- Organizações: Magyar Nemzeti Bank (Banco Central da Hungria)
- Jurisdição: Hungria
- Representantes: Victor (da sua empresa)
- Assunto: Compliance em ativos digitais na Hungria
- Resumo: Discussão sobre os requisitos para a Enterprise X garantir a conformidade com a estrutura regulatória de ativos digitais na Hungria. Discussão incluem o licenciamento VASP, exigências AML-CFT, proteção de dados GDPR, implicações fiscais e estratégias de mitigação de riscos.
- Sentimento: Neutro

Se todas as informações estiverem corretas, digite '1' para confirmar.
"""


app = FastAPI()
api_key = os.getenv("OPENAI_API_KEY")
REDIS_URL = "redis://default:A1ZDEbkF87w7TR0MPTBREnTFOnBgfBw9@redis-14693.c253.us-central1-1.gce.redns.redis-cloud.com:14693/0"

api_key_gemini = os.getenv("GEMINI_API_KEY")
chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key_gemini)



def obter_hora_e_data_atual():
    """Retorna a hora atual e a data de hoje."""
    agora = datetime.now()
    return agora.strftime("%Y-%m-%d - T%H:%M:%S")
data_atual = obter_hora_e_data_atual()


def get_memory_for_user(whatsapp):
    memory = RedisChatMessageHistory(
    session_id=whatsapp, 
    redis_url=REDIS_URL)
            
    return ConversationBufferMemory(return_messages=True, memory_key="memory", chat_memory=memory)

#---------------------------------------------Webhook----------------------------------------------------

@app.post("/webhook")
async def receive_message(request: Request):
    try:
        body = await request.json()
        response = body["n8n_message"]
        whatsapp_id = body['whatsapp_id']

        print("Mensagem recebida:", body)
        print(f"\n----------####### {whatsapp_id} #######------------")
        print(f"----------####### {response} #######------------\n")

        memoria = get_memory_for_user(whatsapp_id)
        print("-----", memoria, "------\n")

        #---------------------------------------------tools--------------------------------------------------

class ExtraiInformacoes(BaseModel):
            """Extrair informações de compliance"""
            data: str = Field(description="Data em que o evento ocorreu (se a pessoa disser algo como ontem ou algo do tipo, perguntar a data específica)")
    contatos: str = Field(
                description="Nome das pessoas contidas no texto", 
        examples=[
            ("Me chamo Rafael e falei com o Junior sobre assunto XYZ.", "Rafael, Junior"), 
                    ("Me chamo Alfredo e falei com o Severino sobre assunto a posse dos EUA.", "Alfredo, Severino")
                ])
            meio: str = Field(description="Meio de contato dos contatos mencionados. Deve perguntar se foi Google Meet, presencial, ou qual meio foi utilizado.")
            cargo: str = Field(description="Cargo dos contatos mencionados")
            organizacoes: str = Field(description="Organização/empresas dos contatos mencionados")
    jurisdicoes: str = Field(
                description="Deve ser identificada através das informações obtidas através da organização do contato. BCRA, já está implícito que se trata de Argentina.",
        examples=[
                    ("Jane Doe é uma Consultora Reguladora Sênior na Autoridade de Conduta Financeira (FCA) no Reino Unido. Eu não tenho o e-mail ou número de telefone dela no momento.", "Reino Unido"),
                    ("BCRA", "está implícito que se trata de Argentina")
                ])
            representantes: str = Field(description="Representantes dos contatos mencionados")
            assunto: str = Field(description="Assunto do texto")
            resumo: str = Field(description="Resumo do texto.")
            sentimento: str = Field(description="Sentimento expresso pelo indivíduo, deve ser 'positivo', 'negativo' ou 'neutro'.")


@tool(args_schema=ExtraiInformacoes)
        def extrutura_informacao(data: str, contatos: str, meio: str, cargo: str, organizacoes: str, jurisdicoes: str, representantes: str, assunto: str, resumo: str, sentimento: str):
            """Estrutura as informações do texto"""
            return data, contatos, meio, cargo, organizacoes, jurisdicoes, representantes, assunto, resumo, sentimento


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

    return {
                "contacts": contacts_list,
                "organizations": organizations_list
            }


        @tool
        def excluir_memoria(whatsapp_id):
            """Exclui memória após o usuário confirmar que a lista de mensagens está correta"""
            r = redis.from_url(REDIS_URL, decode_responses=True)

            session_prefix = f"chat:{whatsapp_id}"
            chaves = r.keys(session_prefix + "*")  
            
            for chave in chaves:
                r.delete(chave)
                print(chave)


        toolls = [extrutura_informacao, buscar_pessoas_tool, excluir_memoria]

        # Adaptação para Gemini: Remover a conversão para o formato OpenAI
        #toolls_json = [convert_to_openai_function(tooll) for tooll in toolls]

        #/---------------------------------------------tools-----------------------------------------------------


prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
                - Você é um assistente juridico que trabalha na Regdoor, seu trabalho é dividido em duas etapas:
                1-Através do input você identifica o nome de quem se esta falando (contato) e onde ele trabalha (organização), para então utilizar a tool 'buscar_pessoas_tool' e obter as demais informações que retornarão do database.
                2-Extrair as informações do texto quando todas as {informações_necessarias} estiverem presentes, após verificar se os textos fornecidos contem todas as informações necessarias, acionar a tool 'extrutura_informacao', caso alguma delas não esteja, pergunte ao usuario antes de acionar o tool.
                - Quando passar as informações listadas ao usuario, peça para confirmar se esta tudo correto, dizendendo que tecle 1 para confirmar, como no exemplo {exemplo_confirmacao} e depois acione a tool 'excluir_memoria' utilizando {whatsapp_id} como argumento.
                - Você responde na lingua/idioma do usuario. 
                REGRA: Você sempre responde na lingua/idioma do usuario.
                - Seja direto e conciso, assim que tiver o nome e empresa de onde o usuario atua, use a tool 'buscar_pessoas_tool' e retorne apenas seu nome completo e a organização em que trabalha, e caso não havendo o nome exato da pessoa naquela organização, retorne nomes similares dentro da mesma organização.
                - Pergunte uma coisa de cada vez até que todas as informações estejam presentes e você possa acionar as devidas tools, fornecendo uma lista com todas informações obtidas com a tool 'extrutura_informacao' ao final da conversa.
                - Faça apenas uma pergunta por vez.
                - Não fale que vai acionar uma tool ou qualquer coisa do tipo, o usuario não deve saber disto.
                - Sempre que o nome de alguém e sua organização estiver presentes na mensagem, pesquise no database utilizando a tool 'buscar_pessoas_tool'. 
                - Para referencia a data atual é {data_atual}. Não utilize formatação markdown. Caso precise, sigo os exemplos em {exemplos}. Não use asteriscos '*' em suas mensagens. Proibido usar asteriscos '*' em suas mensagens. Proibido usar formatação markdown. Quando for listar algo, use '-' ao invés de '.' como nos exemplos {exemplos_listas}. REGRA: Para listar itens use o exemplo de {exemplos_listas}.
        """),
    MessagesPlaceholder(variable_name="memory"),
    ("user", "{input}"),
            #Remova: Formatação específica do OpenAI
            #MessagesPlaceholder(variable_name="agent_scratchpad")
        ])


pass_through = RunnablePassthrough.assign(
            #Remova: Formatação específica do OpenAI
            #agent_scratchpad=lambda x: format_to_openai_function_messages(x["intermediate_steps"])
        )

        # Adaptação para Gemini: Remova OpenAIFunctionsAgentOutputParser e ajuste a chain
        #chain = pass_through | prompt | chat.bind(functions=toolls_json) | OpenAIFunctionsAgentOutputParser()
        chain = prompt | chat

        agent_executor = AgentExecutor(
            agent=chain,
            memory=memoria,
            tools=toolls,
            verbose=True,
            return_intermediate_steps=True  # Útil para depuração
        )

        resposta = agent_executor.invoke({"input": response, "tools": toolls}) # Inclui as tools como parte da invocação
        resposta_final = resposta

        return {"Status": resposta_final}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao processar JSON: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)