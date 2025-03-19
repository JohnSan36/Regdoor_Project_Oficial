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
from langchain_redis import RedisChatMessageHistory
from pydantic import BaseModel, Field
from datetime import datetime
import redis
import requests
import os
load_dotenv(find_dotenv())


informações_necessarias = """
# Date
- It should be captured automatically based on the day the interaction occurred, not when it was recorded, and if the user provides relative terms like "yesterday" or "tomorrow," you should ask for a specific date to avoid ambiguity.
# Contacts
- People present at the meeting.
# Medium
- If the user does not mention the medium, ask what medium was used (e.g., Google Meet, in-person, etc.).
# Organizations
- To which organization the members present at the meeting belonged. If the user mentions only the jurisdiction or a partial name of a regulatory body, you should confirm the full name of the organization and ensure the specificity of the jurisdiction.
# Jurisdiction
- It should be identified through the information obtained from the contact's organization, but never ask this directly (example: BCRA, it is already implied that it is Argentina).
# Company Representatives
- If the user does not explicitly mention company representatives, you should ask them to confirm if someone from their company participated.
"""


informações_necessarias_2 = """
# Position
- If a contact's role is mentioned incompletely (e.g., "consultant," "manager"), the AI Agent should ask for the full title (e.g., "Chief Policy Consultant" instead of just "Consultant").
- If the role is unclear or missing, the AI Agent should ask the user to specify the person's position within their organization.
- If the user provides only a general designation like "executive" or "officer," the AI Agent should clarify the individual's role (e.g., regulatory, compliance, policy, legal).
- If the contact is from a regulatory body, confirm if they are involved in policy-making, supervision, or law enforcement.
- If the contact is from a company, confirm if their role is in compliance, policy, legal, or government relations.
- If the contact is from an association, confirm if they are an industry representative, policy maker, or advocacy specialist.
"""



exemplos = """
<example 1>
# Partial Input Correction:
- User: "I made a call with a UK regulator."
- AI: "Could you specify which regulatory body and the individual's name?"
# Missing Organization Name:
- User: "I met with John Doe."
- AI: "Could you confirm which organization John Doe represents?"
# Confirming Follow-ups:
- User: "We discussed policy updates."
- AI: "Was any follow-up action or specific deadline mentioned?"
</example 1>

<example 2>
1- Date: You mentioned "yesterday." Could you provide the specific date when the conversation occurred?
2- Contacts: You referred to a representative from the Central Bank of Brazil. What is their full name?
3- Jurisdiction: I assume we are talking about Brazil, is that correct?
4- Company Representatives: Did any representative from your company participate in this conversation?
5- Follow-up Actions: You mentioned we need to prepare additional documentation. Are there deadlines or responsible parties for this task?
</example 2>
"""


exemplos_listas = """
# Using '-' instead of '.'.
1-Fulano de ciclano pelinous;
2-Lorem Ipsum is simply a dummy text of the printing and typesetting industry, and has been used since the 16th century, when an unknown printer took a galley of type and scrambled it to make a type specimen book;
3-It became popular in the 60s, when Letraset released sheets containing Lorem Ipsum passages;
4-There are many variations of Lorem Ipsum passages available, but most have suffered some form of alteration, either by the insertion of humor passages or random words that do not seem even slightly believable;
"""


exemplo_confirmacao = """
Great, thank you for the information. Based on what you shared, here are the structured details of the meeting:

- Date: 2025-03-17
- Contacts: Tamas Simonyi
- Medium: Google Meet
- Organizations: Magyar Nemzeti Bank (Central Bank of Hungary)
- Jurisdiction: Hungary
- Representatives: Victor (from your company)
- Subject: Compliance in digital assets in Hungary
- Summary: Discussion on the requirements for Enterprise X to ensure compliance with the digital asset regulatory framework in Hungary. Discussions include VASP licensing, AML-CFT requirements, GDPR data protection, tax implications, and risk mitigation strategies.
- Sentiment: Neutral

If all the information is correct, type '1' to confirm.
"""

nao_fazer = """
I couldn't find an exact contact at the Central Bank of Brazil with the name Luiz, but I have some options that might interest you:

- Luiz Carlos, associated with BCRA
- Luis Senna iii, associated with the Qatar Financial Centre Regulatory Authority (QFCRA)
- Luis Fuenmayor, associated with the Metropolitan Transportation Commission
# Note that here he mentioned random people and organizations, but he can only mention people from the mentioned organization, as the only Luis within BCRA is Luiz Carlos.

If any of these names seem familiar or related to your meeting, please let me know. To continue, confirm if any of these contacts is the correct person and if there was any other representative from your company at the meeting.
"""

app = FastAPI()
api_key = os.getenv("OPENAI_API_KEY")

chat_4o_mini = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0.0)
chat_4o = ChatOpenAI(model="gpt-4o", openai_api_key=api_key, temperature=0.0)

REDIS_URL = "redis://default:A1ZDEbkF87w7TR0MPTBREnTFOnBgfBw9@redis-14693.c253.us-central1-1.gce.redns.redis-cloud.com:14693/0"


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
            """Extract compliance information"""
            data: str = Field(description="Date when the event occurred (if the person says something like yesterday or similar, ask for the specific date)")
            contatos: str = Field(
                description="Names of the people mentioned in the text", 
                examples=[
                    ("My name is Rafael and I spoke with Junior about XYZ topic.", "Rafael, Junior"), 
                    ("My name is Alfredo and I spoke with Severino about the US inauguration topic.", "Alfredo, Severino")
                ])
            meio: str = Field(description="Contact method of the mentioned contacts. Should ask if it was Google Meet, in-person, or which method was used.")
            #cargo: str = Field(description="Position of the mentioned contacts")
            organizacoes: str = Field(description="Organization/companies of the mentioned contacts")
            jurisdicoes: str = Field(
                description="Should be identified through the information obtained from the contact's organization. BCRA, it is already implied that it is Argentina.",
                examples=[
                    ("Jane Doe is a Senior Regulatory Consultant at the Financial Conduct Authority (FCA) in the United Kingdom. I don't have her email or phone number at the moment.", "United Kingdom"),
                    ("BCRA", "it is implied that it is Argentina")
                ])
            representantes: str = Field(description="Representatives of the mentioned contacts")
            assunto: str = Field(description="Subject of the text")
            resumo: str = Field(description="Summary of the text.")
            sentimento: str = Field(description="Sentiment expressed by the individual, should be 'positive', 'negative', or 'neutral'.")


        @tool(args_schema=ExtraiInformacoes) 
        def extrutura_informacao(data: str, contatos: str, meio: str, organizacoes: str, jurisdicoes: str, representantes: str, assunto: str, resumo: str, sentimento: str):
            """Structure the information from the text"""
            return data, contatos, meio, organizacoes, jurisdicoes, representantes, assunto, resumo, sentimento


        class BuscarPessoasSchema(BaseModel):
            contato: str = Field(description="Name or part of the contact's name.")
            organization: str = Field(description="Name of the organization to be searched.")

        @tool(args_schema=BuscarPessoasSchema)
        def buscar_pessoas_tool(contato: str, organization: str):
            """Search contacts and organizations using the Regdoor API."""

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
        def excluir_memoria(whatsapp_id: str):
            """Exclui memória após o usuário confirmar que a lista de mensagens está correta"""
            r = redis.from_url(REDIS_URL, decode_responses=True)

            session_prefix = f"chat:{whatsapp_id}"
            chaves = r.keys(session_prefix + "*")  
            
            for chave in chaves:
                r.delete(chave)
                print(chave)


        toolls = [extrutura_informacao, buscar_pessoas_tool, excluir_memoria]
        toolls_json = [convert_to_openai_function(tooll) for tooll in toolls]

        #/---------------------------------------------tools-----------------------------------------------------


        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
                - You are a legal assistant working at Regdoor, and your work is divided into two stages:
                1-Through the input, you identify the name of the person being referred to (contact) and where they work (organization), then use the tool 'buscar_pessoas_tool' to obtain additional information that will be returned from the database, To then confirm whether the person you found is the same person the user is talking about.
                2-You identify the language of user's input and respond in that language.
                3-Extract information from the text when all {informações_necessarias} are present. After verifying that the provided texts contain all necessary information, trigger the tool 'extrutura_informacao'. If any are missing, ask the user before triggering the tool.
                - When presenting the listed information to the user, ask them to confirm if everything is correct by pressing 1 to confirm, as in the example {exemplo_confirmacao}, and then trigger the tool 'excluir_memoria' using {whatsapp_id} as an argument.
                - You respond in the user's language. 
                RULE: You always respond in the user's language.
                - Be direct and concise. As soon as you have the name and company where the user works, use the tool 'buscar_pessoas_tool' and return only their full name and the organization they work for. If the exact name of the person in that organization is not found, return similar names within the same organization.
                - Ask one thing at a time until all information is present and you can trigger the appropriate tools, providing a list with all information obtained with the tool 'extrutura_informacao' at the end of the conversation.
                - Ask only one question at a time.
                Rule: Only bring into the conversation the names of contacts belonging to the mentioned organization, never bring contacts or organizations that are not related to the conversation. Here is an example of what "NOT TO DO" {nao_fazer}.
                - Do not mention that you are going to trigger a tool or anything like that; the user should not know this.
                - Whenever someone's name and their organization are present in the message, search the database using the tool 'buscar_pessoas_tool'. 
                - For reference, the current date is {data_atual}. Do not use markdown formatting. If needed, follow the examples in {exemplos}. Do not use asterisks '*' in your messages. It is prohibited to use asterisks '*' in your messages. It is prohibited to use markdown formatting. When listing something, use '-' instead of '.' as in the examples {exemplos_listas}. RULE: To list items, use the example from {exemplos_listas}.
            """),
            MessagesPlaceholder(variable_name="memory"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])


        pass_through = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_function_messages(x["intermediate_steps"])
        )
        
        chain = pass_through | prompt | chat_4o.bind(functions=toolls_json) | OpenAIFunctionsAgentOutputParser()
        chain_backup = pass_through | prompt | chat_4o_mini.bind(functions=toolls_json) | OpenAIFunctionsAgentOutputParser()

        chain_fallback = chain.with_fallbacks([chain_backup])

        agent_executor = AgentExecutor(
            agent=chain_fallback,
            memory=memoria,
            tools=toolls,
            verbose=True,
            return_intermediate_steps=True
        )

        resposta = agent_executor.invoke({"input": response})
        resposta_final = resposta

        return {"Status": resposta_final}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao processar JSON: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)