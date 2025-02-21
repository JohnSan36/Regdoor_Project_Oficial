from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from dotenv import load_dotenv, find_dotenv


#Importando o modelo de chat Openai e suas credenciais
load_dotenv(find_dotenv())
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)


#Trecho usado para exemplo
texto = ("""Olá, tive uma reunião mais cedo hoje com representantes tanto da Autoridade Europeia dos Valores Mobiliários e dos 
Mercados (ESMA) quanto da Autoridade de Supervisão Financeira da Suécia (FSA). Discutimos a harmonização das 
regulamentações de ativos digitais em jurisdições da UE e os desafios específicos na implementação de estruturas de 
conformidade.""")


#Criando a tool responsavel por extrair as informações do texto
class ExtrairInformacoes(BaseModel):
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



#Prompt do agente
prompt = ChatPromptTemplate.from_messages([
        ("""system", "Você é um assistente juridico que identifica quem é o sujeito que fala, com quem ele conversou, qual as entidades 
        mencionada no texto e qual o sentimento da conversa. Seja objetivo, sem papo furado. Caso o usuario não forneça alguma destas informações, pergunte"""),
        ("user", "{input}")
    ])


#Criando a chain e convertendo a ferramemnta
chamada_tool = convert_to_openai_function(ExtrairInformacoes)
chain = (prompt | chat.bind_functions([chamada_tool], function_call={"name":"ExtrairInformacoes"}) | JsonOutputFunctionsParser())

#invocando o modelo
resposta = chain.invoke({"input": texto})
print(f"\n\n{resposta}\n\n")