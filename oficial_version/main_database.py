from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
import asyncio
import os


DATABASE_URL = "postgresql://regdoorbduser:LKn67H44ghymn@dev-database-nllo96lo2zax-db-v1x8qclv7fdy.chm8ca0kc5t7.eu-west-1.rds.amazonaws.com:5432/regdoor"
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")


if DATABASE_URL is None:
    raise ValueError("DATABASE_URL não está definida.")


async def main():  
    db = SQLDatabase.from_uri(DATABASE_URL)

    chat = ChatOpenAI(model="gpt-3.5-turbo")
    agent_executer = create_sql_agent(
        chat,
        db=db,
        verbose=True,
        agent_type="tool-calling",
        allow_dangerous_code=True 
    )

    try:
        result = await agent_executer.ainvoke({"input": "Existe alguem chamado Dan em contacts?"})
        print(result["output"])
    except Exception as e:
        print(f"Erro ao executar o agente: {e}")

if __name__ == "__main__":
    asyncio.run(main())  