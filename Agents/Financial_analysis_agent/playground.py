from phi.agent import Agent
from phi.model.groq import Groq
import openai 
from phi.tools.yfinance import YFinanceTools #we can find many such tools like Pubmed in phidata
from phi.tools.duckduckgo import DuckDuckGo

import phi.api
from dotenv import load_dotenv
from phi.model.groq import Groq
import os
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api= os.getenv("PHI_API_KEY")

web_search_agent= Agent(
        name= "Web search agent",
        role="Search the web for info",
        model= Groq(od="llama3-groq-70b-8192-tool-use-preview"),
        tools= [DuckDuckGo(news=True)],
        instructions= ["always include the bibliography or source of info"],
        show_tools_calls=True,
        markdown=True,
)

## finance agent
finance_agent= Agent(
        name= "Finance agent",
        model= Groq(od="llama3-groq-70b-8192-tool-use-preview"),
        tools=[YFinanceTools(stock_price=True, company_info=True, technical_indicators=True, historical_prices=True, analyst_recommendations=True, stock_fundamentals=True)],
        instructions= ["use structured tables to display data"],
        show_tools_calls=True,
        markdown=True,



)


app= Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)


    #playground is the filename and app is given above for file to start execution