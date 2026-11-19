import os
from dotenv import load_dotenv

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7
)

class ChatState(TypedDict):
    message: str
    response: str

def chatbot(state: ChatState):
    try:
        result = llm.invoke(state["message"])

        return {
            "response": result.text
            }
    except Exception as e:
        return{
            "response" : "Sorry, Gemini is Temporarily unavailable. Please try again"
        }

graph = StateGraph(ChatState)
graph.add_node("chatbot",chatbot)

graph.add_edge(START,"chatbot")
graph.add_edge("chatbot",END)

app = graph.compile()

while True:
    user_input = input("\n You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye !")
        break
    result = app.invoke({
        "message": user_input,
        "response": ""
    })
    print("Chatbot: ",result["response"])


