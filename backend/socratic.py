from langchain_openai import ChatOpenAI

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="gpt-3.5-turbo",
    temperature=0.5,
    max_completion_tokens=500
)

prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a student studying assistant. You will ask the student questions 
     about a topic based on their understanding, and you will identify gaps in their explanations."""),
     MessagesPlaceholder(variable_name='history'),
     ("human","{input}")
])

chain = prompt | llm

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def test_multiturn():
    while True:
        user_input = input("You: ")
        if user_input == "exit()":
            break
        response = conversation.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )
        print(response.content)