# import streamlit as st

# st.title("All around bot!")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

    
# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # React to user input
# if prompt := st.chat_input("How may I help you?"):
            
#     # Display user message in chat message container
#     st.chat_message("user").markdown(prompt)
    
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     if 'summarize' in prompt.split(' '):
# #         input_ = st.text_input('Enter the document in text')
#         response = f"Your input is {input_}"

#     elif 'advice' in prompt.split(' '):
#         response = f"My counsel is"
#     else:
#         response = f"Hey there! How may I help you?"
        
#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         st.markdown(response)
        
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})


# # import streamlit as st
# # from langchain.chains import ConversationChain
# # from langchain.chains.conversation.memory import ConversationEntityMemory
# # from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
# # from langchain.chat_models import ChatOpenAI


# # def main():
# #     st.title("ChatGPT ChatBot🤖")
# #     st.markdown(
# #         ''' 
# #             > :black[**A ChatGPT based ChatBot 
# #             that remembers the context of the conversation.**]
# #             ''')

# #     API_KEY = st.text_input(":blue[Enter Your OpenAI API-KEY :]",
# #                             placeholder="Please enter your OpenAI API key here",
# #                             type="password")

# #     if API_KEY:
# #         st.write("API-KEY received.")

# #         MODEL = 'gpt-3.5-turbo'
# #         # An OpenAI instance
# #         llmObj = ChatOpenAI(openai_api_key=API_KEY,
# #                             model_name=MODEL)

# #         # A ConversationEntityMemory object
# #         K = 3  # number of user interactions as context
# #         if 'entity_memory' not in st.session_state:
# #             st.session_state.entity_memory = ConversationEntityMemory(
# #                 llm=llmObj, k=K)

# #         # The ConversationChain object
# #         Conversation = ConversationChain(
# #             llm=llmObj,
# #             prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
# #             memory=st.session_state.entity_memory
# #         )
# #     else:
# #         st.markdown('''
# #             ```
# #             - 1. Please enter the API Key and hit enter.

# #             - 2. Start your conversation with the text input widget below.
# #             ```
# #             ''')
# #         st.warning(
# #             'You need to enter the API key to have a working app.')
        
# #     # Get the user input
# #     user_input = st.text_input("You: ", st.session_state["input"],
# #                                 key="input",
# #                                 placeholder="Your Chatbot friend! Ask away ...",
# #                                 label_visibility='hidden')
    
# #     # Output using the ConversationChain object and the user input, 
# #     # and storing them in the session
# #     if user_input:
# #         output = Conversation.run(input=user_input)
# #         st.session_state.past.append(user_input)
# #         st.session_state.generated.append(output)

# #     # Display the conversation history using an expander
# #     with st.expander("Conversation", expanded=True):
# #         for i in range(len(st.session_state['generated'])-1, -1, -1):
# #             st.info(st.session_state["past"][i], icon="🧐")
# #             st.success(st.session_state["generated"][i], icon="🤖")


# # if __name__ == '__main__':
# #     st.set_page_config(page_title='ChatGPT ChatBot🤖', layout='centered')

# #     if "generated" not in st.session_state:
# #         st.session_state["generated"] = []
# #     if "past" not in st.session_state:
# #         st.session_state["past"] = []
# #     if "input" not in st.session_state:
# #         st.session_state["input"] = ""

# #     main()

import streamlit as st
import pandas as pd

import pyautogui
from PIL import Image
# if st.button("Reset"):
#     pyautogui.hotkey("ctrl","F5")

opt_method = st.sidebar.selectbox('Option', ['Guide', 'Button', 'Chat'])

if opt_method == 'Guide':
    st.markdown('#### This is an AI application that does a variety of text-based tasks.')
    st.markdown('#### This comes in two forms:')
    st.markdown('- Button or Chat interface.')

if opt_method == 'Chat':


    def submit_upon():
        st.session_state.print = st.session_state.input_1.lower()
        st.session_state.input_1 = ''

    with st.chat_message('ai'):
        st.image(Image.open('/Users/gzero/Downloads/mini_robot.png'))
        st.write("Hi, I'm your virtual assistant. I can do a lot of things, just ask!")

    user = st.text_input('Please enter your concern.', key = 'input_1', on_change = submit_upon)

#     try:
    st.text(f"Me: {st.session_state.print}")
    if len(set(['analyze', 'analysis', 'process']).intersection(st.session_state.print.split(' '))) > 0:
#         with st.chat_message('ai'):
        st.write('Definitely, how would you like to provide the data?')
        opt_method = st.text_input('Upload or Input')
        if opt_method == 'upload':
            file_upload = st.file_uploader('Upload the Document.')
            file_upload = file_upload.read().decode()
#             st.text(f'The uploaded file is a {type(file_upload)}')
            st.write(file_upload)
        if opt_method == 'Input':
            file_upload = st.text_input('Upload the Document.')
            st.write(file_upload)

        if opt_method:
            with st.status('Generating Insights'):
                from keybert import KeyBERT
                kb = KeyBERT()
                data_kw = kb.extract_keywords(file_upload)
                st.write(pd.DataFrame(data_kw))
                if st.checkbox('Keyword Pointers:'):
                    # Use a pipeline as a high-level helper
                    from transformers import pipeline
                    pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")
                    pt_dict = {}
                    for i in [x[0] for x in data_kw]:
                        st.write(i)
                        st.write(pipe({'question':f'what is the {i} all about?', 'context':file_upload}))

if st.sidebar.button('Clear'):
    st.text('Clear')
#     if st.button("Reset"):
    pyautogui.hotkey("cmd","R")
            

# #     except:
#         st.text('You will be guided by our intelligent bot as you go along.')

import openai
openai.api_key = 'YOUR_API_KEY'
messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]

message = st.text_input("User : ")
if message:
    messages.append({"role": "user", "content": message},)
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
reply = chat.choices[0].message.content
col2.write(f"ChatGPT: {reply}")
messages.append({"role": "assistant", "content": reply})