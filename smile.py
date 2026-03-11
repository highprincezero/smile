from   PIL          import Image
from   keybert      import KeyBERT
from   transformers import pipeline
import streamlit    as st
import pandas       as pd
# import pyautogui
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

with st.status('Installing packages'):
    os.system("pip install -r ./requirements.txt")

opt_method = st.sidebar.selectbox('Option', ['Guide', 'Button', 'Chat'])

pipeline("summarization", model="facebook/bart-large-cnn")
pipeline("question-answering", model="deepset/roberta-base-squad2")

if opt_method == 'Guide':
    st.markdown('#### This is an AI application that does a variety of text-based tasks.')
    st.markdown('#### This comes in two forms:')
    st.markdown('- Button or Chat interface.')

if opt_method == 'Chat':
    col1, col2, col3 = st.columns(3)
    # if col3.button('Clear'):
    #     pyautogui.hotkey("command", "r")
    
    col3.caption('This bot can perform the following: ')
    col3.caption('    ||  analyze a document or documents,')
    col3.caption('    ||  predict using customized models,')
    col3.caption('    ||  generate inferences based on global intelligence,')
    col3.caption('    ||  locate information, etc.')


    def submit_upon():
        st.session_state.print   = st.session_state.input_1.lower()
        st.session_state.input_1 = ''

#     col1, col2, col3 = st.columns(3)
    with col2.chat_message('ai'):
        col1.image(Image.open('./mini_robot.png'))
#         with st.expander('Read Source'):
        col2.write("Hi, I'm your virtual assistant. I can do a lot of things, just ask!")

    user = col2.text_input('Please enter your concern.', key = 'input_1', on_change = submit_upon)
#     camera = col3.camera_input('Live')
#     col3.write(type(bytes_data))
#     if camera:
#         col3.image(camera)
    
    try:
        col2.markdown(f'<div style="text-align: right;">Me: {st.session_state.print.title()}</div>', unsafe_allow_html=True)
    
        if len(set(['predict', 'infer']).intersection(st.session_state.print.lower().split(' '))) > 0:
            with col2.status('Retrieving Models...'):
                models_ls = os.listdir('models')
                if len(models_ls) == 0:
                    col2.write('No models housed.')
        if len(set(['ask', 'question']).intersection(st.session_state.print.lower().split(' '))) > 0:
            opt_source = col2.radio('Source', ['Domain', 'Global'], horizontal = True)
            message = col2.text_input("User : ")
            if opt_source == 'Global':       
                
                messages       = [{"role": "system", "content": "You are a intelligent assistant."}]
                if message:
                    messages.append({"role": "user", "content": message},)
                    chat = client.chat.completions.create(model=os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo"), messages=messages)
                reply = chat.choices[0].message.content
                col2.caption(reply)
                messages.append({"role": "assistant", "content": reply})
                
            if opt_source == 'Domain':
#                 with col2.status('Source'):
                source_file = col2.file_uploader('Source Upload')
                source_doc  = source_file.read().decode()
                    
                with col3.expander('Read Source'):
                    col3.caption(source_doc)
                
                pipe_src    = pipeline("question-answering", model="deepset/roberta-base-squad2")
                reply = pipe_src({'question':message, 'context':source_doc})
                col2.table(pd.DataFrame(reply))


                
        if len(set(['analyze', 'analysis', 'process']).intersection(st.session_state.print.split(' '))) > 0:
            col2.markdown('<div style="text-align: left;">ai: Definitely, how would you like to provide the data?</div>', unsafe_allow_html=True)
            col2.markdown('----')
            col2.header('DOCUMENT ANALYZER')
            opt_method = col2.radio('Method:', ['Upload', 'Write'], horizontal = True)

            if opt_method.lower() == 'upload':
                col2.markdown('----')
                file_upload = col2.file_uploader('Upload the Document.')
                file_upload = file_upload.read().decode()

            if opt_method.lower() == 'text':
                file_upload = col2.text_input('Upload the Document.')

            if opt_method.lower() in ['upload', 'text']:
                col2.caption(file_upload)
                with col2.status('Generating Insights'):
                    kb      = KeyBERT()
                    data_kw = kb.extract_keywords(file_upload)
                    col2.table(pd.DataFrame(data_kw).sort_values(by = 1, ascending = False).T)

                    if col2.checkbox('What is the corresponding information?'):
                        pipe    = pipeline("question-answering", model="deepset/roberta-base-squad2")
                        pt_dict = []
                        for i in [x[0] for x in data_kw]:
                            res = pipe({'question':f'what is the {i} all about?', 'context':file_upload})
                            pt_dict.append([i, res['answer'], res['score']])
#                             col2.write(i)
#                             col2.write(pipe({'question':f'what is the {i} all about?', 'context':file_upload}))
                        col2.table(pd.DataFrame(pt_dict).T)
                        
            if col2.checkbox('Summarize'):
                pipe_summary = pipeline("summarization", model="facebook/bart-large-cnn")
                summary      = pipe_summary(file_upload)[-1]['summary_text']
                col2.caption(summary)
                        
    except:
        st.write('')
