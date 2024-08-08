import streamlit as st
import os, time, openai
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from save_chat_log import save_chat_log, get_device_info
from scenario_based_answer import load_scenarios, get_response_from_scenario, append_to_scenario, get_closest_match
# from langchain.chains import RetrievalQA

# OpenAI APIキーの読み込み
with open("api_key.txt", "r") as file:
    global OPENAI_API_KEY 
    OPENAI_API_KEY = file.read().strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# チャットGPTのモデルを指定
gpt_model = "gpt-4o-mini"
# gpt_model = "gpt-4o"

# 初期画面表示
def init_page():
    st.set_page_config(
        page_title="BelxChatBot",
        page_icon="🤗"
    )
    st.header("BelxChatBot 🤗")

# 初期化
def init_state():
    if 'stop' not in st.session_state:
        st.session_state['stop'] = False
    if 'chat_message' not in st.session_state:
        st.session_state.chat_message = []
        st.session_state['messages'] = [SystemMessage(content="質問をどうぞ")]
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'chat_history' not in st.session_state:
        st.session_state['history'] = []
    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = []

@st.cache_resource
def save_memory(userid):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return memory

def main():
    init_page()
    init_state()

    # 応答停止ボタン
    if st.sidebar.button('応答停止'):
        st.session_state['stop'] = True

    # 履歴クリアボタン
    if st.sidebar.button("履歴クリア"):
        st.session_state['feedback']=[]
        st.session_state.messages = [
            SystemMessage(content="質問をどうぞ")
        ]
        st.session_state.costs = []

    # LLMモデルとベクトルストアの設定
    llm = ChatOpenAI(
        temperature=0.2, 
        model_name=gpt_model,
        streaming=True, 
    )

    # 回答生成プロセス
    def conversational_chat(query):
        computer_name, user_name, local_ip_address = get_device_info()
        # エンベディングモデル設定
        embeddings = OpenAIEmbeddings()
        # ベクトルストア設定
        vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
        query = "以下の質問に日本語で回答して下さい。回答が得られなかった場合は、「回答NG」と回答してください。" + query
        # chat_qa = ConversationalRetrievalChain(
        # retrieverの設定を変更
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
         # メモリの設定
        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chat_qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            # vectorstore.as_retriever(),
            # memory=memory,
            return_source_documents=True,
            # return_vector_score=True,
            )

    #     chat_qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vectorstore.as_retriever(),
    #     return_source_documents=True
    # )
        
        vectordbkwargs = {"search_distance": 0.9}
        result = chat_qa({'question': query, 'chat_history':st.session_state['history'],"vectordbkwargs": vectordbkwargs})
        # result = chat_qa({'query':query})
        print(f"検索結果:{result}")


        # ベクトル類似度の実験
        # ベクトル類似度の取得方法を変更
        if 'source_documents' in result and result['source_documents']:
            for doc in result['source_documents']:
                print(f"回答: {doc.page_content}")
                if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                    print(f"ベクトル類似度: {doc.metadata['score']}")
                else:
                    print("このドキュメントにはスコアがありません")

        # if 'vector_similarity' in result:
        #     for doc, similarity in zip(result['source_documents'], result['vector_similarity']):
        #         print(f"回答: {doc.text}")
        #         print(f"ベクトル類似度: {similarity}")
        else:
            print("ベクトル類似度が検出できません")

        if 'source_documents' in result:
            source_documents = result['source_documents']
        # 各Documentオブジェクトのmetadataからsourceを取得
            sources = []
            similarities = []
            for doc in source_documents:
                if 'source' in doc.metadata:
                    source = doc.metadata['source']
                    if source not in sources:
                        sources.append(source)
                    if 'similarity' in doc.metadata:
                        similarities.append(doc.metadata['similarity'])
            # sources = set(doc.metadata['source'] for doc in source_documents if 'source' in doc.metadata)
            print(similarities)
            sources_str = '\n'.join(sources)  # sourcesリストを文字列に変換
            print('回答にあたっては、以下を参考にしています。\n', sources_str)
            answer_with_sources = f"{result['answer']} (回答にあたっては、以下を参考にしています。: {sources_str})"
        else:
            print('source_documentsが結果に含まれていません。')

        st.session_state['history'].append((query, answer_with_sources))
        return answer_with_sources

    # 回答がマニュアルに無い場合の会話
    def conversational_chat_whth_fallback(user_input):
        response = conversational_chat(user_input)
        if "回答NG" in response or "お手伝いできません" in response or "回答できません" in response:
            print('社内マニュアルに回答が無いので、チャットGPTが回答を生成します')
            response = get_chatgpt_response(user_input)
        return response
    
    # ChatGPTによる回答生成
    def get_chatgpt_response(message):
        try:
            completion = openai.chat.completions.create(
            model=gpt_model,
             messages=[
                {"role": "user", "content": \
                 "あなたは社内マニュアルの案内を行うアシスタントですが、マニュアルにない質問に対する\
                 回答を行います。冒頭に「社内マニュアルに回答が存在しません。チャットGPTが一般的な回答を行います」\
                と前置きしてから回答してください",},
                {"role": "user", "content": message}
                ],
            max_tokens=1000, # 最大トークン数を設定
            # stream=True,
            )
            # print(completion.choices[0].message.content)
            print("API Response:", completion)  # デバッグ出力
            return completion.choices[0].message.content.strip()

        except Exception as e:
            print("API Error:", e)  # エラー出力
            return f"エラー: {e}\n 管理者へ報告してください"


    messages = st.session_state.get('messages', [])
        # print(messages)

    # 会話の履歴を表示
    for message in messages:
        # AIからの回答の場合
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        # ユーザーからのメッセージの場合
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        # それ以外（エラーメッセージとか）
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")

    # フィードバックの状態に基づいてメッセージを表示
    feedback = st.session_state.get('feedback', None)  # feedbackキーがなければNoneを返す
    if feedback == "good":
        with st.chat_message(''):
            st.success("フィードバックありがとうございます")
            # フィードバックがgoodで、シナリオに基づかない回答だった場合に質問と回答を記録
            append_to_scenario(st.session_state.get('current_question', ''), st.session_state.get('current_response', ''))
            st.session_state['feedback']=[]
            print('append_to_scenario & feedback_reset')

    elif feedback == "bad":
        with st.chat_message(''):
            st.error("フィードバックありがとうございます")
            st.session_state['feedback']=[]
            print('bad feedback')
        # feedbackキーがない場合や、'good'/'bad'以外の場合は何も表示しない

    # テキストボックスの初期値
    user_input = st.chat_input("聞きたいことを入力してね！")
    scenario_df = load_scenarios()
    if scenario_df is None:
        print("シナリオファイル読み込みerror発生")
        exit
    if user_input: # 質問が来たら処理開始
        # print(user_input)

        # 応答生成開始時間を記録
        start_time = time.time()
        if st.session_state['stop']:
        # 新しい入力がある場合、強制停止状態をリセット
            st.session_state['stop'] = False
        # メッセージの履歴に追加
        st.session_state.messages.append(HumanMessage(content=user_input))
        # 質問内容を画面に表示
        st.chat_message("user").markdown(user_input)

        # 回答生成プロセス
        closest_match, scenario_response, is_scenario_based = get_closest_match(user_input, scenario_df)
        with st.spinner("回答生成中"):
            # with st.chat_message('assistant',avatar=img_beorder):
            with st.chat_message('assistant'):
                if is_scenario_based :
                    print('シナリオあり')
                    response = scenario_response + "(シナリオに基づく回答)"
                else:
                    print('シナリオなし')
                    response = conversational_chat_whth_fallback(user_input)
                # print(response)
                # 回答をメッセージ履歴に追加
                st.session_state.messages.append(AIMessage(content=response))
                # print(messages)
                # 回答を画面に表示
                st.markdown(response)
                
        # 回答時間生成プロセス
        end_time = time.time() 
        duration = end_time - start_time

        # フィードバックボタンとフィードバック処理
        col1, col2 = st.columns(2)
        def on_good_button_clicked():
            st.session_state['feedback'] = 'good'
            st.session_state['feedback_given'] = True
            
        def on_bad_button_clicked():
            st.session_state['feedback'] = 'bad'
            st.session_state['feedback_given'] = True

        # 会話の履歴を更新する前に現在の質問と回答をsession_stateに保存
        st.session_state['current_question'] = user_input
        st.session_state['current_response'] = response
        st.session_state['is_scenario_based'] = is_scenario_based

        if col1.button("良かった 👍",on_click=on_good_button_clicked):
            pass
        if col2.button("イマイチ 👎",on_click=on_bad_button_clicked):
            pass
        
        st.info(f'応答にかかった時間：{duration:.2f}秒')

        if st.session_state.get('feedback', '') == 'good' and not is_scenario_based:
        # フィードバックがgoodで、シナリオに基づかない回答だった場合
            append_to_scenario(user_input, response)

        st.session_state['feedback'] = None

        save_chat_log(user_input,response,None) # ログ保存

    elif st.session_state['stop']:
        st.error("応答生成プロセスが停止されました")

if __name__ == '__main__':
    main()