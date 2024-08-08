import pandas as pd
import difflib
from sentence_transformers import SentenceTransformer, util

# モデルのロード
model = SentenceTransformer('all-MiniLM-L6-v2')

# シナリオファイルの読み込み
def load_scenarios():
    try:
        # ファイルを書き込み可能モードで開く
        # with open('scenario.xlsx', 'wb') as f:
        scenario_df = pd.read_excel('./scenario.xlsx', index_col=0)  # 1行目をスキップする
        return scenario_df
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return None

# シナリオに基づく回答を検索
def get_response_from_scenario(user_input, scenario_df):
    # B列（インデックス1）に質問が格納されていると仮定
    matched_row = scenario_df[scenario_df['Question'] == user_input]
    if not matched_row.empty:
        print('シナリオに基づく回答をします')
        # C列（インデックス2）に回答が格納されていると仮定
        # E列（'Usage Count'）に回答で使用された回数をインクリメント
        index = matched_row.index[0]  # 該当する行のインデックスを取得
        print(f"Before update: {scenario_df.at[index, 'Usage Count']}")  # 更新前の値を表示
        scenario_df.at[index, 'Usage Count'] = scenario_df.at[index, 'Usage Count'] + 1 if pd.notna(scenario_df.at[index, 'Usage Count']) else 1
        print(f"After update: {scenario_df.at[index, 'Usage Count']}")  # 更新後の値を表示
        # 変更をファイルに保存
        scenario_df.to_excel('./scenario.xlsx')
        return matched_row['Answer'].values[0], True
    return None, False

# シナリオに回答を追加
def append_to_scenario(user_input, response):
    try:
        # 既存のシナリオファイルを読み込み
        scenario_df = pd.read_excel('./scenario.xlsx', index_col=0)
        # 入力された質問に類似する質問を検索
        closest_match, scenario_response, is_scenario_based = get_closest_match(user_input, scenario_df)
        if is_scenario_based:
            # 類似する質問の Feedback Count をインクリメント
            matching_indexes = scenario_df.index[scenario_df['Question'] == closest_match]
            if not matching_indexes.empty:
                index = matching_indexes[0]
                # Feedback Count が NaN の場合は 1 を設定、そうでなければインクリメント
                # 使用回数はGoodFeedbackボタンを押すと2重計上になるのでデクリメント
                if pd.isna(scenario_df.at[index, 'Good Feedback']):
                    scenario_df.at[index, 'Good Feedback'] = 1
                    scenario_df.at[index, 'Usage Count'] -= 1
                else:
                    scenario_df.at[index, 'Good Feedback'] += 1
                    scenario_df.at[index, 'Usage Count'] -= 1 
                print(f"Good Feedback: {scenario_df.at[index, 'Good Feedback']}")  # 更新後の値を表示
                print(f"Usage Count: {scenario_df.at[index, 'Usage Count']}") # 更新後の値を表示
        else:
            next_index = scenario_df.index.max() + 1 if not scenario_df.empty else 0
            # 新しい行を追加（Good Feedback に 1 を設定）
            scenario_df.loc[next_index] = [user_input, response, 1, 1]  # 新しい行を追加
        scenario_df.to_excel('./scenario.xlsx')
    except Exception as e:
        print(f"ファイルの書き込みに失敗しました: {e}")

# 類似度に基づいて最も近い質問を探す
def get_closest_match(user_input, scenario_df):
    # # Q&A質問の埋め込み
    # questions_embeddings = model.encode(scenario_df['Question'].dropna().astype(str).tolist())
    # # ユーザー質問の埋め込み
    # user_question_embedding = model.encode([user_input])
    # # 類似度の計算
    # similarities = util.cos_sim(user_question_embedding, questions_embeddings)[0]

    # # 類似度の計算と表示
    # for i, similarity in enumerate(similarities):
    #     print(f"質問: {scenario_df.iloc[i]['Question']}, 類似度: {similarity.item():.4f}")

    # # 最も類似度が高い質問の詳細を表示
    # max_similarity_index = similarities.argmax()
    # # print(f"最も類似度が高い質問: {scenario_df.iloc[max_similarity_index]['Question']}")
    # # print(f"類似度: {similarities[max_similarity_index].item():.4f}")
    # # 類似度計算用に質問リストを取得し、元のインデックスも保存
    # questions = scenario_df['Question'].dropna().astype(str).tolist()
    # original_indexes = scenario_df.dropna().index.tolist()  # 元のインデックスリスト
    
    # # 最も類似度が高い質問のインデックスを修正して取得
    # correct_index = original_indexes[max_similarity_index]
    
    # # 正しいインデックスを使用して質問を取得
    # print(f"最も類似度が高い質問: {scenario_df.loc[correct_index]['Question']}")

    # max_similarity = similarities[max_similarity_index]

    # if max_similarity < 0.5:  # 類似度の閾値を設定
    #     return None, False

    # matched_row = scenario_df.iloc[max_similarity_index]
    # return matched_row['Answer'], True

    questions = scenario_df['Question'].dropna().astype(str).tolist()
    # 類似度が0.8以上のものを探す
    closest_matches = difflib.get_close_matches(user_input, questions, n=1, cutoff=0.8)  
    if closest_matches:
        closest_match = closest_matches[0]
        matched_row = scenario_df[scenario_df['Question'] == closest_match]
        if not matched_row.empty:
            index = matched_row.index[0]  # 該当する行のインデックスを取得
            # 'Usage Count' 列の値をチェックし、NaN または 0 の場合は 1 を設定、そうでなければインクリメント
            if pd.isna(scenario_df.at[index, 'Usage Count']) or scenario_df.at[index, 'Usage Count'] == 0:
                scenario_df.at[index, 'Usage Count'] = 1
            else:
                scenario_df.at[index, 'Usage Count'] += 1
            print(f"Usage Count: {int(scenario_df.at[index, 'Usage Count'])}")  # 更新後の値を表示
            # 変更をファイルに保存
            scenario_df.to_excel('./scenario.xlsx')
            return closest_match, matched_row['Answer'].values[0], True
    return None, None, False