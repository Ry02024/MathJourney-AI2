import streamlit as st
import time
from collections import Counter
from utils import (
    check_memory,
    display_memory_usage,
    update_memory_charts,
    check_gpu,
    load_model_and_tokenizer,
    free_memory,
    generate_response,
    evaluate_answer,
    generate_prompt
)

# メモリの確認と可視化
memory_info = check_memory()

# CPUとGPUのメモリ使用状況を表示
col1, col2 = st.columns(2)
fig_cpu = display_memory_usage({
    "total_memory": memory_info['total_memory'],
    "used_memory": memory_info['used_memory'],
    "free_memory": memory_info['free_memory']
}, "CPU Memory Usage")
fig_gpu = display_memory_usage({
    "total_memory": memory_info['gpu_memory'],
    "used_memory": memory_info['gpu_allocated'],
    "free_memory": memory_info['gpu_free']
}, "GPU Memory Usage")

with col1:
    cpu_chart = st.plotly_chart(fig_cpu, use_container_width=True)
with col2:
    gpu_chart = st.plotly_chart(fig_gpu, use_container_width=True)

# GPUの確認
device, gpu_name = check_gpu()
if device is None:
    st.stop()

st.write(f"使用中のデバイス: {gpu_name}")

MODEL_PATH = "deepseek-ai/deepseek-math-7b-rl"

# モデルの読み込みボタン
if st.button('モデルを読み込む'):
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, device, st)
    st.session_state['model'] = model
    st.session_state['tokenizer'] = tokenizer

    # メモリ使用量の更新
    memory_info = check_memory()
    update_memory_charts(memory_info, fig_cpu, fig_gpu, cpu_chart, gpu_chart)

# StreamlitアプリのUI
st.title('Math Problem Solver')
problem = st.text_input('解くべき数学の問題を入力してください:', 'Find the units digit of \( 7^{100} \).')
iterations = st.number_input('試行回数（1-100）:', min_value=1, max_value=100, value=3)
total_time_limit = st.number_input('合計時間制限（秒、10-3600）:', min_value=10, max_value=3600, value=300)
single_attempt_time_limit = st.number_input('1回の試行時間制限（秒、1-300）:', min_value=1, max_value=300, value=30)

if st.button('Solve'):
    if 'model' in st.session_state and 'tokenizer' in st.session_state:
        model = st.session_state['model']
        tokenizer = st.session_state['tokenizer']

        start_time = time.time()
        final_answer = None
        past_outputs = []
        answers = []

        initial_prompt = generate_prompt(problem)
        prompt = initial_prompt
        print(prompt)

        for attempt in range(iterations):
            if time.time() - start_time > total_time_limit:
                st.write("合計時間制限を超えました。")
                break

            attempt_start_time = time.time()
            while time.time() - attempt_start_time < single_attempt_time_limit:
                generated_text = generate_response(prompt, model, tokenizer, device)
                st.write(f"試行 {attempt + 1} の出力:")
                st.write(generated_text)

                past_outputs.append(generated_text)
                answer = evaluate_answer(generated_text)
                if answer is not None:
                    answers.append(answer)
                    break  # 有効な解答が見つかった場合、内側のループを抜ける

            if answers:
                final_answer = Counter(answers).most_common(1)[0][0]
                st.write(f"最終回答: {final_answer}")
                break
        else:
            if answers:
                final_answer = Counter(answers).most_common(1)[0][0]
                st.write(f"最終回答: {final_answer}")
            else:
                st.write("指定された試行回数内に有効な回答が見つかりませんでした。")
                if past_outputs:
                    valid_answers = [evaluate_answer(output) for output in past_outputs if evaluate_answer(output) is not None]
                    if valid_answers:
                        final_answer = choice(valid_answers, 1)[0]
                        st.write(f"過去の試行からランダムに選ばれた回答: {final_answer}")

# メモリ解放ボタン
if st.button('メモリを解放'):
    if 'model' in st.session_state:
        del st.session_state['model']
    if 'tokenizer' in st.session_state:
        del st.session_state['tokenizer']
    free_memory()

    # メモリ使用量の更新
    memory_info = check_memory()
    update_memory_charts(memory_info, fig_cpu, fig_gpu, cpu_chart, gpu_chart)
