import torch
import os
import gc
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import re
import numpy as np
from numpy.random import choice

# メモリ使用状況を確認する関数
def check_memory():
    gpu_memory, gpu_reserved, gpu_allocated, gpu_free = 0, 0, 0, 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        gpu_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        gpu_free = gpu_memory - gpu_allocated

    total_memory = int(os.popen('free -m').readlines()[1].split()[1]) / 1024
    used_memory = int(os.popen('free -m').readlines()[1].split()[2]) / 1024
    free_memory = total_memory - used_memory

    return {
        "gpu_memory": gpu_memory,
        "gpu_reserved": gpu_reserved,
        "gpu_allocated": gpu_allocated,
        "gpu_free": gpu_free,
        "total_memory": total_memory,
        "used_memory": used_memory,
        "free_memory": free_memory,
    }

def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# メモリ使用量を円グラフで表示する関数
def display_memory_usage(memory_info, title):
    fig = go.Figure(data=[go.Pie(labels=['Used Memory', 'Free Memory'],
                                 values=[memory_info['used_memory'], memory_info['free_memory']],
                                 title=title, hole=.3)])
    fig.update_layout(
        annotations=[dict(text=f"Total: {memory_info['total_memory']:.2f} GB<br>Used: {memory_info['used_memory']:.2f} GB",
                          x=0.5, y=0.5, font_size=12, showarrow=False)]
    )
    return fig

# メモリ使用量のグラフを表示および更新する関数
def update_memory_charts(memory_info, fig_cpu, fig_gpu, cpu_chart, gpu_chart):
    fig_cpu.data[0].values = [memory_info['used_memory'], memory_info['free_memory']]
    fig_cpu.update_layout(annotations=[dict(text=f"Total: {memory_info['total_memory']:.2f} GB<br>Used: {memory_info['used_memory']:.2f} GB",
                          x=0.5, y=0.5, font_size=12, showarrow=False)])
    cpu_chart.plotly_chart(fig_cpu, use_container_width=True)

    fig_gpu.data[0].values = [memory_info['gpu_allocated'], memory_info['gpu_free']]
    fig_gpu.update_layout(annotations=[dict(text=f"Total: {memory_info['gpu_memory']:.2f} GB<br>Used: {memory_info['gpu_allocated']:.2f} GB",
                          x=0.5, y=0.5, font_size=12, showarrow=False)])
    gpu_chart.plotly_chart(fig_gpu, use_container_width=True)

# GPUの使用可否と種類を確認
def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), gpu_name
    else:
        return torch.device("cpu"), "CPU"

# モデルとトークナイザーの読み込み
def load_model_and_tokenizer(model_path, device, st):
    st.write("モデルとトークナイザーを読み込んでいます...")
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # モデルを読み込む際に進捗を表示
    with st.spinner('モデルをダウンロードしています...'):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
            config=config
        )

    # モデルを適切なデバイス（GPUまたはCPU）に移動
    model = model.to(device)
    st.write("モデルとトークナイザーの読み込みが完了しました。")
    return model, tokenizer

def generate_response(prompt, model, tokenizer, device):
    TOTAL_TOKENS = 2048
    ALREADY_GEN = 0
    temperature = 0.9
    top_p = 1.0
    model.to(device)
    model_inputs = tokenizer(prompt, return_tensors='pt').to(device)
    # メモリの解放
    free_memory()
    generation_output = model.generate(
        **model_inputs,
        max_new_tokens=TOTAL_TOKENS-ALREADY_GEN,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1
    )

    generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    return generated_text

def evaluate_answer(answer):
    # 答えを評価するロジック（ここでは簡単に出力をパースして数値を探す）
    match = re.search(r'\\boxed{(\d+)}', answer)
    if match:
        return int(match.group(1))
    else:
        return None

def generate_prompt(problem, past_output=None):
    # プロンプトテンプレート
    code_template = """以下は解くべき数学の問題です（正の数値解を求めてください）:
「{}」
これを解決するために、まずSymPyを使った解法を考え、それぞれのステップで呼び出す必要のある関数をリストアップしてください。誰でも理解できるように明確にし、最終的な答えは代数式ではなく正の整数にしてください！
すべてのステップをカバーするスクリプトを書いてください（コメントとドキュメントを含む）結果を表示してください。問題を解いた後、最終的な数値の答えを\\boxed{}で囲んで出力してください。

アプローチ:"""

    cot_template = """以下は解くべき数学の問題です（正の数値解を求めてください！）:
「{}」
この問題を分析し、プログラムを使って段階的に解決策を考えてください。問題を解いた後、最終的な数値の答えを\\boxed{}で囲んで出力してください。\n\n"""

    prompt_options = [code_template, cot_template]

    # 過去の出力を考慮してプロンプトを生成する
    if past_output:
        new_prompt = f"Previous attempt:\n{past_output}\n\n"
        new_prompt += "Analyze the previous attempt and correct any mistakes to solve the problem: \n"
    else:
        new_prompt = choice(prompt_options).format(problem, "{}")
    return f"User: {new_prompt}"
