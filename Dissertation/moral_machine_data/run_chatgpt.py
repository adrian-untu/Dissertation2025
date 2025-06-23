import os
import random
import time
import argparse
import pandas as pd
from tqdm import tqdm

# Import OpenAI
import openai

# Import Google Gemini
import google.generativeai as genai

# Import Hugging Face Transformers
from transformers import pipeline

# Local scenario generator function (you already have this)
from generate_moral_machine_scenarios import generate_moral_machine_scenarios


#### Argument Parser ####
parser = argparse.ArgumentParser(description='Run multi-model moral machine')
parser.add_argument('--model', type=str, choices=['gpt-4o', 'gemini', 'llama2'], default='gpt-4o')
parser.add_argument('--nb_scenarios', type=int, default=100)
parser.add_argument('--random_seed', type=int, default=123)
parser.add_argument('--output', type=str, default='multi_model_results.pickle')
args = parser.parse_args()

random.seed(args.random_seed)


#### API Keys (Set via environment variables or insert manually) ####
OPENAI_API_KEY = "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"
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-key")
# HUGGINGFACE_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # you can change to another


#### Initialize APIs ####
openai.api_key = OPENAI_API_KEY
# genai.configure(api_key=GEMINI_API_KEY)
# llama_pipe = pipeline("text-generation", model=HUGGINGFACE_MODEL, device_map="auto")


#### Model Abstraction ####
def chat_with_model(system_cont, user_cont):
    if args.model == 'gpt-4o':
        try:
            res = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Please respond to binary questions."},
                    {"role": "system", "content": system_cont},
                    {"role": "user", "content": user_cont},
                ]
            )
            return res['choices'][0]['message']['content']
        except Exception as e:
            print(f"[GPT ERROR] Retrying: {e}")
            time.sleep(5)
            return chat_with_model(system_cont, user_cont)

    elif args.model == 'gemini':
        try:
            model = genai.GenerativeModel('gemini-pro')
            convo = model.start_chat(history=[])
            convo.send_message(system_cont + "\n" + user_cont)
            return convo.last.text
        except Exception as e:
            print(f"[Gemini ERROR] Retrying: {e}")
            time.sleep(5)
            return chat_with_model(system_cont, user_cont)

    elif args.model == 'llama2':
        try:
            prompt = f"[System] {system_cont}\n[User] {user_cont}\n[Assistant]"
            res = llama_pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
            return res[0]['generated_text'].split("[Assistant]")[-1].strip()
        except Exception as e:
            print(f"[LLaMA ERROR] Retrying: {e}")
            time.sleep(5)
            return chat_with_model(system_cont, user_cont)

    else:
        raise ValueError("Unsupported model selected.")


#### Run Scenarios ####
scenario_info_list = []

for i in tqdm(range(args.nb_scenarios)):
    dimension = random.choice(["species", "social_value", "gender", "age", "fitness", "utilitarianism"])
    is_interventionism = random.choice([True, False])
    is_in_car = random.choice([True, False])
    is_law = random.choice([True, False])

    system_content, user_content, scenario_info = generate_moral_machine_scenarios(dimension, is_in_car, is_interventionism, is_law)

    try:
        response = chat_with_model(system_content, user_content)
        scenario_info['response'] = response
        scenario_info_list.append(scenario_info)
        print(f"\n[{i+1}] Response: {response}\n")
    except Exception as e:
        print(f"Scenario {i} failed: {e}")

    if (i+1) % 50 == 0:
        pd.DataFrame(scenario_info_list).to_pickle(args.output)

# Final save
pd.DataFrame(scenario_info_list).to_pickle('results_{}_scenarios_seed{}_{}.pickle'.format(args.nb_scenarios, args.random_seed, args.model))
