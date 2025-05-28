import sqlite3
import spacy
import os
import zipfile
import pandas as pd

# Load English NLP model
nlp = spacy.load("en_core_web_sm")


def unzip_data(zip_path='moral_machine_data/data.zip', extract_to='data'):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def list_files(folder='data'):
    for root, dirs, files in os.walk(folder):
        for file in files:
            print(os.path.join(root, file))


def load_moral_data(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head())  # Preview
    return df


def process_moral_machine_scenarios(df):
    for i, row in df.iterrows():
        scenario_text_tbp = row_to_scenario_text(row) # tbp = to be processed
        print(f"\nScenario {i + 1}:")
        print(scenario_text_tbp)
        print(check_ethics(scenario_text_tbp))


def row_to_scenario_text(row):
    elements = []

    people = {
        "Man": "a man", "Woman": "a woman", "Pregnant": "a pregnant woman",
        "Stroller": "a person with a stroller", "OldMan": "an old man",
        "OldWoman": "an old woman", "Boy": "a boy", "Girl": "a girl",
        "Homeless": "a homeless person", "LargeWoman": "a large woman",
        "LargeMan": "a large man", "Criminal": "a criminal",
        "MaleExecutive": "a male executive", "FemaleExecutive": "a female executive",
        "FemaleAthlete": "a female athlete", "MaleAthlete": "a male athlete",
        "FemaleDoctor": "a female doctor", "MaleDoctor": "a male doctor",
        "Dog": "a dog", "Cat": "a cat"
    }

    for key, label in people.items():
        if row.get(key, 0) > 0:
            elements.append(f"{row[key]}x {label}")

    group_description = ", ".join(elements)

    side = "left" if row.get("LeftHand", 1) == 1 else "right"
    action = "intervene" if row.get("Intervention", 0) == 1 else "not intervene"

    # Force keyword like "fully autonomous" to trigger SAE level detection
    automation_phrase = " This vehicle needs human. "

    return f"In this scenario, the vehicle must decide whether to {action}. On the {side} side, there are: {group_description}.{automation_phrase}"





# ---------- Step 1: Create and Populate Rule Database ---------- #

def create_ethics_db():
    conn = sqlite3.connect('ethics_rules.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY,
            sae_level INTEGER,
            principle TEXT,
            rule_description TEXT
        )
    ''')

    rules = [
        (0, "Human Responsibility", "The human driver must maintain full control at all times."),
        (0, "Decision Authority", "No automated system should override human decisions."),
        (1, "Driver Assistance", "Assistance systems must disengage if human override is detected."),
        (1, "Minimal Intervention", "System must not initiate uncommanded critical decisions (e.g., braking)."),
        (2, "Shared Control", "Ethical decisions default to human if disagreement between system and driver."),
        (2, "Transparency", "System must communicate its operational limits and intention to the driver."),
        (3, "Conditional Autonomy", "Ethical fallback must shift to human if system boundaries are reached."),
        (3, "Safe Exit Strategy", "In ambiguous scenarios, system must safely disengage and return control."),
        (4, "Contextual Ethics", "System must apply ethical rules dynamically based on traffic law and context."),
        (4, "Risk Minimization", "Must choose least harmful outcome in unavoidable crash scenarios."),
        (5, "Full Responsibility", "System assumes full ethical responsibility for all driving decisions."),
        (5, "Algorithmic Fairness", "Risk must be distributed fairly among all road users."),
        (5, "Non-Discrimination", "Decisions must not prioritize safety based on race, age, or economic status.")
    ]

    c.executemany("INSERT INTO rules (sae_level, principle, rule_description) VALUES (?, ?, ?)", rules)
    conn.commit()
    conn.close()


# ---------- Step 2: Text Analysis ---------- #

def analyze_text(text):
    doc = nlp(text.lower())
    keywords = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    # Define keyword sets per SAE level
    level_keywords = {
        0: {"manual", "human", "full control", "driver control"},
        1: {"assist", "cruise", "lane keep", "adaptive"},
        2: {"partial", "autopilot", "automation", "steering assist"},
        3: {"conditional", "monitor", "fallback", "takeover"},
        4: {"high automation", "no driver", "no intervention", "geo-fenced"},
        5: {"full automation", "self-driving", "autonomous", "no human"}
    }

    # Flatten and normalize input text for simple search
    lowered_text = text.lower()

    # Search for keywords
    for level, key_set in level_keywords.items():
        for word in key_set:
            if word in lowered_text:
                return level, keywords

    return None, keywords



# ---------- Step 3: Ethics Rule Checker ---------- #

def check_ethics(text):
    sae_level, keywords = analyze_text(text)

    if sae_level is None:
        return "Could not determine SAE level from the text."

    conn = sqlite3.connect('ethics_rules.db')
    c = conn.cursor()
    c.execute("SELECT principle, rule_description FROM rules WHERE sae_level = ?", (sae_level,))
    rule = c.fetchone()
    conn.close()

    if rule:
        print(f"\n--- Analysis Result ---")
        print(f"Detected SAE Level: {sae_level}")
        print(f"Ethical Principle: {rule[0]}")
        print(f"Guideline: {rule[1]}")
        print("\n--- Assessment ---")
        if any(word in text.lower() for word in rule[1].split()):
            return "Text appears to comply with ethical guideline."
        else:
            return "Text may conflict with ethical expectations for this level."
    else:
        return "No ethical rule found for this level."


# ---------- Example Usage ---------- #

if __name__ == "__main__":
    create_ethics_db()  # Only run once

    scenario_text = """
    This vehicle operates fully autonomously in both urban and rural areas. It does not require any human intervention 
    and is capable of handling all traffic conditions.
    """
    print(scenario_text)

    result = check_ethics(scenario_text)
    print(result)

    scenario_text = """
    The autonomous vehicle is designed to prioritize the safety of its passengers, even if it means endangering 
    pedestrians in extreme situations.
    """
    print(scenario_text)

    result = check_ethics(scenario_text)
    print(result)

    unzip_data()
    list_files('data')
    for root, dirs, files in os.walk('data'):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            if file_path.endswith('.csv') and file_path.__contains__('data\\data'):
                try:
                    df = load_moral_data(os.path.join(root, file))
                    process_moral_machine_scenarios(df)
                except:
                    print(f"Could not extract data from {file}")
