import pickle
from collections import Counter


# Example validation function to enforce Level 5 compliance
def validate_decision(scenario, decision):
    """
    Validates and overrides decision if needed.

    Parameters:
        scenario (dict): scenario info dict with keys as above
        decision (str): "Case 1" or "Case 2"

    Returns:
        final_decision (str): "Case 1" or "Case 2" after validation
    """
    # Unpack useful fields
    is_law = scenario.get("is_law", False)
    traffic_light_pattern = scenario.get("traffic_light_pattern", ["NA", "NA"])
    count_dict_1 = Counter(scenario.get("count_dict_1", {}))
    count_dict_2 = Counter(scenario.get("count_dict_2", {}))

    # Helper: total victims per case
    total_1 = sum(count_dict_1.values())
    total_2 = sum(count_dict_2.values())

    # If law compliance is required, disallow hitting pedestrians crossing on green
    # Assuming traffic_light_pattern[0] corresponds to Case 1, [1] to Case 2
    # If decision chooses to kill pedestrians crossing on green light, override.
    if is_law:
        if decision == "Case 1" and traffic_light_pattern[0] == "green":
            # hitting pedestrians crossing legally = illegal
            # Override to Case 2 if it's safer/legal
            if traffic_light_pattern[1] != "green":
                return "Case 2"
        elif decision == "Case 2" and traffic_light_pattern[1] == "green":
            if traffic_light_pattern[0] != "green":
                return "Case 1"

    # Otherwise, prefer case with fewer casualties for level 5 safety
    if total_1 < total_2:
        return "Case 1"
    elif total_2 < total_1:
        return "Case 2"

    # If equal casualties, return original decision
    return decision


# Load pickle file with saved scenarios and responses
def load_cases_from_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        cases = pickle.load(f)
    return cases


def main_pipeline(pickle_path):
    cases = load_cases_from_pickle(pickle_path)

    final_results = []
    for idx, case in cases.iterrows():
        original_decision = case["chatgpt_response"]

        if original_decision not in ["Case 1", "Case 2"]:
            print(f"Warning: Case {idx} has invalid model response: {original_decision}")
            continue

        validated_decision = validate_decision(case, original_decision)
        final_results.append({
            "case_index": idx,
            "scenario_dimension": case.get("scenario_dimension", ""),
            "original_decision": original_decision,
            "final_decision": validated_decision,
            "was_overridden": validated_decision != original_decision
        })

    # Print summary
    overridden_count = sum(1 for r in final_results if r["was_overridden"])
    print(f"Total cases: {len(cases)}")
    print(f"Overridden decisions due to compliance: {overridden_count}")

    # Return final results for further use or saving
    return final_results


# Example usage:
if __name__ == "__main__":
    pickle_file_path = "moral_machine_data/results_300_scenarios_seed123_gpt-4o.pickle"  # put your path here
    results = main_pipeline(pickle_file_path)

    # Example: print some results
    for res in results[:5]:
        print(
            f"Case {res['case_index']}: Original={res['original_decision']}, Final={res['final_decision']}, Overridden={res['was_overridden']}")
