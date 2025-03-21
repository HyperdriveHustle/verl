import re

log_text = """'gen': 279.4479638962075, 'post_processing': 0.07394677586853504, 'old_log_prob': 20.512651755008847, 'ref': 42.639062612783164, 'values': 19.41322973323986, 'adv': 0.8144307578913867, 'update_critic': 84.4035917370915, 'update_actor': 89.4339230619371, 'step': 536.7403232618235, 'collecting': 0.1077086216712737"""

# Parse key-value pairs from the log
time_breakdown = {k: float(v) for k, v in re.findall(r"'(\w+)': ([\d.]+)", log_text)}

# Exclude 'step' and sum the rest
total_time = sum(v for k, v in time_breakdown.items() if k != "step")

print(f"Total time excluding 'step': {total_time}")
print(time_breakdown['step'])
