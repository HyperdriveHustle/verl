import matplotlib.pyplot as plt

GPU = 'A800'
MODEL = 'llama7b'

time_breakdown = {
    'gen': 471.0550225452287 ,
    'post_processing': 0.061001135781407353 ,
    'old_log_prob': 45.843515065708196,
    'ref': 91.02278898411896,
    'values': 44.598032225994395,
    'adv': 1.2991210516076535 ,
    'update_critic': 187.59655480042565 ,
    'update_actor': 198.38269411679357 ,
    #'step': 1039.8598758099834 - 18.42019934319803,
    'collecting': 0.10735758242662996 ,
}


# Define the time breakdown components
#labels = [
#    "gen", "post_processing", "old_log_prob", "ref", "values", "adv",
#    "update_critic", "update_actor", "collecting"
#]
#values = [291.37, 0.0619, 20.94, 42.72, 19.56, 0.77, 85.13, 90.31, 0.0939]
#total_time = 550.87

labels = list(time_breakdown.keys())
values = [x for x in time_breakdown.values()]
total_time = sum(values)

# Verify the sum of breakdown matches the total step time (considering minor floating point differences)
assert abs(
    sum(values) - total_time
) < 1.0, "The sum of breakdown values does not match the total step time."

# Create the pie chart with absolute values and percentages in labels
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    values,
    labels=labels,
    autopct=lambda p: f'{p:.1f}%\n({p * total_time / 100:.2f})',
    startangle=140,
    colors=plt.cm.Paired.colors)

# Improve label visibility
for text in texts + autotexts:
    text.set_fontsize(10)

plt.title(f"{GPU=}, {MODEL=}")
plt.show()
