import matplotlib.pyplot as plt

GPU = 'A800'
MODEL = 'llama7b'

time_breakdown = {
    "gen": [471.055022542287, 21.233553883936978],
    "post_processing": [0.0610113571407353, 0.013105832658047127],
    "old_log_prob": [45.84351506708196, 1.506679798602322],
    "ref": [91.02278898411896, 2.079227286969615],
    "values": [44.598032225994395, 0.8025507943625851],
    "adv": [1.299121051067535, 0.067878702519606764],
    "update_critic": [187.596554842565, 0.9247216280539763],
    "update_actor": [390.583198133117, 1.0988030367877426],
    #"step": [1039.859872462969, 18.42019933419803],
    "collecting": [0.1508021720636, 0.017556559132094251]
}


# Define the time breakdown components
#labels = [
#    "gen", "post_processing", "old_log_prob", "ref", "values", "adv",
#    "update_critic", "update_actor", "collecting"
#]
#values = [291.37, 0.0619, 20.94, 42.72, 19.56, 0.77, 85.13, 90.31, 0.0939]
#total_time = 550.87

labels = list(time_breakdown.keys())
values = [x[0] for x in time_breakdown.values()]
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
