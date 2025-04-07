import matplotlib.pyplot as plt

time_breakdown = {
    "gen": (331.87347317123783, 19.471599649749386),
    "post_processing": (0.0798660227097571, 0.015049873872713348),
    "old_log_prob": (23.980894559808075, 4.744868846860671),
    "ref": (48.11182795548812, 5.812303926062228),
    "values": (23.588751344103365, 4.428156308343343),
    "adv": (0.7970318300649524, 0.057262937263335506),
    "update_critic": (94.75331611009315, 1.0696240718273826),
    "update_actor": (99.7962265675321, 0.6651195843986861),
    "step": (622.982666027308, 5.651579329078492),
    "collecting": (0.09991906676441431, 0.011415995609689549)
}

# Define the time breakdown components
labels = [
    "gen", "post_processing", "old_log_prob", "ref", "values", "adv",
    "update_critic", "update_actor", "collecting"
]
values = [291.37, 0.0619, 20.94, 42.72, 19.56, 0.77, 85.13, 90.31, 0.0939]
total_time = 550.87

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

plt.title("Time Breakdown in Step Execution")
plt.show()
