import matplotlib.pyplot as plt

# Define the time breakdown components
labels = ["gen", "post_processing", "old_log_prob", "ref", "values", "adv", "update_critic", "update_actor", "collecting"]
values = [291.37, 0.0619, 20.94, 42.72, 19.56, 0.77, 85.13, 90.31, 0.0939]
total_time = 550.87

# Verify the sum of breakdown matches the total step time (considering minor floating point differences)
assert abs(sum(values) - total_time) < 1.0, "The sum of breakdown values does not match the total step time."

# Create the pie chart with absolute values and percentages in labels
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    values, labels=labels, autopct=lambda p: f'{p:.1f}%\n({p * total_time / 100:.2f})',
    startangle=140, colors=plt.cm.Paired.colors
)

# Improve label visibility
for text in texts + autotexts:
    text.set_fontsize(10)

plt.title("Time Breakdown in Step Execution")
plt.show()
