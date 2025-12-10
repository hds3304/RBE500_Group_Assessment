import matplotlib.pyplot as plt
import csv

# --- Load CSV data ---
time_list = []
target_list = []
actual_list = []

with open('pd_control_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        time_list.append(float(row['time']))
        target_list.append(float(row['target_ticks']))
        actual_list.append(float(row['actual_ticks']))

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(time_list, actual_list, label='Actual Position', color='blue')
plt.plot(time_list, target_list, '--', label='Target Position', color='red')  # dotted line

plt.xlabel('Time [s]')
plt.ylabel('Position [degrees]')
plt.title('PD Control: Target vs Actual Position')
plt.legend()
plt.grid(True)
plt.show()
