with open("once_a_time/sb3_mas_train_ppo.py", "r") as f:
    lines = f.readlines()

for i in range(575, 693):
    lines[i] = "    " + lines[i]

with open("once_a_time/sb3_mas_train_ppo.py", "w") as f:
    f.writelines(lines)
