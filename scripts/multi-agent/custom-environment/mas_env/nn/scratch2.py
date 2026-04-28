with open("once_a_time/sb3_mas_train_ppo.py", "r") as f:
    lines = f.readlines()

for i in range(566, 636):
    if lines[i].startswith("        "):
        lines[i] = lines[i][8:]

with open("once_a_time/sb3_mas_train_ppo.py", "w") as f:
    f.writelines(lines)
