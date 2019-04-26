import os

"""
Get files
Read each file 100+
find best val
store best val & params + print
opt: return best val & params
"""

files = [file for file in os.listdir("/cluster/home/llorenz/sleep/MT18_LH_human-sleep-classification/bin/") if file.startswith('_exp_')]
print(f"{len(files)} files found")

n_ignore_lines = 110
max_prec1 = 0
min_loss = 5
max_prec1_f = ""
min_loss_f = ""
for file in files:
    with open(file) as op:
        lines = op.readlines()[n_ignore_lines:]
        prec1 = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("Pat"):
                skipped_val = 0
                for j in range(1,7):
                    candidate = lines[i-j]
                    if candidate.startswith("Val"):
                        if not skipped_val:
                            skipped_val = True
                            continue
                        prec1 = float(candidate.split()[7])
                        loss = float(candidate.split()[5])
                        break
                    elif candidate.startswith("Pat"):
                        break
        if prec1 > max_prec1:
            max_prec1 = prec1
            max_prec1_f = file
            print("Prec1:", max_prec1, max_prec1_f)
        if loss < min_loss:
            min_loss = loss
            min_loss_f = file
            print("Loss:", min_loss, min_loss_f)
