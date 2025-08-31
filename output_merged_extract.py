input_file = "automation/outputs_only.txt"

count_WR = 0
count_SL = 0
count_SR = 0

minimum_WR = []
minimum_SL = []
minimum_SR = []

with open(input_file, "r") as infile:
    for line in infile:
        values = eval(line.strip())
        wr, sl, sr = values[2], values[3], values[4]
        min_val = min(wr, sl, sr)
        if min_val == wr:
            count_WR += 1
            minimum_WR.append(wr)
        elif min_val == sl:
            count_SL += 1
            minimum_SL.append(sl)
        else:  # min_val == sr
            count_SR += 1
            minimum_SR.append(sr)

print(f"WR minimum count: {count_WR}")
print(f"SL minimum count: {count_SL}")
print(f"SR minimum count: {count_SR}")

# Save the lists to output_splits.txt
with open("prop_8/output_splits.txt", "w") as f:
    f.write(str(minimum_WR) + "\n")
    f.write(str(minimum_SL) + "\n")
    f.write(str(minimum_SR) + "\n")

print("Minimum WR, SL, SR lists saved to 'output_splits.txt'")