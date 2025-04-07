lines = []
with open("Tutorial_round/so_far_best.log", "r") as f:
    lines = f.readlines()

l = []
r = []

for i in range(10006,14006):
    linedata = lines[i].split(';')
    if linedata[2] != "KELP":
        continue
    timestamp = linedata[1]
    # mid_price = linedata[-2]
    mid_price = 0
    n = 0
    for i in range(3, 14, 2):
        if linedata[i] == "":
            continue

        mid_price += int(linedata[i])*int(linedata[i+1])
        n += int(linedata[i+1])
    mid_price/=n
    l.append((timestamp,mid_price))

for i in range(10006,14005):
    linedata = lines[i].split(';')
    if linedata[2] != "RAINFOREST_RESIN":
        continue
    timestamp = linedata[1]
    mid_price = linedata[-2]
    r.append((timestamp,mid_price))


with open("Tutorial_round/so_far_best_output.csv", "w") as f:
    f.write("timestamp,price\n")
    for i in l:
        f.write(f"{i[0]},{i[1]}\n")

with open("samyak2.csv", "w") as f:
    f.write("timestamp,price\n")
    for i in r:
        f.write(f"{i[0]},{i[1]}\n")