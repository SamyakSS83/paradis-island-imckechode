lines = []
with open("Tutorial_round/samyak.log", "r") as f:
    lines = f.readlines()

l = []
r = []

for i in range(10006,14005):
    linedata = lines[i].split(';')
    if linedata[2] != "KELP":
        continue
    timestamp = linedata[1]
    mid_price = linedata[-2]
    l.append((timestamp,mid_price))

for i in range(10006,14005):
    linedata = lines[i].split(';')
    if linedata[2] != "RAINFOREST_RESIN":
        continue
    timestamp = linedata[1]
    mid_price = linedata[-2]
    r.append((timestamp,mid_price))


with open("Tutorial_round/samyak.csv", "w") as f:
    f.write("timestamp,price\n")
    for i in l:
        f.write(f"{i[0]},{i[1]}\n")

with open("samyak2.csv", "w") as f:
    f.write("timestamp,price\n")
    for i in r:
        f.write(f"{i[0]},{i[1]}\n")