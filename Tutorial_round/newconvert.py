lines = []
with open("initial.log", "r") as f:
    lines = f.readlines()

l = []

for i in range(10006,14005):
    linedata = lines[i].split(';')
    if linedata[2] != "KELP":
        continue
    timestamp = linedata[1]
    mid_price = linedata[-2]
    l.append((timestamp,mid_price))


with open("new_output.csv", "w") as f:
    f.write("timestamp,price\n")
    for i in l:
        f.write(f"{i[0]},{i[1]}\n")