lines = []
with open("initial.log", "r") as f:
    lines = f.readlines()

l = []

for i in range(14012,25659,9):
    assert(lines[i]=="  {\n")
    temp = lines[i+4].rstrip("\",\n").lstrip("\"symbol\": \"")
    print(temp)
    if temp != "KELP":
        continue
    temp1 = int(lines[i+1].rstrip(',\n').lstrip("\"timestamp\": "))
    temp2 = int(lines[i+6].rstrip(',\n').lstrip("\"price\": "))
    temp3 = int(lines[i+7].rstrip(',\n').lstrip("\"quantity\": "))
    l.append((temp1, temp2, temp3))

with open("output.csv", "w") as f:
    f.write("timestamp,price,quantity\n")
    for i in l:
        f.write(f"{i[0]},{i[1]},{i[2]}\n")