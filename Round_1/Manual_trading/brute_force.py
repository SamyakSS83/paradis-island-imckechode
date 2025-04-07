x = [[1,1.45,0.52,0.72], #x[a][b] = a to b
          [0.7, 1, 0.31, 0.48],
          [1.95 ,3.1, 1, 1.49],
          [1.34, 1.98, 0.64, 1]]

ans = {}
start = 2000000
for a in range(4):
    for b in range(4):
        for c in range(4):
            for d in range(4):
                ans[(3,a,b,c,d,3)] = x[3][a]*x[a][b]*x[b][c]*x[c][d]*x[d][3]
for a in range(4):
    for b in range(4):
        for c in range(4):
                ans[(3,a,b,c,3)] = x[3][a]*x[a][b]*x[b][c]*x[c][3]
for a in range(4):
    for b in range(4):
        ans[(3,a,b,3)] = x[3][a]*x[a][b]*x[b][3]
for a in range(4):
    ans[(3,a,3)] = x[3][a]*x[a][3]

print(ans)
print(*sorted(ans.items(), key=lambda x: x[1]), sep="\n")