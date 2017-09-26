# N, M = input().split()
# Tables = input()
# customs = []
# for _ in range(int(M)):
#     x = input().split()
#     customs.append([int(x[0]),int(x[1])])
customs = [[11,3],[3,10],[35,10],[5,9],[12,10],[6,7]]
value_based = sorted(customs, key=lambda x:x[1],reverse=True)
# Tables = [int(x) for x in Tables.split()]
Tables = sorted([12,1,4,7])
print(value_based)

totalvalue = 0
for custom in value_based:
    print(Tables)
    if len(Tables) == 0:
        break
    if custom[0] > max(Tables):
        continue
    else:
        totalvalue+=custom[1]
        for tab in Tables:
            if tab < custom[0]:
                continue
            else:
                Tables.remove(tab)
                break


print(totalvalue)
