n = int(input())
m = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199]

def hash1(a):
    h=0
    b = str(abs(a))
    for i in range(len(b)):
        h = (h + int(b[i] * m[i])) * m[i]
    return h
print(hash1(n))

m2 = 729

def hash2(a):
    h=0
    b = str(abs(a))
    for i in range(len(b)):
        h = (h + int(b[i])) * m2
    return h
print(hash2(n))

