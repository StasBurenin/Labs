from copy import deepcopy

def CaesarCipher(S, k):
	result=''
	for c in S:
		result+=CaesarCipherChar(c, k)
	return result

def CaesarCipherChar(c, k):
	if 'a'<=c<='z':
		return chr((ord(c)-ord('a')+k)%26+ord('a'))
	elif 'A'<=c<='Z':
		return chr ((ord(c)-ord('A')+k)%26+ord('A'))
	return c

alphabet = {'a' : 0, 'b' : 0, 'c' : 0, 'd' : 0, 'e' : 0, 'f' : 0, 'g' : 0, 'h' : 0,
'i' : 0, 'j' : 0, 'k' : 0, 'l' : 0, 'm' : 0, 'n' : 0, 'o' : 0, 'p' : 0, 'q' : 0, 'r' : 0, 's' : 0,
't' : 0, 'u' : 0, 'v' : 0, 'w' : 0, 'x' : 0, 'y' : 0, 'z' : 0}

#Текст, состоящий только из букв
modified=''
with open('text.txt') as file:
	for line in file:
		for c in line:
			if c.lower() in alphabet:
				modified+=c
		modified+='\n'

with open('modified.txt', 'w') as file:
	file.write(modified)

origModified=''
with open('origText.txt') as file:
	for line in file:
		for c in line:
			if c.lower() in alphabet:
				origModified+=c
		origModified+='\n'

with open('origModified.txt', 'w') as file:
	file.write(origModified)

#Зашифрованный текст
ciphered=''
with open('modified.txt') as file:
	for line in file:
		ciphered+=CaesarCipher(line, 1)

with open('ciphered.txt', 'w') as file:
	file.write(ciphered)

#Количество букв
lenf=0

alphabetChapter = deepcopy(alphabet)

alphabetM=deepcopy(alphabet)
#Частота букв в оригинальном тексте
with open('origModified.txt') as file:
	for line in file:
		for c in line:
			if c.lower() in alphabetM:
				alphabetM[c.lower()]+=1
				lenf+=1

for c in alphabetM:
	alphabetM[c]/=lenf


print('Частота букв в оригинальном тексте:', alphabetM, '\n')

alphabetC=deepcopy(alphabet)
#Частота букв в зашифрованном тексте
with open('ciphered.txt') as file:
	for line in file:
		for c in line:
			if c.lower() in alphabetC:
				alphabetC[c.lower()]+=1

for c in alphabetC:
	alphabetC[c]/=lenf

print('Частота букв в зашифрованном тексте:', alphabetC, '\n')

bigram={}
items=[]

for i in alphabet:
	for j in alphabet:
		if i==j:
			continue
		else:
			items.append(i+j)

for i in range(len(items)):
	bigram[items[i]]=0

#Количество биграм
lenb=0

bigramM=deepcopy(bigram)
#Частота биграм в оригинальном тексте
with open('origModified.txt') as file:
	for line in file:
		for i in range(len(line)-1):
			b=line[i]+line[i+1]
			if b.lower() in bigramM:
				bigramM[b.lower()]+=1
				lenb+=1

for c in bigramM:
	bigramM[c]/=lenb

print('Частота биграм в оригинальном тексте:', bigramM, '\n')

bigramC=deepcopy(bigram)
#Частота биграм в зашифрованном тексте
with open('ciphered.txt') as file:
	for line in file:
		for i in range(len(line)-1):
			b=line[i]+line[i+1]
			if b.lower() in bigramM:
				bigramC[b.lower()]+=1

for c in bigramC:
	bigramC[c]/=lenb

print('Частота биграм в зашифрованном тексте:', bigramC, '\n')