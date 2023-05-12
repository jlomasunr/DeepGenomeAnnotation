from Bio import SeqIO
import sys
import math

file = sys.argv[1]
length = int(sys.argv[2])

for record in SeqIO.parse(file, "fasta"):
    num_seqs = math.floor(len(record)/length)
    for i in range(num_seqs):
        print(f">{record.id} | part {i+1}:{(length*i)+1}:{length*(i+1)+1}")
        print(record[length*i:length*(i+1)].seq)
    print(f">{record.id} | part {i+1}:")
    print(record[(length*i)+1:].seq)