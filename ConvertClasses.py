from Bio import SeqIO
import pandas as pd

class_map = {
			"CDS": "0",
			"intron": "1",
			"five_prime_UTR": "2",
			"three_prime_UTR": "2",
			"intergenic": "3"
			}
gff = pd.read_csv("Ath_Test.gff", sep="\t", usecols=[0,2,3,4], names=["chr","feature", "start","end"], comment="#")
gff = gff.loc[gff["feature"].isin(class_map.keys())]
gff.loc[gff["start"] > gff["end"], ('start','end')] = (gff.loc[gff["start"] > gff["end"], ('start','end')].values)

open('BaseWiseClasses.fa', 'w').close()
for record in SeqIO.parse("Ath_Test.fa", "fasta"):
	chromosome = record.id.split(" ")[0]
	record.seq  = "3"*len(record)
	chr_gff = gff.loc[gff["chr"]==chromosome]
	for index, row in chr_gff.iterrows():
		start = int(row["start"])
		end = int(row["end"])
		feat = class_map[row["feature"]]
		record.seq = record.seq[0:start-1] + feat*((end-start)+1) + record.seq[end:]
	record.description  = "Base-wise feature classes"
	with open("BaseWiseClasses.fa", "a") as out:
		out.write(f">{record.id} | {record.description}\n")
		out.write(record.seq + "\n\n")
	
