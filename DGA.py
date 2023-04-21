from Bio import SeqIO
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
from itertools import islice

class GenomeData(Dataset):
	def __init__(self, fasta_path, gff_path, window):
		# Expects the gff to contain at least the following features ('gene','five_prime_UTR','CDS','intron','three_prime_UTR')
		# NOTE: AGAT can be used to add introns (agat_sp_add_introns.pl --gff [] --out [])
		# NOTE: The gff should also be reduced to a single transcripts (agat_sp_keep_longest_isoform.pl -gff [] -o [])
		self.fasta_path = fasta_path
		self.gff_path = gff_path
		self.window = window
		self.class_map = {
			"CDS": 0,
			"intron": 1,
			"five_prime_UTR": 2,
			"three_prime_UTR": 2,
			"intergenic": 3
			}
		self.seq_map = dict(zip("ACGTRYSWKMBDHVN", range(15)))
		
		self.total_length = 0
		self.chr_lengths = [0]
		with open(fasta_path) as fa:
			for record in SeqIO.parse(fa, 'fasta'):
				self.chr_lengths.append(len(record) + self.total_length)
				self.total_length += len(record)

	def __len__(self):
		return self.length - self.window
	
	def __getitem__(self, idx):
		# Get the chromosome sequence corresponding to the index
		# Then update the index to chromosome coordinates
		seqs = SeqIO.parse(self.fasta_path, "fasta")
		chr_bool = [idx < chr_idx for chr_idx in self.chr_lengths]
		for i in range(1,len(chr_bool)):
			if chr_bool[i] and not chr_bool[i-1]:
				record = next(islice(seqs, i-1, None))
				idx -= sum(self.chr_lengths[:i-1])
				if idx + self.window > len(record):
					idx -= (idx + self.window) - len(record)
				break
		
		# Extract the sequence from the chromosome according to the
		# index and the window length
		# Then convert to a one-hot-encoded tensor
		end_idx = idx+self.window
		seq_map = [self.seq_map[i] for i in record[idx:end_idx].seq]
		sequence_onehot = torch.Tensor(np.eye(15)[seq_map])

		# Extract the base-wise classes for the sequence from the gff
		feature_list = ('five_prime_UTR','CDS','intron','three_prime_UTR')
		chromosome = record.id.split(" ")[0]
		gff = (pd.read_csv(self.gff_path, sep="\t", usecols=[0,2,3,4], names=["chr","feature", "start","end"], comment="#")[lambda x: x["chr"] == chromosome])
		gff = gff.loc[gff["feature"].isin(feature_list)]
		gff.loc[gff["start"] > gff["end"], ('start','end')] = (gff.loc[gff["start"] > gff["end"], ('start','end')].values)
		
		#Start with everything 'intergenic' and then update each base
		class_map = [3 for _ in range(len(seq_map))]
		coord_map = [i for i in range(idx, end_idx)]
		for i in range(len(class_map)):
			feat = gff.loc[((gff["start"] <= coord_map[i]) & (gff["end"] >= coord_map[i]))].reset_index(drop=True)
			if len(feat) != 0:
				class_map[i] = self.class_map[feat.loc[0,"feature"]]
		
		classes = torch.Tensor(class_map)
		return sequence_onehot, classes

dataset = GenomeData("Athaliana_167_TAIR10.fa", "Athaliana_167_gene_exons_introns_longest.gff3", 20)
item = dataset.__getitem__(3759)