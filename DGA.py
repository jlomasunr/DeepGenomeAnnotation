from Bio import SeqIO
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from itertools import islice


# Custom dataset class for extracting one-hot-encoded dna sequences
# and basewise classes
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
		return self.total_length - self.window
	
	def __getitem__(self, idx):
		# Get the chromosome sequence corresponding to the index
		# Then update the index to chromosome coordinates
		print(f"Loading from {idx}")
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
		print(f"One-Hot encoding...")
		end_idx = idx+self.window
		seq_map = [self.seq_map[i] for i in record[idx:end_idx].seq]
		sequence_onehot = torch.Tensor(np.eye(15)[seq_map])

		# Extract the base-wise classes for the sequence from the gff
		print("Extracting base-wise classes")
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
		print("Done.")
		
		classes = torch.Tensor(class_map)
		return sequence_onehot, classes

class biLSTM(nn.Module):
	def __init__(self, hidden_size, num_layers, dropout=0.2):
		super(biLSTM, self).__init__()
		self.num_classes = 4 # number of classes to predict
		self.input_size = 15 # number of features for each input vector
		self.num_layers = num_layers # number of LSTM layers
		self.hidden_size = hidden_size #LSTM hidden state
		
		if num_layers > 1:
			self.dropout = dropout
		else:
			self.dropout = 0

		self.lstm = nn.LSTM(input_size=self.input_size, 
							hidden_size=self.hidden_size,
							num_layers=self.num_layers, 
							batch_first=False,
							bidirectional=True,
							dropout = self.dropout)
		self.fc = nn.Linear(2*self.hidden_size, self.num_classes) # Final layer from hidden state to output

	def forward(self, sequence):
		#h_init = Variable(torch.zeros(2*self.num_layers, self.hidden_size, 1)) # Hidden state
		#c_init = Variable(torch.zeros(2*self.num_layers, self.hidden_size, 1)) # Internal state
		out, _ = self.lstm(sequence) # LSTM with input, hidden, and internal state
		out = self.fc(out)
		out = F.log_softmax(out, dim=1)
		return out
	
def trainBiLSTM(model, data_loader, num_epochs, learning_rate):
	objective = torch.nn.NLLLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	for epoch in range(num_epochs):
		loss_trn = 0.0
		for sequence, labels in data_loader:
			print("Forward pass...")
			outputs = model(sequence) # Forward pass	
			print("Computing loss")
			loss = objective(outputs, labels.long()) # Compute loss
		
			optimizer.zero_grad() # Reset gradients 
			print("Computing gradients")
			loss.backward() # Compute gradients
			optimizer.step() # Update parameters via backpropagation
			loss_trn += loss.item()

		#if (epoch == 0) | (epoch % 10 == 0):
		print(f"Epoch: {epoch}, loss: {loss_trn/len(data_loader)}")

def testBiLSTM(model, data_loader):
	
	pred_stats = {
		0:{"total":0, "correct":0, "class":"CDS"},
		1:{"total":0, "correct":0, "class":"Intron"},
		2:{"total":0, "correct":0, "class":"UTR"},
		3:{"total":0, "correct":0, "class":"Intergenic"},
	}

	total = 0
	correct = 0
	with torch.no_grad():
		for sequence, labels in data_loader:
			outputs = model(sequence)
			_, predicted = torch.max(outputs, dim=1)
			total += labels.shape[0]
			correct += int((predicted == labels).sum())
		for i in range(len(labels.tolist())):
			pred_stats[labels[i].item()]["total"] += 1
		if predicted[i].item() == labels[i].item():
			pred_stats[labels[i].item()]["correct"] += 1

# NOTE: Split fasta and gff file into train and test sets
# seqkit grep -n -v -r -p "Chr4" Athaliana_167_TAIR10.fa > Ath_Train.fa
# seqkit grep -n -r -p "Chr4" Athaliana_167_TAIR10.fa > Ath_Test.fa
# grep -v "Chr4" Athaliana_167_gene_exons_introns_longest.gff3 > Ath_Train.gff
# grep "Chr4" Athaliana_167_gene_exons_introns_longest.gff3 > Ath_Test.gff

if __name__ == '__main__':
	torch.manual_seed(1)
	batch_size = 3
	num_workers = 1
	pin_memory = False

	train_dat = GenomeData("Ath_Train.fa", "Ath_Train.gff", 20)
	trainLoader = DataLoader(train_dat,
				batch_size=batch_size, 
				drop_last=False, 
				num_workers=num_workers, 
				pin_memory=pin_memory,
				shuffle = True)

	test_dat = GenomeData("Ath_Test.fa", "Ath_Test.gff", 20000)
	testLoader = DataLoader(train_dat,
				batch_size=batch_size, 
				drop_last=False, 
				num_workers=num_workers, 
				pin_memory=pin_memory,
				shuffle = True)

	lstm_model = biLSTM(20, 1)
	trainBiLSTM(lstm_model, trainLoader, 100, 0.01)
	testBiLSTM(lstm_model, test_dat)