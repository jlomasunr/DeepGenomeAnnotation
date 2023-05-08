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
import time
import argparse


# Custom dataset class for extracting one-hot-encoded dna sequences
# and basewise classes
class GenomeData(Dataset):
	def __init__(self, fasta_path, gff_path, window):
		# Expects the gff to contain at least the following features ('gene','five_prime_UTR','CDS','intron','three_prime_UTR')
		# NOTE: AGAT can be used to add introns (agat_sp_add_introns.pl --gff [] --out [])
		# NOTE: The gff should also be reduced to a single transcripts (agat_sp_keep_longest_isoform.pl -gff [] -o [])
		self.fasta_path = fasta_path
		self.gff = pd.read_csv(gff_path, sep="\t", usecols=[0,2,3,4], names=["chr","feature", "start","end"], comment="#")
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
		for record in SeqIO.parse(fasta_path, "fasta"):
			self.chr_lengths.append(len(record) + self.total_length)
			self.total_length += len(record)

	def __len__(self):
		return self.total_length - self.window
	
	def __getitem__(self, idx):
		# Get the chromosome sequence corresponding to the index
		# Then update the index to chromosome coordinates
		seqs = SeqIO.parse(self.fasta_path, "fasta")
		chr_bool = [idx < chr_idx for chr_idx in self.chr_lengths]
		for i in range(1,len(chr_bool)):
			if chr_bool[i] and not chr_bool[i-1]:
				record = next(islice(seqs, i-1, None))
				idx -= self.chr_lengths[i-1]
				if idx + self.window > len(record):
					idx = len(record) - self.window
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
		gff = self.gff.loc[self.gff["chr"] == chromosome]
		gff = gff.loc[gff["feature"].isin(feature_list)]
		gff.loc[gff["start"] > gff["end"], ('start','end')] = (gff.loc[gff["start"] > gff["end"], ('start','end')].values)
		
		#Start with everything 'intergenic' and then update each base
		class_map = [3 for _ in range(len(seq_map))]
		coord_map = [i for i in range(idx, end_idx)]
		for i in range(len(class_map)):
			feat = gff.loc[((gff["start"] <= coord_map[i]) & (gff["end"] >= coord_map[i]))].reset_index(drop=True)
			if len(feat) != 0:
				class_map[i] = self.class_map[feat.loc[0,"feature"]]
		
		classes = torch.Tensor(class_map).type(torch.LongTensor)
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
		return out
	
def trainBiLSTM(model, data_loader, num_epochs, learning_rate, device, outfile):

	objective = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	epochstart = time.time()
	for epoch in range(num_epochs):
		loss_trn = 0.0
		i = 0
		for sequence, labels in data_loader:
			if device.type == "cpu":
				sequence = sequence.permute(1,0,2)
			else:
				sequence = sequence.permute(1,0,2).cuda()
				labels = labels.cuda()

			batchstart = time.time()
			outputs = model(sequence) # Forward pass	
			loss = objective(outputs.permute(1,2,0), labels) # Compute loss
			optimizer.zero_grad() # Reset gradients 
			loss.backward() # Compute gradients
			optimizer.step() # Update parameters via backpropagation
			batchend = time.time()
			loss_trn += loss.item()
			i += 1
			print(f"Epoch {epoch}, Batch {i} complete. {batchend - batchstart}")

		#if (epoch == 0) | (epoch % 10 == 0):
		print(f"Epoch: {epoch}, loss: {loss_trn/len(data_loader)}... {time.time() - epochstart}")
		torch.save(model.state_dict(), "lstm_state.pt")

def testBiLSTM(model, data_loader):
	#model.eval()
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

def trainModel(fasta, gff, window=20, hidden=10, layers=1, outfile="model_state.pt",batch_size=1, seed=123, num_workers=0, pin_memory=True):
	torch.manual_seed(seed)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	train_dat = GenomeData(fasta, gff, window)
	trainLoader = DataLoader(train_dat,
				batch_size=batch_size, 
				drop_last=False, 
				num_workers=num_workers, 
				pin_memory=pin_memory,
				shuffle = True)
	
	lstm_model = biLSTM(hidden, layers, outfile).to(device)
	print(f"Beginning training on {device}")
	trainBiLSTM(lstm_model, trainLoader, 10, 0.1, device, outfile)


# NOTE: Split fasta and gff file into train and test sets
# seqkit grep -n -v -r -p "Chr4" Athaliana_167_TAIR10.fa > Ath_Train.fa
# seqkit grep -n -r -p "Chr4" Athaliana_167_TAIR10.fa > Ath_Test.fa
# grep -v "Chr4" Athaliana_167_gene_exons_introns_longest.gff3 > Ath_Train.gff
# grep "Chr4" Athaliana_167_gene_exons_introns_longest.gff3 > Ath_Test.gff

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('action', type=str, required=True, help="[train|] : Action to take")
	ap.add_argument("fasta", type=str, required=True, help="Fasta file")
	ap.add_argument('gff', type=str, required=True, help="Gff file")
	ap.add_argument('--window', default=20, type=int, help="Window size for lstm")
	ap.add_argument('--hidden', default=10, type=int, help='Number of hidden units for lstm')
	ap.add_argument('--layers', default=1, type=int, help='Number of lstm layers')
	ap.add_argument('--outfile', default="model_state.pt", type=str, help="File to save model sate to")
	ap.add_argument('--batchsize', default=1, type=int, help="Batch size for data loader")
	ap.add_argument('--seed', default=123, type=int, help="Random seed for torch")
	ap.add_argument('--workers', default=0, type=int, help="Number of subprocesses for data loading")

	if ap.action == 'train':
		trainModel(ap.fasta, 
			ap.gff, 
			window=ap.window, 
			hidden=ap.hidden, 
			layers=ap.layers, 
			outfile=ap.outfile,
			batch_size=ap.batchsize, 
			seed=ap.seed, 
			num_workers=ap.workers, 
			pin_memory=True)