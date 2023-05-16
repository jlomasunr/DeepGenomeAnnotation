import sys, os, time, argparse, torch
from Bio import SeqIO
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from itertools import islice
from socket import gethostname

# Custom dataset class for extracting one-hot-encoded dna sequences and basewise classes
class GenomeData(Dataset):
	def __init__(self, fasta_path, class_path, window):
		# Expects the gff to contain at least the following features ('gene','five_prime_UTR','CDS','intron','three_prime_UTR')
		# NOTE: AGAT can be used to add introns (agat_sp_add_introns.pl --gff [] --out [])
		# NOTE: The gff should also be reduced to a single transcripts (agat_sp_keep_longest_isoform.pl -gff [] -o [])
		self.fasta_path = fasta_path
		self.class_path = class_path
		self.feature_list = ('five_prime_UTR','CDS','intron','three_prime_UTR')
		self.class_map = {
			"CDS": 0,
			"intron": 1,
			"five_prime_UTR": 2,
			"three_prime_UTR": 2,
			"intergenic": 3
			}
		
		self.window = window
		self.seq_map = dict(zip("ACGTRYSWKMBDHVN", range(15)))
		
		self.total_length = 0
		self.chr_lengths = [0]
		for record in SeqIO.parse(fasta_path, "fasta"):
			self.chr_lengths.append(len(record) + self.total_length)
			self.total_length += len(record)
		del record

	def __len__(self):
		return self.total_length - self.window
	
	def __getitem__(self, idx):
		# Get the chromosome sequence corresponding to the index
		# Then update the index to chromosome coordinates
		seqs = SeqIO.parse(self.fasta_path, "fasta")
		clss = SeqIO.parse(self.class_path, "fasta")
		chr_bool = [idx < chr_idx for chr_idx in self.chr_lengths]
		for i in range(1,len(chr_bool)):
			if chr_bool[i] and not chr_bool[i-1]:
				seq_record = next(islice(seqs, i-1, None))
				cls_record = next(islice(clss, i-1, None))
				idx -= self.chr_lengths[i-1]
				if idx + self.window > len(seq_record):
					idx = len(seq_record) - self.window
				break
		del seqs
		del clss
		
		# Extract the dna and class sequences from the chromosome according to the
		# index and the window length. Then convert to a one-hot-encoded tensor
		end_idx = idx+self.window
		sequence_onehot = torch.Tensor(np.eye(15)[[self.seq_map[i] for i in seq_record[idx+1:end_idx+1].seq]])
		class_seq = cls_record[idx+1:end_idx+1].seq
		classes = torch.Tensor(torch.tensor([int(i) for i in class_seq])).type(torch.LongTensor)
		del seq_record
		del cls_record
		
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

	objective = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	
	epochstart = time.time()
	for epoch in range(num_epochs):
		loss_trn = 0.0
		i = 0
		for sequence, labels in data_loader:
			sequence = sequence.permute(1,0,2).to(device)
			labels = labels.to(device)
			
			outputs = model(sequence) # Forward pass	
			loss = objective(outputs.permute(1,2,0), labels) # Compute loss
			optimizer.zero_grad() # Reset gradients 
			loss.backward() # Compute gradients
			optimizer.step() # Update parameters via backpropagation
			loss_trn += loss.item()
			i += 1

		#if (epoch == 0) | (epoch % 10 == 0):
		if rank == 0:
			print(f"Epoch: {epoch}, loss: {loss_trn/len(data_loader)}... {time.time() - epochstart}", file=sys.stderr)
			torch.save(model.module.state_dict(), outfile) 

def testBiLSTM(model, seq, labels):
	model.eval()
	pred_stats = {
		"Overall":{"total":0, "correct":0, "percent":0},
		0:{"class":"CDS", "total":0, "correct":0, "percent":0},
		1:{"class":"Intron", "total":0, "correct":0, "percent":0},
		2:{"class":"UTR","total":0, "correct":0, "percent":0},
		3:{"class":"Intergenic","total":0, "correct":0, "percent":0 },
	}

	with torch.no_grad():
		outputs = model(seq)
		_, predicted = torch.max(outputs, dim=1)
		pred_stats["Overall"]["total"] = labels.size()[0]
		pred_stats["Overall"]["correct"] = int((predicted == labels).sum())
		pred_stats["Overall"]["percent"] = (pred_stats["Overall"]["correct"]/pred_stats["Overall"]["total"])*100
	for i in range(len(labels)):
		pred_stats[labels[i].item()]["total"] += 1
		if predicted[i].item() == labels[i].item():
			pred_stats[labels[i].item()]["correct"] += 1
	
	for i in range(4):
		pred_stats[i]["percent"] = (pred_stats[i]["correct"]/pred_stats[i]["total"])*100
	
	return pred_stats 

def trainModel(fasta, gff, lstm_model=None, window=20, lr=0.1, epochs=100, hidden=10, layers=1, outfile="model_state.pt",batch_size=1, seed=123, num_workers=0, pin_memory=True):
	torch.manual_seed(seed)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	train_dat = GenomeData(fasta, gff, window)
	trainSampler = distributed.DistributedSampler(train_dat, num_replicas=world_size, rank=rank)
	
	if lstm_model:
		lstm_model.load_state_dict(torch.load("lstm_state.pt")) # needs map_location...
	else:
		lstm_model = biLSTM(hidden, layers)
	
	if torch.cuda.device_count() >= 1:
		torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
		local_rank = rank - gpus_per_node * (rank // gpus_per_node)
		torch.cuda.set_device(local_rank)
		lstm_model = nn.parallel.DistributedDataParallel(lstm_model.to(local_rank), 
							device_ids=[local_rank],
							gradient_as_bucket_view=True)
		trainLoader  = DataLoader(train_dat, 
						batch_size=batch_size, 
						sampler=trainSampler,
						num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), 
						pin_memory=pin_memory)
		
		print(f"Beginning training on cuda {local_rank}.")
		trainBiLSTM(lstm_model, trainLoader, epochs, lr, local_rank, outfile)
	else:
		torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=world_size)
		local_rank = rank - cpus_per_node * (rank // cpus_per_node)
		torch.device(local_rank)
		lstm_ddp = nn.parallel.DistributedDataParallel(lstm_model, device_ids=None, output_device=None, gradient_as_bucket_view=True)
		trainLoader  = DataLoader(train_dat, 
						batch_size=batch_size, 
						sampler=trainSampler,
						num_workers=num_workers, 
						pin_memory=pin_memory)

		print(f"Beginning training on {device} with rank {rank}.")
		trainBiLSTM(lstm_ddp, trainLoader, epochs, lr, device, outfile)
	
	torch.distributed.destroy_process_group()

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('action', type=str, help="[train|test] : Action to take")
	ap.add_argument("fasta", type=str, help="Fasta file")
	ap.add_argument('labels', type=str, help="Labels fasta file")
	ap.add_argument('--window', default=20, type=int, help="Window size for lstm")
	ap.add_argument('--hidden', default=10, type=int, help='Number of hidden units for lstm')
	ap.add_argument('--layers', default=1, type=int, help='Number of lstm layers')
	ap.add_argument('--outfile', default="model_state.pt", type=str, help="File to save model sate to")
	ap.add_argument('--batchsize', default=1, type=int, help="Batch size for data loader")
	ap.add_argument('--seed', default=123, type=int, help="Random seed for torch")
	ap.add_argument('--workers', default=0, type=int, help="Number of subprocesses for data loading")
	ap.add_argument('--model', default=None ,type=str, help="Model state_dict")
	ap.add_argument('--lr', default=0.1, type=float, help="Learning rate")
	ap.add_argument('--epochs', default=100, type=int, help="Number of training epochs")
	ap.add_argument('--nocuda', action='store_true', help="CPU only")
	args = ap.parse_args()

	if args.action == 'train':

		rank          = int(os.environ["SLURM_PROCID"])
		world_size    = int(os.environ["WORLD_SIZE"])
		cpus_per_node = int(os.environ["SLURM_CPUS_ON_NODE"])
		

		if not args.nocuda:
			gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
			assert gpus_per_node == torch.cuda.device_count()
			print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
			  f" {gpus_per_node} allocated GPUs per node and {cpus_per_node} CPUs per node.", flush=True)
		else:
			print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
                          f" {cpus_per_node} CPUs per node.", flush=True)
		
		trainModel(args.fasta, 
			args.labels, 
			window=args.window, 
			hidden=args.hidden, 
			layers=args.layers, 
			outfile=args.outfile,
			batch_size=args.batchsize, 
			seed=args.seed, 
			num_workers=args.workers, 
			pin_memory=True,
			lr=args.lr,
			epochs=args.epochs,
			lstm_model=args.model)

	if args.action == 'test':
		# Requires: fasta, labels, --hidden, --layers, --model, --window
		lstm_model = biLSTM(args.hidden, args.layers)
		lstm_model.load_state_dict(torch.load(args.model))

		test_dat = GenomeData(args.fasta, args.labels, args.window)
		(seq, labels) = test_dat.__getitem__(0)
		test = testBiLSTM(lstm_model, seq, labels)
		print(test)
