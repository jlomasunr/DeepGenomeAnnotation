# DeepGenomeAnnotation

Pytorch implementation of a BiLSTM-CRF for constructing genome annotations

## Dataset

Split fasta and gff file into train and test sets
```
seqkit grep -n -v -r -p "Chr4" Athaliana_167_TAIR10.fa > Ath_Train.fa
seqkit grep -n -r -p "Chr4" Athaliana_167_TAIR10.fa > Ath_Test.fa
grep -v "Chr4" Athaliana_167_gene_exons_introns_longest.gff3 > Ath_Train.gff
grep "Chr4" Athaliana_167_gene_exons_introns_longest.gff3 > Ath_Test.gff
```

## Environment 

```
conda create -n bilstm -y && conda activate bilstm
conda install -c bioconda -c pytorch pytorch biopython numpy pandas
```

## Training

```
conda activate bilstm
sbatch train_bilstm.sh
```
