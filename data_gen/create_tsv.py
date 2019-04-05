import csv
import sys

# Chromosome, Position, Strand, IPD Ratio
ipd_f = open("ipd-h5.txt", "r") 
ipd_file = csv.reader(ipd_f, delimiter=",")

# line number -> fold change data
fold_file = open("JM083.fold-change.txt", "r")

# chromosome -> size
c_size_f = open("LtaP_PB.sizes.txt", "r")
c_size_file = csv.reader(c_size_f, delimiter="\t")
c_offset = {}
c_size = {}

# chromosome -> sequence
c_genome_f = open("LtaP_PB.genome.fasta", "r")
c_genome = {}

c_g_curr = ''
for line in c_genome_f:
    if '>' in line:
        c_g_curr = line[1:].strip()
        c_genome[c_g_curr] = ''
    else:
        c_genome[c_g_curr] += line.strip()

# Output file
# chromosome, position, fold change value (non-smooth), ipd top strand, ipd bottom strand
output_file = open("l_tarentolae.tsv", "w")
# output_file.write("Chromosome\tPosition\tFold Change Value (Non-Smooth)\tIPD Top Strand\tIPD Bottom Strand\tIPD Ratio\n")
output_file.write("Chromosome\tPosition\tFold Change\tIPD Top Ratio\tIPD Bottom Ratio\tBase\n")

# Maps a position to a fold value.
pos_fold = {}
fold_index = 0

# Find offsets
off = 0

# Contains all the data needed.
# Chromosome -> (Position -> Data)
data = {}

for row in c_size_file:
    c_offset[row[0]] = off
    c_size[row[0]] = int(row[1])
    off += int(row[1])


LINES_TO_READ = 1000
def load_more_folds():
    global fold_index, LINES_TO_READ
    for i in range(LINES_TO_READ):
        pos_fold[fold_index] = fold_file.readline().strip("\n")
        fold_index += 1

index = -1
# Load IPD data, and write to output
for row in ipd_file:
    index += 1
    if index == 0:
        continue
    if int(row[1]) == 0:
        continue
    pos = (c_offset[row[0]] + int(row[1])) 
    while pos not in pos_fold:
        load_more_folds()
    is_top = int(row[2]) == 0
    if row[0] not in data:
        data[row[0]] = {}
    if row[1] not in data[row[0]]:
        if not is_top:
            data[row[0]][row[1]] = [row[1], pos_fold[pos], "N/A", row[3]]
        else:
            data[row[0]][row[1]] = [row[1], pos_fold[pos], row[3], "N/A"]
    else:
        if not is_top:
            data[row[0]][row[1]][3] = row[3]
        else:
            data[row[0]][row[1]][2] = row[3]
    # output_file.write(row[0] + "\t" + row[1] + "\t" + pos_fold[pos] + "\t" + ("1" if  else "0") + "\t" + ("1" if int(row[2]) == 1 else "0") + "\t" + row[3] + "\n")
    if index % 100000 == 0:
        print(index)

# Loads base character into data
#for c in c_genome:
#    for i in range(len(c_genome[c])):
#        data[c][str(i)][4] = c_genome[c][i]

print("Writing to file")

for c in c_size:
    for i in range(c_size[c]):
        if str(i) not in data[c]:
            # Chromosome\tPosition\tFold Change\tIPD Bottom Ratio\tIPD Top Ratio\tBase\n
            output_file.write(c + "\t" + str(i) + "\tN/A\tN/A\tN/A\t" + c_genome[c][i] + "\n")
        else:
            # Position\tFold Change\tIPD Top Ratio\tIPD Bottom Ratio\tBase
            datum = data[c][str(i)]
            output_file.write(c + "\t" + str(i) + "\t" + datum[1] + "\t" + datum[2] + "\t" + datum[3] + "\t" + c_genome[c][i] + "\n")

ipd_f.close()
fold_file.close()
c_size_f.close()
output_file.close()


# Gather data
# Store data for a single chromosome in a dictionary
# Chromosome -> Data
# Data -> [data entries]
# * Lots of null checking
