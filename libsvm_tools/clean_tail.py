import os
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
feature_size = sys.argv[3]

# Sort features by ID, count positive documents
out_list = []
with open(input_file) as fin:
    for line in fin:
        arr = line.strip().split(' ')
        out_list.append(arr[:int(feature_size) + 2])
        

with open(output_file, 'w') as fout:
    for array in out_list:
        fout.write(" ".join(array))
        fout.write('\n')