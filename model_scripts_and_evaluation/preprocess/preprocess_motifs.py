'''
This file adds additional columns for each motif in the codebook and places an
indicator (1 = motif present, 0 = theme nonpresent) in the column for the specific theme
'''

def contains_motif(row, motif):
	if str(motif) in row['Q8Motifs'].split(','):
		return'1'
	else: 
		return '0'



import pandas as pd
df = pd.read_csv('C1.csv')

df['ContainsMotif1'] = df.apply(lambda row: contains_motif(row, '1'), axis=1)
df['ContainsMotif2'] = df.apply(lambda row: contains_motif(row, '2'), axis=1)
df['ContainsMotif3'] = df.apply(lambda row: contains_motif(row, '3'), axis=1)
df['ContainsMotif4'] = df.apply(lambda row: contains_motif(row, '4'), axis=1)
df['ContainsMotif5'] = df.apply(lambda row: contains_motif(row, '5'), axis=1)
df['ContainsMotif6'] = df.apply(lambda row: contains_motif(row, '6'), axis=1)
df['ContainsMotif7'] = df.apply(lambda row: contains_motif(row, '7'), axis=1)
df['ContainsMotif8'] = df.apply(lambda row: contains_motif(row, '8'), axis=1)
df['ContainsMotif9'] = df.apply(lambda row: contains_motif(row, '9'), axis=1)
df['ContainsMotif10'] = df.apply(lambda row: contains_motif(row, '10'), axis=1)
df['ContainsMotif11'] = df.apply(lambda row: contains_motif(row, '11'), axis=1)

df.to_csv('C1.CSV')

