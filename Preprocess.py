import csv
import numpy as np 
from pubchempy import get_properties


ic_file_location='D:\\programming\\PROJECT\\AI\\Med\\CRXG\\--DATABASE\\PANCANCER_IC.csv'


def load_ic_data(ic_file_location):
	'''parametr: ic_file_location
	return: '''
	csv_file=open(ic_file_location,encoding='utf-8')
	
	ic_csv=csv.reader(csv_file)
	
	next(ic_csv,None)

	cellLine_total_list=[]
	drugs_total_list=[]
	ICs_total_list=[]

	drug_cid_dict={}

	for line in ic_csv:

		cid=line[1]

		drugs_total_list.append(line[0])
		cellLine_total_list.append(line[2])
		ICs_total_list.append(line[7])

		if not line[0] in drug_cid_dict:
			drug_cid_dict[line[0]]=cid

	return drugs_total_list,cellLine_total_list,ICs_total_list,drug_cid_dict

drugs,cellLines,ICs,drug_cid_dict=load_ic_data(ic_file_location)

def t0_Smiles():
	drug_Smile_dict={}

	cids=[v for k,v in drug_cid_dict.items()]
	R_dd_dict={v:k for k,v in drug_cid_dict.items()}

	results=get_properties('CanonicalSMILES', cids , namespace=u'cid', searchtype=None, as_dataframe=False)
	for result in results:
		if not result['CID'] in drug_Smile_dict:

			drug_Smile_dict[R_dd_dict[str(result['CID'])]]=result['CanonicalSMILES']

	return drug_Smile_dict

def pretty(smile):
	i=1
	result=[]
	while i<len(smile):
		ch=smile[i]
		if ch.islower():
			result.append(smile[i-1:i+1])
			i+=1
		else:
			result.append(smile[i-1])
		i+=1
	if not smile[-1].islower():
		result.append(smile[-1])
	return result

def get_chars_MAXlen():
	Smile_dict=t0_Smiles()
	chars_dict={}
	lens=[]
	for smile in [v for k,v in Smile_dict.items()]:
		lens.append(len(smile))
		for ch in pretty(smile):
			if not ch in chars_dict:
				chars_dict[ch]=len(chars_dict)

	return chars_dict,max(lens)

def t0_OneHot():
	drug_Smile_dict=t0_Smiles()
	drug_OneHot_dict={}

	chars,max_len=get_chars_MAXlen()
	R_ds_dict={v:k for k,v in drug_Smile_dict.items()}
	for drug in drug_Smile_dict:
		smile=pretty(drug_Smile_dict[drug])
		onehot=np.zeros([max_len,len(chars)])
		row=0
		for ch in smile:
			col=chars[ch]
			onehot[row,col]=1
			row+=1
		drug_OneHot_dict[drug]=onehot
	return drug_OneHot_dict,onehot.shape

def encode_drugs(drugs):
	drug_OneHot_dict,shp=t0_OneHot()
	y,z=shp
	drugs_encoded=np.zeros([len(drugs),y,z])
	x=0
	for drug in drugs:
		drugs_encoded[x]=drug_OneHot_dict[drug]
		x+=1
	return drugs_encoded

'''
---------------------------------------------------
---------------------------------------------------
---------------------------------------------------
---------------------------------------------------
'''

gentic_file_location='D:\\programming\\PROJECT\\AI\\Med\\CRXG\\--DATABASE\\PANCANCER_Genetic_feature.csv'


def get_cell_dict(cellLine_total_list):
	cell_dict={}
	for cell in cellLine_total_list:
		if not cell in cell_dict:
			cell_dict[cell]=len(cell_dict)
	return cell_dict
def get_gene_dict(gentic_file_location):
	csv_file=open(gentic_file_location,encoding='utf-8')
	genetic_csv=csv.reader(csv_file)
	next(genetic_csv,None)
	genes={}
	for line in genetic_csv:
		if not line[5] in genes:
			genes[line[5]]=len(genes)
	return genes

def t0_cell_feature(gentic_file_location):
	'''parameter: gentic_file_location
	returns: (dict) for cell:(np) genetic_featur
	and genetic_feature.shape'''
	cell_dict=get_cell_dict(cellLines)
	genes=get_gene_dict(gentic_file_location)
	cell_feature_dict={}
	csv_file=open(gentic_file_location,encoding='utf-8')
	genetic_csv=csv.reader(csv_file)
	next(genetic_csv,None)
	for cell in cell_dict:
		cell_feature_dict[cell]=np.zeros([2,len(genes)])
	for line in genetic_csv:
		if line[0] in cell_dict:
			if line[6]=='1':cell_feature_dict[line[0]][0,genes[line[5]]]=1
			if line[7]=='gain':cell_feature_dict[line[0]][1,genes[line[5]]]=1
	return cell_feature_dict, (2,len(genes))


def encode_cells(cellLines,gentic_file_location):
	cell_feature_dict,shp=t0_cell_feature(gentic_file_location)
	y,z=shp
	cells_encoded=np.zeros([len(cellLines),y,z])
	x=0
	for cell in cellLines:
		cells_encoded[x]=cell_feature_dict[cell]
		x+=1
	return cells_encoded
'''
----------------------------------------------------------------
----------------------------------------------------------------
----------------------------------------------------------------
'''
def encode_ICs(ICs):
	ICs_encoded=np.asarray(ICs,dtype=float)
	return ICs_encoded

drugs_encoded=encode_drugs(drugs)
cells_encoded=encode_cells(cellLines,gentic_file_location)
ICs_encoded=encode_ICs(ICs)

print(drugs_encoded.shape)
print(cells_encoded.shape)
print(ICs_encoded.shape)

np.save('drugs_encoded',drugs_encoded)
np.save('cells_encoded',cells_encoded)
np.save('ICs_encoded',ICs_encoded)
