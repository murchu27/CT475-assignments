import pandas as pd
import numpy as np
#user interface to read in file
#should ask for number of classes

#binary svm model
class binarySVM:
	#need to have
	#a hyperplane/separator between classes
	#a measure of the margin between hyperplane and closest data points
	#sep = []
	#features = 1
	#margin = 0	

	def __init__(self, features):
		self.features = features
		self.normal = np.zeros(features)
		self.intercept = 0 

	def train(self, data, target, pos_label):
		self.data = data
		self.target = target
		

	def predict(self, predictdata):
		#to be written by Mark	
		pass

#multiclass svm model
#based on OVA classfication using binary model

#determine how to split given dataset
#also how to repeat this 10 times

def testbinSVM():
	b = binarySVM(3)
	assert b.features == 3
	#print(b.omega)
	#print(b.omega.all())
	#print(b.omega.any())
	#assert list(b.omega) == [0,0,0]
	#assert b.omega_zero == 0

def main():
	owl_features = 4
	owls = pd.read_csv('owls.csv').values
	data = owls[:,:owl_features]
	target = owls[:,owl_features]
	#for i in range(len(owls)):
	#	print(data[i], target[i], sep='\t')

	ai_features = 9 
	ai_label = 'positive'
	ai = pd.read_table('autoimmune_transpose.txt').values
	d = ai[:,:ai_features]
	t = ai[:,ai_features]
	#for i in range(len(ai)):
	#	print(d[i], t[i], sep='\t')

	testbinSVM()

	ai_model = binarySVM(ai_features)
	ai_model.train(d,t,ai_label)
	
if __name__ == "__main__":
	main()
