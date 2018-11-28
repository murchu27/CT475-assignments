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

	def train(self, traindata, trainclass, pos_label):
		data = np.array(1)

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
	f = 'owls.csv'
	filef = open(f)
	print(filef.readline())
	testbinSVM()


if __name__ == "__main__":
	main()
