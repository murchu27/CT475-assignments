import pandas as pd
import numpy as np
#user interface to read in file
#should ask for number of classes

#binary svm model
class binarySVM:
	#need to have
	#a hyperplane/separator between classes
	#a measure of the margin between hyperplane and closest data points

	def __init__(self, features=2):
		self.features = features
		self.normal = np.zeros(self.features)
		self.intercept = 0 

	def train(self, data, target, pos_label, neg_label):
		self.data = data
		self.target = target
		self.pos_label = pos_label
		self.neg_label = neg_label
		#should write some function that scans data for negative label
		#or force user to specify label

		# use find_sv() to determine support vectors
		sv = self.find_sv()

		# sv[i] contains pairs of class A/B of support vectors indices with minimum distance 
		if self.target[sv[0]] == pos_label:
			pos_sv_index = 0
			neg_sv_index = 1
		else:
			pos_sv_index = 1
			neg_sv_index = 0		

		#normal is parallel to line connecting sv pairs 		
		#for now, arbitrarily choose first pair of vectors
		pos_sv = self.data[sv[pos_sv_index]]
		neg_sv = self.data[sv[neg_sv_index]]

		normal_weights = pos_sv-neg_sv

		#explanation of formula
		"""
		use the formula: y_i(dot(w,x_i) + b)
		for pos_sv, becomes dot(w,x_+) + b = 1 ==> b = 1 - dot(w,x_+)
		for neg_sv, becomes dot(w,x_-) + b = -1 ==> b = -1 - dot(w,x_-)
		then by simul. eqns: 1 - dot(w,x_+) = -1 - dot(w,x_-)
		so, 2 = dot(w,x_+) - dot(w,x_-) ; RHS is a scalar multiple of a
		finally, a = 2/(dot(w,x_+) - dot(w,x_-))
		"""

		normal_dot_pos = np.dot(normal_weights,pos_sv)
		normal_dot_neg = np.dot(normal_weights,neg_sv)
		normal_scale = 2/(normal_dot_pos-normal_dot_neg)

		#a is given by normal_scale; finally, w is given by scale*weights
		self.normal = normal_scale * normal_weights

		#now determine intercept using b = y_i - dot(w,x_i), with i referring to one of the sv's
		self.intercept = 1 - np.dot(self.normal,pos_sv)

	def find_sv(self):
		#find minimum distance between points in the two classes
		m = -1
		min_sample = (0,0)
		l = len(self.data)

		#iterate over all samples in the training data
		for i in range(l):
			sample = self.data[i]

			#we compare this sample with all the samples of the opposite label
			for j in range(i+1,l): #start from i+1 so as not to compare to self, or to previously checked samples
				#check if this sample has the opposite label
				if (self.target[j] == self.pos_label and self.target[i] != self.pos_label) or (self.target[j] != self.pos_label and self.target[i] == self.pos_label):
					other = self.data[j]
					
					#check the distance between the samples
					d = np.linalg.norm(sample - other)
					
					#see if less than or equal to previous minimum (if greater than previous minimum, do nothing)
					if (d < m) or (m == -1): #m = -1 at very beginning
						#if less than previous min, scrap the min_samples to this point, and instead add these samples
						#if equal to previous min, change nothing, but do add samples to min_samples    
						m = d
						min_sample = (i,j)

		#now, min_samples_A and min_samples_B contains the samples of class A and B (respectively) to be used as support vectors
		return(min_sample)

	def predict(self, predictdata):
		#explanation of steps
		"""
		predictdata should an array of data points

		u represents a point predictdata
		w represents the normal
		b represents the intercept

		predict using the following formula: 
		dot(w, u) + b < -1 ==> neg_sample
		dot(w, u) + b > 1 ==> pos_sample
		|dot(w, u) + b| < 1 ==> not sure

		assign each resulting label to a predicttarget array
		"""

		#seems to make a lot of false classfications atm
		#come back to this if there is time

		predicttarget = np.array([])
		self.distances = np.array([])
		unsure = 0
		x = True

		for u in predictdata:
			f = np.dot(self.normal, u) + self.intercept

			if f >= 1:
				predicttarget = np.append(predicttarget, self.pos_label)
			elif f <= -1:
				predicttarget = np.append(predicttarget, self.neg_label)
			else:
				predicttarget = np.append(predicttarget, "unsure")
				unsure += 1

			y = np.sign(f)
			self.distances = np.append(self.distances, y*(f)/np.linalg.norm(self.normal))
			if x:
				print(self.normal)
				x=False

		#print(self.distances)

		return predicttarget

class multiclassSVM:
	def __init__(self, c=2, features=2):
		self.features = features
		self.c = c
		self.models = np.array([])

	def train(self, data, target, classes):
		self.data = data
		self.target = target
		self.classes = classes
		assert(len(classes)==self.c)

		for i in range(self.c):
			#build an OVA model, using each class as the positive model
			b = binarySVM(self.features)
			b.train(self.data,self.target,classes[i],'other')
			self.models = np.append(self.models,b)


	def predict(self, predictdata):
	# 	#explanation of steps
	# 	"""
	# 	predictdata should an array of data points

	# 	u represents a point predictdata
	# 	w represents the normal
	# 	b represents the intercept

	# 	predict using the following formula: 
	# 	dot(w, u) + b < -1 ==> neg_sample
	# 	dot(w, u) + b > 1 ==> pos_sample
	# 	|dot(w, u) + b| < 1 ==> not sure

	# 	assign each resulting label to a predicttarget array
	# 	"""

	# 	#seems to make a lot of false classfications atm
	# 	#come back to this if there is time

		# predicttargets = np.array([])
		predicttargets = []

		for model in self.models:
			# predicttargets = np.append(predicttargets, model.predict(predictdata))
			predicttargets.append(model.predict(predictdata))

		# print(predicttargets)
		for i in range(len(predicttargets[0])):
			d = 0
			for j in range(d+1, self.c):
				if self.models[j].distances[i] < self.models[d].distances[i]:
					d = j
					print(i, "||", predicttargets[0][i], ": ", self.models[0].distances[i], "||", predicttargets[1][i], ": ", self.models[1].distances[i], "||", predicttargets[2][i], ": ", self.models[2].distances[i], "||")



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
	owls_classes = ['LongEaredOwl','SnowyOwl','BarnOwl']
	owls = pd.read_csv('owls.csv').values
	data = owls[:,:owl_features]
	target = owls[:,owl_features]
	#for i in range(len(owls)):
	#	print(data[i], target[i], sep='\t')

	train_index = np.array(list(range(0,30)) + list(range(45,75)) + list(range(90,120)))
	train_data = data[train_index]
	train_target = target[train_index]
	
	test_index = np.array(list(range(30,45)) + list(range(75,90)) + list(range(120,135)))
	test_data = data[test_index]
	
	owls_model = multiclassSVM(3, owl_features)
	owls_model.train(train_data,train_target,owls_classes)
	p = owls_model.predict(test_data)

	"""
	ai_features = 9 
	ai_plabel = 'positive'
	ai_nlabel = 'negative'
	ai = pd.read_table('autoimmune_transpose.txt').values
	d = ai[:,:ai_features]
	t = ai[:,ai_features]
	#for i in range(len(ai)):
	#	print(i, d[i], t[i], sep='\t')

	train_index = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,251])

	train_data = d[train_index]
	train_target = t[train_index]
	
	test_index = np.array([250,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375])
	test_data = d[test_index]

	ai_model = binarySVM(ai_features)
	ai_model.train(train_data,train_target,ai_plabel,ai_nlabel)
	a = ai_model.predict(test_data)
	# for i in range(len(a)):
	# 	print(a[i], t[test_index[i]])
	"""


	
	
if __name__ == "__main__":
	main()
