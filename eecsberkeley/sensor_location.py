import numpy as np
import scipy.spatial
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

########################################################################
#########  Data Generating Functions ###################################
########################################################################
def generate_sensors(k = 7, d = 2):
	"""
	Generate sensor locations. 
	Input:
	k: The number of sensors.
	d: The spatial dimension.
	Output:
	sensor_loc: k * d numpy array.
	"""
	sensor_loc = 100*np.random.randn(k,d)
	return sensor_loc

def generate_data(sensor_loc, k = 7, d = 2, 
				 n = 1, original_dist = True):
	"""
	Generate the locations of n points.  

	Input:
	sensor_loc: k * d numpy array. Location of sensor. 
	k: The number of sensors.
	d: The spatial dimension.
	n: The number of points.
	original_dist: Whether the data are generated from the original 
	distribution. 

	Output:
	obj_loc: n * d numpy array. The location of the n objects. 
	distance: n * k numpy array. The distance between object and 
	the k sensors. 
	"""
	assert k, d == sensor_loc.shape

	obj_loc = 100*np.random.randn(n, d)
	if not original_dist:
	   obj_loc += 1000
	   
	distance = scipy.spatial.distance.cdist(obj_loc, 
										   sensor_loc, 
										   metric='euclidean')
	distance += np.random.randn(n, k)  
	return obj_loc, distance
##################################################################
# Starter code for Part (b)
##################################################################

def plotter(point):
	plt.scatter(point[0],point[1],color='red',marker='o')
	plt.pause(0.5)


np.random.seed(0)
sensor_loc = generate_sensors()
obj_loc, distance = generate_data(sensor_loc)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# u = np.linspace(-1, 1, 100)
# a = [0, 100, 200, 300, 400]
# x, y = np.meshgrid(a, a)
# print x, y
#
# # z=<YOUR LOG FUNCTION>
#
# z=np.zeros((len(x),len(x[0])))
# for i, item in enumerate(x):
# 	for j,val in enumerate(item):
#
# 		sum = 0
# 		for k,sensor in enumerate(sensor_loc):
# 			sum+=((((sensor[0]-x[i][j])**2 + (sensor[1]-y[i][j])**2)**.5)-distance[0][k])**2
# 		z[i][j] = -sum
#
#
# print z
# ax.contourf(x, y, z)
# plt.show()

#Trying to normalize distance values and coordinates
#---------NORMALIZE START
# maxd=distance[0][0]
# maxa=sensor_loc[0][0]
# maxb=sensor_loc[0][1]
#
# for i,item in enumerate(sensor_loc):
# 	tempd = distance[0][i]
# 	if tempd<0:
# 		tempd*=-1
# 	if tempd>maxd:
# 		maxd=tempd
# 	tempa = item[0]
# 	if tempa < 0:
# 		tempa *= -1
# 	if tempa > maxa:
# 		maxa = tempa
# 	tempb = item[1]
# 	if tempb < 0:
# 		tempb *= -1
# 	if tempb > maxb:
# 		maxb = tempb
#
# distance[0] = distance[0]/maxd
# sensor_loc[:,0] = sensor_loc[:,0]/tempa
# sensor_loc[:,1] = sensor_loc[:,1]/tempb
#########------NORMALIZE END

single_distance = distance[0]
plt.xlabel("x")
plt.ylabel("y")
plt.title("SENSOR LOCATIONS")
# print sensor_loc,"\n------------",obj_loc,"\n------------",distance,maxa,maxb,maxd
plt.scatter(sensor_loc[:,0],sensor_loc[:,1],color='blue',marker='s',label='sensors')

plotter(obj_loc[0])
learning_rate = 1
initial_loc = np.array([0,0])


def gradient_calculator(sensors,x,d):
	#sensors has ai and bi
	#d has di
	#x has old x1 and y1
	sx=0
	sy=0
	d=d[0]
	for i,sensor in enumerate(sensors):
		sx += 2*(sensor[0]-x[0]) -2*d[i]*((sensor[0]-x[0])/(((sensor[0]-x[0])**2 + (sensor[1]-x[1])**2)**.5))
		sy += 2*(sensor[1]-x[1]) -2*d[i]*((sensor[1]-x[1])/(((sensor[0]-x[0])**2 + (sensor[1]-x[1])**2)**.5))
	return np.array([sx,sy],dtype=float)

def perform_gradient_steps(learning_rate=1,initial_loc=np.array([0,0]),iterations=10):
	i=0
	all_locations = []
	all_locations.append(initial_loc)
	while i<=iterations:
		new_loc = initial_loc - learning_rate*gradient_calculator(sensor_loc,initial_loc, distance)
		plotter(new_loc)
		initial_loc = new_loc
		all_locations.append(new_loc)
		i+=1

	print all_locations
perform_gradient_steps(0.1,initial_loc,10)
#x1 = x1-learningrate*( (d/dx) of L)
plt.legend()
plt.show()






##################################################################
# Starter code for Part (c)
##################################################################
def generate_data_given_location(sensor_loc, obj_loc, k = 7, d = 2):
	"""
	Generate the distance measurements given location of a single object and sensor. 

	Input:
	obj_loc: 1 * d numpy array. Location of object
	sensor_loc: k * d numpy array. Location of sensor. 
	k: The number of sensors.
	d: The spatial dimension. 

	Output: 
	distance: 1 * k numpy array. The distance between object and 
	the k sensors. 
	"""
	assert k, d == sensor_loc.shape 
	 
	distance = scipy.spatial.distance.cdist(obj_loc, 
					   sensor_loc, 
					   metric='euclidean')
	distance += np.random.randn(1, k)  
	return obj_loc, distance

def log_likelihood(obj_loc, sensor_loc, distance): 
	"""
	This function computes the log likelihood (as expressed in Part a).
	Input: 
	obj_loc: shape [1,2]
	sensor_loc: shape [7,2]
	distance: shape [7]
	Output: 
	The log likelihood function value. 
	"""  
	diff_distance = np.sqrt(np.sum((sensor_loc - obj_loc)**2, axis = 1))- distance
	func_value = -sum((diff_distance)**2)/2
	return func_value





np.random.seed(100)
# Sensor locations. 
sensor_loc = generate_sensors()
num_gd_replicates = 100

# Object locations. 
obj_locs = [[[i,i]] for i in np.arange(100,1000,100)]  














