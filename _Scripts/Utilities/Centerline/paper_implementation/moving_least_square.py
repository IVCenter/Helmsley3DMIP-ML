#    moving_least_square.py @ 2018, Wanze Xie
#
#	 Purpose: run the moving least sqaure algorithim.
#    Usage:
#           call the "Moving_Least_Square(target_p, neighbor_points)" function
#			where the target_p should be the current target point, and the 
#			neighnor_points should be target point's neighbor, which could contain
#			the target point itself.
#

import scipy
import numpy as np
from scipy.optimize import minimize

def normalize(v):
	v = np.array(v)
	norm = np.linalg.norm(v)
	if norm == 0:
		return v
	return v / norm

def calc_weight(r, H):
	return np.exp(-(r**2/H**2))

#
# Transform the "point" from the coordinate system established with target_p and tangent_vector (x axis)
# back to the global 2D plane coordinate system
#
# @Return: the 2D coordinate in the global 2D plane in an numpy array [x,y]
#
def transform_point_inverse(point, target_p, tangent_vector):
	# Note that target_p is the origin and the tangent_vector is the new x axis
	tangent_vector = normalize(tangent_vector)

	v = tangent_vector
	u = np.array([-v[1],v[0]]) 
	# v is the new x axis, and u is the new y axis
	rotation_matrix = np.array([v, u])
	#print("[Debug:]",rotation_matrix,point)    
	return (np.linalg.inv(rotation_matrix).dot(point) + target_p)


#
# Transform the whole set of points and return the points with transformed coord from
# the global 2D plane coordinate system to the the coordinate system established with 
# target_p and tangent_vector (x axis)
#
# @Return: the new list of 2D coordinates (2D numpy array)
#
def transform_coordinate_system(coords_list, target_p, tangent_vector, inverse=0):
	# Note that target_p is the origin and the tangent_vector is the new x axis
	tangent_vector = normalize(tangent_vector)
		
	v = tangent_vector
	u = np.array([-v[1],v[0]])
	assert len(v) == 2, "The len of v should be 2"

	# v is the new x axis, and u is the new y axis
	rotation_matrix = np.array([v, u])

	if inverse == 1:
		return np.array([
			(np.matmul(np.linalg.inv(rotation_matrix), np.array([x,y])) + target_p) for (x,y) in coords_list])
	
	return np.array([
		rotation_matrix.dot(np.array([x,y]) - target_p) for (x,y) in coords_list])


#
# Purpose: Compute X1^4 + X2^4 + ... for [x1, x2, x3, ...]
#
# Param:  an array for [x1, x2, x3, ...]
#
# Return: return a float of the sum result
#
def quadri_sum(input_array):
	return sum([ (x * x * x * x) for x in input_array])

#
# Purpose: Compute X1^3 + X2^3 + ... for [x1, x2, x3, ...]
#
# Param:  an array for [x1, x2, x3, ...]
#
# Return: return a float of the sum result
#
def tri_sum(input_array):
	return sum([ (x * x * x) for x in input_array])


#
# Purpose: Compute X1^2 + X2^2 + ... for [x1, x2, x3, ...]
#
# Param:  an array for [x1, x2, x3, ...]
#
# Return: return a float of the sum result
#
def square_sum(input_array):
	return sum([ (x * x) for x in input_array])


#
# Purpose: Get the current linear loss 
#
# Param:
#               y_true:  an array that contains all true y values
#               y_pred:  an array that contains all beta[0]*x + beta[1] result
#               sample_weights: an array of weights
#
# Return: return an array [x,y] to represent the coordinate after moving
def quadratic_weighted_loss(y_pred, y_true, sample_weights=None):
	# Sanity check
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	assert len(y_true) == len(y_pred), "y_true and y_pred should have same length"

	# No weight specified
	if type(sample_weights) == type(None):
		# assign weight to be one by default
		sample_weights = np.full(len(y_true), 1)
	else:
		# weight must be non-negative
		assert len(sample_weights) == len(y_true), "weights should have same dimension as y_true"

	D = np.sum((y_pred - y_true)**2.0 * sample_weights)

	return D

#
# Purpose: Get the current linear loss 
#
# Param:
#               dist: an array of square distance from Pk to each point in array in MST
#               H: prescribed param for weight, it should already be handled that 
#                  all points here are within H
#
# Return: 
def objective_function_linear(beta, X, Y, weights):
	beta = np.array(beta)
	assert len(beta) == 2, "beta should have two elements since it is linear function"
	
	y_pred = beta[0] * X + beta[1]
	y_true = Y

	error = quadratic_weighted_loss(y_pred, y_true, sample_weights=weights)
	return error


# Purpose: Get the current quadratic loss 
#
# Param:

def objective_function_square(beta, X, Y, weights):
	beta = np.array(beta)
	assert len(beta) == 3, "beta should have three elements since it is quadratic regression"

	y_pred = beta[0] * (X**2) + beta[1] * X + beta[2]
	y_true = Y
	
	error = quadratic_weighted_loss(y_pred, y_true, sample_weights=weights)
	return error



#
# Purpose: Compute the Moving Leaset Square of a single point p
#
# Param:
#                    p:  an array [x,y] of the point we need to move
#      neighbor_points:  an 2D array [p1,p2,p3,...] that contains the neighnors of p
#                        include p itself
#
# Return: return an array with two elements [x,y] to represent the coordinate after moving
#
def Moving_Least_Square(target_p, neighbor_points, verbose=0):

	'''
	Find Local Regression Line
	'''
	# extract the array for X coords and Y coords
	X = np.array([x for (x,y) in neighbor_points])
	Y = np.array([y for (x,y) in neighbor_points])

	# calc the distance of neighbors with the target_p
	dists = np.array([np.linalg.norm(p - target_p) for p in neighbor_points])

	# find the largest distance as H and get the weights array
	H = np.amax(dists)
	#print("[MLS]: Found the maximal distance: ", H)

	weights = np.array([calc_weight(r,H) for r in dists])
	# sanity check
	assert len(weights) == len(X), "weights should have same length as coords."

	# initialize two the a and b for y = ax+b and find result
	beta_init = np.random.rand(2)
	result = minimize(objective_function_linear, beta_init, args=(X,Y,weights), method='Nelder-Mead')
	beta_hat = result.x
	#print("[MLS]: Finding the linear regression is successful: ", result.success, " and a and b is: ", beta_hat)


	'''
	Transform the points set based on the local regression line being the x axis
	'''
	target_p = np.array(target_p)
	# randomly pick a point vector that is (1, a+b), where a and b are the obtained linear function
	tangent_vector = np.array([1, beta_hat[0]+beta_hat[1]])
	#print("[MLS]: Tangent vector for target p is: ", tangent_vector)
	#transform the entire point sets
	transformed_neighbor_points = transform_coordinate_system(neighbor_points, target_p, tangent_vector)


	'''
	Find the Local Regression Curve in the new coordinate system
	'''
	X_ = np.array([x for (x,y) in transformed_neighbor_points])
	Y_ = np.array([y for (x,y) in transformed_neighbor_points])

	#initialize three the a, b and c for y = ax^2 + bx + c
	beta_init_ = np.random.rand(3)
	result_ = minimize(objective_function_square, beta_init_, args=(X_,Y_,weights), method='Nelder-Mead')
	beta_hat_ = result_.x
	#print("[MLS]: Finding the quadratic regression is successful: ", result_.success, " and a, b, and c is: ", beta_hat_)


	'''
	Obtain the target coordinates and transformed it back to the orginal coordinate system
	'''
	c_value = beta_hat_[2]
	raw_target_coord = np.array([0,c_value])
	final_target_coord = transform_point_inverse(raw_target_coord, target_p, tangent_vector)

	return final_target_coord



