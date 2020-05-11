import numpy as np
import time
import random

# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.
###################################################################
# Example: Create a zeros vector of size 10 and store variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [0 for i in range(10)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros(10)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


z_1 = None
z_2 = None
################################################################
# 1. Create a zeros array of size (3,5) and store in variable z.
# Python
pythonStartTime = time.time()
z_1 = [[0] * 5 for i in range(3)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
z_2 = np.zeros((3,5), dtype='int')
numPyEndTime = time.time()
print('Question 1: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#################################################
# 2. Set all the elements in first row of z to 7.
# Python
pythonStartTime = time.time()
z_1[0] = [7] * 5
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
z_2[0] = np.full(5, 7)
numPyEndTime = time.time()
print('Question 2: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#####################################################
# 3. Set all the elements in second column of z to 9.
# Python
pythonStartTime = time.time()
for x in range(3):
    z_1[x][1] = 9
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
z_2[:, 1] = 9
numPyEndTime = time.time()
print('Question 3: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

#############################################################
# 4. Set the element at (second row, third column) of z to 5.
# Python
pythonStartTime = time.time()
z_1[1][2] = 5
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
z_2[1, 2] = 5
numPyEndTime = time.time()
print('Question 4: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
# Python
pythonStartTime = time.time()
x_1 = [i+50 for i in range(50)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
x_2 = np.arange(50) + 50
numPyEndTime = time.time()
print('Question 5: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
# Python
pythonStartTime = time.time()
y_1 = [[i+4*j for i in range(4)] for j in range(4)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
y_2 = np.arange(16).reshape((4,4))
numPyEndTime = time.time()
print('Question 6: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


##############
print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [[0]*5 for i in range(5)]
tmp_1[0] = [1] * 5
tmp_1[4] = [1] * 5
for i in range(5):
    tmp_1[i][0] = 1
    tmp_1[i][4] = 1 
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros((5,5), dtype='int')
tmp_2[0,:] = 1
tmp_2[4,:] = 1 
tmp_2[:,0] = 1 
tmp_2[:,4] = 1
numPyEndTime = time.time()
print('Question 7: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


##############
print(tmp_1)
print(tmp_2)
##############


a_1 = None; a_2 = None
b_1 = None; b_2 = None
c_1 = None; c_2 = None
#############################################################################################
# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
# Python
pythonStartTime = time.time()
a_1 = [[i+100*j for i in range(100)] for j in range(50)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
a_2 = np.arange(5000).reshape((50,100))
numPyEndTime = time.time()
print('Question 8: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))

# ?? between 0 to 5000 are 5001 variables, should this be 0 to 4999 ??


###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
pythonStartTime = time.time()
b_1 = [[i+200*j for i in range(200)] for j in range(100)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
b_2 = np.arange(20000).reshape((100,200))
numPyEndTime = time.time()
print('Question 9: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.

# Python
pythonStartTime = time.time()
c_1 = [[0] * 200 for i in range(50)]
for i in range(50):
    for j in range(200):
        for k in range(100):
            c_1[i][j] += a_1[i][k] * b_1[k][j]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
c_2 = np.dot(a_2, b_2)
numPyEndTime = time.time()
print('Question 10: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
print(c_1[0][1])
print(c_2[0,1])
print(c_2.shape)


d_1 = None; d_2 = None
################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
# Python
pythonStartTime = time.time()
d_1 = [[random.random() for i in range(3)] for j in range(3)]
min_1 = 100
max_1 = -1
for i in range(3):
    for j in range(3):
        min_1 = min(min_1, d_1[i][j])
        max_1 = max(max_1, d_1[i][j])
for i in range(3):
    for j in range(3):
        d_1[i][j] = (d_1[i][j] - min_1) / (max_1 - min_1)
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
d_2 = np.random.random((3,3))
min_2 = d_2.min()
max_2 = d_2.max()
d_2 = (d_2 - min_2) / (max_2 - min_2)
numPyEndTime = time.time()
print('Question 11: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


##########
print(d_1)
print(d_2)
#########


################################################
# 12. Subtract the mean of each row of matrix a.
# Python
pythonStartTime = time.time()
avg_firstLine = 0
for i in range(100):
    avg_firstLine += a_1[0][i]
avg_firstLine = avg_firstLine / 100
for i in range(50):
    for j in range(100):
        a_1[i][j] = a_1[i][j] - (avg_firstLine + 100*i)
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
a_2 = a_2 - a_2.mean(axis = 1, keepdims = True)
numPyEndTime = time.time()
print('Question 12: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


###################################################
# 13. Subtract the mean of each column of matrix b.
# Python
pythonStartTime = time.time()
avg_firstRow = 0
for i in range(100):
    avg_firstRow += b_1[i][0]
avg_firstRow = avg_firstRow / 100

for j in range(200):
    for i in range(100):
        b_1[i][j] = b_1[i][j] - (avg_firstRow + 1*j)

pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
b_2 = b_2 - b_2.mean(axis = 0, keepdims = True)
numPyEndTime = time.time()
print('Question 13: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
# Python
pythonStartTime = time.time()
e_1 = [[0]*50 for i in range(200)]
for i in range(50):
    for j in range(200):
        e_1[j][i] = c_1[i][j] + 5
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
e_2 = c_2.transpose() + 5
numPyEndTime = time.time()
print('Question 14: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


##################
print (np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
# Python
pythonStartTime = time.time()
f_1 = [[0] for i in range(10000)]
for i in range(200):
    for j in range(50):
        f_1[50*i+j] = e_1[i][j]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
f_2 = e_2.reshape(1,-1)
numPyEndTime = time.time()
print('Question 15: ')
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))
print(f_2.shape)
print (np.sum(f_1 == f_2))