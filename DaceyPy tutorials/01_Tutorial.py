
from daceypy import DA

DA.init(20,1)   # Initialise the DA to the 20th order with one variable

x = DA(1)           #set the first DA number variable as x

y = x.sin()        #Completed via approximating elementary functions - compute DA sin of x.

#print x and y to screen

print('x\n' + str(x) + "\n")
print('y = sin(x) \n' + str(y) + "\n")






