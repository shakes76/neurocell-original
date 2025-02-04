'''
Tests for working with binary
'''
b1='100010' #Decimal value: 34
b2='101001' #Decimal value: 41

res = bin(int(b1,2) + int(b2,2)) #Passing the base value of 2 for binary to the int() function
res_int = int(res,2)
print(res, res_int)

#If the user wants to get rid of the suffix '0b' at the start
print(res[2:])

#to list
print(list(res[2:]))
