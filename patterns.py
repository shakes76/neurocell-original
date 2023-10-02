'''
Different patterns of cells known
''' 

def insertBlinker(grid, index=(0,0), value=1):
    '''
    Insert a blinker oscillator construct at the index position
    '''
    grid[index[0], index[1]+1] = value
    grid[index[0]+1, index[1]+1] = value
    grid[index[0]+2, index[1]+1] = value

    return grid
    
def insertGlider(grid, index=(0,0), value=1):
    '''
    Insert a glider construct at the index position
    '''
    grid[index[0], index[1]+1] = value
    grid[index[0]+1, index[1]+2] = value
    grid[index[0]+2, index[1]] = value
    grid[index[0]+2, index[1]+1] = value
    grid[index[0]+2, index[1]+2] = value

    return grid
    
def insertGliderGun(grid, iindex=(0,0), value=1):
    '''
    Insert a glider construct at the index position
    '''
    grid[index[0]+1, index[1]+25] = value
    
    grid[index[0]+2, index[1]+23] = value
    grid[index[0]+2, index[1]+25] = value
    
    grid[index[0]+3, index[1]+13] = value
    grid[index[0]+3, index[1]+14] = value
    grid[index[0]+3, index[1]+21] = value
    grid[index[0]+3, index[1]+22] = value
    grid[index[0]+3, index[1]+35] = value
    grid[index[0]+3, index[1]+36] = value
    
    grid[index[0]+4, index[1]+12] = value
    grid[index[0]+4, index[1]+16] = value
    grid[index[0]+4, index[1]+21] = value
    grid[index[0]+4, index[1]+22] = value
    grid[index[0]+4, index[1]+35] = value
    grid[index[0]+4, index[1]+36] = value
    
    grid[index[0]+5, index[1]+1] = value
    grid[index[0]+5, index[1]+2] = value
    grid[index[0]+5, index[1]+11] = value
    grid[index[0]+5, index[1]+17] = value
    grid[index[0]+5, index[1]+21] = value
    grid[index[0]+5, index[1]+22] = value
    
    grid[index[0]+6, index[1]+1] = value
    grid[index[0]+6, index[1]+2] = value
    grid[index[0]+6, index[1]+11] = value
    grid[index[0]+6, index[1]+15] = value
    grid[index[0]+6, index[1]+17] = value
    grid[index[0]+6, index[1]+18] = value
    grid[index[0]+6, index[1]+23] = value
    grid[index[0]+6, index[1]+25] = value
    
    grid[index[0]+7, index[1]+11] = value
    grid[index[0]+7, index[1]+17] = value
    grid[index[0]+7, index[1]+25] = value
    
    grid[index[0]+8, index[1]+12] = value
    grid[index[0]+8, index[1]+16] = value
    
    grid[index[0]+9, index[1]+13] = value
    grid[index[0]+9, index[1]+14] = value

    return grid

