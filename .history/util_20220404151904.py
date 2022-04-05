
def generateFixedActions():
    actions = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'LEFT', 'LEFT','LEFT','LEFT', 'UP','UP','UP','UP']
    #actions = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'LEFT', 'UP','UP','UP','UP','UP']
    #actions = ['RIGHT']*63
    return actions

def distance(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)    