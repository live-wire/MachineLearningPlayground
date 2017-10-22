from PIL import Image
import numpy as np

#Always import matplotlib like this bruh
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt



i = Image.open('images/numbers/0.4.png')
iar = np.asarray(i)
i2 = Image.open('images/numbers/1.4.png')
iar2 = np.asarray(i2)
i3 = Image.open('images/numbers/y0.3.png')
iar3 = np.asarray(i3)
i4 = Image.open('images/sentdex.png')
iar4 = np.asarray(i4)





def threshold(imageArray):
    balanceAr = []
    newArr = imageArray.copy()

    for eachRow in imageArray:
        for eachPix in eachRow:
            avgNum = reduce(lambda x, y: int(x)+int(y), eachPix[:3])/3
            balanceAr.append(avgNum)

    balance = reduce(lambda x,y: x + y, balanceAr)/ len(balanceAr)

    for eachRow in newArr:
        for eachPix in eachRow:
            if reduce(lambda x,y:int(x)+int(y),eachPix[:3])/len(eachPix[:3]) > balance:
                eachPix[0] = 255
                eachPix[1] = 255
                eachPix[2] = 255
                eachPix[3] = 255
            else:
                eachPix[0] = 0
                eachPix[1] = 0
                eachPix[2] = 0
                eachPix[3] = 255
    return newArr


threshold(iar)
threshold(iar2)
threshold(iar3)
iar4=threshold(iar4)


fig = plt.figure()
ax1 = plt.subplot2grid((8,6),(0,0),rowspan=4, colspan=3)
ax2 = plt.subplot2grid((8,6),(0,3),rowspan=4, colspan=3)
ax3 = plt.subplot2grid((8,6),(4,0),rowspan=4, colspan=3)
ax4 = plt.subplot2grid((8,6),(4,3),rowspan=4, colspan=3)

ax1.imshow(iar)
ax2.imshow(iar2)
ax3.imshow(iar3)
ax4.imshow(iar4)
plt.show()