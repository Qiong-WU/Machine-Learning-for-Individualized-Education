import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.cluster import KMeans
imgMatrix = np.zeros((256*512,21))
for i in range(21):
    t = misc.imread('/Users/wuqiong/Desktop/new/'+str(i+1)+'.jpg')
    t = np.reshape(t,(256*512,))
    imgMatrix[:,i] = t

imgMatrix = np.transpose(imgMatrix)
kmeans = KMeans(n_clusters = 2).fit(imgMatrix)


kmeans.labels_
print kmeans.labels_
classes = kmeans.cluster_centers_
#quality_val=100
class1 = classes[0,:]
class2 = classes[1,:]

class1 = np.reshape(class1,(256,512))
class2 = np.reshape(class2,(256,512))
#class1.save('/Users/wuqiong/Desktop/new/class1.jpg', 'JPEG', quality=quality_val)
fig = plt.figure()
a = fig.add_subplot(2,1,1)
plt.imshow(class1,cmap = 'gray')
a.set_title('Class1')
a = fig.add_subplot(2,1,2)
plt.imshow(class2,cmap = 'gray')
a.set_title('Class2')
plt.show()