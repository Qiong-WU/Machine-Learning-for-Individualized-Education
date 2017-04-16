import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import lda
import cv2
import os,os.path
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from skimage.transform import AffineTransform
from skimage.transform import warp

ProjectFolder = "G:\\vs2010\\59NetWork\\identity_state_project\\"
dataFolder = "data\\"
eyeDataFolder = "eye_data\\"
mouthDataFolder = "mouth_data\\"
reference_number = 10
ref_eye_img = misc.imread(ProjectFolder+dataFolder+eyeDataFolder+str(reference_number)+'.bmp')
ref_mouth_img = misc.imread(ProjectFolder+dataFolder+mouthDataFolder+str(reference_number)+'.bmp')

data_number = len(os.listdir(ProjectFolder+dataFolder+eyeDataFolder))
feature_amount = 5
feature_dimension = 2
eye_featureMatrix = np.zeros(shape=(data_number,feature_amount*feature_dimension))
mouth_featureMatrix = np.zeros(shape=(data_number,feature_amount*feature_dimension))
for i in xrange(data_number):
	eye_image = misc.imread(ProjectFolder+dataFolder+eyeDataFolder+str(i)+'.bmp')
	mouth_image = misc.imread(ProjectFolder+dataFolder+mouthDataFolder+str(i)+'.bmp')
	
	# registration
	shift_eye, error_eye, diffphase_eye = register_translation(ref_eye_img,eye_image)
	shift_mouth,error_mouth,diffphase_mouth = register_translation(ref_mouth_img,mouth_image)
	shift_eye = [-shift_eye[1],-shift_eye[0]]
	shift_mouth = [-shift_mouth[1],-shift_mouth[0]]
	trans_eye = AffineTransform(translation=shift_mouth)
	trans_mouth = AffineTransform(translation=shift_mouth)
	trans_eye_img = warp(eye_image,trans_eye,mode='edge')
	trans_eye_img = trans_eye_img.astype('float32')
	trans_mouth_img = warp(mouth_image,trans_mouth,mode='edge')
	trans_mouth_img = trans_mouth_img.astype('float32')
	eye_corners = cv2.goodFeaturesToTrack(trans_eye_img,feature_amount,0.001,2)
	mouth_corners = cv2.goodFeaturesToTrack(trans_mouth_img,feature_amount,0.001,2)
	for j,content in enumerate(zip(eye_corners,mouth_corners)):
		eyes = content[0]
		mouths = content[1]
		eyes = eyes.flatten()
		mouths = mouths.flatten()
		for k in xrange(feature_dimension):
			eye_featureMatrix[i][j*feature_dimension:j*feature_dimension+k] = eyes.ravel()[k]
			mouth_featureMatrix[i][j*feature_dimension:j*feature_dimension+k] = mouths.ravel()[k]

eye_featureMatrix = eye_featureMatrix.astype(int)
model = lda.LDA(n_topics=2,n_iter=1500,random_state=1)
model.fit(eye_featureMatrix)
eye_probability = model.doc_topic_

mouth_featureMatrix = mouth_featureMatrix.astype(int)
model = lda.LDA(n_topics=2,n_iter=1500,random_state=1)
model.fit(mouth_featureMatrix)
mouth_probability = model.doc_topic_

#HMM session
observed = np.concatenate((eye_probability,mouth_probability),axis=1)
from hmmlearn.hmm import GaussianHMM
# specific the amount of states for HMM
HMM_components = 2  # sleep or not
model = GaussianHMM(n_components=HMM_components, covariance_type="diag", n_iter=1000).fit(observed)
hidden_states = model.predict(observed)


