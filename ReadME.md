# Cosine Similarity

1. Prepare pre-trained model
2. Preprocess your employees image(source imgage)
3. Create Cosine Similarity function
4. Use cv2 detect face(test image)
5. Compare source image and test image
6. Result
7. Conclusion

## 1.Prepare pre-trained model
* You can choose Vggface or other pre-trained model, in this case I use vggface.

## 2.Preprocess your employees image
* Collect your employees face image.
* Define function to preprocess your image. Resize image to (224,224), because vggface input image size use (224, 224).

## 3.Create Cosine Similarity function
* Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them.
* The cosine similarity function we can use scikit-learn.
	
		from sklearn.metrics.pairwise import cosine_similarity

* Or you can define cosine function by yourself.

![cos function](https://github.com/chunyuan0221/Real-time-face-detection/blob/master/Cosine%20Similarity/cos.png)
	
		def findCosineSimilarity(source_representation, test_representation):
			a = np.matmul(np.transpose(source_representation), test_representation)
    		b = np.sum(np.multiply(source_representation, source_representation))
    		c = np.sum(np.multiply(test_representation, test_representation))
    		return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

## 4.Use cv2 detect face(test image)
* Use cv2.CascadeClassifier to detect the face image.

		face_cascade = cv2.CascadeClassifier('C:/Anaconda/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
		faces = face_cascade.detectMultiScale(img, 1.3, 5)

## 5.Compare source image and test image
* Use CosineSimilarity function compare source images and test image.
* Given similarity threshold = 0.6, if the cosineSimilarity > 0.6 we determine that the test image is the same person as the source image.

## 6.Result
* Place the image we want to recognize in the source image folder.
* Real time face-detection result:

![cos function](https://github.com/chunyuan0221/Real-time-face-detection/blob/master/Cosine%20Similarity/result/result.png)

## 7.Conclusion
* Although the face recognition is done, sometimes the effect of face recognition by opencv is not very good.
* Maybe we can use YOLO to train a model that detects faces, and then use cosine similarity analysis.
* In this case, it's not suitable for smart door locks. Because I put mine photo in front of the camera and the recognition result is still successful.
* The next reserch will use 3D image detection.
