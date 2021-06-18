# Face_Detection
Using Opencv and Haar Cascade algorithm to detect the face by comparing the value of haar faeatures.

# Dataset
Capturing the face(region of interest) and then using cv2.write() writing it in the dataset directory in  the 
format of "data_set/user<user_id>.<img_id>.jpg".The dataset contains 5000 images.


# Face-Recognition
Using the LBPH Recognizer to recognize the faces.If the face is recognized the function returns the user id and the confidence value.The
lower the confidenec,the better the match.A confidence value below 50-60 is considered a good confidence value for face determination.
Once the id is returned ,it is mapped to the repective user name and the username along with the accuracy value (100-confidence) is shown on the output screen.
