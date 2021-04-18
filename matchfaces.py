import face_recognition as fr
import os

image_of_jashan = fr.load_image_file('D:/new prjt on face recognition/IMAGES/IMG_0081.JPG')
jashan_face_encoding = fr.face_encodings(image_of_jashan)[0]

unknown_image = fr.load_image_file('D:/new prjt on face recognition/ENEK1885.JPG')
unkown_encoding= fr.face_encodings(unknown_image)[0]

resuts = fr.compare_faces([jashan_face_encoding],unkown_encoding)
if resuts[0]== True:
    print('THIS IS JASHAN')

else:
    print('THIS IS NOT JASHAN')