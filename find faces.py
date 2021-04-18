import os
import face_recognition as fr

images = fr.load_image_file('D:/AA/IMAGES/IMG_0081.JPG')
face_locations = fr.face_locations(images)
print(face_locations)

print (f'there are {len(face_locations)} peoples in this images')