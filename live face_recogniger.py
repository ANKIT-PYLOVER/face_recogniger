import cv2
import face_recognition as fr
import numpy as np
import os

my_dir = "D:/face_recogniger/faces/"
known_encodings=[]
images = []
faces_names=[]

for cl in os.listdir(my_dir):
    curimage = cv2.imread(f'{my_dir}/{cl}')
    images.append(curimage)
    faces_names.append(os.path.splitext(cl)[0])
print(faces_names)
def face_en(images):
    encoded = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encoding = fr.face_encodings(img)[0]
        encoded.append(encoding)
    return encoded
known_encodings = face_en(images)
print(len(known_encodings))

video_cap = cv2.VideoCapture(0)
print("streaming....")

while True:
    ret , frame = video_cap.read()
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces_locations = fr.face_locations(rgb_frame)
    current_encodings = fr.face_encodings(rgb_frame,faces_locations)
    for face_encoding,faceloc in zip(current_encodings,faces_locations):
        matches = fr.compare_faces(known_encodings, face_encoding)
        name = "unknown"
        
        facee_distances = fr.face_distance(known_encodings,face_encoding)
        best_match_index = np.argmin(facee_distances)
        
        if matches[best_match_index]:
            name = faces_names[best_match_index].upper()
        top,right,bottom,left = faceloc       
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame,name,(left +6,bottom-6),font,1.0,(255,255,255),1)
        if matches == True:
            os.startfile()
            
    cv2.imshow('Recogniger',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
        
    