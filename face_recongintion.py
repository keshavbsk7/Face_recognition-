import face_recognition as fr
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_c=cv2.VideoCapture(0)

rohit_img=fr.load_image_file("pic/rohit.jpg")
rohit_enc=fr.face_encodings(rohit_img)[0]

virat_img=fr.load_image_file("pic/virat.jpg")
virat_enc=fr.face_encodings(virat_img)[0]

dhoni_img=fr.load_image_file("pic/dhoni.jpg")
dhoni_enc=fr.face_encodings(dhoni_img)[0]

keshav_img=fr.load_image_file("pic/keshav.jpg")
keshav_enc=fr.face_encodings(keshav_img)[0]

anu_img=fr.load_image_file("pic/anu.jpeg")
anu_enc=fr.face_encodings(anu_img)[0]

gliffy_img=fr.load_image_file("pic/gliffy.jpg")
gliffy_enc=fr.face_encodings(gliffy_img)[0]



know_encoding=[rohit_enc,virat_enc,dhoni_enc,keshav_enc,anu_enc,gliffy_enc]
know_img=["rohit","virat","dhoni","keshav","Anu","sindhu","gliffy"]

student=know_img.copy()

face_location=[]
face_encoding=[]
face_name=[]
s=True

now=datetime.now()
current=now.strftime("%Y-%m-%d")

f=open(current+'.csv','w+',newline='')
lnwritter=csv.writer(f)

while True:
    _,frame=video_c.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
        face_locations=fr.face_locations(rgb_small_frame)
        face_encodings=fr.face_encodings(rgb_small_frame,face_locations)
        face_name=[]
        for i in face_encodings:
            matches=fr.compare_faces(know_encoding,i)
            name=''
            face_dist=fr.face_distance(know_encoding,i)
            best_index=np.argmin(face_dist)
            if matches[best_index]:
                name=know_img[best_index]
            face_name.append(name)
            if name in know_img:
                if name in student:
                    student.remove(name)
                    print(student)
                    current_time=now.strftime("%H-%M-%S")
                    lnwritter.writerow([name,current_time])
    cv2.imshow("attendace",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_c.release()
cv2.destroyAllWindows()
f.close()
