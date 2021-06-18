from cv2 import cv2

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def generate_dataset(frame,id,img_id):
    cv2.imwrite("data_set/user."+str(id)+"."+str(img_id)+".jpg",frame)



def draw_boundary(frame,face_cascade,scaleFactors,minNeighbours,text, clf,number):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactors,minNeighbours)
    coords=[]
    for x,y,w,h in faces:
        if number==1:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
            id,confidence =clf.predict(gray[y:y+h,x:x+w])
            if confidence<60:
                probability=100-confidence
                if id==1:
                    cv2.putText(frame,"Anju-1's accuracy ="+str(probability),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
                if id==2:
                    cv2.putText(frame,"Kritika-2's accuracy ="+str(probability),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
                if id==3:
                    cv2.putText(frame,"Nikunj-3's accuracy ="+str(probability),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
                if id==4:
                    cv2.putText(frame,"Pooja-4's accuracy ="+str(probability),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
                if id==5:
                    cv2.putText(frame,"Ritika-5's accuracy ="+str(probability),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
            else:
                cv2.putText(frame,"Unknown Fellow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
                  
            coords=[x,y,w,h]

    return coords, frame

number=0
def recognize(frame,clf,face_cascade):
    number=1
    coords=draw_boundary(frame,face_cascade,1.3,4,"Face",clf,number)
    return frame


def detect(frame,face_cascade,img_id):
    number=2
    coords,frame=draw_boundary(frame,face_cascade,1.3,4,"Face",number)

    if len(coords)==4:
        roi_img=frame[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        user_id=1
        generate_dataset(roi_img,user_id,img_id)
    return frame


img_id=0
clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("trainerr.yml")

while True:
    ret ,frame=cap.read()
    #frame = detect(frame,face_cascade,img_id)
    frame=recognize(frame,clf,face_cascade)
    cv2.imshow('pic',frame)
    img_id+=1
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
