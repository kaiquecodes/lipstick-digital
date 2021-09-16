# Lipstick Digital

## Descrição

O provador de cores de batons, Lipstick Digital é um projeto desenvolvido para a disciplina de Processamento Digital de Imagens da Universidade Federal do Rio Grande do Norte(UFRN), foi baseado em um vídeo do canal <a href="https://www.youtube.com/watch?v=V2gmgkSqyi8&t=3s" target="_blank">Murtaza's Workshop</a>. 

## Introdução

A proposta do projeto é simular a cor de um batom através de um arquivo de imagem ou em tempo real com auxílio de uma webcam. É utilizado uma Machine learning treinada que detecta o rosto na imagem e atribui pontos de coordenadas para cada parte dele. Além disso é possível selecionar um efeito em escala de cinza na imagem exceto na região dos lábios.

## Manual do Usuário

Ao executar o programa, o usuário será solicitado a inserir uma imagem que será logo exibida. Depois isso, ele deverá digitar ctrl + p para abrir o menu de opções. Entre as opções estão as denominadas cores: Aphrodite, Athena e Hera, o efeito Gray Effect, a opção com cores, o botão para capturar a imagem da câmera, a opção de salvar a imagem, o botão de cancelar a cor do batom e a opção sair.

## Código e Funcionamento
~~~py
import cv2
import dlib
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

root = Tk()
root.withdraw()

webcam = False
Egray = False
set = False

path = askopenfilename() 

color = 0,0,0

def aphrodite_color(*args):
    global color 
    color =153,0,157
    return color

def athena_color(*args):
    global color 
    color = 255,20,80
    return color

def hera_color(*args):
    global color 
    color = 0,0,150
    return color   

def gray_color(*args):
    global set 
    set = True 

def colors(*args):
    global set 
    set = False    

def cam(*args):
    global webcam 
    webcam = True   

def save_color(*args):
    cv2.imwrite('result.png',ImgColor)
    return color    

def cancel_color(*args):
    global color 
    color = 0,0,0
    return color

def button_exit(*args):
    exit()

cv2.namedWindow("Lipstick Digital")
cv2.createButton("Aphrodite",aphrodite_color)
cv2.createButton("Athena",athena_color)
cv2.createButton("Hera",hera_color)
cv2.createButton("Gray Effect",gray_color)
cv2.createButton("Colors",colors)
cv2.createButton("Câmera",cam)
cv2.createButton("Save Image",save_color)
cv2.createButton("Cancel",cancel_color)
cv2.createButton("Exit",button_exit)

cap = cv2.VideoCapture(0)

def gray(img,set=False):
        if set:
            ImgOG = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ImgOG = cv2.cvtColor(ImgOG,cv2.COLOR_GRAY2BGR)
            img = cv2.addWeighted(ImgOG,1,ImgColor,0.4,0)
            return img 
        else:     
            img = cv2.addWeighted(ImgO,1,ImgColor,0.4,0)
            return img

def createBox(img,points,scale=5,masked=False,cropped = True): 
    if masked: 
     mask = np.zeros_like(img) 
     mask = cv2.fillPoly(mask,[points],(255,255,255)) 
     img = cv2.bitwise_and(img,mask)
     
    if cropped: 
     bbox = cv2.boundingRect(points)
     x,y,w,h = bbox 
     imgCrop = img[y:y+h,x:x+w] 
     imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
     return imgCrop
    else:
     return mask 

while True:  
       
    if webcam: success ,img = cap.read()
    else: img = cv2.imread(path) 
       
    img = cv2.resize(img,(0,0),None,0.5,0.5) 
    ImgO = img.copy()

    detector = dlib.get_frontal_face_detector() 

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray) 

    for face in faces:
            x1 , y1 = face.left(), face.top()
            x2 , y2 = face.right(), face.bottom()      
            landmarks = predictor (imgGray,face) 
            myPoints = [] 
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                myPoints.append([x,y])
                
            myPoints = np.array(myPoints) 
            imgLips = createBox(img,myPoints[48:61],3,True,False)   
                        
            ImgColor = np.zeros_like(imgLips) 
            ImgColor [:] = color
        
            ImgColor = cv2.bitwise_and(imgLips,ImgColor) 
            
            ImgColor = cv2.GaussianBlur(ImgColor,(7,7),10) 
            
            ImgColor = gray(ImgO,set) 

            cv2.imshow('Lipstick Digital',ImgColor)

    cv2.waitKey(30)
~~~
    


               

    


