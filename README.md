# Lipstick Digital

## Descrição

O Lipstick Digital é um projeto desenvolvido para a disciplina de Processamento Digital de Imagens da Universidade Federal do Rio Grande do Norte(UFRN), foi inspirado no canal <a href="https://www.youtube.com/watch?v=V2gmgkSqyi8&t=3s" target="_blank">Murtaza's Workshop</a>. 

## Introdução

A proposta do projeto é simular a cor de um batom através de um arquivo de imagem ou em tempo real com auxílio de uma webcam. É utilizado uma Machine learning treinada que detecta o rosto na imagem e atribui pontos de coordenadas para cada parte dele. Além disso, é possível selecionar um efeito em escala de cinza na imagem exceto na região dos lábios.

## Instalação

1. Instale o Python Versão 3.8.
3. Instale a biblioteca OpenCV.
3. Instale a biblioteca dlib para a Machine Learning.
4. Garanta que seu ambiente tenha as bibliotecas tkinter e numpy.
5. Baixe o arquivo: <a href="https://github.com/davisking/dlib-models" target="_blank">shape_predictor_68_face_landmarks.dat</a>.
6. Execute na sua IDE favorita.

## Manual do Usuário

Ao executar o programa, o usuário será solicitado a inserir uma imagem que será logo exibida. Depois isso, ele deverá digitar 
**CTRL + P** para abrir o menu de opções. Entre as opções estão as denominadas cores: Aphrodite, Athena e Hera, o efeito 
Gray Effect, a opção com cores, o botão para capturar a imagem da câmera, a opção de salvar a imagem, o botão de cancelar
a cor do batom e a opção sair.

![Interface Gráfica](/images/interface.png)
###### Figura 1: Interface Gráfica.

## Código 

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
                #cv2.circle(ImgO,(x,y),2,(0,255,0),cv2.FILLED)  
                              
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
 ## Funcionamento

A seguir teremos um breve explicação dos conceitos e ferramentas utilizadas no projeto.


 ~~~py
    detector = dlib.get_frontal_face_detector() 
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray) 
 ~~~
 Nas linhas de código acima, temos a parte responsável por instanciar os elementos da biblioteca. O método **detector** irá nos retornar os rostos encontrados na imagem e o **predictor** irá retornar os pontos de referência do rosto em questão com o arquivo **"shape_predictor_68_face_landmarks.dat"** teremos a rede neural treinada para detectar 68 pontos.
 
~~~py
for face in faces:
    landmarks = predictor (imgGray,face) 
    myPoints = [] 
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x,y])
        #cv2.circle(ImgO,(x,y),2,(0,255,0),cv2.FILLED)  
 ~~~
 Neste trecho de código, atribuímos os pontos do rosto na variável **landmarks**, logo em seguida colocamos todos eles em uma lista **myPoints**. Por fim, temos a opção de ativar a linha comentada para ver os pontos marcados na imagem com o  método **circle**.
 
 ~~~py
 
myPoints = np.array(myPoints) 
imgLips = createBox(img,myPoints[48:61],3,True,False)   

ImgColor = np.zeros_like(imgLips) 
ImgColor [:] = color

ImgColor = cv2.bitwise_and(imgLips,ImgColor) 

ImgColor = cv2.GaussianBlur(ImgColor,(7,7),10) 

ImgColor = gray(ImgO,set) 

cv2.imshow('Lipstick Digital',ImgColor)
            
~~~            
 
Nessa parte, temos o núcleo do sistema, transformamos em um **myPoints** em **np.array** para que seja posssível passar os pontos do rosto para a função **createBox** (será explicava) que retornará nossa região alvo que são os lábios. Com o retorno dela setado em **imgLips** podemos fazer uma operação **bitwise_and** para colorir nosso lábio com a cor que está em **ImgColor**.

>Observe que todas as imagens citadas até agora, contém as mesmas dimensões da imagem de entrada.
>Isto é fundamental para que nosso "merge" de imagens tenha sucesso.

Além disso, teremos que desfocar a imagem **ImgColor** que contém a forma do lábio já pintado. Para isso, utilizamos o método **GaussianBlur** do OpenCV que irá desfocar as bordas do nosso lábio tornando mais natural seu contorno.

Por fim, nosso ponto mais importante: devemos juntar a imagem que contém nosso lábio colorido com a imagem de entrada. Isso é possível com a função **gray()**, com ela fizemos uso do método **addWeighted** do OpenCV, para somar essas imagens com os respectivos pesos escolhidos. Também ela nós dá a oportunidade de setar o efeito Gray Effect que irá transforma a imagem em tons de cinza, exceto o lábio.

               
~~~py
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
~~~
A idéia da função **createBox** é capturar a região do lábio. Para isso, passamos nossa imagem de entrada e os pontos referentes a região do lábio. No corpo da função, criamos uma máscara que terá as mesmas dimenções da imagem de entrada e na cor preta, isso é necessário para desenharmos os pontos desejados na mesma posição da imagem de entrada. A função **fillPoly** do OpenCV irá desenhar o contorno do lábio de forma poligonal com o preenchimento na cor branca, de forma a torna a aproximação da região mais precisa. Com a opção **masked = True e crooped = false**, a função retorna a máscara com lábio branco e o fundo preto. 

>A linha que consta um novo **bitwise_and** dá a possibilidade de termos a região do lábio a cores como na imagem de entrada. Já opção com **cropped = True** >servirá para pŕóximas versões do projeto, com ela podemos fazer o recorte do lábio diretamente da imagem de entrada. Ambas linhas, serão aproveitadas em uma nova versão do programa que implementará mais requisitos
    


