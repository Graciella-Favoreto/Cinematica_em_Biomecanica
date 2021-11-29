# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:41:52 2021

@author: graciellafavoreto
"""

import time
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#importando pastas
pose_path = "C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\pose"
imagens_path = "C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\imagens"
modulos_path = "C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\modulos"
modulos_path

sys.path.append("C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\modulos\\modulos")
sys.path

import extrator_POSICAO as posicao
import extrator_ALTURA as altura
import extrator_PROXIMIDADE as proximidade
import alfabeto


#import extrator_POSICAO as posicao 
#posicao = "C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\modulos\\modulos\\extrator_POSICAO.py"
#altura = "C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\modulos\\modulos\\extrator_ALTURA.py"
#proximidade = "C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\modulos\\modulos\\extrator_PROXIMIDADE.py"
#alfabeto = "C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\modulos\\modulos\\alfabeto.py"

#Carregando o modelo e estruturas da rede neural pré-treinada
arquivo_proto = "C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\pose\\pose\\hand\\pose_deploy.prototxt"
arquivo_pesos = "C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\pose\\pose\\hand\\pose_iter_102000.caffemodel"
numero_pontos = 22
pares_pose = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], 
              [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], 
              [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 
          'Q', 'R', 'S', 'T', 'U', 'V', 'W']

#ler modelo que já foi carregado
modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)

cor_pontoA, cor_pontoB, cor_linha = (14, 201, 255), (255, 0, 128), (192, 192, 192)
cor_txtponto, cor_txtinicial, cor_txtandamento = (10, 216, 245), (255, 0, 128), (192, 192, 192)

tamanho_fonte, tamanho_linha, tamanho_circulo, espessura = 1, 1, 4, 2
fonte = cv2.FONT_HERSHEY_SIMPLEX

#carregando um video
video = "C:\\Users\\graciellafavoreto\\Desktop\\Reconhecimento de Gestos\\imagens\\imagens\\hand\\letraA.mp4"
captura = cv2.VideoCapture(video)
ret, frame = captura.read()
ret

imagem_largura = frame.shape[1]
imagem_altura = frame.shape[0]
proporcao = imagem_largura / imagem_altura

#print (imagem_largura, imagem_altura, proporcao)

#Definir as dimensões da imagem de entrada.
entrada_altura = 368
entrada_largura = int(((proporcao * entrada_altura) * 8) // 8)
#print (entrada_largura, entrada_altura)

#Criando a variável para salvar os resultados no Drive
resultado = './libras.avi'
gravar_video = cv2.VideoWriter(resultado, cv2.VideoWriter_fourcc(*'XVID'), 10,
                              (frame.shape[1], frame.shape[0]))

#Lendo o modelo carregado na linha 25 adiante
modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)

#Exibindo as saídas
limite = 0.1
while (cv2.waitKey(1) < 0):
    t = time.time()
    conectado, frame = captura.read()
    frame_copia = np.copy(frame)

    tamanho = cv2.resize(frame, (imagem_largura, imagem_altura))
    mapa_suave = cv2.GaussianBlur(tamanho, (3, 3), 0, 0)
    fundo = np.uint8(mapa_suave > limite)

    if not conectado:
        cv2.waitKey()
        break

    entrada_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, 
                                         (entrada_largura, entrada_altura),
                                    (0, 0, 0), swapRB=False, crop=False)

    modelo.setInput(entrada_blob)

    saida = modelo.forward()

    pontos = []

    for i in range(numero_pontos):
        mapa_confianca = saida[0, i, :, :]
        mapa_confianca = cv2.resize(mapa_confianca, (imagem_largura, imagem_altura))

        _, confianca, _, point = cv2.minMaxLoc(mapa_confianca)

        if confianca > limite:
            cv2.circle(frame_copia, (int(point[0]), int(point[1])), 
                       tamanho_circulo, cor_pontoA, thickness=espessura,
                       lineType=cv2.FILLED)
            cv2.putText(frame_copia, "{}".format(i), (int(point[0]), int(point[1])),
                        fonte, .8,
                        cor_txtponto, 2, lineType=cv2.LINE_AA)

            pontos.append((int(point[0]), int(point[1])))
        else:
            pontos.append((0, 0))

    for par in pares_pose:
        parteA = par[0]
        parteB = par[1]

        if pontos[parteA] != (0, 0) and pontos[parteB] != (0, 0):
            cv2.line(frame, pontos[parteA], pontos[parteB], cor_linha, 
                     tamanho_linha, lineType=cv2.LINE_AA)
            cv2.circle(frame, pontos[parteA], tamanho_circulo, cor_pontoA,
                       thickness=espessura, lineType=cv2.FILLED)
            cv2.circle(frame, pontos[parteB], tamanho_circulo, cor_pontoB,
                       thickness=espessura, lineType=cv2.FILLED)

            cv2.line(fundo, pontos[parteA], pontos[parteB], cor_linha, 
                     tamanho_linha, lineType=cv2.LINE_AA)
            cv2.circle(fundo, pontos[parteA], tamanho_circulo, cor_pontoA, 
                       thickness=espessura, lineType=cv2.FILLED)
            cv2.circle(fundo, pontos[parteB], tamanho_circulo, cor_pontoB, 
                       thickness=espessura, lineType=cv2.FILLED)

    posicao.posicoes = []

    # dedo polegar
    posicao.verificar_posicao_DEDOS(pontos[1:5], 'polegar', altura.verificar_altura_MAO(pontos))

    # dedo indicador
    posicao.verificar_posicao_DEDOS(pontos[5:9], 'indicador', altura.verificar_altura_MAO(pontos))

    # dedo médio
    posicao.verificar_posicao_DEDOS(pontos[9:13], 'medio', altura.verificar_altura_MAO(pontos))

    # dedo anelar
    posicao.verificar_posicao_DEDOS(pontos[13:17], 'anelar', altura.verificar_altura_MAO(pontos))

    # dedo mínimo
    posicao.verificar_posicao_DEDOS(pontos[17:21], 'minimo', altura.verificar_altura_MAO(pontos))

    for i, a in enumerate(alfabeto.letras):
        if proximidade.verificar_proximidade_DEDOS(pontos) == alfabeto.letras[i]:
            cv2.putText(frame, 'Letra: ' + letras[i], (50, 50), fonte, 1, 
                        cor_txtinicial, tamanho_fonte,
                        lineType=cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Analisando', (250, 50), fonte, 1, 
                        cor_txtandamento, tamanho_fonte,
                        lineType=cv2.LINE_AA)
    
    
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    cv2.imshow("preview", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    print("Tempo total = {:.2f}seg".format(time.time() - t))
    gravar_video.write(frame)
gravar_video.release()
