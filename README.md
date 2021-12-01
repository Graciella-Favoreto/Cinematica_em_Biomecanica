# Cinematica_em_Biomecanica

Dificuldades e limitações

-> mais de uma pessoa no video

-> confusão do cotovelo/partes do rosto como dedos

-> Exige bastante processamento, o tempo médio de processamento por frame aqui foi em torno de 7s

Utilização de rede neural pré treinada para fazer a detecção dos pontos chaves da mão, modelo Coco Cafee. Este modelo possui 22 pontos-chave sendo os primeiros 21 localizados na mão o 22 representando o fundo da imagem

A saída, logo, possui 22 matrizes sendo que cada uma irá representar a probabilidade (mapa de confiança) de cada ponto chave.

![pontos_chaves_mão](https://user-images.githubusercontent.com/68204317/144255845-25ca9700-2441-440b-9caa-967adc8a32c8.PNG)

As letras H, J, K, X, Y e Z exigem movimentação, necessário fazer comparação de movimento de transição entre os pontos



https://user-images.githubusercontent.com/68204317/144264747-94be5b5d-be62-472e-9a83-a4bcb2847091.mp4




