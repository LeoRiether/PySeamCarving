###
##
#
# Introdução ao Processamento de Imagens
# Leonardo Alves Riether - 190032413
# Trabalho Final - Seam Carving
#
##
###

import numpy as np
import cv2 as cv

imname = "img/onecat.jfif"
imid = imname.split('/')[-1].split('.')[0]
MAXWIDTH = 1000

def sobel(src):
    ddepth = cv.CV_16S
    scale = 1.0
    delta = 0
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def build_energy(src):
    h, w = src.shape
    res = np.zeros((h, w), dtype=np.int)

    # Última linha da matriz de programação dinâmica
    for j in range(w):
        res[h-1, j] = src[h-1, j]

    # Linhas de cima 
    for i in range(h-2, -1, -1):
        for j in range(w-1, -1, -1):
            res[i, j] = min(
                res[i+1, k]
                for k in range(j-1, j+2)
                if k >= 0 and k < w
            ) + src[i, j]

    return res

def get_seam(src, energy):
    h, w, d = src.shape
    seam = np.zeros(h, dtype=np.int) # seam[i] = coluna que será removida na linha i 

    _, seam[0] = min( (energy[0, j], j) for j in range(w) )
    for i in range (1, h):
        up_seam = seam[i-1]
        _, seam[i] = min( (energy[i, j], j) for j in range(up_seam-1, up_seam+2) if j >= 0 and j < w )

    return seam

def remove_seam(src, energy, seam):
    h, w, d = src.shape
    res = np.zeros((h, w-1, d))
    renergy = np.zeros((h, w-1))

    for i in range(h):
        s = seam[i]
        for j in range(w):
            if j < s:
                res[i, j] = src[i, j]
                renergy[i, j] = energy[i, j]
            elif j > s:
                res[i, j-1] = src[i, j]
                renergy[i, j-1] = energy[i, j]

    return res, renergy

def main():
    im = cv.imread(imname)
    h, w, d = im.shape
    if w > MAXWIDTH:
        scaling = MAXWIDTH / w
        im = cv.resize(im, None, fx=scaling, fy=scaling, interpolation=cv.INTER_CUBIC)

    sob = sobel(im)
    energy = build_energy(sob)

    cv.imwrite(f"out/{imid}.input.png", im)
    cv.imwrite(f"out/{imid}.sobel.png", sob)
    cv.imwrite(f"out/{imid}.energy.png", cv.normalize(energy, None, 0, 255, cv.NORM_MINMAX))

    for k in range(500):
        seam = get_seam(im, energy)
        im, energy = remove_seam(im, energy, seam)

    cv.imwrite(f"out/{imid}.scaled.png", im)

if __name__ == '__main__':
    main()
