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
from numba import jit
import argparse

MAXWIDTH = 1000

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Imagem de entrada")
    parser.add_argument("-m", "--mask", required=False, help="Mask para alteração de energia")
    parser.add_argument("-n", "--iterations", required=False, default=500, help="Número de seams a serem retirados")
    parser.add_argument("--show", required=False, default=False, nargs='?',
                        const=True, type=bool, help="Mostra as imagens com cv.imshow")
    parser.add_argument("-r", "--remove", required=False, default=False, nargs='?',
                        const=True, type=bool, help="Remove um objeto da imagem")
    return parser.parse_args()

# Calcula a função de energia e(I) = |dI/dx| + |dI/dy|
def image_energy(src):
    ddepth = cv.CV_16S
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dx = cv.Sobel(src, ddepth, 1, 0, ksize=3, borderType=cv.BORDER_DEFAULT)
    dy = cv.Sobel(src, ddepth, 0, 1, ksize=3, borderType=cv.BORDER_DEFAULT)
    dx_abs = cv.convertScaleAbs(dx)
    dy_abs = cv.convertScaleAbs(dy)
    return dx_abs + dy_abs

# Aplica o algoritmo de programação dinâmica na imagem de energia, gerando, para cada
# posição res[i, j], a energia total do melhor caminho saindo de (i, j)
@jit
def build_energy_paths(src):
    h, w = src.shape[:2]
    res = np.zeros((h, w))

    # Última linha da matriz de programação dinâmica
    for j in range(w):
        res[h-1, j] = src[h-1, j]

    # Linhas de cima
    for i in range(h-2, -1, -1):
        for j in range(w-1, -1, -1):
            res[i, j] = src[i, j] + np.amin(res[i+1, max(0, j-1):j+2])

    return res

# Retorna o array seam, tal que seam[i] = coluna j que deve ser removida na linha i
@jit
def get_seam(src, energy):
    h, w, d = src.shape
    seam = np.zeros(h, dtype=np.int32)

    seam[0] = np.argmin(energy[0, :])
    for i in range (1, h):
        up = seam[i-1]
        seam[i] = max(0, up-1) + np.argmin(energy[i, max(0, up-1):up+2])

    return seam

@jit
def remove_seam(src, mask, seam):
    h, w, d = src.shape
    res = np.zeros((h, w-1, d), dtype=src.dtype)

    rmask = None
    if mask is not None:
        rmask = np.zeros((h, w-1), dtype=mask.dtype)

    for i in range(h):
        s = seam[i]

        res[i, 0:s] = src[i, 0:s] # antes do seam[i]
        res[i, s:w-1] = src[i, s+1:w] # depois do seam[i]

        if mask is not None:
            rmask[i, 0:s] = mask[i, 0:s]
            rmask[i, s:w-1] = mask[i, s+1:w]

    return res, rmask

# Aplica uma mask de remoção sobre a função de energia
def apply_mask(sob, mask):
    copy = np.copy(sob).astype(np.float64)
    mx = np.max(copy)
    copy[mask == 255] = 10000 * mx
    copy[mask == 0]   = -10000 * mx
    return copy
    # return sob * mask

# Remove 1 seam da imagem
def one_step(im, mask):
    sob = image_energy(im)
    if mask is not None:
        sob = apply_mask(sob, mask)
    energy = build_energy_paths(sob)
    seam = get_seam(im, energy)
    im, mask = remove_seam(im, mask, seam)

    return im, mask

# Remove `n` seams da imagem
# A `mask` é opcional
def iterate(im, mask, n):
    for _ in range(n):
        im, mask = one_step(im, mask)
    return im, mask

# Remove seams da imagem até que a mask não possua mais pixeis 255
def remove_object(im, mask):
    while np.any(mask == 0):
        im, mask = one_step(im, mask)
    return im, mask

def main():
    args = arguments()
    imid = args.image.split('/')[-1].split('.')[0]

    print(args)
    im = cv.imread(args.image)
    h, w, d = im.shape
    if w > MAXWIDTH:
        scaling = MAXWIDTH / w
        im = cv.resize(im, None, fx=scaling, fy=scaling, interpolation=cv.INTER_CUBIC)

    sob = image_energy(im)

    mask = None
    if args.mask:
        h, w, d = im.shape
        mask = cv.imread(args.mask)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.resize(mask, (w, h), interpolation=cv.INTER_LINEAR)
        # lo = 0
        # hi = 1.0
        # mask = mask / np.amax(mask) * (hi - lo) + lo
        mask = np.uint8(mask)
        sob = apply_mask(sob, mask)

    energy = build_energy_paths(sob)

    cv.imwrite(f"out/{imid}.input.png", im)
    cv.imwrite(f"out/{imid}.sobel.png", sob)
    cv.imwrite(f"out/{imid}.energy.png", cv.normalize(energy, None, 0, 255, cv.NORM_MINMAX))

    if args.show:
        cv.imshow("Original", im)
        cv.imshow("Sobel", sob)
        cv.imshow("Energy", np.uint8(cv.normalize(energy, None, 0, 255, cv.NORM_MINMAX)))

    if args.remove:
        im, mask = remove_object(im, mask)
    else:
        im, mask = iterate(im, mask, int(args.iterations))

    cv.imwrite(f"out/{imid}.scaled.png", im)

    if args.show:
        cv.imshow("Resultado", im)
        cv.waitKey()

if __name__ == '__main__':
    main()
