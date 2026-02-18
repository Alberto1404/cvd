import numpy as np
import matplotlib.pyplot as plt
import imageio

import process

from skimage import io,img_as_float64
from matplotlib.pyplot import imread

def compute_difference(ima_RGB, ima_simulated):

    ima_difference = ima_RGB - ima_simulated
    # Para que salga igual hay que restar la original de la simulada.

    return ima_difference

def compute_RRGGBB(dif, defecto):

    RRGGBB = np.zeros(np.shape(dif))

    matrix_change = np.zeros((3,3))

    if defecto == 'protanopia':
        matrix_change[0,0] = 0
        matrix_change[0,1] = 0
        matrix_change[0,2] = 0
        matrix_change[1,0] = 0.5
        matrix_change[1,1] = 1
        matrix_change[1,2] = 0
        matrix_change[2,0] = 0.5
        matrix_change[2,1] = 0
        matrix_change[2,2] = 1

    elif defecto == 'deuteranopia':
        matrix_change[0,0] = 1
        matrix_change[0,1] = 0.5
        matrix_change[0,2] = 0
        matrix_change[1,0] = 0
        matrix_change[1,1] = 0
        matrix_change[1,2] = 0
        matrix_change[2,0] = 0
        matrix_change[2,1] = 0.5
        matrix_change[2,2] = 1

    elif defecto == 'tritanopia':
        matrix_change[0,0] = 1
        matrix_change[0,1] = 0
        matrix_change[0,2] = 0.7
        matrix_change[1,0] = 0
        matrix_change[1,1] = 1
        matrix_change[1,2] = 0.7
        matrix_change[2,0] = 0
        matrix_change[2,1] = 0
        matrix_change[2,2] = 0

    else:
        raise ValueError('Defecto desconocido.\n')  

    aux = np.dot(matrix_change,process.reshape_matrix(dif))
    for i in range(3):
        RRGGBB[:,:,i] = aux[i,:].reshape((np.shape(dif)[0],np.shape(dif)[1]))
    
    return RRGGBB

def compute_image_recolored(RRGGBB, ima_simulated):
    
    ima_recolored = np.zeros(np.shape(ima_simulated))

    RR = RRGGBB[:,:,0]
    GG = RRGGBB[:,:,1]
    BB = RRGGBB[:,:,2]

    r = ima_simulated[:,:,0]
    g = ima_simulated[:,:,1]
    b = ima_simulated[:,:,2]

    ima_recolored[:,:,0] = RR+r
    ima_recolored[:,:,1] = GG+g
    ima_recolored[:,:,2] = BB+b

    return ima_recolored

def recolor_melillo(ima_RGB, defecto):
    ima_RGB = process.norm(ima_RGB)
    ima_sim = process.norm(process.process_image(ima_RGB, defecto, 'Melillo'))
    dif = compute_difference(ima_RGB, ima_sim)
    RRGGBB = compute_RRGGBB(dif, defecto)
    ima_recolored = compute_image_recolored(RRGGBB, ima_sim)
    np.clip(ima_recolored, 0, 1, out = ima_recolored)

    return process.renorm(ima_recolored)


if __name__ == "__main__":
    
    name = 'iso'+'.jpg'
    ima_ori = io.imread('/Users/alberto/Desktop/'+name)

    k = process.split_string(name, ".")
    ima_prot = process.process_image(process.norm(ima_ori), 'protanopia', 'Melillo')
    ima_deut = process.process_image(process.norm(ima_ori), 'deuteranopia', 'Melillo')
    ima_trit = process.process_image(process.norm(ima_ori), 'tritanopia', 'Melillo')

    # imageio.imwrite("/Users/alberto/Downloads/pruebas/"+k[0]+"_"+'sim_'+'protanopia'+".jpg", ima_prot)
    # imageio.imwrite("/Users/alberto/Downloads/pruebas/"+k[0]+"_"+'sim_'+'deuteranopia'+".jpg", ima_deut)
    # imageio.imwrite("/Users/alberto/Downloads/pruebas/"+k[0]+"_"+'sim_'+'tritanopia_Machado'+".jpg", ima_trit)

    recolored_prot = recolor_melillo(ima_ori, 'protanopia')
    recolored_deut = recolor_melillo(ima_ori, 'deuteranopia')
    recolored_trit = recolor_melillo(ima_ori, 'tritanopia')

    # imageio.imwrite("/Users/alberto/Downloads/pruebas/"+k[0]+"_"+'recolor_'+'protanopia'+'_Melillo'+".jpg", recolored_prot)
    # imageio.imwrite("/Users/alberto/Downloads/pruebas/"+k[0]+"_"+'recolor'+'deuteranopia'+'_Melillo'+".jpg", recolored_deut)
    imageio.imwrite("/Users/alberto/Downloads/pruebas/"+k[0]+"_"+'recolor'+'tritanopia'+'_trL_Melillo'+".jpg", recolored_trit)

    sim_rec_prot = process.process_image(recolored_prot, 'protanopia', 'Melillo')
    sim_rec_deut = process.process_image(recolored_deut, 'deuteranopia', 'Melillo')
    sim_rec_trit = process.process_image(recolored_trit, 'tritanopia', 'Melillo')

    # imageio.imwrite("/Users/alberto/Downloads/pruebas/"+k[0]+"_"+'recolor_sim_'+'protanopia'+'_Melillo'+".jpg", sim_rec_prot)
    # imageio.imwrite("/Users/alberto/Downloads/pruebas/"+k[0]+"_"+'recolor_sim_'+'deuteranopia'+'_Melillo'+".jpg", sim_rec_deut)
    # imageio.imwrite("/Users/alberto/Downloads/pruebas/"+k[0]+"_"+'recolor_sim_'+'tritanopia'+'_trL_Melillo'+".jpg", sim_rec_trit)


    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(ima_ori)
    plt.title('Imagen original')
    plt.subplot(2,2,2)
    plt.imshow(ima_prot)
    plt.title('Simulación de protanopia')
    plt.subplot(2,2,3)
    plt.imshow(recolored_prot)
    plt.title('Recoloreado')
    plt.subplot(2,2,4)
    plt.imshow(sim_rec_prot)
    plt.title('Simulación de la imagen recoloreada')

    plt.show()






