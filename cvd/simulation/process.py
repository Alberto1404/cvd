# SIMULATION FILE

import numpy as np
import matplotlib.pyplot as plt
import imageio
import PIL

from skimage import io,img_as_float64
from matplotlib.pyplot import imread
from PIL import Image

def reshape_matrix(matrix):
	reshaped = np.zeros((3,np.shape(matrix)[0]*np.shape(matrix)[1]))
	for i in range(np.shape(matrix)[2]):
		reshaped[i,:] = matrix[:,:,i].flatten()
		
	return reshaped

def ratio_image(ima_RGB):
	filas = np.shape(ima_RGB)[0]
	columnas = np.shape(ima_RGB)[1]

	el_max = max(filas, columnas)
	el_min = min(filas, columnas)

	if el_max > filas:
		# COLUMNAS >
		ratio = el_max/filas
	else:
		# FILAS >
		ratio = el_max/columnas

	if el_max >= 1500 and el_max < 2000:
		ima = Image.fromarray(ima_RGB)
		ima = ima.resize((int(columnas/2),int(filas/2)), PIL.Image.ANTIALIAS)

	elif el_max >= 2000:
		ima = Image.fromarray(ima_RGB)
		ima = ima.resize((int(columnas/4),int(filas/4)), PIL.Image.ANTIALIAS)
	
	else:
		ima = Image.fromarray(ima_RGB)
	
	return ima

def RGB2LMS_Vienot():
	matrix_change = np.zeros((3,3))

	matrix_change[0,0] = 17.8824
	matrix_change[0,1] = 43.5161
	matrix_change[0,2] = 4.11935
	matrix_change[1,0] = 3.45565
	matrix_change[1,1] = 27.1554
	matrix_change[1,2] = 3.86714
	matrix_change[2,0] = 0.0299566
	matrix_change[2,1] = 0.184309
	matrix_change[2,2] = 1.46709
	
	return matrix_change

def RGB2LMS_Melillo():

	matrix_change = RGB2LMS_Vienot()
	return matrix_change


def RGB2LMS(image_RGB, opcion):
	matrix_change = np.zeros((3,3))
	image_LMS = np.zeros((np.shape(image_RGB)))

	if opcion == 'Vienot':
		matrix_change = RGB2LMS_Vienot()
	
	elif opcion == 'Melillo':

		matrix_change = RGB2LMS_Melillo()

	elif opcion == 'Machado':
		print("Aviso... El método de propuesto por Machado no realiza conversión de espacios de color.\n")
		return image_RGB
		
	else:
		raise ValueError('Opción incorrecta.\n')

	print('Obteniendo imagen LMS...\n')
	"""for i in range(np.shape(image_LMS)[0]):
		for j in range(np.shape(image_LMS)[1]):
			image_LMS[i,j,:] = np.dot(matrix_change, image_RGB[i,j,:])"""
	ima_LMS = np.dot(matrix_change,reshape_matrix(image_RGB))
	for i in range(3):
		image_LMS[:,:,i] = ima_LMS[i,:].reshape((np.shape(image_RGB)[0],np.shape(image_RGB)[1]))
	print('Done¡\n')
	return image_LMS

def LMS2RGB_Vienot():
	matrix_change = np.zeros((3,3))
	matrix_change[0,0] = 0.080944
	matrix_change[0,1] = -0.130504
	matrix_change[0,2] = 0.116721
	matrix_change[1,0] = -0.0102485
	matrix_change[1,1] = 0.0540194
	matrix_change[1,2] = -0.113615
	matrix_change[2,0] = -0.000365294
	matrix_change[2,1] = -0.00412163
	matrix_change[2,2] = 0.693513

	return matrix_change

def LMS2RGB_Melillo():
	matrix_change = LMS2RGB_Vienot()

	return matrix_change


def LMS2RGB(image_LMS, opcion):
	matrix_change = np.zeros((3,3))
	image_RGB = np.zeros((np.shape(image_LMS)))

	if opcion == 'Vienot':

		matrix_change = LMS2RGB_Vienot()
	
	elif opcion == 'Melillo':

		matrix_change = LMS2RGB_Melillo()

	elif opcion == 'Machado':
		print("Aviso... El método de propuesto por Machado no realiza conversión de espacios de color.\n")
		return image_LMS

	else:
		raise ValueError('Opción no válida.\n')

	print('Reconvirtiendo imagen LMS con defecto de visión a RGB...\n')
	"""for i in range(np.shape(image_LMS)[0]):
		for j in range(np.shape(image_LMS)[1]):
			image_RGB[i,j,:] = np.dot(matrix_change, image_LMS[i,j,:])"""
	ima_RGB = np.dot(matrix_change,reshape_matrix(image_LMS))
	for i in range(3):
		image_RGB[:,:,i] = ima_RGB[i,:].reshape((np.shape(image_LMS)[0],np.shape(image_LMS)[1]))
	print('Done¡\n')
	return image_RGB

def protanopia_Vienot():
	matrix_change = np.zeros((3,3))
	matrix_change[0,0] = 0
	matrix_change[0,1] = 2.02344
	matrix_change[0,2] = -2.52581
	matrix_change[1,0] = 0
	matrix_change[1,1] = 1
	matrix_change[1,2] = 0
	matrix_change[2,0] = 0
	matrix_change[2,1] = 0
	matrix_change[2,2] = 1
	return matrix_change

def protanopia_Melillo():
	matrix_change = protanopia_Vienot()
	return matrix_change


def protanopia_Machado():
	matrix_change = np.zeros((3,3))
	matrix_change[0,0] = 0.152286
	matrix_change[0,1] = 1.052583
	matrix_change[0,2] = -0.204868
	matrix_change[1,0] = 0.114503
	matrix_change[1,1] = 0.786281
	matrix_change[1,2] = 0.099216
	matrix_change[2,0] = -0.003882
	matrix_change[2,1] = -0.048116
	matrix_change[2,2] = 1.051998
	return matrix_change


def protanopia(image_LMS_or_RGB, opcion):
	matrix_change = np.zeros((3,3))
	ima_protanopia = np.zeros((np.shape(image_LMS_or_RGB)))

	if opcion == 'Vienot':
		
		# Vienot -> Melillo >< Thakkar
		matrix_change = protanopia_Vienot()

	elif opcion == 'Melillo':
		matrix_change = protanopia_Melillo()
		
	elif opcion == 'Machado':
		
		# Machado (DIRECTAMENTE SOBRE RGB¡¡¡)
		matrix_change = protanopia_Machado()

	else:
		raise ValueError('Opción no válida.\n')

	print('Aplicando el defecto protanopia a imagen LMS...\n')
	"""for i in range(np.shape(image_LMS_or_RGB)[0]):
		for j in range(np.shape(image_LMS_or_RGB)[1]):
			ima_protanopia[i,j,:] = np.dot(matrix_change, image_LMS_or_RGB[i,j,:])"""
	ima_prot = np.dot(matrix_change,reshape_matrix(image_LMS_or_RGB))
	for i in range(3):
		ima_protanopia[:,:,i] = ima_prot[i,:].reshape((np.shape(image_LMS_or_RGB)[0],np.shape(image_LMS_or_RGB)[1]))
	print('Done¡\n')
	return ima_protanopia

def deuteranopia_Vienot():
	matrix_change = np.zeros((3,3))
	matrix_change[0,0] = 1
	matrix_change[0,1] = 0
	matrix_change[0,2] = 0
	matrix_change[1,0] = 0.494207
	matrix_change[1,1] = 0
	matrix_change[1,2] = 1.24827
	matrix_change[2,0] = 0
	matrix_change[2,1] = 0
	matrix_change[2,2] = 1
	return matrix_change

def deuteranopia_Melillo():
	matrix_change = np.zeros((3,3))
	matrix_change[0,0] = 1.42319
	matrix_change[0,1] = -0.88995
	matrix_change[0,2] = 1.77557
	matrix_change[1,0] = 0.67558
	matrix_change[1,1] = -0.42203
	matrix_change[1,2] = 2.82788
	matrix_change[2,0] = 0.00267
	matrix_change[2,1] = -0.00504
	matrix_change[2,2] = 0.99914
	return matrix_change

def deuteranopia_Machado():
	matrix_change = np.zeros((3,3))
	matrix_change[0,0] = 0.367322
	matrix_change[0,1] = 0.860646
	matrix_change[0,2] = -0.227968
	matrix_change[1,0] = 0.280085
	matrix_change[1,1] = 0.672501
	matrix_change[1,2] = 0.047413
	matrix_change[2,0] = -0.01182
	matrix_change[2,1] = 0.04294
	matrix_change[2,2] = 0.968881
	return matrix_change

	
def deuteranopia(image_LMS_or_RGB, opcion):
	matrix_change = np.zeros((3,3))
	ima_deuteranopia = np.zeros((np.shape(image_LMS_or_RGB)))

	if opcion == 'Vienot':

		# Vienot (Usada por Thakkar)
		matrix_change = deuteranopia_Vienot()

	elif opcion == 'Melillo':

		# Melillo
		matrix_change = deuteranopia_Melillo()

	elif opcion == 'Machado':

		# Oliveria (DIRECTAMENTE SOBRE RGB¡¡¡)
		matrix_change = deuteranopia_Machado()

	else:
		raise ValueError('Opción no válida.\n')  

	print('Aplicando el defecto deuteranopia a imagen LMS...\n')
	"""for i in range(np.shape(image_LMS_or_RGB)[0]):
		for j in range(np.shape(image_LMS_or_RGB)[1]):
			ima_deuteranopia[i,j,:] = np.dot(matrix_change, image_LMS_or_RGB[i,j,:])"""
	ima_deut = np.dot(matrix_change,reshape_matrix(image_LMS_or_RGB))
	for i in range(3):
		ima_deuteranopia[:,:,i] = ima_deut[i,:].reshape((np.shape(image_LMS_or_RGB)[0],np.shape(image_LMS_or_RGB)[1]))
	print('Done¡\n')
	return ima_deuteranopia

def tritanopia_Melillo():
	matrix_change = np.zeros((3,3))
	matrix_change[0,0] = 0.95451
	matrix_change[0,1] = -0.04719
	matrix_change[0,2] = 2.74872
	matrix_change[1,0] = -0.00447
	matrix_change[1,1] = 0.96543
	matrix_change[1,2] = 0.88835
	matrix_change[2,0] = -0.01251
	matrix_change[2,1] = 0.07312
	matrix_change[2,2] = -0.01161
	return matrix_change

def tritanopia_Machado():
	matrix_change = np.zeros((3,3))
	matrix_change[0,0] = 1.255528
	matrix_change[0,1] = -0.076749
	matrix_change[0,2] = -0.178779
	matrix_change[1,0] = -0.078411
	matrix_change[1,1] = 0.930809
	matrix_change[1,2] = 0.147602
	matrix_change[2,0] = 0.004733
	matrix_change[2,1] = 0.691367
	matrix_change[2,2] = 0.3039
	return matrix_change

def tritanopia(image_LMS_or_RGB, opcion):
	matrix_change = np.zeros((3,3))
	ima_tritanopia = np.zeros((np.shape(image_LMS_or_RGB)))

	# No vale para nada, usadda por Thakkar
	"""
	matrix_change[0,0] = 1
	matrix_change[0,1] = 0
	matrix_change[0,2] = 0
	matrix_change[1,0] = 0
	matrix_change[1,1] = 1
	matrix_change[1,2] = 0
	matrix_change[2,0] = -0.86744736
	matrix_change[2,1] = 1.86727089
	matrix_change[2,2] = 0
	"""
	if opcion == 'Melillo':

		# Melillo
		matrix_change = tritanopia_Melillo()

	elif opcion == 'Machado':

		# Oliveria (DIRECTAMENTE SOBRE RGB¡¡¡)
		matrix_change = tritanopia_Machado()

	print('Aplicando el defecto tritanopia a imagen LMS...\n')
	
	ima_trit = np.dot(matrix_change,reshape_matrix(image_LMS_or_RGB))
	for i in range(3):
		ima_tritanopia[:,:,i] = ima_trit[i,:].reshape((np.shape(image_LMS_or_RGB)[0],np.shape(image_LMS_or_RGB)[1]))
	print('Done¡\n')
	return ima_tritanopia

def renorm(ima_observada):
	ima_renorm = 255*ima_observada
	# ima_renorm = np.round(ima_renorm)
	np.clip(ima_renorm, 0, 255, out = ima_renorm)
	ima_renorm = ima_renorm.astype(np.uint8)
	return ima_renorm

def norm(ima_RGB):
	ima_norm = img_as_float64(ima_RGB)
	return ima_norm

def process_image(image_RGB, defecto, opcion):

	print('Obteniendo imagen simulada...\n')

	image_RGB = norm(image_RGB)

	ima_observada = np.zeros((np.shape(image_RGB)))
	if defecto == 'protanopia':
		if opcion == 'Machado':
			ima_protanopia = protanopia(image_RGB, opcion)
			ima_observada = renorm(ima_protanopia)
			return ima_observada
		elif opcion == 'Melillo':
			ima_protanopia = np.dot(LMS2RGB_Melillo(),protanopia_Melillo())
			ima_protanopia = np.dot(ima_protanopia,RGB2LMS_Melillo())
			"""for i in range(np.shape(image_RGB)[0]):
				for j in range(np.shape(image_RGB)[1]):
					ima_observada[i,j,:] = np.dot(ima_protanopia, image_RGB[i,j,:])"""
			ima_observad = np.dot(ima_protanopia,reshape_matrix(image_RGB))
			for i in range(3):
				ima_observada[:,:,i] = ima_observad[i,:].reshape((np.shape(image_RGB)[0],np.shape(image_RGB)[1]))
			ima_observada = renorm(ima_observada)
		elif opcion == 'Vienot':
			ima_protanopia = np.dot(LMS2RGB_Vienot(),protanopia_Vienot())
			ima_protanopia = np.dot(ima_protanopia,RGB2LMS_Vienot())
			"""for i in range(np.shape(image_RGB)[0]):
				for j in range(np.shape(image_RGB)[1]):
					ima_observada[i,j,:] = np.dot(ima_protanopia, image_RGB[i,j,:])"""
			ima_observad = np.dot(ima_protanopia,reshape_matrix(image_RGB))
			for i in range(3):
				ima_observada[:,:,i] = ima_observad[i,:].reshape((np.shape(image_RGB)[0],np.shape(image_RGB)[1]))
			ima_observada = renorm(ima_observada)
		
			
	
	elif defecto == 'deuteranopia':
		if opcion == 'Machado':
			ima_deuteranopia = deuteranopia(image_RGB, opcion)
			ima_observada = renorm(ima_deuteranopia)
			return ima_observada
		elif opcion == 'Melillo':
			ima_deuteranopia = np.dot(LMS2RGB_Melillo(),deuteranopia_Melillo())
			ima_deuteranopia = np.dot(ima_deuteranopia,RGB2LMS_Melillo())
			"""for i in range(np.shape(image_RGB)[0]):
				for j in range(np.shape(image_RGB)[1]):
					ima_observada[i,j,:] = np.dot(ima_deuteranopia, image_RGB[i,j,:])"""
			ima_observad = np.dot(ima_deuteranopia,reshape_matrix(image_RGB))
			for i in range(3):
				ima_observada[:,:,i] = ima_observad[i,:].reshape((np.shape(image_RGB)[0],np.shape(image_RGB)[1]))
			ima_observada = renorm(ima_observada)
		elif opcion == 'Vienot':
			ima_deuteranopia = np.dot(LMS2RGB_Vienot(),deuteranopia_Vienot())
			ima_deuteranopia = np.dot(ima_deuteranopia,RGB2LMS_Vienot())
			"""for i in range(np.shape(image_RGB)[0]):
				for j in range(np.shape(image_RGB)[1]):
					ima_observada[i,j,:] = np.dot(ima_deuteranopia, image_RGB[i,j,:])"""
			ima_observad = np.dot(ima_deuteranopia,reshape_matrix(image_RGB))
			for i in range(3):
				ima_observada[:,:,i] = ima_observad[i,:].reshape((np.shape(image_RGB)[0],np.shape(image_RGB)[1]))
			ima_observada = renorm(ima_observada)


	elif defecto == 'tritanopia':
		if opcion == 'Machado':
			ima_tritanopia = tritanopia(image_RGB, opcion)
			ima_observada = renorm(ima_tritanopia)
			return ima_observada
		elif opcion == 'Melillo':
			ima_tritanopia = np.dot(LMS2RGB_Melillo(),tritanopia_Melillo())
			ima_tritanopia = np.dot(ima_tritanopia,RGB2LMS_Melillo())
			"""for i in range(np.shape(image_RGB)[0]):
				for j in range(np.shape(image_RGB)[1]):
					ima_observada[i,j,:] = np.dot(ima_tritanopia, image_RGB[i,j,:])"""
			ima_observad = np.dot(ima_tritanopia,reshape_matrix(image_RGB))
			for i in range(3):
				ima_observada[:,:,i] = ima_observad[i,:].reshape((np.shape(image_RGB)[0],np.shape(image_RGB)[1]))
			ima_observada = renorm(ima_observada)

	else:
		raise ValueError('Opción no válida.\n')
		
	return ima_observada

def simulation(ima_original, defecto, opcion, name):

	ima_defecto = process_image(ima_original, defecto, opcion)
	option = input(' ¿Desea guardar o representar las imágenes original y simulada? [save(s)/show(sh)/both(b)/*] \n')

	if option == 'save' or option == 's':
		print('\nGuardando imagen simulada...')
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_ori/"+opcion+'/'+ name.split('.')[0]+"_"+opcion+"_"+defecto+".jpg", ima_defecto)
	
	elif option == 'show' or option == 'sh':
		print('\nRepresentando...\n')
		plt.figure()
		plt.subplot(1,2,1)
		plt.imshow(renorm(ima_original))
		plt.title('Imagen original')
		plt.subplot(1,2,2)
		plt.imshow(ima_defecto)
		plt.title('Visualización de '+defecto)
		plt.show()

	elif option == 'both' or option == 'b':
		print('\nGuardando imagen simulada...')
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_ori/"+opcion+'/'+ name.split('.')[0]+"_"+opcion+"_"+defecto+".jpg", ima_defecto)
		print('\n\tImágenes guardadas correctamente.\n')
		print('\nRepresentando...\n')
		plt.figure()
		plt.subplot(1,2,1)
		plt.imshow(renorm(ima_original))
		plt.title('Imagen original')
		plt.subplot(1,2,2)
		plt.imshow(ima_defecto)
		plt.title('Visualización de '+defecto)
		plt.show()
