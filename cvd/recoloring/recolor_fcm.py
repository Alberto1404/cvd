
import numpy as np
import matplotlib.pyplot as plt
import math
import process
import skfuzzy as fuzz
import imageio
from mpl_toolkits.mplot3d import Axes3D
import recolor_melillo

from skimage import io, img_as_float64
from matplotlib.pyplot import imread
from matplotlib import colors

import PIL
from PIL import Image

def get_theta(ima_original, ima_observed, PSI, defecto):
	###### ARGUMENTOS DE ENTRADA ######
	# ima_original: imagen original RGB (Filas,Columnas,3)
	# ima_observed: imagen con el defecto de visión RGB (Filas,Columnas,3)

	###### FUNCIONAMIENTO ######
	# La función get_theta realiza una comaración componente a componente entre cada píxel diferencia entre la imagen original 
	# y la del defecto con el 1% del cada píxel original.

	###### ARGUMENTOS DE SALIDA ###### 
	# result: matriz que tomará o 0 o 1 dependiendo de si la comparación realizada anteriormente es mayor o menor la una de la 
	#         otra, respecticamente (Filas, Columnas)

	if PSI == 'PSI_1':

		print('Obteniendo φ(X_o,X_d,ψ1)...\n')
		ima_dif = abs(ima_original - ima_observed)
		PSI_1 = 0.01*ima_original
		filas, columnas, _ = np.shape(ima_dif)

		ima_dif = process.reshape_matrix(ima_dif)
		PSI_1 = process.reshape_matrix(PSI_1)
		result = []

		for i in range(np.shape(PSI_1)[1]):
			if ima_dif[0,i] <= PSI_1[0,i] and ima_dif[1,i] <= PSI_1[1,i] and ima_dif[2,i] <= PSI_1[2,i]:
				result.append(1)
			else:
				result.append(0)
		print('\n\tφ(X_o,X_d,ψ1) obtenido correctamente.\n')
		result = np.asarray(result, dtype=np.float64)
		print('Done\n')
		return np.reshape(result, (filas, columnas))


	elif PSI == 'PSI_2':

		###### ARGUMENTOS DE ENTRADA ######
		# ima_original: centroides originales recoloreados G1^(3,N); N = nº centroides/clusters
		# ima_observed: centroides originales G2 percibidos correctamente por daltónico (3,N); N = nº centroides/clusters

		###### FUNCIONAMIENTO ######
		# La función get_theta realiza una comparación de cada componente de cada centroide, y comprueba si es menor que una
		# variable PSI_2, llamada umbral de calidad visual, inicializada a (20,20,20) 

		###### ARGUMENTOS DE SALIDA ###### 
		# result: lista de valores que indicará si hay discriminación de color o no tras el recoloreado

		print('Obteniendo φ(X_o,X_d,ψ2)...\n')
		result = np.zeros((np.shape(ima_original)[1],np.shape(ima_original)[1]))

		for i in range(np.shape(result)[0]):
			for j in range(np.shape(result)[0]):
				if abs(ima_original[0,i]-ima_observed[0,j]) < np.float64(20/255) and abs(ima_original[1,i]-ima_observed[1,j]) < np.float64(20/255) and abs(ima_original[2,i]-ima_observed[2,j]) < np.float64(20/255):
					result[i,j] = 1
				else:
					result[i,j] = 0

		return result

def get_gamma_color(ima_original, opcion):
	###### ARGUMENTOS DE ENTRADA ######
	# ima_original: imagen original RGB (Filas, Columnas, 3)
	# opción: cadena en la que se indica si se quiere trabajar protanopia, deuteranopia, o tritanopia

	###### FUNCIONAMIENTO ######
	# PROTANOPIA: devuelve 1 para cada píxel de la imagen en el que se cumple que la componente roja es, a la vez, mayor que 
	# las otras 2, 0 en caso contrario. (extracción de rojos)
	# DEUTERANOPIA: devuelve 1 para cada píxel de la imagen en el que se cumple que la componente roja es, a la vez, mayor que 
	# las otras 2, 0 en caso contrario. (extracción de rojos)
	# TRITANOPIA: devuelve 1 para cada píxel de la imagen en el que se cumple que la componente azul es, a la vez, mayor que 
	# las otras 2, 0 en caso contrario. (extracción de azules)

	###### ARGUMENTOS DE SALIDA ######
	# result: matriz (Filas, Columnas) que tomará valores 0 o 1 en función de lo obtenido anteriormente. 
	print('Calculando ɣ(Xo)...\n')
	filas, columnas,_ = np.shape(ima_original)

	ima_original = process.reshape_matrix(ima_original)
	result = []

	if opcion == 'protanopia': # EXTRACCIÓN DE ROJOS
		for i in range(np.shape(ima_original)[1]):
			if ima_original[0,i] > ima_original[1,i] and ima_original[0,i] > ima_original[2,i]:
				result.append(1)
			else:
				result.append(0)
	elif opcion == 'deuteranopia': # EXTRACCIÓN DE ROJOS
		for i in range(np.shape(ima_original)[1]):
			if ima_original[0,i] > ima_original[1,i] and ima_original[0,i] > ima_original[2,i]:
				result.append(1)
			else:
				result.append(0)
	elif opcion == 'tritanopia': # EXTRACCIÓN DE AZULES
		for i in range(np.shape(ima_original)[1]):
			if ima_original[2,i] > ima_original[1,i] and ima_original[2,i] > ima_original[0,i]:
				result.append(1)
			else:
				result.append(0)
	else:
		raise ValueError('\n\tDiscromatopsia desconocida. No se obtendrá ɣ(Xo).\n')

	result = np.asarray(result,dtype=np.float64)

	print('\n\tɣ(Xo) obtenido correctamente\n')
	return np.reshape(result,(filas, columnas))

def get_O_1andO_2(theta, gamma_color, ima_original, defecto, nombre_imagen):
	###### ARGUMENTOS DE ENTRADA ######
	# theta: función diferencia de color. Vale 0 cuando no lo ve el daltónico y 1 cuando sí lo ve, aunque en este punto del
	#        algoritmo no tiene sentido todavía hablar de esto. (Filas, Columnas)
	# gamma_color: Función dependiente del tipo de daltonismo. (Filas, columnas)
	# ima_original: imagen original RGB, aunque solo la usaremos para calcular las dimensiones de las variables de salida
	# defecto: protanopia, deuteranopia o tritanopia

	###### FUNCIONAMIENTO ######
	# Para cada píxel, si la función theta vale 0 y la función gamma vale 1, ese color de píxel (de momento ese píxel)
	# pertenecerá al conjunto O_1. O_2 es el complementario de O_1

	###### ARGUMENTOS DE SALIDA ######
	# O_1: subconjunto en el que incluimos los colores (de momento los píxeles, posteriormente extraeremos el color del píxel) 
	# que el daltónico no ve.
	print('Obteniendo conjuntos O1 y O2\n\tO1 = [Xk | φ(X_o,X_d,ψ1) = 0 and ɣ(Xo) = 1]\n\tO2 = /O1')
	filas, columnas, _ = np.shape(ima_original)

	theta = theta[:,:].flatten()
	gamma_color = gamma_color[:,:].flatten()
	O_1 = []

	for i in range(len(theta)):
		if theta[i] == 0 and gamma_color[i] == 1:
			O_1.append(1)
		else:
			O_1.append(0)

	O_1 = np.reshape(np.asarray(O_1,dtype=np.float64),(filas, columnas))

	O_2 = 1 - O_1 # IMPORTA SOLO CUANDO VALE 1, ES DECIR, CUANDO EL DALTÓNICO SI VE ESE COLOR
	# O_2 = np.reshape(np.asarray(O_2,dtype=np.float64),(filas,columnas))

	"""print('\n\tConjuntos O1 y O2 obtenidos con éxito.\n')
	print('\nObteniendo colores que sí/no se pueden ver por un daltónico...\n')
	print('Colocando colores que sí/no se pueden ver por un daltónico en su correspondiente lugar en la imagen...\n')"""

	no = np.argwhere(O_1 == 1) # Contiene posición de colores que el daltónico NO ve
	yes = np.argwhere(O_2 == 1) # Contiene posición de colores que el daltónico SÍ ve

	colors_CB_no = np.zeros((np.shape(no)[0],3))
	colors_CB_yes = np.zeros((np.shape(yes)[0],3))

	image_no = np.zeros((filas, columnas, 3))
	image_yes = np.zeros((filas, columnas, 3))

	for index, each in enumerate(no):
		colors_CB_no[index,:] = ima_original[each[0], each[1],:] # COLORES QUE EL DALTÓNICO NO VE
		image_no[each[0], each[1],:] = ima_original[each[0], each[1],:] # IMAGEN DE LOS COLORES QUE NO VE

	for index, each in enumerate(yes):
		colors_CB_yes[index,:] = ima_original[each[0], each[1],:] # COLORES QUE EL DALTÓNICO NO VE
		image_yes[each[0], each[1],:] = ima_original[each[0], each[1],:] # IMAGEN DE LOS COLORES QUE SI VE

	"""print('\n\tColores e imágenes obtenidas correctamente. ')
	print('\n\tGuardando conjuntos O1 y O2...')
	imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/O_1_2/"+'O1_'+nombre_imagen+".jpg", process.renorm(image_no))
	imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/O_1_2/"+'O2_'+nombre_imagen+".jpg", process.renorm(image_yes))
	option = input(' ¿Desea guardar o representar las imágenes de los conjuntos O1 y O2? [save(s)/show(sh)/both(b)/*] \n')

	if option == 'save' or option == 's':
		print('\nGuardando conjuntos O1 y O2...')
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/O_1_2/"+'O1'+nombre_imagen+".jpg", process.renorm(image_no))
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/O_1_2/"+'O2'+nombre_imagen+".jpg", process.renorm(image_yes))

	elif option == 'show' or option == 'sh':
		print('\nRepresentando...\n')
		plt.figure()
		plt.subplot(2,1,1)
		plt.imshow(process.renorm(image_no))
		plt.title('O1 (colores no percibidos correctamente)')
		plt.subplot(2,1,2)
		plt.imshow(process.renorm(image_yes))
		plt.title('O2 (colores percibidos correctamente)')
		plt.show()

	elif option == 'both' or option == 'b':
		print('\nGuardando conjuntos O1 y O2...')
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/O_1_2/"+'O1'+nombre_imagen+".jpg", process.renorm(image_no))
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/O_1_2/"+'O2'+nombre_imagen+".jpg", process.renorm(image_yes))
		print('\n\tImágenes guardadas correctamente.\n')
		print('\nRepresentando...\n')
		plt.figure()
		plt.subplot(2,1,1)
		plt.imshow(process.renorm(image_no))
		plt.title('O1 (colores no percibidos correctamente)')
		plt.subplot(2,1,2)
		plt.imshow(process.renorm(image_yes))
		plt.title('O2 (colores percibidos correctamente)')
		plt.show()"""

	return O_1, O_2, colors_CB_no.T, colors_CB_yes.T, no,yes, image_yes, image_no

def get_clustered_model(nclusters, dataset, fuzz_idx, alpha, beta):
	print('Agrupando colores en '+str(nclusters)+' clusters con índice de dispersión '+str(fuzz_idx)+', α = '+str(alpha)+' y β = '+str(beta)+'...\n')
	centroids, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(dataset, nclusters, fuzz_idx, 0.005, 100)
	print('Done\n')
	return centroids.T, u_orig

def get_G1andG2(nclusters, colors_CB, fuzz_idx, alpha, beta):

	if np.shape(colors_CB) == (3,0):
		raise ValueError('Todos los colores de la imagen se perciben correctamente. No se procederá a agrupar.\n')

	else:
		print('Obteniendo G1 a partir de O1...\n')
		centroids_no_ori, u_orig_no_ori = get_clustered_model(nclusters, colors_CB, fuzz_idx, alpha, beta)
		# print('Obteniendo G2 a partir de O2...\n')
		# centroids_yes_ori, u_orig_yes_ori= get_clustered_model(nclusters, O_2, fuzz_idx)

	return centroids_no_ori, u_orig_no_ori

def simulate_centroids(centroids, defecto, opcion):

	print('Simulando centroides...\n')
	print('Simulación de centroides de '+defecto +'según'+opcion+'...\n')
	if defecto == 'protanopia':
		if opcion == 'Melillo':
			centroids_sim = np.dot(process.LMS2RGB_Melillo(),process.protanopia_Melillo())
			centroids_sim = np.dot(centroids_sim,process.RGB2LMS_Melillo())
			centroids_sim = np.dot(centroids_sim,centroids)
		elif opcion == 'Vienot':
			centroids_sim = np.dot(process.LMS2RGB_Vienot(),process.protanopia_Vienot())
			centroids_sim = np.dot(centroids_sim,process.RGB2LMS_Vienot())
			centroids_sim = np.dot(centroids_sim,centroids)
		else:
			centroids_sim = np.dot(process.protanopia_Machado(),centroids)

	elif defecto == 'deuteranopia':
		if opcion == 'Melillo':
			centroids_sim = np.dot(process.LMS2RGB_Melillo(),process.deuteranopia_Melillo())
			centroids_sim = np.dot(centroids_sim,process.RGB2LMS_Melillo())
			centroids_sim = np.dot(centroids_sim,centroids)
		elif opcion == 'Vienot':
			centroids_sim = np.dot(process.LMS2RGB_Vienot(),process.deuteranopia_Vienot())
			centroids_sim = np.dot(centroids_sim,process.RGB2LMS_Vienot())
			centroids_sim = np.dot(centroids_sim,centroids)
		else:
			centroids_sim = np.dot(process.deuteranopia_Machado(),centroids)

	elif defecto == 'tritanopia':
		if opcion == 'Vienot':
			raise ValueError('Vienot no realiza simulación de tritanopia. No se procederá a recolorear.\n')
		elif opcion == 'Melillo':
			centroids_sim = np.dot(process.LMS2RGB_Melillo(),process.tritanopia_Melillo())
			centroids_sim = np.dot(centroids_sim,process.RGB2LMS_Melillo())
			centroids_sim = np.dot(centroids_sim,centroids)
		else:
			centroids_sim = np.dot(process.tritanopia_Machado(),centroids)

	else:
		raise ValueError('Defecto desconocido. No se simularon correctamente los centroides.\n')
	print('\n\tCentroides simulados correctamente.\n')
	return centroids_sim


def w(centroids_ori, centroids_sim):

	print('Calculando cantidad de diferencia de color entre centroides de la imagen original y tras aplicar matrices de simulación...\n')
	x = np.shape(centroids_ori)
	y = np.shape(centroids_sim)

	assert x == y, 'Error, tiene que haber el mismo número de clusters¡¡¡'
	c_list = []
	for centroide in range(x[1]):
		c = abs(centroids_ori[0,centroide]-centroids_sim[0,centroide])+abs(centroids_ori[1,centroide]-centroids_sim[1,centroide])+abs(centroids_ori[2,centroide]-centroids_sim[2,centroide])
		c_list.append(c)
	print('\n\tCantidad de diferencia de color obtenida con éxito.\n')
	return c_list

def get_rotate_angle(w, centroids_no_ori, defecto, alpha, beta):

	print('Calculando ángulo de rotación..\n')
	theta_ov_list = []
	theta_rv_list = []
	theta_pv_list = []

	for centroide in range(np.shape(centroids_no_ori)[1]):
		if defecto == 'protanopia' or defecto == 'deuteranopia':
			theta_ov = np.arctan(centroids_no_ori[2,centroide]/np.sqrt(np.power(centroids_no_ori[0,centroide],2)+np.power(centroids_no_ori[1,centroide],2)))
		elif defecto == 'tritanopia':
			theta_ov = np.arctan(centroids_no_ori[1,centroide]/np.sqrt(np.power(centroids_no_ori[0,centroide],2)+np.power(centroids_no_ori[2,centroide],2)))
		else:
			raise ValueError('Defecto desconocido. Ángulo de rotación no calculado.\n')

		theta_ov_list.append(theta_ov)
		theta_rv = 0.5*math.pi - theta_ov
		theta_rv_list.append(theta_rv)
		theta_pv = np.deg2rad(np.trunc((alpha*w[centroide]+beta)*np.rad2deg(theta_rv)))
		theta_pv_list.append(theta_pv)

	print('\n\tÁngulo de rotación calculado correctamente.\n')
	return theta_pv_list

def get_P_matrix(theta_pv_list, centroids_no_ori, defecto):

	print('Calculando matriz de rotación P...\n')
	P = np.zeros((3,3,np.shape(centroids_no_ori)[1]))
	if defecto == 'protanopia': # ROTACIÓN HACIA AZUL
		eje_giro = np.zeros((3,np.shape(centroids_no_ori)[1]))
		eje_giro[0,:] = centroids_no_ori[1,:]
		eje_giro[1,:] = -centroids_no_ori[0,:]

	elif defecto == 'deuteranopia': # ROTACIÓN HACIA AZUL
		"""eje_giro = np.zeros((3,np.shape(centroids_no_ori)[1])) # ROTACION HACIA ROJO
		eje_giro[1,:] = centroids_no_ori[2,:]
		eje_giro[2,:] = -centroids_no_ori[1,:] """
		eje_giro = np.zeros((3,np.shape(centroids_no_ori)[1]))
		eje_giro[0,:] = centroids_no_ori[1,:]
		eje_giro[1,:] = -centroids_no_ori[0,:]

	elif defecto == 'tritanopia': # ROTACIÓN HACIA VERDE
		eje_giro = np.zeros((3,np.shape(centroids_no_ori)[1]))
		eje_giro[0,:] = -centroids_no_ori[2,:]
		eje_giro[2,:] = centroids_no_ori[0,:] 

	else:
		raise ValueError('Defecto desconocido. Matriz P no hallada.\n')

	for centroide in range(np.shape(centroids_no_ori)[1]):
		q0 = np.cos(theta_pv_list[centroide]/2)
		q1 = np.sin(theta_pv_list[centroide]/2)*eje_giro[0,centroide]/np.linalg.norm(eje_giro[:,centroide])
		q2 = np.sin(theta_pv_list[centroide]/2)*eje_giro[1,centroide]/np.linalg.norm(eje_giro[:,centroide])
		q3 = np.sin(theta_pv_list[centroide]/2)*eje_giro[2,centroide]/np.linalg.norm(eje_giro[:,centroide])
		# a = np.sqrt(np.power(q0,2)+np.power(q1,2)+np.power(q2,2)+np.power(q3,2))

		P[0,0,centroide] = 1-2*(np.power(q2,2)+np.power(q3,2))
		P[0,1,centroide] = 2*(q1*q2-q0*q3)
		P[0,2,centroide] = 2*(q0*q2+q1*q3)

		P[1,0,centroide] = 2*(q1*q2+q0*q3)
		P[1,1,centroide] = 1-2*(np.power(q1,2)+np.power(q3,2))
		P[1,2,centroide] = 2*(q2*q3-q0*q1)

		P[2,0,centroide] = 2*(q1*q3-q0*q2)
		P[2,1,centroide] = 2*(q0*q1+q2*q3)
		P[2,2,centroide] =1-2*(np.power(q1,2)+np.power(q2,2))

	print('\n\tMatriz P obtenida correctamente.\n')
	return P


def get_rotated_centroid(P,centroids_no_ori):

	print('Obteniendo centroides rotados...\n')
	centroids_rotated_ori = np.zeros((3,np.shape(centroids_no_ori)[1]))

	for centroide in range(np.shape(centroids_no_ori)[1]):
		centroids_rotated_ori[:,centroide] = np.dot(P[:,:,centroide], centroids_no_ori[:,centroide])

	np.clip(centroids_rotated_ori, 0, 1, out = centroids_rotated_ori)
	# centroids_rotated_ori[0:2,:] = centroids_no_ori[0:2,:]
	print('Rotación realizada con éxito.\n')
	return centroids_rotated_ori


def print_clustered_model(clusters, centroids, u_orig, dataset):
	# centroids, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(dataset, clusters, 2.0, 0.005, 100)

	print('Representando modelo 3D de clustering...\n')
	dataset = process.renorm(dataset)
	centroids = process.renorm(centroids)
	u_orig = process.renorm(u_orig)
	_, ax = plt.subplots()
	ax = plt.axes(projection = '3d')
	for i in range(clusters):
		# Representamos todos los puntos pertenecientes a cada cluster de un color
		ax.plot(dataset[0, u_orig.argmax(axis=0) == i], dataset[1, u_orig.argmax(axis=0) == i], dataset[2, u_orig.argmax(axis=0) == i], '.',label='Cluster ' + str(i+1), markersize = 2)
		# Representamos cada centroide del cluster
		ax.scatter(centroids[0,i], centroids[1,i], centroids[2,i], c = 'black',marker = 'x', label = 'Centroid '+str(i+1))
		# ax.legend(loc = 'best', ncol = int(2.0), shadow = 'True', fontsize = 'small')
	ax.set_xlabel('RED')
	ax.set_ylabel('GREEN')
	ax.set_zlabel('BLUE')
	ax.set_xlim(0,255)
	ax.set_ylim(0,255)
	ax.set_zlim(0,255)
	print('Done\n')    
	plt.show()


def get_recolored_image(reshaped_yes, reshaped_no,u_no,positions,centroids, dimensiones_ori):

	"""
	fig, ax = plt.subplots()
	_, u= get_clustered_model(16, image)
	# image = process.renorm(image)
	# u = process.renorm(u)
	# centroids = process.renorm(centroids)
	# new_ima = np.zeros((np.shape(ima)[0], np.shape(ima)[1]))

	ax = plt.axes(projection = '3d')

	for i in range(np.shape(centroids)[1]):
	ax.plot(image[0, u.argmax(axis=0) == i], image[1, u.argmax(axis=0) == i], image[2, u.argmax(axis=0) == i], '.',label='Cluster ' + str(i+1), markersize = 2, c = centroids[:,i])

	plt.show()
	"""
	print('Obteniendo nueva imagen recoloreada...\n')
	ima_rotated = np.zeros((dimensiones_ori))

	etiquetas = np.argmax(u_no, axis=0)  # Hardening for visualization

	for i in range(np.shape(positions)[0]):
		ima_rotated[positions[i,0],positions[i,1],:] = centroids[:,etiquetas[i]]

	ima_rotated = process.renorm(ima_rotated)
	reshaped_yes = process.renorm(reshaped_yes)
	new_ima = ima_rotated+reshaped_yes
	print('********IMAGEN RECOLOREADA OBTENIDA CON ÉXITO********.\n')

	return new_ima

def recolor(nombre_ima,ima_original, ima_observed, defecto, opcion, alpha,beta,nclusters, fuzz_idx):

	ima_original = process.norm(ima_original)
	ima_observed = process.norm(ima_observed)

	theta_1 = get_theta(ima_original, ima_observed, 'PSI_1', defecto)
	gamma = get_gamma_color(ima_original, defecto)
	_, _, colors_CB_no, _, no,_, image_yes, image_no = get_O_1andO_2(theta_1, gamma, ima_original, defecto, nombre_ima)
	centroids_no_ori, u_orig_no_ori = get_G1andG2(nclusters, colors_CB_no, fuzz_idx=fuzz_idx, alpha=alpha, beta = beta)
	centroids_no_sim = simulate_centroids(centroids_no_ori, defecto, opcion)
	w_ = w(centroids_no_ori,centroids_no_sim)
	theta_pv = get_rotate_angle(w_, centroids_no_ori, defecto, alpha, beta)
	P = get_P_matrix(theta_pv, centroids_no_ori, defecto)
	centroids_rotated_ori = get_rotated_centroid(P, centroids_no_ori)
	"""centroids_rotated_sim = simulate_centroids(centroids_rotated_ori,defecto, opcion)
	discriminacion = get_theta(centroids_rotated_sim, centroids_yes_ori, 'PSI_2')
	if np.any(discriminacion) == 1:
		print('Recoloreamos de nuevo\n')
		recolor(nombre_ima,ima_original, ima_observed, defecto, opcion)

	else:
		new_ima = get_recolored_image(image_yes, image_no, u_orig_no_ori, no, centroids_rotated_ori, np.shape(image_yes))"""

	new_ima = get_recolored_image(image_yes, image_no, u_orig_no_ori, no, centroids_rotated_ori, np.shape(image_yes))

	return new_ima

	