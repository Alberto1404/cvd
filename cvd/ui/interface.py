from tkinter import *
from tkinter import filedialog,messagebox
import PIL
from PIL import ImageTk,Image
import process
import recolor_fcm
import recolor_melillo
import numpy as np
import matplotlib.pyplot as plt
import imageio
import re

def validate_float(string):
	regex = re.compile(r"[0-9.]*$")
	result = regex.match(string)
	return (string == ""
			or (string.count('.') <= 1
				and result is not None
				and result.group(0) != ""))

def validate_int(string):
	regex = re.compile(r"[0-9]*$")
	result = regex.match(string)
	return (string == ""
			or (result is not None
				and result.group(0) != ""))

def on_validate_float(P):
	return validate_float(P)  

def on_validate_int(P):
	return validate_int(P)  

def all_children (window) :
	_list = window.winfo_children()

	for item in _list :
		if item.winfo_children() :
			_list.extend(item.winfo_children())

	return _list

def forget_window():
	widget_list = all_children(root)
	for item in widget_list:
		item.grid_forget()

def clear_window():
	widget_list = all_children(root)
	for item in widget_list:
		item.destroy()

def Melillo(flag, alpha, beta, n_clusters, fuzz_idx):
	global defecto
	global opcion
	global ima

	opcion = 'Melillo'

	if flag == 1: # SIMULACIÓN
		ima_defecto = process.process_image(np.asarray(ima),defecto,opcion)

		plt.figure()
		plt.subplot(1,2,1)
		plt.imshow(np.asarray(ima))
		plt.axis('off')
		plt.title('Imagen original')
		plt.subplot(1,2,2)
		plt.imshow(ima_defecto)
		plt.axis('off')
		plt.title('Visualización de '+defecto)
		plt.show()
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_ori/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+"_"+opcion+"_"+defecto+".jpg", ima_defecto)
		menu_principal()
	
	elif flag == 2: # RECOLOR JEONG

		ima = process.ratio_image(np.asarray(ima))

		ima_defecto = process.process_image(np.asarray(ima),defecto,opcion)
		ima_recolored = recolor_fcm.recolor(root.filename.split('/')[-1].split('.')[0],np.asarray(ima), ima_defecto, defecto, opcion, alpha = alpha, beta = beta, nclusters=n_clusters, fuzz_idx=fuzz_idx)
		# Guardamos imagen recoloreada y su correspondiente simulación
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/recolored/jeong/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_recolor_J'+"_"+opcion+"_"+defecto+".jpg", ima_recolored)
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_rec/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_sim_recolor_J'+"_"+opcion+"_"+defecto+".jpg",process.process_image(ima_recolored,defecto, opcion))

		plt.figure()
		plt.subplot(2,2,1)
		plt.imshow(np.asarray(ima))
		plt.axis('off')
		plt.title('Imagen original')
		plt.subplot(2,2,3)
		plt.imshow(ima_defecto)
		plt.axis('off')
		plt.title('Simulación de '+defecto)
		plt.subplot(2,2,2)
		plt.imshow(ima_recolored)
		plt.axis('off')
		plt.title('Imagen recoloreada')
		plt.subplot(2,2,4)
		plt.imshow(process.process_image(ima_recolored,defecto, opcion))
		plt.axis('off')
		plt.title('Simulación del recoloreado')
		plt.show()
		menu_principal()

	else:
		# RECOLOR MELILLO
		ima_defecto = process.process_image(np.asarray(ima),defecto,opcion)
		ima_recolored = recolor_melillo.recolor_melillo(np.asarray(ima), defecto)
		# Guardamos imagen recoloreada y su correspondiente simulación
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/recolored/melillo/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_recolor_M'+"_"+opcion+"_"+defecto+".jpg", ima_recolored)
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_rec/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_sim_recolor_M'+"_"+opcion+"_"+defecto+".jpg", process.process_image(ima_recolored,defecto, opcion))

		plt.figure()
		plt.subplot(2,2,1)
		plt.imshow(np.asarray(ima))
		plt.axis('off')
		plt.title('Imagen original')
		plt.subplot(2,2,3)
		plt.imshow(ima_defecto)
		plt.axis('off')
		plt.title('Simulación de '+defecto)
		plt.subplot(2,2,2)
		plt.imshow(ima_recolored)
		plt.axis('off')
		plt.title('Imagen recoloreada')
		plt.subplot(2,2,4)
		plt.imshow(process.process_image(ima_recolored,defecto, opcion))
		plt.axis('off')
		plt.title('Simulación del recoloreado')
		plt.show()
		menu_principal()

def Vienot(flag, alpha, beta, n_clusters, fuzz_idx):
	global defecto
	global ima
	global opcion

	opcion = 'Vienot'
	if flag == 1: # SIMULACIÓN
		ima_defecto = process.process_image(np.asarray(ima),defecto,opcion)

		plt.figure()
		plt.subplot(1,2,1)
		plt.imshow(np.asarray(ima))
		plt.axis('off')
		plt.title('Imagen original')
		plt.subplot(1,2,2)
		plt.imshow(ima_defecto)
		plt.axis('off')
		plt.title('Visualización de '+defecto)
		plt.show()
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_ori/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+"_"+opcion+"_"+defecto+".jpg", ima_defecto)
		menu_principal()
	
	elif flag == 2: # RECOLOR JEONG
		ima = process.ratio_image(np.asarray(ima))

		ima_defecto = process.process_image(np.asarray(ima),defecto,opcion)
		ima_recolored = recolor_fcm.recolor(root.filename.split('/')[-1].split('.')[0],np.asarray(ima), ima_defecto, defecto, opcion, alpha = alpha, beta=beta, nclusters=n_clusters, fuzz_idx=fuzz_idx)
		# Guardamos imagen recoloreada y su correspondiente simulación
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/recolored/jeong/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_recolor_J'+"_"+opcion+"_"+defecto+".jpg", ima_recolored)
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_rec/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_sim_recolor_J'+"_"+opcion+"_"+defecto+".jpg",process.process_image(ima_recolored,defecto, opcion))

		plt.figure()
		plt.subplot(2,2,1)
		plt.imshow(np.asarray(ima))
		plt.axis('off')
		plt.title('Imagen original')
		plt.subplot(2,2,3)
		plt.imshow(ima_defecto)
		plt.axis('off')
		plt.title('Simulación de '+defecto)
		plt.subplot(2,2,2)
		plt.imshow(ima_recolored)
		plt.axis('off')
		plt.title('Imagen recoloreada')
		plt.subplot(2,2,4)
		plt.imshow(process.process_image(ima_recolored,defecto, opcion))
		plt.axis('off')
		plt.title('Simulación del recoloreado')
		plt.show()
		menu_principal()

	else:
		# RECOLOR MELILLO
		ima_defecto = process.process_image(np.asarray(ima),defecto,opcion)
		ima_recolored = recolor_melillo.recolor_melillo(np.asarray(ima), defecto)
		# Guardamos imagen recoloreada y su correspondiente simulación
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/recolored/melillo/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_recolor_M'+"_"+opcion+"_"+defecto+".jpg", ima_recolored)
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_rec/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_sim_recolor_M'+"_"+opcion+"_"+defecto+".jpg", process.process_image(ima_recolored,defecto, opcion))

		plt.figure()
		plt.subplot(2,2,1)
		plt.imshow(np.asarray(ima))
		plt.axis('off')
		plt.title('Imagen original')
		plt.subplot(2,2,3)
		plt.imshow(ima_defecto)
		plt.axis('off')
		plt.title('Simulación de '+defecto)
		plt.subplot(2,2,2)
		plt.imshow(ima_recolored)
		plt.axis('off')
		plt.title('Imagen recoloreada')
		plt.subplot(2,2,4)
		plt.imshow(process.process_image(ima_recolored,defecto, opcion))
		plt.axis('off')
		plt.title('Simulación del recoloreado')
		plt.show()
		menu_principal()

def Machado(flag, alpha, beta, n_clusters, fuzz_idx):
	global defecto
	global ima 
	global opcion

	opcion = 'Machado'

	if flag == 1: # SIMULACIÓN
		ima_defecto = process.process_image(np.asarray(ima),defecto,opcion)

		plt.figure()
		plt.subplot(1,2,1)
		plt.imshow(np.asarray(ima))
		plt.axis('off')
		plt.title('Imagen original')
		plt.subplot(1,2,2)
		plt.imshow(ima_defecto)
		plt.axis('off')
		plt.title('Visualización de '+defecto)
		plt.show()
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_ori/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+"_"+opcion+"_"+defecto+".jpg", ima_defecto)
		menu_principal()
	
	elif flag == 2: # RECOLOR JEONG
		ima = process.ratio_image(np.asarray(ima))

		ima_defecto = process.process_image(np.asarray(ima),defecto,opcion)
		ima_recolored = recolor_fcm.recolor(root.filename.split('/')[-1].split('.')[0],np.asarray(ima), ima_defecto, defecto, opcion, alpha = alpha, beta=beta, nclusters=n_clusters, fuzz_idx=fuzz_idx)
		# Guardamos imagen recoloreada y su correspondiente simulación
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/recolored/jeong/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_recolor_J'+"_"+opcion+"_"+defecto+".jpg", ima_recolored)
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_rec/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_sim_recolor_J'+"_"+opcion+"_"+defecto+".jpg",process.process_image(ima_recolored,defecto, opcion))

		plt.figure()
		plt.subplot(2,2,1)
		plt.imshow(np.asarray(ima))
		plt.axis('off')
		plt.title('Imagen original')
		plt.subplot(2,2,3)
		plt.imshow(ima_defecto)
		plt.axis('off')
		plt.title('Simulación de '+defecto)
		plt.subplot(2,2,2)
		plt.imshow(ima_recolored)
		plt.axis('off')
		plt.title('Imagen recoloreada')
		plt.subplot(2,2,4)
		plt.imshow(process.process_image(ima_recolored,defecto, opcion))
		plt.axis('off')
		plt.title('Simulación del recoloreado')
		plt.show()
		menu_principal()

	else:
		ima_defecto = process.process_image(np.asarray(ima),defecto,opcion)
		ima_recolored = recolor_melillo.recolor_melillo(np.asarray(ima), defecto)
		# Guardamos imagen recoloreada y su correspondiente simulación
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/recolored/melillo/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_recolor_M'+"_"+opcion+"_"+defecto+".jpg", ima_recolored)
		imageio.imwrite("/Users/alberto/Documents/PracticasTSV/TFG/simulated_rec/"+opcion+'/'+ root.filename.split('/')[-1].split('.')[0]+'_sim_recolor_M'+"_"+opcion+"_"+defecto+".jpg", process.process_image(ima_recolored,defecto, opcion))

		plt.figure()
		plt.subplot(2,2,1)
		plt.imshow(np.asarray(ima))
		plt.axis('off')
		plt.title('Imagen original')
		plt.subplot(2,2,3)
		plt.imshow(ima_defecto)
		plt.axis('off')
		plt.title('Simulación de '+defecto)
		plt.subplot(2,2,2)
		plt.imshow(ima_recolored)
		plt.axis('off')
		plt.title('Imagen recoloreada')
		plt.subplot(2,2,4)
		plt.imshow(process.process_image(ima_recolored,defecto, opcion))
		plt.axis('off')
		plt.title('Simulación del recoloreado')
		plt.show()
		menu_principal()

def protanopia(flag, alpha, beta, n_clusters, fuzz_idx):
	# PROTANOPIA CASE
	global protanopia_def
	global deuteranopia_def
	global tritanopia_def
	global texto
	global defecto
	global espacio
	global btn_menu
	global intro

	defecto = 'protanopia'
	if flag == 1:
		# SIMULACIÓN CASE
		"""protanopia_def.grid_forget()
		deuteranopia_def.grid_forget()
		tritanopia_def.grid_forget()"""
		forget_window()

		texto = Label(root, text = 'Seleccione ahora la simulación a aplicar según el autor')
		texto.grid(row = 0, column = 1, columnspan = 3)
		Melillo_option = Button(root, text = 'Melillo', command = lambda:Melillo(1,'','','',''))
		Vienot_option = Button(root, text = 'Vienot', command = lambda:Vienot(1,'','','',''))
		Machado_option = Button(root, text = 'Machado', command = lambda:Machado(1,'','','',''))
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')
		espacio = Label(root, text = '\t')
		intro = Label(root, text = '\n\n')

		espacio.grid(row = 0, column = 0)
		Melillo_option.grid(row = 1, column = 1, columnspan = 3)
		Vienot_option.grid(row = 2, column = 1, columnspan = 3)
		Machado_option.grid(row = 3, column = 1, columnspan = 3)
		intro.grid(row = 4, column = 1)
		btn_menu.grid(row = 5, column = 1, columnspan = 3)

	elif flag == 2:
		# RECOLOR JEONG CASE
		"""protanopia_def.grid_forget()
		deuteranopia_def.grid_forget()
		tritanopia_def.grid_forget()"""
		if alpha >=1 or alpha < 0 or beta >=1 or beta < 0:
			messagebox.showinfo('Aviso', 'Valores de α y β sin sentido')
		
		forget_window()

		texto = Label(root, text = 'Seleccione ahora la simulación a aplicar según el autor')
		texto.grid(row = 0, column = 1, columnspan = 3)
		Melillo_option = Button(root, text = 'Melillo', command = lambda:Melillo(2,alpha,beta,n_clusters,fuzz_idx))
		Vienot_option = Button(root, text = 'Vienot', command = lambda:Vienot(2,alpha,beta,n_clusters,fuzz_idx))
		Machado_option = Button(root, text = 'Machado', command = lambda:Machado(2,alpha,beta,n_clusters,fuzz_idx))
		espacio = Label(root, text = '\t')
		intro = Label(root, text = '\n')
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

		espacio.grid(row = 0, column = 0)
		Melillo_option.grid(row = 1, column = 1, columnspan = 3)
		Vienot_option.grid(row = 2, column = 1, columnspan = 3)
		Machado_option.grid(row = 3, column = 1, columnspan = 3)
		intro.grid(row = 4, column = 1)
		btn_menu.grid(row = 5, column = 1, columnspan = 3)
	
	else:
		# RECOLOR MELILLO CASE
		"""protanopia_def.grid_forget()
		deuteranopia_def.grid_forget()
		tritanopia_def.grid_forget()"""
		forget_window()

		texto = Label(root, text = 'Seleccione ahora la simulación a aplicar según el autor')
		texto.grid(row = 0, column = 1, columnspan = 3)
		Melillo_option = Button(root, text = 'Melillo', command = lambda:Melillo(3,'','','',''))
		Vienot_option = Button(root, text = 'Vienot', command = lambda:Vienot(3,'','','',''))
		Machado_option = Button(root, text = 'Machado', command = lambda:Machado(3,'','','',''))
		espacio = Label(root, text = '\t')
		intro = Label(root, text = '\n')
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

		espacio.grid(row = 0, column = 0)
		Melillo_option.grid(row = 1, column = 1, columnspan = 3)
		Vienot_option.grid(row = 2, column = 1, columnspan = 3)
		Machado_option.grid(row = 3, column = 1, columnspan = 3)
		intro.grid(row = 4, column = 1)
		btn_menu.grid(row = 5, column = 1, columnspan = 3)

def deuteranopia(flag, alpha, beta, n_clusters, fuzz_idx):
	# PROTANOPIA CASE
	global protanopia_def
	global deuteranopia_def
	global tritanopia_def
	global texto
	global defecto
	global espacio
	global btn_menu
	global intro

	defecto = 'deuteranopia'
	if flag == 1:
		# SIMULACIÓN CASE
		"""protanopia_def.grid_forget()
		deuteranopia_def.grid_forget()
		tritanopia_def.grid_forget()"""
		forget_window()

		texto = Label(root, text = 'Seleccione ahora la simulación a aplicar según el autor')
		texto.grid(row = 0, column = 1, columnspan = 3)
		Melillo_option = Button(root, text = 'Melillo', command = lambda:Melillo(1,'','','',''))
		Vienot_option = Button(root, text = 'Vienot', command = lambda:Vienot(1,'','','',''))
		Machado_option = Button(root, text = 'Machado', command = lambda:Machado(1,'','','',''))
		espacio = Label(root, text = '\t')
		intro = Label(root, text = '\n')
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

		espacio.grid(row = 0, column = 0)
		Melillo_option.grid(row = 1, column = 1, columnspan = 3)
		Vienot_option.grid(row = 2, column = 1, columnspan = 3)
		Machado_option.grid(row = 3, column = 1, columnspan = 3)
		intro.grid(row = 4, column = 1)
		btn_menu.grid(row = 5, column = 1, columnspan = 3)
	
	elif flag == 2:
		# RECOLOR JEONG CASE
		"""protanopia_def.grid_forget()
		deuteranopia_def.grid_forget()
		tritanopia_def.grid_forget()"""
		forget_window()
		if alpha >=1 or alpha < 0 or beta >=1 or beta < 0:
			messagebox.showinfo('Aviso', 'Valores de α y β sin sentido')

		texto = Label(root, text = 'Seleccione ahora la simulación a aplicar según el autor')
		texto.grid(row = 0, column = 1, columnspan = 3)
		Melillo_option = Button(root, text = 'Melillo', command = lambda:Melillo(2,alpha,beta,n_clusters,fuzz_idx))
		Vienot_option = Button(root, text = 'Vienot', command = lambda:Vienot(2,alpha,beta,n_clusters,fuzz_idx))
		Machado_option = Button(root, text = 'Machado', command = lambda:Machado(2,alpha,beta,n_clusters,fuzz_idx))
		espacio = Label(root, text = '\t')
		intro = Label(root, text = '\n')
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

		espacio.grid(row = 0, column = 0)
		Melillo_option.grid(row = 1, column = 1, columnspan = 3)
		Vienot_option.grid(row = 2, column = 1, columnspan = 3)
		Machado_option.grid(row = 3, column = 1, columnspan = 3)
		intro.grid(row = 4, column = 1)
		btn_menu.grid(row = 5, column = 1, columnspan = 3)
	
	else:
		# RECOLOR MELILLO CASE
		"""protanopia_def.grid_forget()
		deuteranopia_def.grid_forget()
		tritanopia_def.grid_forget()"""
		forget_window()

		texto = Label(root, text = 'Seleccione ahora la simulación a aplicar según el autor')
		texto.grid(row = 0, column = 1, columnspan = 3)
		Melillo_option = Button(root, text = 'Melillo', command = lambda:Melillo(3,'','','',''))
		Vienot_option = Button(root, text = 'Vienot', command = lambda:Vienot(3,'','','',''))
		Machado_option = Button(root, text = 'Machado', command = lambda:Machado(3,'','','',''))
		espacio = Label(root, text = '\t')
		intro = Label(root, text = '\n')
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

		espacio.grid(row = 0, column = 0)
		Melillo_option.grid(row = 1, column = 1, columnspan = 3)
		Vienot_option.grid(row = 2, column = 1, columnspan = 3)
		Machado_option.grid(row = 3, column = 1, columnspan = 3)
		intro.grid(row = 4, column = 1)
		btn_menu.grid(row = 5, column = 1, columnspan = 3)

def tritanopia(flag, alpha, beta, n_clusters, fuzz_idx):
	# PROTANOPIA CASE
	global protanopia_def
	global deuteranopia_def
	global tritanopia_def
	global texto
	global defecto
	global espacio
	global btn_menu
	global intro

	defecto = 'tritanopia'
	if flag == 1:
		# SIMULACIÓN CASE
		"""protanopia_def.grid_forget()
		deuteranopia_def.grid_forget()
		tritanopia_def.grid_forget()"""
		forget_window()

		texto = Label(root, text = 'Seleccione ahora la simulación a aplicar según el autor')
		texto.grid(row = 0, column = 1, columnspan = 3)
		Melillo_option = Button(root, text = 'Melillo', command = lambda:Melillo(1,'','','',''))
		Vienot_option = Button(root, text = 'Vienot',state = DISABLED)
		Machado_option = Button(root, text = 'Machado', command = lambda:Machado(1,'','','',''))
		espacio = Label(root, text = '\t')
		intro = Label(root, text = '\n')
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

		espacio.grid(row = 0, column = 0)
		Melillo_option.grid(row = 1, column = 1, columnspan = 3)
		Vienot_option.grid(row = 2, column = 1, columnspan = 3)
		Machado_option.grid(row = 3, column = 1, columnspan = 3)
		intro.grid(row = 4, column = 1)
		btn_menu.grid(row = 5, column = 1, columnspan = 3)
	
	elif flag == 2:
		# RECOLOR JEONG CASE
		forget_window()
		if alpha >=1 or alpha < 0 or beta >=1 or beta < 0:
			messagebox.showinfo('Aviso', 'Valores de α y β sin sentido')

		texto = Label(root, text = 'Seleccione ahora la simulación a aplicar según el autor')
		texto.grid(row = 0, column = 1, columnspan = 3)
		Melillo_option = Button(root, text = 'Melillo', command = lambda:Melillo(2,alpha,beta,n_clusters,fuzz_idx))
		Vienot_option = Button(root, text = 'Vienot', state = DISABLED)
		Machado_option = Button(root, text = 'Machado', command = lambda:Machado(2,alpha,beta,n_clusters,fuzz_idx))
		espacio = Label(root, text = '\t')
		intro = Label(root, text = '\n')
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')
		intro = Label(root, text = '\n')

		espacio.grid(row = 0, column = 0)
		Melillo_option.grid(row = 1, column = 1, columnspan = 3)
		Vienot_option.grid(row = 2, column = 1, columnspan = 3)
		Machado_option.grid(row = 3, column = 1, columnspan = 3)
		intro.grid(row = 4, column = 1)
		btn_menu.grid(row = 5, column = 1, columnspan = 3)
	
	else:
		# RECOLOR MELILLO CASE
		"""protanopia_def.grid_forget()
		deuteranopia_def.grid_forget()
		tritanopia_def.grid_forget()"""
		forget_window()

		texto = Label(root, text = 'Seleccione ahora la simulación a aplicar según el autor')
		texto.grid(row = 0, column = 1, columnspan = 3)
		Melillo_option = Button(root, text = 'Melillo', command = lambda:Melillo(3,'','','',''))
		Vienot_option = Button(root, text = 'Vienot',state=DISABLED)
		Machado_option = Button(root, text = 'Machado', command = lambda:Machado(3,'','','',''))
		espacio = Label(root, text = '\t')
		intro = Label(root, text = '\n')
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

		espacio.grid(row = 0, column = 0)
		Melillo_option.grid(row = 1, column = 1, columnspan = 3)
		Vienot_option.grid(row = 2, column = 1, columnspan = 3)
		Machado_option.grid(row = 3, column = 1, columnspan = 3)
		intro.grid(row = 4, column = 1)
		btn_menu.grid(row = 5, column = 1, columnspan = 3)

def set_params(flag, state):
	global alpha_txt
	global beta_txt
	global cluster_txt
	global fuzz_txt
	global siguiente
	global espacio

	if (flag == 1 or flag == 3) and state != 0:
		# PROTANOPIA OR TRITANOPIA CASE AND CONTINUE TRUE
		alpha_txt.delete(0,END)
		alpha_txt.insert(0,"0.25")
		beta_txt.delete(0,END)
		beta_txt.insert(0,"0.05")
		cluster_txt.delete(0,END)
		cluster_txt.insert(0,"32")
		fuzz_txt.delete(0,END)
		fuzz_txt.insert(0,"1.1")

		if flag == 1:
			# PROTANOPIA
			siguiente.config(state = NORMAL, command = lambda:protanopia(2,float(alpha_txt.get()),float(beta_txt.get()),int(cluster_txt.get()),float(fuzz_txt.get())))
		else:
			# TRITANOPIA
			siguiente.config(state = NORMAL, command = lambda:tritanopia(2,float(alpha_txt.get()),float(beta_txt.get()),int(cluster_txt.get()),float(fuzz_txt.get())))
	
	elif (flag == 1 or flag == 3) and state == 0:
		# PROTANOPIA OR TRITANOPIA CASE AND CONTINUE FALSE
		alpha_txt.delete(0,END)
		alpha_txt.insert(0,"0.25")
		beta_txt.delete(0,END)
		beta_txt.insert(0,"0.05")
		cluster_txt.delete(0,END)
		cluster_txt.insert(0,"32")
		fuzz_txt.delete(0,END)
		fuzz_txt.insert(0,"1.1")
		siguiente.config(state = DISABLED)

	elif flag == 2 and state != 0:
		# DEUTERANOPIA CASE AND CONTINUE TRUE
		alpha_txt.delete(0,END)
		alpha_txt.insert(0,"0.37")
		beta_txt.delete(0,END)
		beta_txt.insert(0,"0.03")
		cluster_txt.delete(0,END)
		cluster_txt.insert(0,"32")
		fuzz_txt.delete(0,END)
		fuzz_txt.insert(0,"1.1")

		siguiente.config(state = NORMAL, command = lambda:deuteranopia(2,float(alpha_txt.get()),float(beta_txt.get()),int(cluster_txt.get()),float(fuzz_txt.get())))
	elif flag == 2 and state == 0:
		# DEUTERANOPIA CASE AND CONTINUE FALSE
		alpha_txt.delete(0,END)
		alpha_txt.insert(0,"0.37")
		beta_txt.delete(0,END)
		beta_txt.insert(0,"0.03")
		cluster_txt.delete(0,END)
		cluster_txt.insert(0,"32")
		fuzz_txt.delete(0,END)
		fuzz_txt.insert(0,"1.1")
		siguiente.config(state = DISABLED)

def jeong():
	global btn_sim
	global btn_rec
	global btn_exit
	global texto
	global opcion
	global defecto
	global ima
	global alpha_txt
	global beta_txt
	global cluster_txt
	global fuzz_txt
	global siguiente
	global espacio
	global btn_menu
	global intro

	forget_window()

	texto = Label(root,text = 'Selecciona el defecto de visión')
	v = IntVar()
	protanopia_def = Checkbutton(root, text = 'Protanopía', variable = v, onvalue = 1, offvalue = 0,command =lambda:set_params(1, v.get()))
	deuteranopia_def = Checkbutton(root, text = 'Deuteranopía', variable = v, onvalue = 2,offvalue = 0,command =lambda:set_params(2,v.get()))
	tritanopia_def = Checkbutton(root, text = 'Tritanopía', variable = v, onvalue = 3,offvalue = 0,command =lambda:set_params(3,v.get()))
	space = Label(root, text = '\n\n')
	alpha_label = Label(root, text = 'α')
	alpha_txt = Entry(root, width = 5, validate = 'key')
	vcmd = (alpha_txt.register(on_validate_float), '%P')
	alpha_txt.config(validatecommand=vcmd)
	alpha_info = Label(root, text = '[0-1]')
	beta_label = Label(root, text = 'β')
	beta_txt = Entry(root, width = 5, validate = 'key')
	vcmd = (beta_txt.register(on_validate_float), '%P')
	beta_txt.config(validatecommand=vcmd)
	beta_info = Label(root, text = '[0-1]')
	cluster_label = Label(root, text = 'Clusters')
	cluster_txt = Entry(root, width = 5, validate = 'key')
	vcmd = (cluster_txt.register(on_validate_int), '%P')
	cluster_txt.config(validatecommand=vcmd)
	cluster_info = Label(root, text = '[8-64]')
	fuzz_idx  = Label(root, text = 'Índice de dispersión')
	fuzz_txt = Entry(root, width = 5, validate = 'key')
	vcmd = (fuzz_txt.register(on_validate_float), '%P')
	fuzz_txt.config(validatecommand=vcmd)
	fuzz_info = Label(root, text ='[1.1-10]')
	siguiente = Button(root, text = 'Siguiente', state = DISABLED)
	espacio = Label(root, text = '\t')
	intro = Label(root, text = '\n')
	btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

	espacio.grid(row = 0, column = 0)
	texto.grid(row = 0, column = 1, columnspan = 3)
	protanopia_def.grid(row = 1, column = 1, columnspan = 3)
	deuteranopia_def.grid(row = 2, column = 1, columnspan = 3)
	tritanopia_def.grid(row = 3, column = 1, columnspan = 3)
	space.grid(row = 4, column = 1, columnspan = 3)
	alpha_label.grid(row = 5, column = 1)
	alpha_txt.grid(row = 5, column = 2)
	alpha_info.grid(row = 5, column = 3)
	beta_label.grid(row = 6, column = 1)
	beta_txt.grid(row = 6, column = 2)
	beta_info.grid(row = 6, column = 3)
	cluster_label.grid(row = 7, column = 1)
	cluster_txt.grid(row = 7, column = 2)
	cluster_info.grid(row = 7, column = 3)
	fuzz_idx.grid(row = 8, column = 1)
	fuzz_txt.grid(row = 8, column = 2)
	fuzz_info.grid(row = 8, column = 3)
	intro.grid(row = 9, column = 1)
	siguiente.grid(row = 10, column = 1)
	btn_menu.grid(row = 10, column = 2)

def melillo():
	global btn_sim
	global btn_rec
	global btn_exit
	global texto
	global opcion
	global defecto
	global ima
	global espacio
	global btn_menu
	global intro

	forget_window()

	texto = Label(root,text = 'Selecciona la opción')
	texto = Label(root,text = 'Selecciona el defecto de visión')
	protanopia_def = Button(root, text = 'Protanopía', command = lambda:protanopia(3,'','','',''))
	deuteranopia_def = Button(root, text = 'Deuteranopía', command = lambda:deuteranopia(3,'','','',''))
	tritanopia_def = Button(root, text = 'Tritanopía', command = lambda:tritanopia(3,'','','',''))
	espacio = Label(root, text = '\t')
	intro = Label(root, text = '\n\n')
	btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

	espacio.grid(row = 0, column = 0)
	texto.grid(row = 0, column = 1, columnspan = 3)
	protanopia_def.grid(row = 1, column = 1, columnspan = 3)
	deuteranopia_def.grid(row = 2, column = 1, columnspan = 3)
	tritanopia_def.grid(row = 3, column = 1, columnspan = 3)
	intro.grid(row = 4, column = 1)
	btn_menu.grid(row = 5, column = 1, columnspan = 3)

def recol():
	global btn_sim
	global btn_rec
	global btn_exit
	global texto
	global opcion
	global defecto
	global ima
	global espacio
	global btn_menu
	global intro

	root.filename = filedialog.askopenfilename(initialdir='./TFG/database/Fotos', title="Select A File", filetypes=(("jpg files", "*.jpg"),("all files", "*.*")))
	
	if root.filename == '':
		menu_principal()

	else:
		# my_label = Label(root, text=root.filename).pack() # Esto me indica el directorio de lectura
		ima = Image.open(root.filename)
		# ima_RGB = np.asarray(ima)

		"""btn_sim.grid_forget()
		btn_rec.grid_forget()
		btn_exit.grid_forget()
		label_ima.grid_forget()"""
		forget_window()

		root.title('Recoloreado')

		texto = Label(root,text = 'Selecciona el tipo de recoloreado')

		jeong_rec = Button(text = 'Jeong', command = jeong)
		melillo_rec = Button(text = 'Melillo', command = melillo)
		espacio = Label(root, text = '\t\t')
		intro = Label(root, text = '\n\n')
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

		espacio.grid(row = 0, column = 0)
		texto.grid(row = 0, column = 1, columnspan = 3)
		jeong_rec.grid(row = 1, column = 1, columnspan = 3)
		melillo_rec.grid(row = 2, column = 1, columnspan = 3)
		intro.grid(row = 3, column = 1)
		btn_menu.grid(row = 4, column = 1, columnspan = 3)

def simul():
	global btn_sim
	global btn_rec
	global btn_exit
	global texto
	global opcion
	global defecto
	global ima
	global espacio
	global btn_menu
	global intro

	root.filename = filedialog.askopenfilename(initialdir='./TFG/database/Fotos', title="Select A File", filetypes=(("jpg files", "*.jpg"),("all files", "*.*")))

	if root.filename == '':
		menu_principal()
	
	else:
		# my_label = Label(root, text=root.filename).pack() # Esto me indica el directorio de lectura
		ima = Image.open(root.filename)
		forget_window()
		root.title('Simulación')

		texto = Label(root,text = 'Selecciona el defecto de visión')
		protanopia_def = Button(root, text = 'Protanopía', command = lambda:protanopia(1,'','','',''))
		deuteranopia_def = Button(root, text = 'Deuteranopía', command = lambda:deuteranopia(1,'','','',''))
		tritanopia_def = Button(root, text = 'Tritanopía', command = lambda:tritanopia(1,'','','',''))
		espacio = Label(root, text = '\t\t')
		intro = Label(root, text = '\n\n')
		btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

		espacio.grid(row = 0, column = 0)
		texto.grid(row = 0, column = 1, columnspan = 3)
		protanopia_def.grid(row = 1, column = 1, columnspan = 3)
		deuteranopia_def.grid(row = 2, column = 1, columnspan = 3)
		tritanopia_def.grid(row = 3, column = 1, columnspan = 3)
		intro.grid(row = 4, column = 1)
		btn_menu.grid(row = 5, column = 1, columnspan = 3)

def show_info():
	messagebox.showinfo('Acerca de', 'App de simulación y recoloreado de imágenes para sujetos dicromáticos\n- Autor: Alberto Ruiz Guijosa\n')


def menu_principal():

	global btn_sim
	global btn_rec
	global btn_exit
	global texto
	global opcion
	global defecto

	root.title('Menú principal')
	
	clear_window()

	imagen = Image.open('./TFG/database/discromatopsia.jpg')
	imagen = imagen.resize((int(round(imagen.size[0]/2.2)),int(round(imagen.size[1]/2.2))), PIL.Image.ANTIALIAS)
	ima = ImageTk.PhotoImage(imagen)
	ima_btn = Button(root, image = ima, command = show_info)
	# imagen = ImageTk.PhotoImage(imagen)
	# ima_label = Label(root, image = imagen)
	# ima_label.image = imagen # KEEP THE REFERENCE!!!
	# ima_label.grid(row = 0, column = 0, columnspan = 3)
	ima_btn.image = ima # KEEP THE REFERENCE!!!
	ima_btn.grid(row = 0, column = 0, columnspan = 3)
	btn_sim = Button(root, text = 'Simulación', command = simul, height = 2)
	btn_rec = Button(root, text = 'Recoloreado', command = recol, height = 2)
	btn_exit = Button(root, text="Exit",fg="red", command=root.quit, height = 2)
	btn_sim.grid(row = 1, column = 0)
	btn_rec.grid(row = 1, column = 1)
	btn_exit.grid(row = 1, column = 2)
	opcion = ''
	defecto = ''
	
if __name__ == "__main__":
    # Primero nos creamos la ventana de inicio. 
	root = Tk()
	root.title('Menú principal')
	root.geometry('560x460')
	root.resizable(False, False)

	texto = Label(root,text = '     App de simulación y recoloreado de discromatopsia')
	texto.grid(row = 0, column = 0)
	Melillo_option = Button(root, text = 'Melillo')
	Vienot_option = Button(root, text = 'Vienot')
	Machado_option = Button(root, text = 'Machado')
	protanopia_def = Button(root, text = 'protanopía')
	deuteranopia_def = Button(root, text = 'deuteranopía')
	tritanopia_def = Button(root, text = 'tritanopía')
	Melillo_option.grid(row = 1, column = 2,rowspan = 3)
	Vienot_option.grid(row = 2, column = 2,rowspan = 3)
	Machado_option.grid(row = 3, column = 2,rowspan = 3)
	Melillo_option.grid_forget()
	Vienot_option.grid_forget()
	Machado_option.grid_forget()
	protanopia_def.grid(row = 12, column = 12,rowspan = 3)
	deuteranopia_def.grid(row = 331, column = 12,rowspan = 3)
	tritanopia_def.grid(row = 23, column = 11,rowspan = 3)
	protanopia_def.grid_forget()
	deuteranopia_def.grid_forget()
	tritanopia_def.grid_forget()
	var = StringVar()
	param = Checkbutton(root, text = 'Parametrización', variable = var, onvalue = 'yes', offvalue = 'no')
	param.deselect()
	space = Label(root, text = '\n\n')
	alpha_label = Label(root, text = 'α')
	alpha_txt = Entry(root, width = 5, state = DISABLED)
	beta_label = Label(root, text = 'β')
	beta_txt = Entry(root, width = 5, state = DISABLED)
	cluster_label = Label(root, text = 'Clusters')
	cluster_txt = Entry(root, width = 5, state = DISABLED)
	fuzz_idx  = Label(root, text = 'Índice de dispersión')
	fuzz_txt = Entry(root, width = 5)
	texto.grid(row = 0, column = 1, columnspan = 3)
	protanopia_def.grid(row = 1, column = 1, columnspan = 3)
	deuteranopia_def.grid(row = 2, column = 1, columnspan = 3)
	tritanopia_def.grid(row = 3, column = 1, columnspan = 3)
	space.grid(row = 4, column = 1, columnspan = 3)
	param.grid(row = 5, column = 1, columnspan = 3)
	alpha_label.grid(row = 6, column = 1, columnspan = 1)
	alpha_txt.grid(row = 6, column = 2, columnspan = 3)
	beta_label.grid(row = 7, column = 1, columnspan = 1)
	beta_txt.grid(row = 7, column = 2, columnspan = 3)
	cluster_label.grid(row = 8, column = 1, columnspan = 1)
	cluster_txt.grid(row = 8, column = 2, columnspan = 3)
	fuzz_idx.grid(row = 8, column = 1, columnspan = 1)
	fuzz_txt.grid(row = 8, column = 2, columnspan =3)
	siguiente = Button(root, text = 'Siguiente', state = DISABLED)
	siguiente.grid(row = 9, column = 1)
	espacio = Label(root, text = '\t')
	espacio.grid(row = 0, column = 0)
	intro = Label(root, text = '\n\n')
	btn_menu = Button(root, text = 'Menú principal', command = menu_principal, fg = 'red')

	clear_window()
	
	imagen = Image.open('./TFG/database/discromatopsia.jpg')
	imagen = imagen.resize((int(round(imagen.size[0]/2.2)),int(round(imagen.size[1]/2.2))), PIL.Image.ANTIALIAS)
	ima = ImageTk.PhotoImage(imagen)
	ima_btn = Button(root, image = ima, command = show_info)
	ima_btn.image = ima # KEEP THE REFERENCE!!!
	# imagen = ImageTk.PhotoImage(imagen)
	# ima_label = Label(root, image = imagen)
	# ima_label.image = imagen # KEEP THE REFERENCE!!!
	# ima_label.grid(row = 0, column = 0, columnspan = 3)
	# ima_btn.image = imagen # KEEP THE REFERENCE!!!
	ima_btn.grid(row = 0, column = 0, columnspan = 3)
	btn_sim = Button(root, text = 'Simulación', command = simul, height = 2)
	btn_rec = Button(root,text = 'Recoloreado', command = recol, height = 2)
	btn_exit = Button(root, text="Exit",fg="red", command=root.quit, height =2)
	btn_sim.grid(row = 1, column = 0)
	btn_rec.grid(row = 1, column = 1)
	btn_exit.grid(row = 1, column = 2)
	opcion = ''
	defecto = ''

	root.mainloop()

    





