#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:32:23 2020

@author: carlotal
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import proper

def objet_def(D,D_ratio,N):
    x = np.linspace(-D/2,D/2,N,endpoint=False) + D/(2.0*N)
    X,Y = np.meshgrid(x,x,sparse = False)
    R = np.sqrt(X**2 + Y**2) 
    x0 = 0.3*D*D_ratio
    y0 = 0.1*D*D_ratio
    objet = (R<(D*D_ratio/2))*(1-(np.sqrt((X-x0)**2+(Y-y0)**2)<D*D_ratio/2/50))
    return objet

plt.close('all')

#erreur_position = 2*1e-3 #erreur de positionnement des optiques, en mètres (FWHM d'une gausienne centrée sur la valeur vraie)
lbd = 0.55*1e-6 #longueur d'onde (m)

#focal length of lenses
foc_library = np.array([-100,50,100,200,250,500])
foc_library_rand = np.zeros(len(foc_library))
for k in range(len(foc_library)):
    foc_library_rand[k] = np.random.normal(foc_library[k], np.abs(foc_library[k]*20/100/2.355)) #students will need to measure the focal length!
#lens_rand_index = np.random.permutation(len(foc_library))
#foc_library_rand = foc_library[lens_rand_index] #randomization of the lens library: 
    
fig,ax = plt.subplots(1,3,figsize=[16,8])
plt.subplots_adjust(bottom=0.32)

# Parameters (distances in mm)
N_simu = 500
object_size_0 = 2
D_ratio = 0.8
window_size = object_size_0/D_ratio

#object_size_0 += np.random.normal(0,1e-3/2.355)

number_L1_0 = 3
number_L2_0 = 3
number_L3_0 = 3
distance_object_L1_0 = 220
distance_object_L3_0 = 200
distance_L3_L3_0 = 100
distance_L1_L2_0 = 300
distance_L2_screen_0 = 200

#delta focus (in mm)
delta_focus = 1

#initialization of function
objet = objet_def(window_size,D_ratio,N_simu)

onde_1 = proper.prop_begin(window_size*1e-3, lbd, N_simu, D_ratio)
proper.prop_multiply(onde_1, objet)
proper.prop_define_entrance(onde_1)
A_object_1 = np.abs(proper.prop_get_amplitude(onde_1))**2
A_object_1 /= np.max(A_object_1)
L_object_1 = D_ratio*N_simu*proper.prop_get_sampling(onde_1)*1e3
proper.prop_propagate(onde_1, distance_object_L1_0*1e-3)
proper.prop_lens(onde_1,foc_library_rand[number_L1_0]*1e-3)
proper.prop_propagate(onde_1, distance_L1_L2_0*1e-3)
proper.prop_lens(onde_1, foc_library_rand[number_L2_0]*1e-3)
proper.prop_propagate(onde_1, distance_L2_screen_0*1e-3)
A_screen = np.abs(proper.prop_get_amplitude(onde_1))**2
L_screen = D_ratio*N_simu*proper.prop_get_sampling(onde_1)*1e3
A_screen *= np.sum(A_object_1)/np.sum(A_screen)#*(L_object/L_screen)**2

onde_3 = proper.prop_begin(window_size*1e-3, lbd, N_simu, D_ratio)
proper.prop_multiply(onde_3, objet)
proper.prop_define_entrance(onde_3)
A_object_3 = np.abs(proper.prop_get_amplitude(onde_3))**2
A_object_3 /= np.max(A_object_3)
L_object_3 = D_ratio*N_simu*proper.prop_get_sampling(onde_3)*1e3 
proper.prop_propagate(onde_3, distance_object_L3_0*1e-3)
proper.prop_lens(onde_3,foc_library_rand[number_L3_0]*1e-3)
proper.prop_propagate(onde_3, distance_L3_L3_0*1e-3)
proper.prop_lens(onde_3, foc_library_rand[number_L3_0]*1e-3)
proper.prop_propagate(onde_3, distance_object_L3_0*1e-3)
A_collim = np.abs(proper.prop_get_amplitude(onde_3))**2
L_collim = D_ratio*N_simu*proper.prop_get_sampling(onde_3)*1e3
A_collim *= np.sum(A_object_3)/np.sum(A_collim)

#plot
l_1 = ax[0].imshow(A_object_3,extent=(-L_object_3/2,L_object_3/2,-L_object_3/2,L_object_3/2),cmap='gray',vmin=0,vmax=1.2)
ax[0].set_title('Objet physique')
l_2 = ax[1].imshow(A_screen,extent=(-L_screen/2,L_screen/2,-L_screen/2,L_screen/2),cmap='gray',vmin=0,vmax=1.2)
ax[1].set_title('Lumière sur l\'écran après L2')
l_3 = ax[2].imshow(A_collim+A_object_3,extent=(-L_collim/2,L_collim/2,-L_collim/2,L_collim/2),cmap='gray',vmin=0,vmax=2.4)
ax[2].set_title('Image autocollimation L3')
ax[2].set_axis_off()
#ax.margins(x=0)

axcolor = 'white'

ax_object_size = plt.axes([0.25, 0.1, 0.65, 0.02], facecolor=axcolor)
ax_L1_focal_length = plt.axes([0.25, 0.13, 0.65, 0.02], facecolor=axcolor)
ax_L2_focal_length = plt.axes([0.25, 0.16, 0.65, 0.02], facecolor=axcolor)
ax_L3_focal_length = plt.axes([0.25, 0.19, 0.65, 0.02], facecolor=axcolor)
ax_distance_object_L1 = plt.axes([0.25, 0.22, 0.65, 0.02], facecolor=axcolor)
ax_distance_object_L3 = plt.axes([0.25, 0.25, 0.65, 0.02], facecolor=axcolor)
ax_distance_L1_L2 = plt.axes([0.25, 0.28, 0.65, 0.02], facecolor=axcolor)
ax_distance_L2_screen = plt.axes([0.25, 0.31, 0.65, 0.02], facecolor=axcolor)

#Sliders
s_object_size = Slider(ax_object_size,'Taille objet',1,10,valinit=object_size_0,valstep = 0.5)
s_number_L1 = Slider(ax_L1_focal_length,'Numéro lentille L1',1,len(foc_library),valinit=number_L1_0,valstep=delta_focus)
s_number_L2 = Slider(ax_L2_focal_length,'Numéro lentille L2',1,len(foc_library),valinit=number_L2_0,valstep=delta_focus)
s_number_L3 = Slider(ax_L3_focal_length,'Numéro lentille L3 (autocollimation)',1,len(foc_library),valinit=number_L3_0,valstep=delta_focus)
s_distance_object_L1 = Slider(ax_distance_object_L1,'Distance Objet L1',0,600,valinit=distance_object_L1_0,valstep = 1)
s_distance_object_L3 = Slider(ax_distance_object_L3,'Distance Objet L3 (autocollimation)',0,600,valinit=distance_object_L3_0,valstep = 1)
s_distance_L1_L2 = Slider(ax_distance_L1_L2,'Distance L1 L2',0,1000,valinit=distance_L1_L2_0,valstep = 1)
s_distance_L2_screen = Slider(ax_distance_L2_screen,'Distance L2 écran',0,600,valinit=distance_L2_screen_0,valstep = 1)

def update(val):  

    n_object_size = s_object_size.val*1e-3
    n_window_size = n_object_size/D_ratio
    
    n_number_L1 = int(s_number_L1.val-1)
    n_number_L2 = int(s_number_L2.val-1)
    n_number_L3 = int(s_number_L3.val-1)
    n_distance_object_L1 = s_distance_object_L1.val*1e-3
    n_distance_object_L3 = s_distance_object_L3.val*1e-3
    n_distance_L1_L2 = s_distance_L1_L2.val*1e-3
    n_distance_L3_L3 = distance_L3_L3_0*1e-3
    n_distance_L2_screen = s_distance_L2_screen.val*1e-3
    
    n_objet = objet_def(n_window_size,D_ratio,N_simu)
    
    onde_1 = proper.prop_begin(n_window_size, lbd, N_simu, D_ratio)
    proper.prop_multiply(onde_1, n_objet)
    proper.prop_define_entrance(onde_1)
    A_object_1 = np.abs(proper.prop_get_amplitude(onde_1))**2
    A_object_1 /= np.max(A_object_1)
    #L_object_1 = D_ratio*N_simu*proper.prop_get_sampling(onde_1)*1e3
    proper.prop_propagate(onde_1, n_distance_object_L1)
    proper.prop_lens(onde_1,foc_library_rand[n_number_L1]*1e-3)
    proper.prop_propagate(onde_1, n_distance_L1_L2)
    proper.prop_lens(onde_1, foc_library_rand[n_number_L2]*1e-3)
    proper.prop_propagate(onde_1, n_distance_L2_screen)
    A_screen = np.abs(proper.prop_get_amplitude(onde_1))**2
    L_screen = D_ratio*N_simu*proper.prop_get_sampling(onde_1)*1e3
    A_screen *= np.sum(A_object_1)/np.sum(A_screen)#*(L_object/L_screen)**2
    
    onde_3 = proper.prop_begin(n_window_size, lbd, N_simu, D_ratio)
    proper.prop_multiply(onde_3, n_objet)
    proper.prop_define_entrance(onde_3)
    A_object_3 = np.abs(proper.prop_get_amplitude(onde_3))**2
    A_object_3 /= np.max(A_object_3)
    L_object_3 = D_ratio*N_simu*proper.prop_get_sampling(onde_3)*1e3 
    proper.prop_propagate(onde_3, n_distance_object_L3)
    proper.prop_lens(onde_3,foc_library_rand[n_number_L3]*1e-3)
    proper.prop_propagate(onde_3, n_distance_L3_L3)
    proper.prop_lens(onde_3, foc_library_rand[n_number_L3]*1e-3)
    proper.prop_propagate(onde_3, n_distance_object_L3)
    A_collim = np.abs(proper.prop_get_amplitude(onde_3))**2
    L_collim = D_ratio*N_simu*proper.prop_get_sampling(onde_3)*1e3
    A_collim *= np.sum(A_object_3)/np.sum(A_collim)

    l_1.set_data(A_object_3)
    l_1.set_extent((-L_object_3/2,L_object_3/2,-L_object_3/2,L_object_3/2))
    
    l_2.set_data(A_screen)
    l_2.set_extent((-L_screen/2,L_screen/2,-L_screen/2,L_screen/2))
    
    l_3.set_data(A_collim+A_object_3)
    l_3.set_extent((-L_collim/2,L_collim/2,-L_collim/2,L_collim/2))
    
    fig.canvas.draw_idle()

s_object_size.on_changed(update)
s_number_L1.on_changed(update)
s_number_L2.on_changed(update)
s_number_L3.on_changed(update)
s_distance_object_L1.on_changed(update)
s_distance_object_L3.on_changed(update)
s_distance_L1_L2.on_changed(update)
s_distance_L2_screen.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_object_size.reset()
    s_number_L1.reset()
    s_number_L2.reset()
    s_number_L3.reset()
    s_distance_object_L1.reset()
    s_distance_object_L3.reset()
    s_distance_L1_L2.reset()
    s_distance_L2_screen.reset()
button.on_clicked(reset)

# plt.show()
