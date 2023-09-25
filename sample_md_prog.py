# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 07:52:31 2022

@author: SUCHETA GHOSH
"""

import matplotlib.pyplot as plt
import numpy as np
import math

def init_pos(npart):#Assigning initial position 
  a=np.zeros((npart,3))
  k=0
  n=6
  for z in range(n):
    for i in range(n):
      for j in range(n):
        if (k< npart):
          a[k][0]=i*1.1 #sigma=1.1
          a[k][1]=(j)*1.1
          a[k][2]=z*1.1
          k+=1
          if(z<n-1 and j<n-1):
            a[k][0]=i*1.1
            a[k][1]=0.55+j*1.1
            a[k][2]=0.55+z*1.1
            k+=1
          if (i<n-1 and j<n-1):
            a[k][0]=0.55+i*1.1
            a[k][1]=0.55+j*1.1
            a[k][2]=z*1.1
            k+=1
          if (i<n-1 and z<n-1):
            a[k][0]=0.55+i*1.1
            a[k][1]=j*1.1
            a[k][2]=0.55+z*1.1
            k+=1
        else:
          break
  return a

def init_vel(pos,npart,temp,dt):#Assigning initial velocity 
  vel= np.random.rand(npart,3)-0.5
  posm= np.zeros((npart,3))
  sumvx=sumvy=sumvz=0
  sumvx2=sumvy2=sumvz2=0
  for i in range (npart):
    sumvx = sumvx + vel[i][0]
    sumvy = sumvy + vel[i][1]
    sumvz = sumvz + vel[i][2]
    sumvx2 = sumvx2 + vel[i][0]**2
    sumvy2 = sumvy2 + vel[i][1]**2
    sumvz2 = sumvz2 + vel[i][2]**2

  sumvx=sumvx/npart
  sumvy=sumvy/npart
  sumvz=sumvz/npart
  sumvx2=sumvx2/npart
  sumvy2=sumvy2/npart
  sumvz2=sumvz2/npart

  #scale factors for velocities
  fsx=math.sqrt(3*temp/sumvx2)
  fsy=math.sqrt(3*temp/sumvy2)
  fsz=math.sqrt(3*temp/sumvz2)

  for i in range (npart):
    vel[i][0]=(vel[i][0]-sumvx)*fsx
    vel[i][1]=(vel[i][1]-sumvy)*fsy
    vel[i][2]=(vel[i][2]-sumvz)*fsz
    posm[i][0]=pos[i][0] - (vel[i][0]*dt)
    posm[i][1]=pos[i][1] - (vel[i][1]*dt)
    posm[i][2]=pos[i][2] - (vel[i][2]*dt)

  return vel,posm

def force(pos,npart,box_len,rc,Vc):#incorporating the interactions between the particles
    pnrg=0
    f = np.zeros((npart,3))
    fij=V=fij=Vm=0
    for i in range (npart-1):
        for j in range(i+1, npart):
          dx=pos[i][0]-pos[j][0]
          dy=pos[i][1]-pos[j][1]
          dz=pos[i][2]-pos[j][2]
          if abs(dx)> (box_len/2):
              dx=(box_len-abs(dx))*(-dx)/(abs(dx))
          if abs(dy)> (box_len/2):
              dy=(box_len-abs(dy))*(-dy)/(abs(dy))
          if abs(dz)> (box_len/2):
              dz=(box_len-abs(dz))*(-dz)/(abs(dz))
          r = math.sqrt((dx**2)+(dy**2)+(dz**2))
          if r<=rc:
            fij = (24/r**7)*((2/r**6) - 1)
            f[i][0]+=fij*(dx/r)
            f[i][1]+=fij*(dy/r)
            f[i][2]+=fij*(dz/r)
            f[j][0]-=fij*(dx/r)
            f[j][1]-=fij*(dy/r)
            f[j][2]-=fij*(dz/r)
            V= 4*((1/r**12)-(1/r**6))
            Vm=V-Vc
            pnrg+=Vm
    return f, pnrg

def integrate(pos,posm,f,dt,box_len,npart):#Assigning the update to the position and velocity of the fcc lattice particles.
  knrg=sumv2=0
  posf= np.zeros((npart,3))
  for i in range (npart):
    posf[i][0]= 2*pos[i][0] - posm[i][0] + (f[i][0]*dt*dt)
    posf[i][1]= 2*pos[i][1] - posm[i][1] + (f[i][1]*dt*dt)
    posf[i][2]= 2*pos[i][2] - posm[i][2] + (f[i][2]*dt*dt)
    if posf[i][0]>box_len:
      posf[i][0]-=box_len
    elif posf[i][0]<0:
      posf[i][0]+=box_len
    if posf[i][1]>box_len:
      posf[i][1]-=box_len
    elif posf[i][1]<0:
      posf[i][1]+=box_len
    if posf[i][2]>box_len:
      posf[i][2]-=box_len
    elif posf[i][2]<0:
      posf[i][2]+=box_len
    vel[i][0]= (posf[i][0]-posm[i][0])/(2*dt)
    vel[i][1]= (posf[i][1]-posm[i][1])/(2*dt)
    vel[i][2]= (posf[i][2]-posm[i][2])/(2*dt)
    sumv2 +=((vel[i][0])**2+(vel[i][1])**2+(vel[i][2])**2)
  knrg= 0.5*sumv2
  temp=sumv2/(3*npart)
  posm=pos
  pos=posf
  return posm,pos,vel,knrg,temp

#The Main simulation part 

temp=3  #The temperature of the system at hand 
dt=0.0009  #The timestep that has been considered
npart=500
box_len=6.6
rc=box_len/2.5 #The parameters chosen are sigma=1.0, m=1, epsilon=1
itr=500
Vc=(4/rc**6)*((1/rc**6)-1)
velocity=np.zeros((npart,1))
part_no=np.zeros((npart,1))
a = init_pos(npart)
outputFile1 = open('initial_position_fcc.xyz','w')
outputFile1.write(str(npart)+"\n")
outputFile1.write('\n')
for i in range(npart):
    outputFile1.write('He'+"  "+str(a[i][0])+"  "+str(a[i][1])+"  "+str(a[i][2])+'\n')
outputFile1 = open('initial_velocity_coordinate_fcc.xyz','w')
outputFile1.write(str(npart)+"\n")
outputFile1.write('\n')
outputFile2 = open('initial_velocity_fcc.xyz','w')
outputFile2.write(str(npart)+"\n")
outputFile2.write('\n')
vel,am= init_vel(a,npart,temp,dt)
for i in range(npart):
    velocity[i] = math.sqrt(((vel[i][0])**2)+((vel[i][0])**2)+((vel[i][2])**2))
    outputFile1.write('He'+"  "+str(vel[i][0])+"  "+str(vel[i][1])+"  "+str(vel[i][2])+'\n\n')
    outputFile2.write('He'+"  "+str(velocity[i])+'\n\n')
    part_no[i]=i
plt.figure(5,figsize=(18, 6))
plt.xlabel('npart')
plt.ylabel('velocity')
plt.title('velocity curve')
plt.scatter(part_no[:],velocity[:],s=10)
plt.show() 
p_enrg = np.zeros((itr,1))
k_enrg = np.zeros((itr,1))
tot_enrg = np.zeros((itr,1))
temp_step = np.zeros((itr,1))
itrc= np.zeros((itr,1))
outputFile1 = open('data.txt','w')
outputFile1.write(str(npart)+"\n")
outputFile1.write('\n')
for i in range (itr):
  f, pnrg = force(a, npart, box_len, rc,Vc)
  am,a,vel,knrg,tempt=integrate(a,am,f,dt,box_len,npart)
  temp_step[i]=tempt
  p_enrg[i]=pnrg/npart
  k_enrg[i]=knrg/npart
  tot_enrg[i]=p_enrg[i]+k_enrg[i] 
  itrc[i]=i
  print
  print(str(itrc[i])+"     "+str(p_enrg[i])+"   "+str(k_enrg[i])+"   "+str(tot_enrg[i])+"   "+str(temp_step[i]))
  outputFile1.write(str(itrc[i])+"     "+str(p_enrg[i])+"   "+str(k_enrg[i])+"   "+str(tot_enrg[i])+"   "+str(temp_step[i])+'\n')
plt.figure(1,figsize=(18, 6))
plt.xlabel('iteration')
plt.ylabel('PE')
plt.title('PE curve')
plt.scatter(itrc[:],p_enrg[:],s=10)
plt.show() 
plt.figure(2,figsize=(18, 6))
plt.xlabel('iteration')
plt.ylabel('KE')
plt.title('KE curve')
plt.scatter(itrc[:],k_enrg[:],s=10)
plt.show() 
plt.figure(3,figsize=(18, 6))
plt.xlabel('iteration')
plt.ylabel('TE')
plt.title('TE curve')
plt.scatter(itrc[:],tot_enrg[:],s=10)
plt.show()
plt.figure(4,figsize=(18, 6))
plt.xlabel('iteration')
plt.ylabel('Temperture')
plt.title('Temperature curve')
plt.scatter(itrc[:],temp_step[:],s=10)
plt.show()  

