# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 08:46:14 2022

@author: 
"""

import numpy as np

A=np.array([[0,0,1],[0,0,0],[4,0,0]])

Oi=[1,1,2,2,3]
Oj=[1,3,1,3,2]

Ii=[1,2,3,3]
Ij=[2,2,1,3]

O,I=[],[]

for i in range(0,5):
    O.append(A[Oi[i]-1,Oj[i]-1])    

"获取需求值"
for j in range(0,4):
    I.append(A[Ii[j]-1,Ij[j]-1])

""
#while max(O)>0 and max(I)>0:
"求输出与输入两两之间的距离，并计算最短路径"    
d=[]
di1=[]
dj1=[]


for j1 in range(0,4):
    if I[j1]==0:
        continue
    dj1.append(j1)

for i1 in range(0,5):
    if O[i1]==0:
        continue
    di1.append(i1)
    for j1 in range(0,4):
        if I[j1]==0:
            continue
        d.append(((Oi[i1]-Ii[j1])**2+(Oj[i1]-Ij[j1])**2)**0.5)
    mind1=min(d)

"记录出现最短距离的所有输入输出的行列号"

to=[]
ti=[]
for i2 in range(0,len(di1)):
    for j2 in range(0,len(dj1)):
        if ((Oi[di1[i2]]-Ii[dj1[j2]])**2+(Oj[di1[i2]]-Ij[dj1[j2]])**2)**0.5==mind1:
            to.append(di1[i2])
            ti.append(dj1[j2])

"找到需求最大的位置"

vti=[]
index2=[]

for i3 in range(0,len(ti)):
    vti.append(A[Ii[ti[i3]]-1,Ij[ti[i3]]-1])
    max1=max(vti)

for i4 in range(0,len(vti)):
    if vti[i4]==max1:
        index2.append(i4)
    else:
        continue

"找到供应最大的位置"
t_o=0

for i5 in range(0,len(index2)):
    if A[Oi[to[index2[i5]]]-1,Oj[to[index2[i5]]]-1]>=t_o:
        t_o=A[Oi[to[index2[i5]]]-1,Oj[to[index2[i5]]]-1]
        index1=index2[i5]
    else:
        continue
        
"最终计算搜索的结果"    

if A[Oi[to[index1]]-1,Oj[to[index1]]-1]>=A[Ii[ti[index1]]-1,Ij[ti[index1]]-1]:
    A[Oi[to[index1]]-1,Oj[to[index1]]-1]=A[Oi[to[index1]]-1,Oj[to[index1]]-1]-A[Ii[ti[index1]]-1,Ij[ti[index1]]-1]
    A[Ii[ti[index1]]-1,Ij[ti[index1]]-1]=0
else:
    A[Ii[ti[index1]]-1,Ij[ti[index1]]-1]=A[Ii[ti[index1]]-1,Ij[ti[index1]]-1]-A[Oi[to[index1]]-1,Oj[to[index1]]-1]
    A[Oi[to[index1]]-1,Oj[to[index1]]-1]=0






