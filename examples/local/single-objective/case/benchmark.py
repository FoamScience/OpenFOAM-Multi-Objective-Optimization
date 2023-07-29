#!/usr/bin/env python3
"""
    Python implementation of some nice benchmark functions described in:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9684455
"""

import sys, os
import numpy as np
import plotly.express as px
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--F', type=str, help='Function to test')
parser.add_argument('--k', type=float, help='k')
parser.add_argument('--m', type=float, help='m')
parser.add_argument('--lb', type=float, help='lambda')
args = parser.parse_args()

def z(x,k,m,lb):
    cond= np.abs(x)/k - np.floor(np.abs(x)/k)
    return [1-m+(m/lb)*i if i<lb else 1-m+(m/(1-lb))*(1-i) for i in cond]

def F1(x,k,m,lb):
    c=z(x,k,m,lb)
    p=(x-40)*(x-185)*x*(x+50)*(x+180)
    return 3e-9*np.abs(p)*c+10*np.abs(np.sin(0.1*x))

def F2(x,k,m,lb):
    return F1(F1(x,k,m,lb),k,m,lb)

def F3(x,k,m,lb):
    return 3*np.abs(np.log(1000*np.abs(x)+1))*z(x,k,m,lb)+30-30*np.abs(np.cos(x/(10*np.pi)))

def F4(x, k, m, lb):
    return F3(F3(x,k,m,lb),k,m,lb)

def plot1D(x, Func, k,m,lb):
    y=Func(x,k,m,lb)
    fig = px.line(x=x, y=y)
    fig.update_layout(xaxis_title='x', yaxis_title=args.F,width=2048,height=768)
    fig.write_image(f"{args.F}.png")

Fs = {
    "F1": F1,
    "F2": F2,
    "F3": F3,
    "F4": F4,
}

x = np.linspace(-200,200,1000)
plot1D(x, Fs[args.F], args.k,args.m,args.lb)

if __name__ == '__main__':
    pFile = ParsedParameterFile(name=os.getcwd()+"/FxDict")
    x=float(pFile['x'])
    print(Fs[args.F](np.array([x,x]), args.k,args.m,args.lb)[0])
