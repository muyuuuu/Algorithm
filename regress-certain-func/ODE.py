from scipy.integrate import odeint
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

t = np.arange(0,100,0.1)

def deriv(w,t,a,b,c,d,e,f): 
    x,y = w
    return np.array([ a*(1-b*x)*x-c*y*x, d*(1-e*y)*y-f*x*y])

p=[0.1, 0.002, 0.0001, 0.3, 0.003, 0.0002, 100, 150]

a,b,c,d,e,f,x0,y0=p
yinit = np.array([x0,y0]) # 初值
yyy = odeint(deriv,yinit,t,args=(a,b,c,d,e,f))

plt.figure(figsize=(7,5))
plt.plot(t,yyy[:,0],"b-",label="$x_1$变化曲线")
plt.plot(t,yyy[:,1],"r-",label="$x_2$变化曲线")
plt.plot([0,100],[250,250],"g--")
plt.plot([0,100],[375,375],"g--")
plt.xlabel(u'时间t')
plt.ylabel(u'物种量')
plt.title(u'两竞争物种的变化曲线')
plt.legend(loc=4)
plt.show()