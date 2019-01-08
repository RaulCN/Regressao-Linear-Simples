
# coding: utf-8

# In[1]:

# Seguindo este tutorial  https://www.youtube.com/watch?v=9uS0qiMeZu0&index=5&list=PLbmt8d_ueDMVUVlw9VZSdgAIi6W3u-7Zg
# Versão no collaboratory https://colab.research.google.com/drive/1hG77wMKlnvKjsIKxgFHL3hJDvbUic96p


import numpy as np
import matplotlib.pyplot as plt
import math
#get_ipython().run_line_magic('matplotlib', 'inline')



# Dados 
# 

# In[2]:


X = [0.5, 2.2, 2.0]
y = [2.0, 2.5, 1.4]



# TAXA DE APRENDIZADO ( velocidade de descida )
# 

# In[3]:



a=0.01
w0=0.1
w1=0.1


# In[4]:



def y_hat(x, w0, w1):
    return w0+w1*x



# In[5]:


y_hat(1.5, w0, w1)


# In[6]:


def plot_line(X, y, w0, w1):
    x_values = [i for i in range(int(min(X))-1,int(max(X))+2)]
    y_values = [y_hat(x, w0, w1) for x in x_values]
    plt.plot(x_values,y_values,'r')
    plt.plot(X,y,'bo')
    


# In[7]:


plot_line(X,y,w0,w1)


# FUNÇÃO MSE (Média do erro ao quadrado)¶
# 
# 
# 
# 

# In[8]:


def MSE (X, y , w0 , w1):
    custo = 0
    m = float(len(X))
    for i in range(0, len(X)):
        custo += (y_hat(X[i], w0, w1)-y[i])**2
        
    return custo/m


# #ALGORITMO DO GRADIENTE DESCENDENTE
# 
# GRADIENTE DESCENDENTE STEP
# 

# In[9]:


MSE(X,y, w0, w1)


# In[10]:


def gradient_descent_step(w0, w1, X, y, a):
    
    erro_w0 = 0
    erro_w1 = 0
    m = float(len(X))
    
    for i in range(0,len(X)):
        erro_w0 += y_hat(X[i], w0, w1) - y[i]
        erro_w1 += (y_hat(X[i], w0, w1) - y[i]) * X[i]
        
    new_w0 = w0 - a * (1/m) * erro_w0
    new_w1 = w1 - a * (1/m) * erro_w1

    return new_w0, new_w1


# In[11]:


epoch = 800



# In[12]:


def gradient_descent(w0, w1, X, y, a, epoch):
    custo = np.zeros(epoch)
    for i in range(epoch):
        w0 , w1 = gradient_descent_step(w0, w1, X, y , a)
        custo[i] = MSE(X,y,w0,w1)
        
    return w0, w1, custo


# In[13]:


w0 , w1 , custo = gradient_descent(w0, w1, X, y, a, epoch)


# In[15]:


custo


# In[16]:


print("w0={}, w1={}".format(w0,w1))


# Plotando o custo
# 
# 

# In[17]:


fig, ax = plt.subplots()  
ax.plot(np.arange(epoch), custo, 'r')  
ax.set_xlabel('Iterações')  
ax.set_ylabel('Custo')  
ax.set_title('MSE vs. Epoch')


# 
# PLOTANDO A HIPÓTESE OTIMIZADA
# 

# In[19]:


plot_line(X,y, w0, w1)


# REALIZADO UMA PREVISÃO¶
# 
# 
# 

# In[20]:


y_hat(1.5, w0, w1)


# In[21]:


print("w0={}, w1={}".format(w0,w1))


# In[22]:


w0 + w1*1.5

