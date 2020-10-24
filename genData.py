import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
nFrames = 500
plt.figure(figsize=(10,10))
n = 10000
m = 200
flexion_task_data = np.zeros((n,nFrames))
flexion_task_time = np.zeros((n,nFrames))
extension_task_data = np.zeros((n,nFrames))
extension_task_time = np.zeros((n,nFrames))

flexion_task_test_data = np.zeros((m,nFrames))
flexion_task_test_time = np.zeros((m,nFrames))
extension_task_test_data = np.zeros((m,nFrames))
extension_task_test_time = np.zeros((m,nFrames))
random_task_test_data = np.zeros((m,nFrames))
random_task_test_time = np.zeros((m,nFrames))

for i in range(n):
    noise = 4*np.random.normal(0,1,nFrames)
    t_end = .9 + .1*np.random.randn(1)
    start = 10*np.random.randn(1)
    max = 70 + 10*np.random.randn(1)
    shift = (np.pi/2) +(np.pi/2)*.1*np.random.randn(1)
    amp = max/2
    t = np.linspace(0,t_end[0],nFrames)
    y = start +  amp*(1 + np.sin(-shift + np.pi*t))
    flexion_task_data[i] = y + noise
    flexion_task_time[i] = t
    plt.plot(t,y + noise)

for i in range(n):
    noise = 4*np.random.normal(0,1,nFrames)
    t_end = .9 + .1*np.random.randn(1)
    start = 10*np.random.randn(1)
    max = 70 + 10*np.random.randn(1)
    amp = max/2
    t = np.linspace(0,t_end[0],nFrames)
    y = start +  amp*(1 + np.sin(shift + np.pi*t))
    extension_task_data[i] = y + noise
    extension_task_time[i] = t
    plt.plot(t,y + noise)    
plt.show()

pd.DataFrame(flexion_task_data).to_csv("./flexion_task_data.csv", header=None, index=None)
pd.DataFrame(extension_task_data).to_csv("./extension_task_data.csv", header=None, index=None)

pd.DataFrame(flexion_task_time).to_csv("./flexion_task_time.csv", header=None, index=None)
pd.DataFrame(extension_task_time).to_csv("./extension_task_time.csv", header=None, index=None)

plt.figure(figsize=(10,10))

for i in range(m):
    noise = 4*np.random.normal(0,1,nFrames)
    t_end = .9 + .1*np.random.randn(1)
    start = 10*np.random.randn(1)
    max = 70 + 10*np.random.randn(1)
    amp = max/2
    t = np.linspace(0,t_end[0],nFrames)
    y = start +  amp*(1 + np.sin(shift + np.pi*t))
    extension_task_test_data[i] = y + noise
    extension_task_test_time[i] = t
    plt.plot(t,y + noise)  
pd.DataFrame(extension_task_test_data).to_csv("./extension_task_test_data.csv", header=None, index=None)
pd.DataFrame(extension_task_test_time).to_csv("./extension_task_test_time.csv", header=None, index=None)

for i in range(m):
    noise = 4*np.random.normal(0,1,nFrames)
    t_end = .9 + .1*np.random.randn(1)
    start = 10*np.random.randn(1)
    max = 70 + 10*np.random.randn(1)
    amp = max/2
    t = np.linspace(0,t_end[0],nFrames)
    y = start +  amp*(1 + np.sin(-shift + np.pi*t))
    flexion_task_test_data[i] = y + noise
    flexion_task_test_time[i] = t
    plt.plot(t,y + noise)  
plt.show()
pd.DataFrame(flexion_task_test_data).to_csv("./flexion_task_test_data.csv", header=None, index=None)
pd.DataFrame(flexion_task_test_time).to_csv("./flexion_task_test_time.csv", header=None, index=None)

for i in range(m):
    noise = 4*np.random.normal(0,1,nFrames)
    t_end = .9 + .1*np.random.randn(1)
    start = 10*np.random.randn(1)
    max = 70 + 10*np.random.randn(1)
    amp = max/2
    t = np.linspace(0,t_end[0],nFrames)
    y = start +  amp*(1 + np.sin((np.random.choice([-1,1]))*shift + np.pi*t))
    random_task_test_data[i] = y + noise
    random_task_test_time[i] = t
    plt.plot(t,y + noise)   
plt.show()
pd.DataFrame(random_task_test_data).to_csv("./random_task_test_data.csv", header=None, index=None)
pd.DataFrame(random_task_test_time).to_csv("./random_task_test_time.csv", header=None, index=None)