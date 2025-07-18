'''
設定任務量(task),也就是csv裡有幾個狀態
'''

#這邊先預設我只跳點跳一層
import numpy as np
import csv

K1_throughput = []
K2_throughput = []
K3_throughput = []
packet_size = []
K1_direction = []
K2_direction = []
K3_direction = []
K1_beta = []
K2_beta = []
K3_beta = []
beta1 = 1 #跳點權重因子,跳一層
beta2 = -0.1 #跳點權重因子,跳二層
state = []
task = 100

for i in range(task):
    K1_throughput.append(round(np.random.uniform(300,550),3))#action = 300,350,400,450,500
    K2_throughput.append(round(np.random.uniform(300,550),3))
    K3_throughput.append(round(np.random.uniform(300,550),3))
    packet_size.append(np.random.randint(400,500))
    K1_direction.append(round(np.random.uniform(50,100),3))
    K2_direction.append(round(np.random.uniform(50,100),3))
    K3_direction.append(round(np.random.uniform(50,100),3))
    if (K1_direction[i] < 100) :
        K1_beta.append(beta1)
    elif (100 < K1_direction[i] < 200) :
        K1_beta.append(beta2)
    
    if (K2_direction[i] < 100) :
        K2_beta.append(beta1)
    elif (100 < K2_direction[i] < 200) :
        K2_beta.append(beta2)
    
    if (K3_direction[i] < 100) :
        K3_beta.append(beta1)
    elif (100 < K3_direction[i] < 200) :
        K3_beta.append(beta2)
        
state.append(K1_throughput)
state.append(K2_throughput)
state.append(K3_throughput)
state.append(packet_size)
state.append(K1_direction)
state.append(K2_direction)
state.append(K3_direction)
state.append(K1_beta)
state.append(K2_beta)
state.append(K3_beta)
state_T = np.transpose(state)
#print(np.shape(state_T))

with open('single_state_hop1.csv','w',newline='') as file: #'w'為覆寫
    write = csv.writer(file)
    write.writerow(["K1_throughput","K2_throughput","K3_throughput","packet_size","K1_direction","K2_direction","K2_direction","K1_beta","K2_beta","K3_beta"])
    for i in range(task):    
        write.writerow(state_T[i])
    print('build stete file')
