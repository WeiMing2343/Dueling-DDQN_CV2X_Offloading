##### 5G 傳輸
import numpy as np
from dueling_ddqn_tf2 import Agent
from utils import plotLearning
from enviroment_offloading import offloading_env
import tensorflow as tf
import pandas as pd
import csv
import time
import pydot
from tensorflow.keras.utils import plot_model

NR_date_rate = 8*8*(948/1024)*(1*12)*(0.001*(14*2**1))*(1-0.14)

VtoM_trans = NR_date_rate*7  #110Mbps以5G傳輸速度為範本
VtoV_trans = NR_date_rate*10  #165Mbps
VtoC_trans = NR_date_rate*7  #110Mbps

car_process = 40  #Mbps
MEC_process = 165 #Mbps


if __name__ == '__main__':
    start = time.time()
    #tf.compat.v1.disable_eager_execution()#關閉tensorflow2裡的eager_execution，eager_execution這會讓tf執行的很慢
    #env = gym.make('LunarLander-v2')
    tf.compat.v1.enable_eager_execution()
    env = offloading_env(VtoM_trans,VtoV_trans,VtoC_trans,car_process,MEC_process)
    agent = Agent(lr=0.0001, gamma=0.9, n_actions = env.action_space, epsilon=1.0,
                  batch_size=64, input_dims=env.observation_space)

    
    n_games = 2000
    break_count = 0
    ddqn_scores = []
    eps_history = []
    best_score = -1.6#平均要超過才能保存
    print("break_count = ",break_count)
    for i in range(n_games):
        
        action0_number = 0
        action1_number = 0
        action2_number = 0
        action3_number = 0
        done = False
        observation = env.reset()
        setp_number = -1
        car1data = 0
        car2data = 0
        MECdata = 0
        score = 0
        done = False
        score = 0
        observation = env.reset()
        while not done:
            setp_number =setp_number +1
            action = agent.choose_action(observation)
            observation_, reward, done, info,car1data ,car2data ,MECdata  ,csv_latency = env.step(action,setp_number,car1data ,car2data ,MECdata)            
#             print("第",setp_number,"個action = ", action)
#             print("延遲 = ",reward)
#             print("第一台車現在的數據輛 = ",observation_[0])
#             print("第一台車之前的數據輛 = ",car1data)  
            
#             print("第二台車現在的數據輛 = ",observation_[1])
#             print("第二台車之前的數據輛 = ",car2data)
            
#             print("MEC現在的數據輛 = ",observation_[2])   
#             print("MEC之前的數據輛 = ",MECdata)
#             print("========================================")
            score += reward #這次得到的分數就是這一圈epoch的所有reward加總的值,然後迴圈一直更新,直到done為true,就是跑完這次epoch的分數
            #score = score - reward
#             print("score = ",score)
            agent.store_transition(observation, action, reward, observation_, done) #把這個迴圈的資料(觀察,動作,獎勵,下一個觀察,終端狀態)存進replay buffer
            observation = observation_ #要在while裡更新新的狀態,不然會一直是舊得observation進行learn
            
            agent.learn() #迭代中進行TD學習
            if action  == 0 :
                action0_number += 1
            elif action  == 1:
                action1_number += 1
            elif action  == 2:
                action2_number += 1
            elif action  == 3:
                action3_number += 1
        print("action_number = ", action0_number,action1_number,action2_number,action3_number)
        
        
        eps_history.append(agent.epsilon)
        score = score/(env.data_time_slot)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[-50:])
        

                
        
        
        if score > best_score : #如果平均分數大於最佳分數
            best_score = score #覆蓋掉最佳分數
            best_latency_csv=pd.DataFrame(data=csv_latency)
            print("best_score loading = ",best_score)
            best_latency_csv.to_csv('plots/other_data/best_score_latency_0.8.csv',encoding='utf-8', index = False)
            
        print('episode: ', i,'score: %.5f' % score,' average score %.5f' % avg_score , 'epsilon %.2f' % agent.epsilon )
        if i>1000: 
            if avg_score > -0.62:
                break_count = break_count + 1
                print("break_count = ",break_count)
        if break_count > 100 and avg_score > -0.62 :
            n_games = i+1
            print("n_games = ",n_games)
            break
    filename = 'offloading_ddqn_0.8.png'
    end = time.time()
    print("總共所花的時間",format(end-start))
    x = [i+1 for i in range(n_games)]
    plotLearning(x, ddqn_scores, eps_history, filename)
    for i in range(n_games):
        ddqn_scores[i] =-ddqn_scores[i] 
        
    N = len(ddqn_scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(ddqn_scores[max(0, t-50):(t+1)])
    print(running_avg)    
    test=pd.DataFrame(data=running_avg)
    test.to_csv('plots/learning_rate/DDQN_avg_score_0.8.csv',encoding='utf-8', index = False)
