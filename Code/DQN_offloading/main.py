from DQN_networks import Agent
import numpy as np
import gym
import tensorflow as tf
from utils import plotLearning
from enviroment_csv import single_env

if __name__ =='__main__':
    tf.compat.v1.disable_eager_execution()#關閉tensorflow2裡的eager_execution，eager_execution這會讓tf執行的很慢
    
    #創建環境
    #env = gym.make('LunarLander-v2')
    #pig 這一段要換成自己寫的車載環境 
    env = single_env()
    
    lr = 0.001 #學習率
    n_games = 5000 #epoch
    #agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=env.observation_space.shape,
    #              n_actions = env.action_space.n, mem_size=1000000, batch_size=64, epsilon_end=0.01)
    #pig n_actions這個要換成自己從車載環境回傳的動作空間,或是直接給值,ex:直接給5(env.action_space.n 輸出為常數)
    #pig env.observation_space.shape 要換成自己觀察向量的維度，但要注意gym裡是列表型態，要設成一個值的話，DQN_networks 裡buffer裡的*要去掉。不然就是env.observation_space.shape要為列表型態
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=env.observation_space,
                  n_actions = env.action_space, mem_size=1000000, batch_size=64, epsilon_end=0.01)
    
    score_history = [] #紀錄得分歷史
    eps_history = [] #紀錄epsilon歷史
    
    for i in range(n_games): #主訓練迴圈,訓練幾圈由epoch決定
        done = False #每次迴圈開始,done為0
        score = 0 #每次迴圈,分數從計算,代表重新計算這個epoch所得到的分數
        observation = env.reset() #由env.reset給出環境的狀態
        #pig env.reset()要在自己的車載環境裡寫,就是讀出環境現在的狀態(return要回傳狀態)
        
        nextstep_count = 1
        while not done: #done如果false, not done=not false=true=1 ,while判斷為1就等於是無限迴圈,要終止就要done的值為true,not done才會為false
            nextstep_count = nextstep_count + 1
            action = agent.choose_action(observation) #根據環境當下的狀態選擇出一個動作,choose_action在class agent已經寫好了
            
            #observation_, reward, done, info = env.step(action) #將選擇出來的動作帶入env.step()會得到下一個狀態,獎勵,最終狀態,訊息
            #pig env.step()要在自己的車載環境裡寫,可以參考gym和學長是怎麼寫的,總之return要回傳[下一個狀態(應該是可以直接去讀狀態csv的下一列),獎勵(我的reward function應該寫在這),done(是否跑完全檔案),info]
            #pig 注意 env.step()要回傳的是下一個狀態
            #pig 我的車載環境是動態改變所以observation_直接random state的參數,直接讀下一列的csv
            observation_, reward, done, info = env.step(action,nextstep_count)
            
            
            score += reward #這次得到的分數就是這一圈epoch的所有reward加總的值,然後迴圈一直更新,直到done為true,就是跑完這次epoch的分數
            
            agent.store_transition(observation, action, reward, observation_, done) #把這個迴圈的資料(觀察,動作,獎勵,下一個觀察,終端狀態)存進replay buffer
            observation = observation_ #要在while裡更新新的狀態,不然會一直是舊得observation進行learn
            
            agent.learn() #迭代中進行TD學習
        
        eps_history.append(agent.epsilon) #把每一個epoch的epsilon記錄下來
        score_history.append(score) #把每一個epoch的分數記錄下來
        avg_score = np.mean(score_history[-100:]) #計算平均分數,為最新一百場比賽的分數加起來平均
        
        print('episode: ', i, 'score %.2f' % score, 'average_score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
        
        
    filename = 'single_DQN.png' #生成的收斂圖名稱,有調整超參數的話可以改變圖的名稱比較好分辨
    figure_file = 'plots/' + filename
    x = [i+1 for i in range(n_games)] #x軸,i+1=1
    plotLearning(x, score_history, eps_history, figure_file) #畫圖
        
            
            
            
            
            
            
            
            
            
            
            
        
        

