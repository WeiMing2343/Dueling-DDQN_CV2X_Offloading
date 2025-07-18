import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

class ReplayBuffer(): #定義緩衝暫存器
    def __init__(self, max_size, input_dims): #初始化包刮最大內存大小,輸入維度
        self.mem_size = max_size
        self.mem_cntr = 0 #內存計數器,跟蹤最後一個未保存的內存位置
        #利用self.mem_cntr將新的記憶存入緩衝區
        
        #self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)#狀態內存
        # *在python裡是解壓,可以解壓list裡面的元素, ex: test = [1], *test = 1
        # pig,範例裡gym的input_dims是列表型態，所以要用*取值，但我如果輸入的維度直接是一個值的話，這邊的*就要刪掉
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        
        #self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32) #下一個狀態的內存
        # pig,範例裡gym的input_dims是列表型態，所以要用*取值，但我如果輸入的維度直接是一個值的話，這邊的*就要刪掉
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32) #動作內存
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32) #獎勵內存
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32) #終端內存
    
    #這個def要做的事是把讀進buffer的資料儲存下來
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size #計算第一個可用的內存位置
        
        #內存處理,將資料存進內存
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1-int(done)
        
        self.mem_cntr += 1
    
    #這個def要做的事是選定緩衝區的其中一筆資料回傳到主程式
    def sample_buffer(self, batch_size):#對緩衝區進行取樣
        max_mem = min(self.mem_cntr, self.mem_size)#要得知現在內存的最上面那筆是第幾個位置
        #看緩存是否填滿了,如果已經填滿了,會對整個內存進行採樣。如果沒填滿，將採樣到 mem_cntr 的位置，這樣才不會對一堆0進行採樣
        
        #從緩衝區內隨機選擇一個作為輸出，包刮狀態、獎勵、新狀態、終端狀態
        batch = np.random.choice(max_mem, batch_size, replace=False) #replace=False 確保不會兩次獲得相同內存
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, terminal
    
# 建構DQN模型
def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims): #輸入包括(學習率, 動作數量, 輸入維度, 第一個全連接的維度, 第二個全連階層維度)
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    
    return model

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000, fname='dqn_model.h5'):
    #輸入包刮(學習率, gamma, 動作數量, epsilon, 批量大小, 輸入維度大小, epsilon減量, epsilon最小, 內存大小, 儲存模型名稱)
    #epsilon_dec 表示隨著時間推移讓探索率下降
    #epsilon_end=0.01 表示不要讓探索率為0
        
        self.action_space = [i for i in range(n_actions)] #創建動作空間列表,方便選擇隨機動作    
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)#建構DQN模型
    
    
    def store_transition(self, state, action, reward, new_state, done): #agent和內存的接口函數
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def choose_action(self, observation):#選擇動作,環境的觀察作為輸入
        if np.random.random() < self.epsilon :  #如果隨機數小於探索率的話
            action = np.random.choice(self.action_space) #隨機從動作空間裡選一個動作
            #print('random action')
            #讓DQN去探索新的動作
        else:
            state = np.array([observation]) #將觀察向量添加一個維度
            actions = self.q_eval.predict(state) #用DQN預測動作向量機率
            # '注意'在DQN裡 輸入環境狀態 輸出為動作 後續才給動作評分
            # '注意' 因為self.q_eval是使用 keras建的,而且回傳是整個model,所以內建就有predict函數
            # '注意' 這裡的actions是機率矩陣
            #print('actions:',actions)
            
            action = np.argmax(actions) #回傳最大機率的動作,貪婪
            # np.argmax()是返回機率最大的索引值。假如actions = [0.1, 0.2, 0.3, 0.1]，np.argmax(actions) = 2。
        #print(action)
        return action
        # 這裡好像預測比較慢了些

    def learn(self): #學習,DQN用的是時間差異學習(TD)
        if self.memory.mem_cntr < self.batch_size: #如果沒有填充滿最小批量的內存就不進行學習
            return
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size) #從內存緩衝區裡取值出來
        
        q_eval = self.q_eval.predict(states) #從緩衝區讀出來的狀態，使用DQN預測出狀態的動作價值函數的估計
        q_next = self.q_eval.predict(states_) #從緩衝區讀出來的下一個狀態，使用DQN預測出下一個狀態的動作價值函數的估計
        
        q_target = np.copy(q_eval) #動作價值函數的目標值,為從緩衝區讀出來的狀態進行DQN預測後的值
        #創建目標，讓學習更新代理人估計動作價值函數的權重移動方向往target移動，已收斂到正確的值
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)#以等差數列方式產生陣列,ex: self.batch_size=64，np.arange(self.batch_size, dtype=np.int32)=[0 1 2 ~63]
        
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)*dones #計算Q值
        #'注意' 因為他這裡乘上了dones，可以往上看dones是怎麼設計的
        
        self.q_eval.train_on_batch(states, q_target) #計算delta，train_on_batch 這個 api，是對一個 mini-batch 的數據進行梯度更新
        
        #讓探索率下降
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
    def save_model(self):#儲存模型
        print('... saving model ...')
        self.q_eval.save(self.model_file)
    
    def load_model(self):#讀取模型
        print('... loading model ...')
        self.q_eval = load_model(self.model_file)
    
'''
A = Agent(0.001, 0.9, 4, 0.1, 64, [2])
#A = Agent(0.001, 0.9, 5, 0.1, 64, 2)
observation = [0.1,0.3]
A.choose_action(observation)
'''  
        
        
        
        
        