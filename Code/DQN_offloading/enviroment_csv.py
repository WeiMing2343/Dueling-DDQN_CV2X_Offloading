'''
寫兩種方式
第一種為先random生成好資料class的環境直接讀(這樣讀的資料會是固定的)
第二種為在class的環境裡random資料(這樣每次迴圈執行資料就會隨機產生不是固定的)

此為第一種

看情況補負獎勵判斷
'''
import csv
import numpy as np
import pandas as pd
import math

class single_env():
    def __init__(self):
        self.action_space = 3
        self.observation_space = 10
        #self.reward_origin = -float('inf')
        self.beta1 = 1
        self.beta2 = -0.1
        self.observation =[]
        self.observation_ =[]
        
        self.veh_throughput = None
        self.veh_direction = None
        self.veh_packet_size = None
        self.beta = None
        
        self.SINR = 10
        
        self.max_delay = 0.49 #封包帶500,頻寬帶300
        self.min_delay = 0.23 #封包帶400,頻寬帶500
        
        #self.max_bitrate = 5 #Bitrate不做正規化不然我的賽局會爆
        self.price = 0
        self.max_bitrate = 500
        
        self.c = 1 #money_token
        self.rd = 0.1655 #每1Mbit的功耗
        self.rt = 0.7438 #每個時間段的功耗
        self.max_power_consumption = 50
        self.min_power_consumption = 84
        
    def minmax_norm(self,data, max_data, min_data):
        norm_data = (data-min_data) / (max_data-min_data)
        return norm_data
    
    def action_correspond(self, action, step_count): #動作對應,選擇出來的動作對應哪台車輛,先假設只有3台
        #print('step_count:',step_count)
        with open('single_state_hop1.csv','r',newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]
        float_row = list(map(self.convert, rows[step_count])) #去讀第step_count行
        #print(float_row)
        if action == 0: #如果選出來的action為0,代表選擇K1車做傳輸
            throughput = float_row[0]
            packet_size = float_row[3]
            direction = float_row[4]
            
        elif action == 1 : #如果選出來的action為1,代表選擇K2車做傳輸
            throughput = float_row[1]
            packet_size = float_row[3]
            direction = float_row[5]
            
        elif action == 2: #如果選出來的action為2,代表選擇K3車做傳輸
            throughput = float_row[2]
            packet_size = float_row[3]
            direction = float_row[6]
        
        return throughput, packet_size, direction
    
    def convert(self,string): #文字轉換成float
        try:
            string=float(string)
        except :
            pass    
        return string
    
    def Game_price_set(self, bitrate): #賽局的價格制定,依照bitrate給定
        if (300 <= bitrate < 350) :
            price = 1
        elif (350 <= bitrate < 400) :
            price = 2
        elif (400 <= bitrate < 450) :
            price = 3
        elif (450 <= bitrate < 500) :
            price = 4
        elif (500 <= bitrate ) :
            price = 5
        else:
            price = 0
        return price
    
    def choose_bitrate(self, throughput):
        bitrate = np.random.uniform(300,throughput)
        '''
        if 300 < throughput < 350:
            bitrate = 300
        elif 350 < throughput < 400:
            bitrate = 350
        elif 400 < throughput < 450:
            bitrate = 400
        elif 450 < throughput < 500:
            bitrate = 450
        elif 500 < throughput < 550:
            bitrate = 500
        else :
            print('bitrate_choose error')
        '''
        return bitrate
    
    def bitrate_norm(self, bitrate):
        norm_bitrate = bitrate / self.max_bitrate
        return norm_bitrate
    
    def reset(self):#讀取csv第一行
        with open('single_state_hop1.csv','r',newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]
        
        float_row = list(map(self.convert, rows[1])) #第0列是title,用map直接全部轉換
        self.observation = float_row 
        ''' 另一種list文字轉換float的方法
        float_row = []
        for obj in rows[1]:
            float_row.append(self.convert(obj))
        print(float_row)
        '''
        #print(self.observation)
        return self.observation
        
    def step(self, action, nextstep_count):#要回傳(下一個狀態,獎勵,done,info)
        #寫done
        terminate = False #設定done
        if nextstep_count == 100: #因為我csv的資料只有100筆,所以當跑到第100筆時就讓done維true結束迴圈
            terminate = True #所以csv裡最後一筆資料訓練不到
        
        #寫observation_ 下一個狀態
        with open('single_state_hop1.csv','r',newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = [row for row in reader]
        float_row = list(map(self.convert, rows[nextstep_count])) #去讀第nextstep_count行
        self.observation_ = float_row #為下一個狀態
        
        #動作進行對應
        self.veh_throughput, self.veh_packet_size, self.veh_direction  = self.action_correspond(action, nextstep_count-1) #因為我設的nextstep_count是記錄下一個狀態,要找這個狀態的對應要nextstep_count-1
        #print('self.veh_throughput:',self.veh_throughput)
        #print('self.veh_direction:',self.veh_direction)
        #print('self.veh_packet_size:',self.veh_packet_size)
        if (self.veh_direction < 100):
            self.beta = self.beta1
        elif (100 < self.veh_direction < 200):
            self.beta = self.beta2
        #print('self.beta:',self.beta)
        
        #寫reward
        #計算車載傳輸延遲分數
        #bitrate = np.random.randint(1,self.veh_throughput +1 )#np.random.randint(a,b) 的範圍為a ~ b-1,所以self.veh_throughput 要+1
        bitrate = self.choose_bitrate(self.veh_throughput)
        #bitrate = max(bitrate-200,300)
        #print('bitrate:',bitrate)
        data_rate = bitrate * math.log2(1 + self.SINR) #計算數據接收率
        #print('data_rate:',data_rate)
        #print('self.veh_packet_size:',self.veh_packet_size)
        Transmission_delay = self.veh_packet_size / data_rate #計算車載傳輸延遲
        #print('Transmission_delay:',Transmission_delay)
        norm_Transmission_delay = self.minmax_norm(Transmission_delay, self.max_delay, self.min_delay) #做min_max正歸化
        #print('norm_Transmission_delay:',norm_Transmission_delay)
        
        #計算bitrate分數
        self.price = self.Game_price_set(bitrate)
        #print('price:',self.price)
        norm_bitrate = self.bitrate_norm(bitrate)
        bitrate_score = self.price * norm_bitrate
        #print('bitrate_score:',bitrate_score)
        
        #計算功耗分數,pig注意 考慮一下這段要不要做正規化,做正規化的話bitrate怎麼選利潤都會大於成本,不做正規化的話bitrate要選3以上利潤才會大於成本
        power_consumption = self.rd * bitrate + self.rt
        #print('power_consumption:',power_consumption)
        norm_power_consumption = self.minmax_norm(power_consumption, self.max_power_consumption, self.min_power_consumption)
        #print('norm_power_consumption:',norm_power_consumption)
        
        #計算reward
        reward = self.beta*(-norm_Transmission_delay + bitrate_score - (self.c * norm_power_consumption))
        #print('reward:',reward)
        return self.observation_, reward, terminate, {} #要回傳(下一個狀態,獎勵,done,info)
    
    
#A = single_env()
#observation_,reward, done, info = A.step(0, 2)
#print('observation_ :',observation_)
#print('done:',done)
#print('reward:',reward)
#print('info:',info)