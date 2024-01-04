import numpy as np
import random
import pandas as pd



class offloading_env():
    def __init__(self,VtoM_trans,VtoV_trans,VtoC_trans,car_process,MEC_process):
        self.action_space = 4
        self.observation_space = 3
        self.data_time_slot = 500
        #傳輸速率Mbps
        VtoM , sigma_VtoM = VtoM_trans,5
        self.VtoM = np.random.normal(VtoM, sigma_VtoM, self.data_time_slot)
        self.VtoM = np.clip(self.VtoM, VtoM_trans-5, VtoM_trans+5)
        self.VtoM = self.VtoM.astype(int)
        
        VtoV , sigma_VtoV = VtoV_trans ,5
        self.VtoV = np.random.normal(VtoV, sigma_VtoV, self.data_time_slot)
        self.VtoV = np.clip(self.VtoV, VtoV_trans-5, VtoV_trans+5)
        self.VtoV = self.VtoV.astype(int)
        
        VtoC , sigma_VtoC = VtoC_trans,5
        self.VtoC = np.random.normal(VtoC, sigma_VtoC, self.data_time_slot)
        self.VtoC = np.clip(self.VtoC, VtoC_trans-5, VtoC_trans+5)
        self.VtoC = self.VtoC.astype(int)
        
        #car_cpu處理速率
        mu_cpu, sigma_cpu = 40, 5
        self.car_cpu = np.random.normal(mu_cpu, sigma_cpu, self.data_time_slot)
        self.car_cpu = np.clip(self.car_cpu , 40-5 , 40+5)
        self.car_cpu = self.car_cpu.astype(int)
        #MEC_cpu處理速率
        mu_mec, sigma_mec = MEC_process, 5
        self.mec_cpu = np.random.normal(mu_mec, sigma_mec, self.data_time_slot)
        self.mec_cpu = np.clip(self.mec_cpu , MEC_process-5 , MEC_process+5)
        self.mec_cpu = self.mec_cpu.astype(int)
        #每秒產生100~900Mbps數據大小
        mu, sigma = car_process, 5
        self.car1_data = np.random.normal(mu, sigma, self.data_time_slot)
        self.car1_data = np.clip(self.car1_data, car_process-20, car_process+20) # 将数据限制在500到1000之间
        self.car1_data = self.car1_data.astype(int) # 将数据转换为整数
        
        self.car2_data = np.random.normal(mu, sigma, self.data_time_slot)
        self.car2_data = np.clip(self.car2_data, car_process-20, car_process+20) # 将数据限制在500到1000之间
        self.car2_dataa = self.car2_data.astype(int) # 将数据转换为整数

        
        #讀重複的資料
        #arr = np.genfromtxt("datasize_car.csv",delimiter=",", dtype=int,encoding="utf-8")
        #self.car1_data = arr[:,1]
        #self.car2_data = arr[:,2]
        
        self.mec_data = np.zeros(self.data_time_slot)
        self.cloud = np.zeros(self.data_time_slot)
        
        self.before_car1_data = 0
        self.before_car2_data = 0
        self.before_MEC_data = 0    
        self.latency_total = 0
        self.observation = np.zeros([3],dtype="float32")
        self.observation_ = np.zeros([3],dtype="float32")
        self.latency_csv = np.zeros([self.data_time_slot],dtype="float32")
        
    def action_offload(self,action,setp_number,car1data ,car2data ,MECdata):
        self.before_car1_data = car1data 
        self.before_car2_data = car2data 
        self.before_MEC_data = MECdata
        if action == 0:#在本地計算
            waiting_data_number = self.before_car1_data#這邊為上一秒的時間段有沒有排隊的數據
                
            latency_process = self.car1_data[setp_number]/self.car_cpu[setp_number] #計算本地處理所花的時間 
            latency_trasnport = 0  #本地計算不會有傳輸問題
            latency_queue = waiting_data_number/self.car_cpu[setp_number]
            
            self.latency_total = latency_process + latency_queue
            self.before_car1_data = (self.latency_total - 1)*self.car_cpu[setp_number]#這邊計算1秒時間段後還剩下多少資料，因為我是設定每秒產生多少數據
            
            if self.before_car1_data < 0 :
                self.before_car1_data = 0
                
#             if latency_queue >= 2 :
#                 trans_time1 = (waiting_data_number + self.car1_data[setp_number] ) /self.VtoM[setp_number]
#                 process_time = (self.before_MEC_data + waiting_data_number + self.car1_data[setp_number]) / self.mec_cpu[setp_number]
#                 self.latency_total = trans_time1 + process_time
#                 self.before_car1_data = 0
        
        elif action == 1:#傳輸到MEC計算
            waiting_data_number = self.before_MEC_data#這邊為上一秒的時間段mec有沒有排隊的數據
            car1_data = self.car1_data[setp_number] + self.before_car1_data #丟過來的data是包含在car1等待的data
            
            latency_process = (car1_data)/self.mec_cpu[setp_number] #計算MEC處理這次過來的數據所花的時間 
            latency_trasnport = car1_data/self.VtoM[setp_number]   #數據大小傳輸到MEC的速率
            latency_queue = waiting_data_number/self.mec_cpu[setp_number]   #計算MEC本來排隊的等待時間
            
            self.latency_total = latency_process + latency_queue + latency_trasnport
            #這邊計算1秒時間段後還剩下多少資料，因為我是設定每秒產生多少數據
            self.before_MEC_data = (latency_process + latency_queue - 1)*self.mec_cpu[setp_number]
            #每個time slot每個地方都在計算所以car2也在計算要減掉這秒car2計算的數據
            self.before_car2_data = self.before_car2_data-self.car_cpu[setp_number]
            
            if self.before_car2_data < 0 :
                self.before_car2_data = 0 
            if self.before_MEC_data < 0 :
                self.before_MEC_data = 0
                
#             if  (self.before_MEC_data + car1_data) / self.mec_cpu[setp_number]  >= 1.6:
#                 self.latency_total = (self.before_MEC_data + car1_data) /self.VtoC[setp_number]
#                 self.before_MEC_data = 0
            
            self.before_car1_data = 0 #car1車輛數據丟過來MEC後就不會有數據在排隊了
                
        elif action == 2:#傳輸到旁邊車輛計算
            waiting_data_number = self.before_car2_data  #這邊先計算上一秒的時間段有car2沒有排隊的數據
            
            car1_data = self.car1_data[setp_number] + self.before_car1_data#丟過來的data是包含在car1等待的data
            
            latency_process = (car1_data+self.car2_data[setp_number])/self.car_cpu[setp_number] #計算MEC處理所花的時間 
            latency_trasnport =  car1_data/self.VtoV[setp_number]   #數據大小傳輸到MEC的速率
            latency_queue =  waiting_data_number/self.car_cpu[setp_number]  
            
            self.latency_total = latency_process + latency_queue + latency_trasnport
            #這邊計算1秒時間段後還剩下多少資料，因為我是設定每秒產生多少數據
            self.before_car2_data = (latency_process + latency_queue - 1)*self.car_cpu[setp_number]
            #每個time slot每個地方都在計算所以MEC也在計算要減掉這秒MEC計算的數據
            self.before_MEC_data = self.before_MEC_data-self.mec_cpu[setp_number]
            
            if self.before_car2_data < 0 :
                self.before_car2_data = 0 
            if self.before_MEC_data  < 0 :
                self.before_MEC_data = 0
            
            if  self.before_car2_data / self.car_cpu[setp_number]  >= 2:
                trans_time1 = (car1_data + self.car2_data[setp_number]+waiting_data_number) /self.VtoM[setp_number]
                trans_time2 = (car1_data + self.car2_data[setp_number]+waiting_data_number) /self.VtoC[setp_number]
                self.latency_total = trans_time1 + trans_time2 + latency_trasnport
                self.before_car2_data = 0

 
            self.before_car1_data = 0 #car1車輛數據丟過來car2後就不會有數據在排隊了
              
                
        elif action == 3:#傳輸到cloud計算假設CLOUD很強所以直接忽略掉排隊跟計算，只算傳輸時間
            car1_data = self.car1_data[setp_number] + self.before_car1_data

            #這邊數據會先傳輸到MEC再傳輸到cloud
            latency_trasnport_V2M =  car1_data/self.VtoM[setp_number]   
            latency_trasnport_M2C =  car1_data/self.VtoC[setp_number]
            
            self.latency_total = latency_trasnport_V2M + latency_trasnport_M2C
            self.before_car1_data = 0
            
            #每個time slot每個地方都在計算所以car2也在計算要減掉這秒car2計算的數據
            self.before_car2_data = self.before_car2_data-self.car_cpu[setp_number]
            
            if self.before_car2_data < 0 :
                self.before_car2_data = 0 
            self.before_MEC_data = 0
            
            
        self.latency_csv[setp_number] =  self.latency_total
        
        return -self.latency_total, self.before_car1_data, self.before_car2_data, self.before_MEC_data,self.latency_csv
    
    def reset(self):
        self.observation[0] = (self.car1_data[0])
        self.observation[1] = (self.car2_data[0])
        self.observation[2] = (self.mec_data[0])
        #print(self.observation)

        return self.observation
    
    def step(self,action,setp_number,car1data ,car2data ,MECdata):
        #這邊設定跑完一次整個數據結束迴圈，取決於來了多少秒=data_time_slot
        done = False
        if setp_number == self.data_time_slot-1 :
            done = True
        #print("setp_number",setp_number)    
        #下一個狀態
        #print("self.car1_data[setp_number]",setp_number,self.car1_data[setp_number])
        self.observation_[0] = self.car1_data[setp_number]
        self.observation_[1] = self.car2_data[setp_number]
        self.observation_[2] = self.mec_data[setp_number]

        
        #計算reward 
        reward ,car1data ,car2data ,MECdata , total_latency = self.action_offload(action,setp_number,car1data ,car2data ,MECdata)
        
        return  self.observation_ ,reward, done ,{} ,car1data ,car2data ,MECdata ,total_latency
    
  
        
