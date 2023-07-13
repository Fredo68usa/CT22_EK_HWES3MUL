import pandas as pd
import numpy
import json
import datetime as dt
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# holt winters 
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error
from elasticsearch import Elasticsearch, helpers
import ct22posgresql as psg
import math

class TSPredGen:

    def __init__(self, param_jason):
       print ("init")
       with open(param_jason) as f:
          self.param_data = json.load(f)

       self.myTS_data = pd.DataFrame()
       self.myTS = pd.DataFrame()
       self.myTS_train = pd.DataFrame()
       self.myTS_fitted = pd.DataFrame()
       self.myTS_global = pd.DataFrame()
       self.preds_df = pd.DataFrame()
       self.alpha = float()
       self.predsHWES3MUL = pd.Series()
       self.predsHWES1 = pd.Series()
       self.SQLHash = self.param_data["SQLHash"]
       self.DayOfWeek = self.param_data["DayOfWeek"]
       self.esTZ = self.param_data["ESTZ"]
       self.esInIndex = self.param_data["ESInIndex"]
       self.esOutIndex = self.param_data["ESOutIndex"]
       self.SeasonalPeriod = self.param_data["SeasonalPeriod"]
       self.Alpha = self.param_data["Alpha"]
       self.IndexFreq = self.param_data["IndexFreq"]
       self.Train = self.param_data["Train"]
       self.Forecast = self.param_data["Forecast"]
       self.rmse_HWES3MUL = float()

       self.rmse_HWES1 = float()
       self.preds_interval_HWES3MUL = float()
       self.preds_interval_HWES1 = float()
       self.now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
       print (self.now , type(self.now))

    def getPreds(self,p1):
       # body = """SELECT * from currentpreds where hash='%s';"""
       body = """SELECT * from currentpreds;"""
       # self.cursor.execute("SELECT version();")
       # cur.execute(sql, (value1,value2))
       # p1.cursor.execute(body,self.SQLHash)
       p1.cursor.execute(body)
       records = p1.cursor.fetchall()
       # print("Read Predictions - ", records,"\n")
       # print("Read Predictions - ","\n")
       self.preds_df = pd.DataFrame(records,columns=['hash','year','dayofyear','predstype','preds','preds_interval','anomaly','excessqty'])
       # print(self.preds_df)

       # exit(0)

    def elkOpen(self):
       self.esServer = self.param_data["ESServer"]
       self.esUser = self.param_data["ESUser"]
       self.esPwd = self.param_data["ESPwd"]
       self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))
       print ("After connection to ES")

    # ---- getting the TS
    def getTS(self):
        print ("Getting the Time Series")
        body={"query" : { "bool" : { "must" : [{"match": {"HashHash" : self.SQLHash}} ,{ "match": {"DayOfWeek" : "Thursday"} } ] } } , "size" : 1000}
        ts_ready_tmp = self.es.search(index=self.esInIndex, body=body)
        data = (ts_ready_tmp['hits']['hits'])
        # myTS_data = pd.DataFrame()
        for row in range(len(data)) :
            src =  data[row]['_source']
            self.myTS_data=self.myTS_data.append(pd.json_normalize(src))
            # self.myTS_data['Time of Run'] = self.now
            # print(self.myTS_data)
        # myTS_data.reset_index(drop=True)
        # myTS_data.index=myTS_data['HashHash']
        self.myTS_data.reset_index(drop=True,inplace=True)
        # print('self.myTS_data', self.myTS_data)
        # print('self.myTS_data', self.myTS_data)
        # exit(0)

        # --- Taking into account the Anomalies
        for i in range(len(self.myTS_data)):
            predFoundDf = self.preds_df[(self.preds_df.hash == self.myTS_data.iloc[i]['HashHash']) & (self.preds_df.year == self.myTS_data.iloc[i]['Year'])& (self.preds_df.dayofyear == self.myTS_data.iloc[i]['DayOfYear'])]
            if predFoundDf.empty == False :
               # print (predFoundDf['excessqty'])
               # print (self.myTS_data.loc[i]['Records Affected'])
               # print (self.myTS_data[i]['Records Affected'])
               self.myTS_data.at[i,'Records Affected'] = self.myTS_data.loc[i]['Records Affected'] - predFoundDf['excessqty']

        self.myTS_data.rename(columns = {'Timestamp Local Time':'Date', 'Records Affected':'Quantity'}, inplace = True)
        self.myTS_data = self.myTS_data.set_index('Date')
        self.myTS_data = self.myTS_data.sort_index()

        # print(" ---> self.myTS_data")
        # print(self.myTS_data)
        # exit(0)

    def put_TS_Train_Test_Init(self):

      # print(" data : " , self.myTS_data)
      nbrTrainPeriod = int(self.Train*(len(self.myTS_data)))
      self.nbrTestPeriod = len(self.myTS_data) - nbrTrainPeriod
      self.myTS_train = self.myTS_data[:nbrTrainPeriod].copy()
      self.myTS_test = self.myTS_data[nbrTrainPeriod:].copy()


      # Prepare for Preds Computation
      self.TS_test_fit = self.myTS_test.copy()
      # print ("===========================================================================")

      self.TS_test_fit.drop(['Quantity'], axis=1,inplace = True )
      # print(self.TS_test_fit)
      # print(self.myTS_test)


      # exit(0)


    def put_TS_Train_Test(self,algo):
      print ( " ============== ")
      print ( " In put_TS_Train_test ")
      print ( " ============== ")

      TS_train = self.myTS_train.copy()
      TS_test = self.myTS_test.copy()
      TS_train["Type Of Data"] = "Train"
      TS_train["Time Of Simul"] = self.now
      TS_train["Algorithm"] = algo
      TS_train["DataTypeAlgo"] = "Train " + algo
      TS_test["Type Of Data"] = "Test"
      TS_test["Time Of Simul"] = self.now
      TS_test["Algorithm"] = algo
      TS_test["DataTypeAlgo"] = "Test " + algo

      print(TS_train)
      print(TS_test)

      # Record Train set
      TS_train.reset_index(inplace=True)
      df_json=TS_train.to_json(orient='records', date_format = 'iso')
      parsed=json.loads(df_json)
      # print (parsed)
      self.insertES_bulk(parsed)

      # record Test set
      TS_test.reset_index(inplace=True)
      df_json=TS_test.to_json(orient='records', date_format = 'iso')
      parsed=json.loads(df_json)
      # print (parsed)
      self.insertES_bulk(parsed)

    def frameShape(self):
        self.myTS = self.myTS_data[['Quantity']]
        # decompose_result = seasonal_decompose(self.myTS['Quantity'],model='multiplicative',period=12)

    def setFreq(self):
        # Set the frequency of the date time index as Monthly start as indicated by the data
        self.myTS_data.index.freq = self.IndexFreq
        self.myTS.index.freq = self.IndexFreq
        m = self.Alpha
        self.alpha = 1/(2*m)

    def singleHWES(self):
        print("  In  Simple HWES ")
        self.put_TS_Train_Test("HWES1")
        fitted_model = SimpleExpSmoothing(self.myTS_train['Quantity']).fit(smoothing_level=self.alpha,optimized=False,use_brute=True)
        self.myTS_fitted['HWES1'] = fitted_model.fittedvalues
        self.predsHWES1 = fitted_model.forecast(self.nbrTestPeriod + self.Forecast)
        print ("Preds dates/Quantities HWES1 : \n" , self.predsHWES1, " ", type(self.predsHWES1))
        TS_fit = self.predsHWES1.to_frame()
        nbr = self.put_Fit("HWES1",TS_fit)

    def doubleHWES(self):
        print("  ==============")
        print("  In  doubleHWES ")
        print("  ==============")

        # --- HWES2ADD ----
        self.put_TS_Train_Test("HWES2ADD")

        fitted_model = ExponentialSmoothing(self.myTS_train['Quantity'],trend='add').fit()
        self.myTS_fitted['HWES2_ADD'] = fitted_model.fittedvalues
        self.predsHWES2ADD = fitted_model.forecast(self.nbrTestPeriod + self.Forecast)
        print ("===========================================================================")
        print ("Preds dates/Quantities Double HWES ADD : \n" , self.predsHWES2ADD, " ", type(self.predsHWES2ADD))
        print ("===========================================================================")
        TS_fit = self.predsHWES2ADD.to_frame()
        nbr = self.put_Fit("HWES2ADD",TS_fit)

        # exit(0)

        # --- HWES2MUL ----
        self.put_TS_Train_Test("HWES2MUL")

        fitted_model = ExponentialSmoothing(self.myTS_train['Quantity'],trend='mul').fit()
        self.myTS_fitted['HWES2_MUL'] = fitted_model.fittedvalues
        # test predictions 
        self.predsHWES2MUL = fitted_model.forecast(self.nbrTestPeriod + self.Forecast)

        print ("===========================================================================")
        print ("Preds dates/Quantities Double HWES MUL : \n" , self.predsHWES2MUL, " ", type(self.predsHWES2MUL))
        print ("===========================================================================")

        TS_fit = self.predsHWES2MUL.to_frame()
        nbr = self.put_Fit("HWES2MUL",TS_fit)



    def tripleHWES(self):
        print("  ==============")
        print("  In  tripleHWES ")
        print("  ==============")

        # --- HWES3MUL ----
        self.put_TS_Train_Test("HWES3ADD")

        fitted_model = ExponentialSmoothing(self.myTS_train['Quantity'],trend='add',seasonal='add',seasonal_periods=self.SeasonalPeriod).fit()
        self.myTS_fitted['HWES3_ADD'] = fitted_model.fittedvalues
        self.predsHWES3ADD = fitted_model.forecast(self.nbrTestPeriod + self.Forecast)
        print ("===========================================================================")
        print ("Preds dates/Quantities Triple HWES ADD : \n" , self.predsHWES3ADD, " ", type(self.predsHWES3ADD))
        print ("===========================================================================")
        TS_fit = self.predsHWES3ADD.to_frame()
        nbr = self.put_Fit("HWES3ADD",TS_fit)

        # exit(0)


        # --- HWES3MUL ----
        self.put_TS_Train_Test("HWES3MUL")

        fitted_model = ExponentialSmoothing(self.myTS_train['Quantity'],trend='mul',seasonal='mul',seasonal_periods=self.SeasonalPeriod).fit()
        self.myTS_fitted['HWES3_MUL'] = fitted_model.fittedvalues
        # test predictions 
        self.predsHWES3MUL = fitted_model.forecast(self.nbrTestPeriod + self.Forecast)

        print ("===========================================================================")
        print ("Preds dates/Quantities Triple HWES : \n" , self.predsHWES3MUL, " ", type(self.predsHWES3MUL))
        print ("===========================================================================")

        TS_fit = self.predsHWES3MUL.to_frame()
        nbr = self.put_Fit("HWES3MUL",TS_fit)

    def put_Fit(self, algo, TS_fit):
      TS_fit.reset_index(inplace=True)
      # print(TS_fit)
      # exit(0)
      TS_fit.rename(columns = {'index':'Date' , 0 :'Quantity'}, inplace = True)
      TS_fit['Date'] = pd.to_datetime(TS_fit.Date)
      TS_fit['Date'] = TS_fit['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

      TS_fit.set_index('Date', inplace = True )
      print(TS_fit)

      print("---- In put Fit -----")
      print(type(self.TS_test_fit))
      print(self.TS_test_fit)
      # TS_test_fit.drop(['Quantity'], axis=1,inplace = True )
      # print(TS_test_fit)
      TS_test_fit = pd.concat([self.TS_test_fit,TS_fit[:len(self.myTS_test)]], axis=1)

      print(TS_test_fit)
      # print(TS_test)

      # exit(0)

      # record Test Fit set
      TS_test_fit["Type Of Data"] = "Test Fit"
      TS_test_fit["Time Of Simul"] = self.now
      TS_test_fit["Algorithm"] = algo
      TS_test_fit["DataTypeAlgo"] = "Test Fit " + algo
      print(TS_test_fit)

      # exit(0)

      TS_test_fit.reset_index(inplace=True)
      df_json=TS_test_fit.to_json(orient='records', date_format = 'iso')
      parsed=json.loads(df_json)
      print (parsed)
      self.insertES_bulk(parsed)

      # exit(0)



    def mainProcess(self):

        # Opening the postGresql DB
        p1 = psg.CT22PosGreSQL()
        p1.open_PostGres()

        # Getting the Preds
        self.getPreds(p1)

        # Getting the TS
        self.elkOpen()
        self.getTS()
        # self.frameShape()
        # print("After frameShape")

        # Recording of the TS in ES as Train/Test sets
        self.put_TS_Train_Test_Init()
        # self.put_TS_Train_Test("TonThon")
        # exit(0)

        # Computing the Preds
        self.frameShape()
        self.setFreq()
        self.singleHWES()
        # exit(0)
        self.doubleHWES()
        exit(0)
        # self.frameShape()
        self.tripleHWES()

        # exit(0)
        # Computng the pred interval
        # self.rmse_HWES3MUL, self.preds_interval_HWES3MUL = self.errorComp(self.myTS['Quantity'],self.myTS_fitted['HWES3_MUL'])
        # self.rmse_HWES1 , self.preds_interval_HWES1 = self.errorComp(self.myTS['Quantity'],self.myTS_fitted['HWES1'])


        # Recording the reference pred
        preType = input (" Select type of pred to record : 1-HWES3MUL ")
        # p1 = psg.CT22PosGreSQL()
        # p1.open_PostGres()

        # print(p1.postgres_connect)
        # print(p1.cursor)
        # self.write_PosGres(p1)
        p1.close_PosGres()


    def insertES_bulk(self,parsed):
      # exit(0)
      try:
         response = helpers.bulk(self.es,parsed, index=self.esOutIndex)
         print ("ES response : ", response )
      except Exception as e:
         print ("ES Error :", e)

