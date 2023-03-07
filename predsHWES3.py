# import pdb;pdb.set_trace()
# dataframe opertations - pandas
import pandas as pd
import numpy
import json
from datetime import datetime
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# holt winters 
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error
from elasticsearch import Elasticsearch, helpers

class HWESSimple:

    def __init__(self, param_jason):
       print ("init")
       with open(param_jason) as f:
          self.param_data = json.load(f)

       self.myTS_data = pd.DataFrame()
       self.myTS = pd.DataFrame()
       self.alpha = float()
       self.preds = float()


    def elkOpen(self):
       self.esServer = self.param_data["ESServer"]
       self.esUser = self.param_data["ESUser"]
       self.esPwd = self.param_data["ESPwd"]
       self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))
       print ("After connection to ES")



    # ---- getting the TS
    def getTS(self):
        print ("Getting the Time Series")
        ts_ready_tmp = self.es.search(index="ts_ready", body={"query": {"match_all" : {}}, "size" : 1000})

        data = (ts_ready_tmp['hits']['hits'])
        # print(ts_ready_tmp['hits']['hits'])
        self.myTS_data=pd.json_normalize(data)


    def frameShape(self):
        # finding shape of the dataframe
        print(self.myTS_data.shape)
        self.myTS = self.myTS_data[['_source.Date','_source.Quantity']]
        self.myTS.rename(columns = {'_source.Date':'Date', '_source.Quantity':'Quantity'}, inplace = True)

        decompose_result = seasonal_decompose(self.myTS_data['_source.Quantity'],model='multiplicative',period=12)

    def setFreq(self):
        # Set the frequency of the date time index as Monthly start as indicated by the data
        self.myTS_data.index.freq = 'D'
        self.myTS.index.freq = 'D'
        # airline.index.freq = 'MS'
        # Set the value of Alpha and define m (Time Period)
        m = 12
        self.alpha = 1/(2*m)

    def tripleHWES(self):
        # print (self.myTS)
        # forecast_data.index.freq = 'D'
        self.myTS = self.myTS.set_index('Date')
        self.myTS.index.freq = 'D'
        fitted_model = ExponentialSmoothing(self.myTS['Quantity'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
        self.myTS['HWES3_MUL'] = fitted_model.fittedvalues
        # test predictions 
        self.preds = fitted_model.forecast(1)

        print(self.preds)
        self.myTS.reset_index(inplace=True)
        # print(type(self.myTS))
        # print(self.myTS.shape)

    def errorComp(self,ref_data, tbc_data):
        print(f'Mean Absolute Error = {mean_absolute_error(ref_data,tbc_data)}')
        print(f'Mean Squared Error = {mean_squared_error(ref_data,tbc_data)}')


    def mainProcess(self):
        self.elkOpen() 
        self.getTS() 
        self.frameShape()

        self.setFreq()
        # self.singleHWES()
        # self.doubleHWES()
        self.tripleHWES()
        self.errorComp(self.myTS['Quantity'],self.myTS['HWES3_MUL'])
        nbr = self.put_Preds()

    def put_Preds(self):
      pred = self.preds.to_numpy()[0]
      # print (pred[0])
      pred_date_tmp = self.preds.index.to_numpy()[0]
      # print (str(pred_date_tmp)[0:10])
      pred_date = str(pred_date_tmp)[0:10]
      # print (' Pred Date ' , pred_date) 
      df_json=self.myTS.to_json(orient='records', date_format = 'iso')
      parsed=json.loads(df_json)
      # parsed.append({'Date': pred_date, 'Quantity': 0 , 'HWES3_MUL': pred })
      parsed.append({'Date': pred_date, 'HWES3_MUL': pred })
      try:
         response = helpers.bulk(self.es,parsed, index='preds_ts')
         print ("ES response : ", response )
      except Exception as e:
         print ("ES Error :", e)

      return(len(parsed))

