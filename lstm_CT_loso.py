import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from keras.wrappers.scikit_learn import KerasRegressor

# Adjust for no tkinter
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

calib_path = './data/calibs/Calib_hrrrHourly_summerCT_townFeat_totCounts_withREFD_2016-07-15--2018-09-30_filled.csv'
results_path = './results/lstm_predictions.csv'
events_path = './data/tStormList_18hr_CT_2016+.csv'

STORM_DURATION = 18

####################
##--TRANSFORMERS--##
####################

# Adds lag features
class Lookback(TransformerMixin):
    def __init__(self, n_hours_past):
        self.n_hours_past = n_hours_past

    def fit(self, X, y):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)

        lookbacks = []
        for i in range(self.n_hours_past, len(df)):
            lookback = df.iloc[(i-self.n_hours_past):(i+1)]
            lookbacks.append(lookback.values.tolist())

        lookbacks = np.array(lookbacks)
        print('training input shape:', lookbacks.shape)
        return lookbacks

################################
##--PROCESS CALIBRATION FILE--##
################################

print('\n\n...processing calib...')
dataset = pd.read_csv(calib_path)

# Set formatted DT as index
DTs = [datetime.strptime(str(dt), "%Y%m%d%H") for dt in list(dataset['dateTime'])]
df = dataset.set_index('dateTime')
print('dataframe shape:', df.shape)

# Cyclically encode hour, month features
df['hour_sin'] = np.sin(2*np.pi*df['hour']/23)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/23)
df.drop('hour', axis=1, inplace=True)

# Move response to last column for easy access later
df['countts'] = df.pop('countts')

# Restrict training set to prediction window
intStartDT, intEndDT = 2016071500, 2018093023
df = df.loc[intStartDT:intEndDT]
DTs = [dt for dt in DTs if intStartDT <= int(datetime.strftime(dt, "%Y%m%d%H")) <= intEndDT]
print('Length of time series: ', len(DTs))



###########################
##--PROCESS EVENTS FILE--##
###########################

# Select event DTs
print('\n...processing events...')

eventDataset = pd.read_csv(events_path)

# Format file, timestamps
strEventDs = [str(d) for d in list(eventDataset['StartDate'])]
eventTs = list(eventDataset['StartUTChourtime'])
strEventDTs = [strEventDs[i]+' '+str(eventTs[i]) for i in range(0, len(strEventDs))]
eventDTs = [datetime.strptime(strDT, "%m/%d/%Y %H") for strDT in strEventDTs]

# Restrict events set
startDT, endDT = datetime.strptime(str(intStartDT), "%Y%m%d%H"), datetime.strptime(str(intEndDT), "%Y%m%d%H")
eventDTs = [dt for dt in eventDTs if startDT <= dt <= endDT]

#############
##--MODEL--##
#############

# Set feature/response window
n_hours_past = 16

# Set up LSTM
def create_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(75, input_shape=input_shape, return_sequences=False))
    model.add(Dense(4))
    model.add(Dense(1, activation="relu"))
    model.compile(loss='mse', optimizer='adam')
    return model

# Set number of PCA components
n_pca = 100

# Set up early stopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=4,
                          verbose=0, mode='auto')

##################################
##--REPEATED K-FOLD VALIDATION--##
##################################

# Set up final actual, predicted
pred_y_all = np.array([])

r = 1  # Set repeated prediction
for _ in range(0, r):
    # Set up predictions timeseries for rth run
    pred_y_r = np.array([])

    for eventDT in eventDTs:
        # Set critical points around event
        intEventStartDT = int(datetime.strftime(eventDT - timedelta(hours=n_hours_past), "%Y%m%d%H"))
        if intEventStartDT < intStartDT: continue # Check if event is too close to start of timeseries for setting up a lookback
        intEventEndDT = int(datetime.strftime(eventDT + timedelta(hours=STORM_DURATION-1), "%Y%m%d%H"))
        intAfterEventDT = int(datetime.strftime(eventDT + timedelta(hours=STORM_DURATION), "%Y%m%d%H"))

        # Set event-appropriate train, test intervals
        train_X = pd.concat([df.loc[intStartDT:intEventStartDT], df.loc[intAfterEventDT:intEndDT]]).values[:,:-1]
        train_y = np.append(df.loc[intStartDT:intEventStartDT].values[:,-1], df.loc[intAfterEventDT:intEndDT].values[:,-1])
        test_X, test_y = df.loc[intEventStartDT:intEventEndDT].values[:,:-1], df.loc[intEventStartDT:intEventEndDT].values[:,-1]
        print("test input shape:", test_y.shape)

        train_y = train_y.reshape(train_y.shape[0], 1)[n_hours_past:,:]
        test_y = test_y.reshape(test_y.shape[0], 1)[n_hours_past:,:]

        sample_weight = np.array([.5*(np.tanh((y-20)/2)[0]+1)+.001 for y in train_y])

        # Create base lSTM model
        lstm = KerasRegressor(build_fn=create_lstm, input_shape=(n_hours_past+1, n_pca),
                              epochs=25, sample_weight=sample_weight,
                              verbose=0, shuffle=False)

        # Create LSTM pipeline with scaling and lookback transformations
        lstm_model = make_pipeline(StandardScaler(), PCA(n_pca),
                                   Lookback(n_hours_past), lstm)

        # Fit model
        print('...fitting model around event ' + str(intEventStartDT) + '...')
        fitted = lstm_model.fit(train_X, train_y)

        print('PCA explained variance:',
              sum(fitted.named_steps['pca'].explained_variance_ratio_))

        # Get prediction for event
        pred_y = lstm_model.predict(test_X)
        print('predicted:', np.sum(pred_y))
        print("predicted shape:", pred_y.shape)

        # Append to rth prediction timeseries
        pred_y_r = np.append(pred_y_r, pred_y)

    # Stack kth timeseries
    pred_y_all = np.vstack((pred_y_all, pred_y_r)) if pred_y_all.size else pred_y_r

# Take mean predictions
pred_y_mean = np.average(pred_y_all, axis=0) if pred_y_all.ndim > 1 else pred_y_all

############
##--SAVE--##
############

# Get datetimes, actuals for prediction range
def during_event(dt):
    for eventDT in eventDTs:
        if eventDT <= dt < (eventDT + timedelta(hours=STORM_DURATION)): return True
resultDTs = [dt for dt in DTs if during_event(dt)]
strResultDTs = [datetime.strftime(dt, "%Y%m%d%H") for dt in resultDTs]
test_y_all = df.loc[[int(strResultDT) for strResultDT in strResultDTs]].values[:,-1]

# Write results to output file
print('\n\n...writing results to %s...' % results_path)
results = pd.DataFrame(np.column_stack((strResultDTs, pred_y_mean, test_y_all)))
results.columns = ['dateTime', 'Predicted', 'Actual']
results.to_csv(results_path, index=False)

############
##--PLOT--##
############

# Plot actual and predicted for test interval
print('\n\n...plotting...')
plt.plot(resultDTs, test_y_all, label='actual')
plt.plot(resultDTs, pred_y_mean, label='predicted', alpha=.7)
plt.title('Storm-Caused Outage Timeseries')
plt.legend()
plt.savefig("results.png")
