import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import numpy as np
from sklearn.pipeline import make_pipeline

calib_path = 'data/Calib_hrrrHourly_summerCT_townFeat_totCounts.csv'
events_path = 'data/tStormList_18hr_CT_2016+.csv'

STORM_DURATION = 18

##############################
## PROCESS CALIBRATION FILE ##
##############################

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

#########################
## PROCESS EVENTS FILE ##
#########################

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

#########
## PCA ##
#########

for eventNum, eventDT in enumerate(eventDTs):
    print("Saving PCA for training set of", eventNum, eventDT)
    
    # Set critical points around event
    intEventStartDT = int(datetime.strftime(eventDT, "%Y%m%d%H"))
    if intEventStartDT < intStartDT: continue # Check if event is too close to start of timeseries for setting up a lookback
    intEventEndDT = int(datetime.strftime(eventDT + timedelta(hours=STORM_DURATION-1), "%Y%m%d%H"))
    intAfterEventDT = int(datetime.strftime(eventDT + timedelta(hours=STORM_DURATION), "%Y%m%d%H"))
    
    train_X = pd.concat([df.loc[intStartDT:intEventStartDT], df.loc[intAfterEventDT:intEndDT]]).iloc[:,:-1]

    scale_pca = make_pipeline(StandardScaler(), PCA(100)).fit(train_X)
    print('PCA explained variance:', sum(scale_pca.named_steps['pca'].explained_variance_ratio_))
    df_pca = pd.DataFrame(scale_pca.transform(df.iloc[:,:-1]))
    df_pca.index = df.index
    df_pca.columns = ["PC"+str(i) for i in range(1,101)]
    df_pca["countts"] = df["countts"]
    df_pca.to_csv("data/LOSO/Calib_hrrrHourly_summerCT_townFeat_totCounts_storm{:02}.csv".format(eventNum+1))
