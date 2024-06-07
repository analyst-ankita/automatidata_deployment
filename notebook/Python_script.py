
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

import pickle


# %%
df = pd.read_csv(r'data\2017_Yellow_Taxi_Trip_Data.csv')

pd.set_option('display.max_columns', None)

# %%
df.head(10)

# %%
df.info()

# %%
df.describe()

# Convert data columns to datetime

df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# %%
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])/np.timedelta64(1,'m')

# %%
# Create box plot of trip_distance
plt.figure(figsize=(7,2))
plt.title('trip_distance')
sns.boxplot(data=None, x=df['trip_distance'], fliersize=1);

# %%
# Create histogram of trip_distance
plt.figure(figsize=(10,5))
sns.histplot(df['trip_distance'], bins=range(0,26,1))
plt.title('Trip distance histogram');

# %% [markdown]
# The majority of trips were journeys of less than two miles. The number of trips falls away steeply as the distance traveled increases beyond two miles.

# %%
# Create box plot of total_amount
plt.figure(figsize=(7,2))
plt.title('total_amount')
sns.boxplot(x=df['total_amount'], fliersize=1)

# %%
# Create histogram of total_amount
plt.figure(figsize=(12,6))
ax = sns.histplot(df['total_amount'], bins=range(-10,101,5))
ax.set_xticks(range(-10,101,5))
ax.set_xticklabels(range(-10,101,5))
plt.title('Total amount histogram');

# %% [markdown]
# The total cost of each trip also has a distribution that skews right, with most costs falling in the $5-15 range.

# %%
# Create box plot of tip_amount
plt.figure(figsize=(7,2))
plt.title('tip_amount')
sns.boxplot(x=df['tip_amount'], fliersize=1);

# %%
# Create histogram of tip_amount
plt.figure(figsize=(12,6))
ax = sns.histplot(df['tip_amount'], bins=range(0,21,1))
ax.set_xticks(range(0,21,2))
ax.set_xticklabels(range(0,21,2))
plt.title('Tip amount histogram');

df['passenger_count'].value_counts()

# Create a month column
df['month'] = df['tpep_pickup_datetime'].dt.strftime('%b').str.lower()
# Create a day column
df['day'] = df['tpep_pickup_datetime'].dt.day_name().str.lower()

# Total number of rides for each month
monthly_rides = df['month'].value_counts()
monthly_rides

# %%
# Reorder the monthly ride list so months go in order
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
         'aug', 'sep', 'oct', 'nov', 'dec']

monthly_rides = monthly_rides.reindex(index=month_order)
monthly_rides

# %%
# Create a bar plot of total rides per month
plt.figure(figsize=(12,7))
ax = sns.barplot(x=monthly_rides.index, y=monthly_rides)
ax.set_xticklabels(month_order)
plt.title('Ride count by month', fontsize=16);

# %% [markdown]
#  Monthly rides are fairly consistent, with notable dips in the summer months of July, August, and September, and also in February.

# %%
# Repeating the above process, this time for rides by day
daily_rides = df['day'].value_counts()
day_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
daily_rides = daily_rides.reindex(index=day_order)
daily_rides

# %%
# Create bar plot for ride count by day
plt.figure(figsize=(12,7))
ax = sns.barplot(x=daily_rides.index, y=daily_rides)
ax.set_xticklabels(day_order)
ax.set_ylabel('Count')
plt.title('Ride count by day', fontsize=16);

# %% [markdown]
# Suprisingly, Wednesday through Saturday had the highest number of daily rides, while Sunday and Monday had the least.

# %%
# Repeating the process, this time for total revenue by day

total_amount_day = df.groupby('day')[['total_amount']].sum()
total_amount_day = total_amount_day.reindex(index=day_order)
total_amount_day

# %%
# Create bar plot of total revenue by day
plt.figure(figsize=(12,7))
ax = sns.barplot(x=total_amount_day.index, y=total_amount_day['total_amount'])
ax.set_xticklabels(day_order)
ax.set_ylabel('Revenue (USD)')
plt.title('Total revenue by day', fontsize=16);

# %% [markdown]
# Thursday had the highest gross revenue of all days, and Sunday and Monday had the least. Interestingly, although Saturday had only 35 fewer rides than Thursday, its gross revenue was ~\$6,000 less than Thursday's—more than a 10% drop.

# %%
# Repeating the process, this time for total revenue by month
total_amount_month = df.groupby('month')[['total_amount']].sum()
total_amount_month = total_amount_month.reindex(index=month_order)
total_amount_month

# %%
# Create a bar plot of total revenue by month
plt.figure(figsize=(12,7))
ax = sns.barplot(x=total_amount_month.index, y=total_amount_month['total_amount'])
plt.title('Total revenue by month', fontsize=16);

# %% [markdown]
# Monthly revenue generally follows the pattern of monthly rides, with noticeable dips in the summer months of July, August, and September, and also one in February.

# %% [markdown]
# ## **Building a multiple linear regression model:**
# 
# Keeping in mind that many of the features will not be used to fit our model, the most important columns to check for outliers are likely to be:
# * `trip_distance`
# * `fare_amount`
# * `duration`
# 

# %% [markdown]
# #### **Box plots**

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 2))
fig.suptitle('Boxplots for outlier detection')
sns.boxplot(ax=axes[0], x=df['trip_distance'],fliersize=2)
sns.boxplot(ax=axes[1], x=df['fare_amount'],fliersize=2)
sns.boxplot(ax=axes[2], x=df['duration'],fliersize=2)
plt.show();

# Are trip distances of 0 bad data or very short trips rounded down?
sorted(set(df['trip_distance']))[:10]

# are there enough zero values in the data to pose a problem?
sum(df['trip_distance']==0)


# %%
df['fare_amount'].describe()

# Imputing the maximum value as `Q3 + (6 * IQR)`.

def outlier_imputer(column_list, iqr_factor):
    '''
    Impute upper-limit values in specified columns based on their interquartile range.

    Arguments:
        column_list: A list of columns to iterate over
        iqr_factor: A number representing x in the formula:
                    Q3 + (x * IQR). Used to determine maximum threshold,
                    beyond which a point is considered an outlier.

    The IQR is computed for each column in column_list and values exceeding
    the upper threshold for each column are imputed with the upper threshold value.
    '''
    for col in column_list:
        # Reassign minimum to zero
        df.loc[df[col] < 0, col] = 0

        # Calculate upper threshold
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_threshold = q3 + (iqr_factor * iqr)
        print(col)
        print('q3:', q3)
        print('upper_threshold:', upper_threshold)

        # Reassign values > threshold to threshold
        df.loc[df[col] > upper_threshold, col] = upper_threshold
        print(df[col].describe())
        print()

# %%
outlier_imputer(['fare_amount'], 6)

# %% [markdown]
# #### **`duration` outliers**

# %%
df['duration'].describe()

# Impute the high outliers
outlier_imputer(['duration'], 6)

# Create `pickup_dropoff` column
df['pickup_dropoff'] = df['PULocationID'].astype(str) + ' ' + df['DOLocationID'].astype(str)
df['pickup_dropoff'].head(2)

# %%
grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
grouped[:5]

# %%
# 1. Convert `grouped` to a dictionary
grouped_dict = grouped.to_dict()

# 2. Reassign to only contain the inner dictionary
grouped_dict = grouped_dict['trip_distance']

# %%
# 1. Create a mean_distance column that is a copy of the pickup_dropoff helper column
df['mean_distance'] = df['pickup_dropoff']

# 2. Map `grouped_dict` to the `mean_distance` column
df['mean_distance'] = df['mean_distance'].map(grouped_dict)

# Confirm that it worked
df[(df['PULocationID']==100) & (df['DOLocationID']==231)][['mean_distance']]

# %% [markdown]
# #### Creating `mean_duration` column
# 
# Repeating the above process to calculate mean duration.

# %%
grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['duration']]
grouped[:5]

# %%
grouped_dict = grouped.to_dict()

grouped_dict = grouped_dict['duration']

# %%
df['mean_duration'] = df['pickup_dropoff']
df['mean_duration'] = df['mean_duration'].map(grouped_dict)
df[(df['PULocationID']==100) & (df['DOLocationID']== 100)][['mean_duration']]

# %%
df.head()

# Creating 'rush_hour' col
df['rush_hour'] = df['tpep_pickup_datetime'].dt.hour

# If day is Saturday or Sunday, imputing 0 in `rush_hour` column
df.loc[df['day'].isin(['Saturday', 'Sunday']), 'rush_hour'] = 0

# %%
def rush_hourizer(hour):
    if 6 <= hour['rush_hour'] < 10:
        val = 1
    elif 16 <= hour['rush_hour'] < 20:
        val = 1
    else:
        val = 0
    return val

# %%
# Applying the `rush_hourizer()` function to the new column
df.loc[(df.day != 'saturday') & (df.day != 'sunday'), 'rush_hour'] = df.apply(rush_hourizer, axis=1)
df.head()

# Creating a scatter plot of duration and trip_distance, with a line of best fit
sns.set(style='whitegrid')
f = plt.figure()
f.set_figwidth(5)
f.set_figheight(5)
sns.regplot(x=df['mean_duration'], y=df['fare_amount'],
            scatter_kws={'alpha':0.5, 's':5},
            line_kws={'color':'red'})
plt.ylim(0, 70)
plt.xlim(0, 70)
plt.title('Mean duration x fare amount')
plt.show()

df[df['fare_amount'] > 50]['fare_amount'].value_counts().head()

# %%
df[df['fare_amount']==52].head()

# ### Isolating modeling variables
# 
# Droping features that are redundant, irrelevant, or that will not be available in a deployed environment.

# %%
df2 = df.copy()

df2 = df2.drop(['Unnamed: 0', 'tpep_dropoff_datetime', 'tpep_pickup_datetime',
               'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
               'payment_type', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
               'total_amount', 'tpep_dropoff_datetime', 'tpep_pickup_datetime', 'duration',
               'pickup_dropoff', 'day', 'month'
               ], axis=1)

df2.info()

# %%
# Creating a pairplot to visualize pairwise relationships between variables in the data
sns.pairplot(df2[['fare_amount', 'mean_duration', 'mean_distance']],
             plot_kws={'alpha':0.4, 'size':5},
             );

# %%
# Creating correlation heatmap

plt.figure(figsize=(6,4))
sns.heatmap(df2.corr(method='pearson'), annot=True, cmap='Reds')
plt.title('Correlation heatmap',
          fontsize=18)
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)

# %%
# Removing the target column from the features
X = df2.drop(columns=['fare_amount'])

# Setting y variable
y = df2[['fare_amount']]

# %%
X = pd.get_dummies(X, columns = ['VendorID'], dtype= int)
X.head()

# %%
# Creating training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
# Standardize the X variables
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

# %%
# Fitting the model to the training data
lr=LinearRegression()
lr.fit(X_train_scaled, y_train)

# %%
# Evaluating the model performance on the training data
r_sq = lr.score(X_train_scaled, y_train)
print('Coefficient of determination:', r_sq)
print()
y_pred_train = lr.predict(X_train_scaled)

print('R^2:', r2_score(y_train, y_pred_train))
print()
print('MAE:', mean_absolute_error(y_train, y_pred_train))
print()
print('MSE:', mean_squared_error(y_train, y_pred_train))
print()
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_pred_train)))

# %%
# Scaling the X_test data
X_test_scaled = scaler.transform(X_test)

# %%
# Evaluating the model performance on the testing data
r_sq_test = lr.score(X_test_scaled, y_test)
print('Coefficient of determination:', r_sq_test)
print()
y_pred_test = lr.predict(X_test_scaled)

print('R^2:', r2_score(y_test, y_pred_test))
print()
print('MAE:', mean_absolute_error(y_test,y_pred_test))
print()
print('MSE:', mean_squared_error(y_test, y_pred_test))
print()
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred_test)))

# Getting model coefficients
coefficients = pd.DataFrame(lr.coef_, columns=X.columns)
coefficients

# %% [markdown]
# ### 1. Predict on full dataset

# %%
X_scaled = scaler.transform(X)
y_preds_full = lr.predict(X_scaled)

# %%
with open('lr.pkl','wb') as f:
    pickle.dump(lr, f)


