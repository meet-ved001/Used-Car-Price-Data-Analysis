import pandas as pd
import numpy as np
car_train_data = pd.read_csv("train-data.csv")
## Data Cleaning Methods
# renaming price columns
car_train_data = car_train_data.rename(columns={'Price': 'Price(Lakh)'})
print(car_train_data)
car_train_data.info()

# remove Lakh word from New_Price column
car_train_data['New_Price'] = car_train_data['New_Price'].str.replace('Lakh', '')
car_train_data['New_Price'] = car_train_data['New_Price'].str.replace('Cr', '')
car_train_data = car_train_data.rename(columns={'New_Price': 'New_Price(Lakh)'})

print(car_train_data)

car_train_data.info()
car_train_data = car_train_data.drop(['Unnamed: 0'],axis=1)
print(car_train_data.isnull().sum())

# converting object to float data type
car_train_data['New_Price(Lakh)'] = pd.to_numeric(car_train_data['New_Price(Lakh)'], errors='coerce')
mean_value = car_train_data['New_Price(Lakh)'].mean()

# Replace NaNs in column New Price with the mean value
# mean of values in the same column
car_train_data['New_Price(Lakh)'].fillna(value=mean_value, inplace=True)
car_train_data['New_Price(Lakh)'] = car_train_data['New_Price(Lakh)'].round(2)
print(car_train_data.isnull().sum())


# 1
# updating Mileage column and replacing null values with mean
car_train_data = car_train_data.rename(columns={'Mileage': 'Mileage(kmpl)'})
car_train_data['Mileage(kmpl)'] = car_train_data['Mileage(kmpl)'].str.replace('kmpl', '')
car_train_data['Mileage(kmpl)'] = car_train_data['Mileage(kmpl)'].str.replace('km/kg', '')
car_train_data['Mileage(kmpl)'] = pd.to_numeric(car_train_data['Mileage(kmpl)'], errors='coerce')

mean_value = car_train_data['Mileage(kmpl)'].mean()
# Replace NaNs in column Mileage with the mean value
# mean of values in the same column
car_train_data['Mileage(kmpl)'].fillna(value=mean_value, inplace=True)
car_train_data['Mileage(kmpl)'] = car_train_data['Mileage(kmpl)'].round(2)

# 2
# updating Engine column

car_train_data = car_train_data.rename(columns={'Engine': 'Engine(CC)'})
car_train_data['Engine(CC)'] = car_train_data['Engine(CC)'].str.replace('CC', '')
car_train_data['Engine(CC)'] = pd.to_numeric(car_train_data['Engine(CC)'], errors='coerce')

mean_value = car_train_data['Engine(CC)'].mean()
# Replace NaNs in column Engine with the mean value
# mean of values in the same column
car_train_data['Engine(CC)'].fillna(value=mean_value, inplace=True)
car_train_data['Engine(CC)'] = car_train_data['Engine(CC)'].round(2)

# 3
# updating power column

car_train_data = car_train_data.rename(columns={'Power': 'Power(bhp)'})
car_train_data['Power(bhp)'] = car_train_data['Power(bhp)'].str.replace('bhp', '')
car_train_data['Power(bhp)'] = pd.to_numeric(car_train_data['Power(bhp)'], errors='coerce')

mean_value = car_train_data['Power(bhp)'].mean()
# Replace NaNs in column Power(bhp) with the mean value
# mean of values in the same column
car_train_data['Power(bhp)'].fillna(value=mean_value, inplace=True)
car_train_data['Power(bhp)'] = car_train_data['Power(bhp)'].round(2)

# 4
# updating seats column

mean_value = car_train_data['Seats'].mean()
# Replace NaNs in column Seats with the mean value
# mean of values in the same column
car_train_data['Seats'].fillna(value=mean_value, inplace=True)
car_train_data.info()

print(car_train_data.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Location and New Price Bar graph
plt.figure(figsize=(12,6))
bar_width = 0.4

ax=sns.barplot(x='Location', y='New_Price(Lakh)',data=car_train_data, palette='viridis',ci=None)
# ax.patches stores value of each bar. This was used to show count of each Locations.
for p in ax.patches:
    height_rounded = round(p.get_height(), 2)
    ax.annotate(f'{height_rounded}', (p.get_x() + p.get_width() / 2., height_rounded),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
# Add labels and title
plt.xlabel('Location',color='blue',fontweight='bold')
plt.ylabel('New Price(Lakh)',color='blue',fontweight='bold')
plt.title('Location vs New Price(Lakh) observations',color='blue',fontweight='bold')
# Show the plot
plt.show()


# 2nd Location vs Price(Lakh) Strip plot
plt.figure(figsize=(12,6))
sns.stripplot(x='Location', y='Price(Lakh)',data=car_train_data, jitter=True, palette='muted')

plt.xlabel('Location',color='blue',fontweight='bold')
plt.ylabel('Price(Lakh)',color='blue',fontweight='bold')
plt.title('Strip Plot [ Location vs Price(Lakh) ]',color='blue',fontweight='bold')
plt.show()

# 3rd plot Transmission vs Mileage

ax=sns.boxplot(x='Transmission',y='Mileage(kmpl)',data=car_train_data, palette = 'RdBu')

# showing median values in the boxplot
# for i, transmission_type in enumerate(set(car_train_data['Transmission'])):
#     data = np.array([car_train_data['Mileage(kmpl)'][j] for j in range(len(car_train_data['Mileage(kmpl)'])) if car_train_data['Transmission'][j] == transmission_type])
#     median_val = np.median(data)
#     q1_val = np.percentile(data, 25)
#     q3_val = np.percentile(data, 75)

#     ax.annotate(f'Median: {median_val:.2f}\nQ1: {q1_val:.2f}\nQ3: {q3_val:.2f}',
#                 xy=(i, max(q1_val, min(data))),
#                 ha='center', va='center', color='white', size=10,
#                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='steelblue'))

plt.xlabel('Transmission Type of a Car',color='blue',fontweight='bold')
plt.ylabel('Mileage Values (kmpl)',color='blue',fontweight='bold')
plt.title('Boxplot for Transmission vs Mileage',color='blue',fontweight='bold')

# Show the plot
plt.show()

# 4th plot Fuel type vs Mileage
ax=sns.boxplot(x='Fuel_Type',y='Mileage(kmpl)',data=car_train_data, palette = 'colorblind')
plt.xlabel('Fuel Type of a Car',color='blue',fontweight='bold')
plt.ylabel('Mileage Values (kmpl)',color='blue',fontweight='bold')
plt.title('Boxplot for Fuel Type vs Mileage',color='blue',fontweight='bold')

# Show the plot
plt.show()

# 5th plot co-relation between mileage and engine (scatter plot and correlation function)
plt.figure(figsize=(12,6))
# sns.scatterplot(x='Engine(CC)',y='Mileage(kmpl)',data=car_train_data)
sns.regplot(x='Engine(CC)', y='Mileage(kmpl)', data=car_train_data, scatter=True,line_kws={'color':'green'})
plt.xlabel("Engine(CC)",color='blue',fontweight='bold')
plt.ylabel("Mileage(kmpl)",color='blue',fontweight='bold')
plt.title("Scatter plot of Engine vs Mileage",color='blue',fontweight='bold')
correlation = car_train_data['Engine(CC)'].corr(car_train_data['Mileage(kmpl)'])
print(f"Correlation between Engine and Mileage is {correlation.round(2)}")

# 6th plot  year vs price (histogram)
sns.histplot(x='Year',y='Price(Lakh)',data=car_train_data, bins=200, kde=False, color='purple',cbar=True)
plt.xlabel('Year',color='blue',fontweight='bold')
plt.ylabel('Price(Lakh)',color='blue',fontweight='bold')
plt.title('Histogram with 200 Bins of Year vs Price(Lakh)',color='blue',fontweight='bold')
plt.show()
print("Minimum year",car_train_data['Year'].min())
print("Maximum year",car_train_data['Year'].max())

# 7th Plot: owner type vs price (violin plot)
sns.violinplot(x='Owner_Type',y='Price(Lakh)',data=car_train_data, palette='pastel')

# Add labels and title
plt.xlabel('Owner Type',color='blue',fontweight='bold')
plt.ylabel('Price(Lakh)',color='blue',fontweight='bold')
plt.title('Violin Plot of owner type vs price',color='blue',fontweight='bold')

# Show the plot
plt.show()

ax=sns.boxplot(x='Owner_Type',y='Price(Lakh)',data=car_train_data, palette='pastel')
plt.xlabel('Owner Type',color='blue',fontweight='bold')
plt.ylabel('Price(Lakh)',color='blue',fontweight='bold')
plt.title('Box Plot of owner type vs price',color='blue',fontweight='bold')

# Show the plot
plt.show()
# from boxplot and violin, I say that as owner is new the price of the car will increase.
# Also, there are outliers in Third owner type, therefore in violine plot its line is higher compared to Second owner type.

# 8th plot Km vs price (joint vs line vs bubble vs pair plot)
sns.jointplot(x='Kilometers_Driven', y='Price(Lakh)', data=car_train_data, kind='scatter')
plt.xlabel('KM Driven',color='blue',fontweight='bold')
plt.ylabel('Price(Lakh)',color='blue',fontweight='bold')
plt.title('Joint Plot of KM Driven vs Price(Lakh)',y=1.2,color='blue',fontweight='bold')

# sns.lineplot(x='Kilometers_Driven', y='Price(Lakh)', data=car_train_data)
sns.scatterplot(x='Kilometers_Driven', y='Price(Lakh)', data=car_train_data,  palette='viridis')

# plt.figure(figsize=(12,6))

# sns.pairplot(
#     car_train_data,
#     x_vars=["Kilometers_Driven"],
#     y_vars=["Price(Lakh)"]
# )
# plt.xlabel('Km driven',color='blue',fontweight='bold')
# plt.ylabel('Price(Lakh)',color='blue',fontweight='bold')
# plt.title('Pair Plot of Km driven vs price',color='blue',fontweight='bold')

# # Show the plot
# plt.show()

car_train_data.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
car_train_data['Fuel_Type'] = le.fit_transform(car_train_data['Fuel_Type'])
car_train_data['Transmission'] = le.fit_transform(car_train_data['Transmission'])
car_train_data['Owner_Type'] = le.fit_transform(car_train_data['Owner_Type'])

car_train_data=car_train_data.drop('Location',axis=1)
print(car_train_data)

from sklearn.model_selection import train_test_split, cross_val_score
# List of all column names in the DataFrame
# car_train_data=car_train_data.drop('Location',axis=1)
# car_train_data = pd.get_dummies(car_train_data, columns=['Location'], drop_first=True)
# car_train_data = pd.get_dummies(car_train_data, columns=['Fuel_Type'], drop_first=True)
# car_train_data = pd.get_dummies(car_train_data, columns=['Transmission'], drop_first=True)
# car_train_data = pd.get_dummies(car_train_data, columns=['Owner_Type'], drop_first=True)

x=car_train_data.iloc[:,1:]
y=car_train_data['Price(Lakh)']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=70)

# feature scaling method
from sklearn.preprocessing import MinMaxScaler
numeric_columns = car_train_data.select_dtypes(include=['float64', 'int64']).columns
car_train_data_numeric = car_train_data[numeric_columns]

scaler = MinMaxScaler()
# Fit and transform the data
car_train_data_numeric_normalized = pd.DataFrame(scaler.fit_transform(car_train_data_numeric), columns=car_train_data_numeric.columns)

# Concatenate the non-numeric columns (if any) with the normalized numeric columns
df_concat_normalized = pd.concat([car_train_data[car_train_data.columns.difference(numeric_columns)], car_train_data_numeric_normalized], axis=1)

# Display the normalized DataFrame
df_concat_normalized.head()


from sklearn.metrics import mean_absolute_error

def do_prediction(classifier):

    # training the classifier on the dataset
    classifier.fit(X_train,y_train)

    #Do prediction and evaluting the prediction
    prediction = classifier.predict(X_test)
    cross_validation_score = cross_val(X_train, y_train, classifier)
    error = mean_absolute_error(y_test, prediction)

    return error, cross_validation_score

def cross_val(xtrain, ytrain, classifier):

    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
    return accuracies.mean()

# Linear regression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Create a linear regression model
model_linear = LinearRegression()

# Train the model
model_linear.fit(X_train,y_train)

# Make predictions on the test set
y_pred = model_linear.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the coefficients and performance metrics
print("Coefficients:", model_linear.coef_)
print("Intercept:", model_linear.intercept_)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Plot the regression line
# plt.scatter(X_test, y_test, color='black', label='Actual Data')
# plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Linear Regression Example')
# plt.legend()
# plt.show()
error, score = do_prediction(model_linear)
print("Linear regression  MAE: {}".format(round(error,2)))
print("Cross Validation Score  MAE: {}".format(round(score,2)))


# Decision Tree
from sklearn.tree import DecisionTreeRegressor
model_decision = DecisionTreeRegressor()
error, score = do_prediction(model_decision)
print("Decision Tree regression  MAE: {}".format(round(error,2)))
print("Cross Validation Score  MAE: {}".format(round(score,2)))

# Random Forest
from sklearn.ensemble import RandomForestRegressor
model_random_forest = RandomForestRegressor()
error, score = do_prediction(model_random_forest)
print("Random Forest regression  MAE: {}".format(round(error,2)))
print("Cross Validation Score  MAE: {}".format(round(score,2)))

