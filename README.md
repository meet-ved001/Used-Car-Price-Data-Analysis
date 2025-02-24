**Dataset Description:
**
The dataset consists of used car listings, containing information about various car attributes such as make, location, year, mileage, engine capacity, power, price, and more. Below are the columns present in the dataset:

Name: The name of the car model.
Location: The location where the car is being sold.
Year: The manufacturing year of the car.
Kilometers_Driven: The total kilometers the car has been driven.
Fuel_Type: The type of fuel used by the car (e.g., Petrol, Diesel, LPG, CNG).
Transmission: The transmission type of the car (Manual or Automatic).
Owner_Type: The ownership status of the car (e.g., First, Second).
Mileage: The mileage of the car, expressed in km per liter (kmpl) or km per kg (km/kg).
Engine: The engine capacity of the car in CC (cubic centimeters).
Power: The power output of the car in bhp (brake horsepower).
Seats: The number of seats in the car.
New_Price: The original price of the car when it was new.
Price(Lakh): The current price of the car in Lakh (Indian currency unit).
Data Cleaning and Preprocessing Code:
The code focuses on cleaning the dataset, handling missing values, and transforming certain columns into suitable formats for machine learning models.

**Renaming Columns:**

The Price column is renamed to Price(Lakh) for clarity.
The New_Price column is cleaned by removing the word "Lakh" and "Cr" (if present) and is renamed to New_Price(Lakh).
Handling Missing Values:

Missing values in columns like New_Price(Lakh), Mileage(kmpl), Engine(CC), Power(bhp), and Seats are replaced by the mean of each column.
Any rows with NaN values are handled to ensure clean data for analysis.
Data Transformation:

The Mileage, Engine, and Power columns, which contain unit labels (like "kmpl" and "bhp"), are cleaned by removing these units and converting them into numerical formats.
The Mileage column is converted to a numeric value representing kilometers per liter (kmpl).
The Engine column is converted to a numeric value representing the engine capacity in CC.
The Power column is converted to a numeric value representing the engine power in bhp.
Encoding Categorical Variables:

The categorical variables (Fuel_Type, Transmission, and Owner_Type) are label-encoded, transforming them into numeric representations suitable for machine learning models.

**Data Normalization:**

Numeric columns are scaled using MinMaxScaler, which transforms the data into a range between 0 and 1 to improve model performance.
Feature and Target Variables:

The independent variables (features) include all columns except Price(Lakh), which is used as the target variable for prediction.
Visualization:

Various plots (bar plot, strip plot, boxplot, scatter plot, violin plot) are used to visualize relationships between different columns, such as:
1. **Location vs. Price:** Shows how car prices vary across different locations.
2. **Transmission vs. Mileage**: Displays the relationship between the type of transmission and mileage.
3. **Fuel Type vs. Mileage**: Shows how mileage varies based on the fuel type.
4. **Engine vs. Mileage:** Correlation between engine size and mileage.
5. **Owner Type vs. Price**: Highlights how the car's ownership affects its price.
6. **Kilometers Driven vs. Price:** Explores the relationship between car usage and price.

**Train-Test Split**:

The data is split into training and testing sets (80% training, 20% testing) to evaluate the performance of machine learning models.

**Model Training and Evaluation**:
The dataset is used to train regression models (e.g., Linear Regression, Decision Trees, Random Forests) to predict the price of used cars based on the provided features. The code also implements various techniques for model evaluation, including:

**Mean Absolute Error (MAE)**: Used to evaluate the performance of regression models.
**Cross-Validation**: Used for model validation to ensure robustness.

**Usage:**

This cleaned and preprocessed dataset can be used to build predictive models to estimate the price of used cars based on their attributes, making it valuable for car dealerships, individual buyers, or analysts in the automobile industry.







