import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap


# Load the dataset
file_path = r'D:\vidit\prodigy infotech internship\task 5\US_Accidents_March23_sampled_500k.csv'
df = pd.read_csv(file_path)

# Inspect the dataset
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values (drop or fill missing data as needed)
# Dropping rows with any missing values for simplicity, adjust as necessary
df.dropna(inplace=True)

# Convert the 'Start_Time' and 'End_Time' columns to datetime format
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Check for any rows where the conversion failed
invalid_start_times = df['Start_Time'].isna().sum()
invalid_end_times = df['End_Time'].isna().sum()

print(f'Number of invalid Start_Time entries: {invalid_start_times}')
print(f'Number of invalid End_Time entries: {invalid_end_times}')

# Feature extraction: Extract day, hour, and month from Start_Time for further analysis
df['Hour'] = df['Start_Time'].dt.hour
df['Day'] = df['Start_Time'].dt.dayofweek  # Monday=0, Sunday=6
df['Month'] = df['Start_Time'].dt.month

# Enhanced EDA: Plot the number of accidents by hour of the day
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Hour', palette='viridis')
plt.title('Number of Accidents by Hour of the Day', fontsize=16)
plt.xlabel('Hour of the Day', fontsize=14)
plt.ylabel('Number of Accidents', fontsize=14)
plt.grid(True)
plt.show()

# Enhanced EDA: Plot the number of accidents by day of the week
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Day', palette='plasma')
plt.title('Number of Accidents by Day of the Week', fontsize=16)
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Number of Accidents', fontsize=14)
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.grid(True)
plt.show()

# Enhanced EDA: Plot the number of accidents by month
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Month', palette='coolwarm')
plt.title('Number of Accidents by Month', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Accidents', fontsize=14)
plt.grid(True)
plt.show()

# Enhanced EDA: Investigate road conditions and weather impact
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Weather_Condition', order=df['Weather_Condition'].value_counts().iloc[:10].index, palette='magma')
plt.title('Top 10 Weather Conditions During Accidents', fontsize=16)
plt.xlabel('Weather Condition', fontsize=14)
plt.ylabel('Number of Accidents', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Enhanced EDA: Analyze the severity of accidents in different weather conditions using a violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(data=df, x='Weather_Condition', y='Severity', order=df['Weather_Condition'].value_counts().iloc[:10].index, palette='Set3')
plt.title('Accident Severity by Weather Condition', fontsize=16)
plt.xlabel('Weather Condition', fontsize=14)
plt.ylabel('Severity', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Enhanced Visualization: Create a heatmap of accident locations
heatmap_data = df[['Start_Lat', 'Start_Lng']].dropna()
m = folium.Map(location=[df['Start_Lat'].median(), df['Start_Lng'].median()], zoom_start=5)
HeatMap(heatmap_data.values, radius=8, max_zoom=13).add_to(m)
m.save('accident_heatmap.html')

# Count the number of accidents by state
state_counts = df['State'].value_counts()

# Plot the number of accidents by state
plt.figure(figsize=(16, 10))
sns.barplot(x=state_counts.values, y=state_counts.index, palette='viridis')

# Add title and labels
plt.title('Number of Accidents by State', fontsize=18)
plt.xlabel('Number of Accidents', fontsize=14)
plt.ylabel('State', fontsize=14)

# Add gridlines for better readability
plt.grid(True, axis='x')

plt.show()


# Save the cleaned and processed dataset (optional)
df.to_csv('cleaned_US_Accidents_March23.csv', index=False)




