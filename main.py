# All the code for the analysis is in this file

## Importing the libraries
import pandas as pd  # Used for data manipulation and analysis
import numpy as np  # Used for scientific computing
import matplotlib.pyplot as plt  # Used for plotting graphs
import seaborn as sns  # Used for plotting graphs
import re  # Used for regular expression

palette = sns.color_palette("Oranges_r")
# Display all the columns of the dataframe
pd.set_option("display.max_columns", None)

# Read the dataset into a dataframe named data
data = pd.read_csv("../Hotels_Analysis/dataset/hotel_bookings.csv")

# # Let's see number of rows and columns in our dataset
data.shape

# Display top 5 rows of the table.
data.head(5)


## Data Preprocessing
# Copy the dataset
df = data.copy()

# Clean column names
df.columns = [re.sub(r"([A-Z])", r"_\1", col).lower() for col in df.columns]

# Let's check for null values and there percentage in each column
# Also as we have too many columns, let's sort the columns in descending order of percentage of null values

null_value = pd.DataFrame(
    {
        "Null Values": df.isna().sum(),
        "Percentage Null Values": (df.isna().sum()) / (df.shape[0]) * (100),
    }
)
null_value.sort_values(by="Percentage Null Values", ascending=False)[:10]


# Replace missing values in `agent` and `company` columns with 0.0`
df[["agent", "company"]] = df[["agent", "company"]].fillna(0.0)

# Replace the null values in Children column with the mean.
df["children"].fillna(round(data.children.mean()), inplace=True)

# Replace missing values in the country column with the mode.
df["country"].fillna(data.country.mode().to_string(), inplace=True)

# Check the data with no adults, no children and no babies
df[(df.adults + df.children + df.babies) == 0].shape

# Drop the data with no adults, no children and no babies
df.drop(df[(df.adults + df.children + df.babies) == 0].index, inplace=True)

# Let's check the data types of each column
dtypes = pd.DataFrame({"Dtypes": df.dtypes})
dtypes

# Convert the datatype of Childern, Company and Agent from float to int
df[["children", "company", "agent"]] = df[["children", "company", "agent"]].astype(
    "int64"
)


## EDA and Data Visualization

# Find the top 10 countries with the highest number of customers
top_countries_customers_count = df["country"].value_counts().head(5)
# Print top 5 countries with the highest number of customers
top_countries_customers_count


# Plot the top countries with the highest number of customers
plt.figure(figsize=(15, 8))
sns.barplot(
    x=top_countries_customers_count.index,
    y=top_countries_customers_count.values,
    palette=palette,
)

# Set labels and title for the plot
plt.xlabel("Countries")
plt.ylabel("Number of Customers")
plt.title("Top 10 Countries with the highest number of customers")

# Display the plot
plt.show()


# Let's create a resort column to find the number of customers who booked a resort hotel and who booked a city hotel

resort_hotel = df.loc[(df["hotel"] == "Resort Hotel")
                      & (df["is_canceled"] == 0)]
city_hotel = df.loc[(df["hotel"] == "City Hotel") & (df["is_canceled"] == 0)]


# Calculate the Average Daily Rate (ADR) per paying guest (excluding babies) for resort hotel
resort_hotel.loc[:, "adr_per_paying_guest"] = resort_hotel["adr"] / (
    resort_hotel["adults"] + resort_hotel["children"]
)

# Calculate the Average Daily Rate (ADR) per paying guest (excluding babies) for city hotel
city_hotel.loc[:, "adr_per_paying_guest"] = city_hotel["adr"] / (
    city_hotel["adults"] + city_hotel["children"]
)


# Calculate and display the average nightly rates per person for non-canceled bookings.
print(
    "For all non-canceled reservations, encompassing various room types and meal plans:"
)
print(
    f"Resort Hotel: Average nightly rate per person is €{resort_hotel['adr_per_paying_guest'].mean():.2f}."
)
print(
    f"City Hotel: Average nightly rate per person is €{city_hotel['adr_per_paying_guest'].mean():.2f}."
)


# Calculate the normalized Average Daily Rate (ADR) per guest by dividing ADR by the total number of guests (adults + children).
df["adr_per_paying_guest"] = df["adr"] / (df["adults"] + df["children"])

# Filter the dataset to include only non-canceled bookings.
non_canceled_bookings = df[df["is_canceled"] == 0]

# Display the resulting DataFrame containing non-canceled bookings.
non_canceled_bookings


# Selecting data for actual guests, sorting by reserved room type
guests_room_prices = non_canceled_bookings[
    ["hotel", "reserved_room_type", "adr_per_paying_guest"]
].sort_values("reserved_room_type")
guests_room_prices


# Create a boxplot to visualize room prices per night and person by room type and hotel
plt.figure(figsize=(14, 10))

sns.boxplot(
    x="reserved_room_type",
    y="adr_per_paying_guest",
    hue="hotel",
    data=guests_room_prices,
    hue_order=["City Hotel", "Resort Hotel"],
    fliersize=0,
)

plt.title("Price Distribution of Room Types per Night and Person", fontsize=16)
plt.xlabel("Reserved Room Type", fontsize=16)
plt.ylabel("Price (EUR)", fontsize=16)
plt.legend(loc="upper right")
plt.ylim(0, 160)
plt.show()


# Select relevant columns and sort by arrival_date_month
room_price_monthly = non_canceled_bookings[
    ["hotel", "arrival_date_month", "adr_per_paying_guest"]
].sort_values("arrival_date_month")


# Let's order the months by their logical order
ordered_months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
room_price_monthly["arrival_date_month"] = pd.Categorical(
    room_price_monthly["arrival_date_month"], categories=ordered_months, ordered=True
)

room_price_monthly


# Let's plot a line graph to visualize the average price per night and person over the months
plt.figure(figsize=(12, 8))
sns.lineplot(
    x="arrival_date_month",
    y="adr_per_paying_guest",
    hue="hotel",
    data=room_price_monthly,
    hue_order=["City Hotel", "Resort Hotel"],
    err_style="band",
    size="hotel",
    sizes=(2, 2),
)
plt.title("Room price per night and person over the year", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=90)
plt.ylabel("Price (EUR)", fontsize=16)
plt.show()


# Let's find out number of cancelled booking and then plot a bar chart for cancelled and no cancelled bookings for each hotel
canceled_bookings = df[df["is_canceled"] == 1]
no_canceled_bookings = df[df["is_canceled"] == 0]

# Let's plot a bar chart for cancelled and no cancelled bookings for each hotel
plt.figure(figsize=(12, 8))
sns.countplot(x="hotel", hue="is_canceled", data=df)
plt.title("Canceled and non-canceled bookings for each hotel", fontsize=16)
plt.xlabel("Hotel", fontsize=16)
plt.ylabel("Number of Bookings", fontsize=16)
plt.show()


# Calculate canceled booking percentage for each hotel
canceled_percentage = (
    canceled_bookings["hotel"].value_counts(
    ) / df["hotel"].value_counts() * 100
).round(2)

# Create a DataFrame with hotel and cancellation percentage
canceled_df = canceled_percentage.reset_index()

# Rename the columns for clarity
canceled_df.columns = ["Hotel", "Cancellation Percentage"]

# Display the result without the index column
print(canceled_df.to_string(index=False))


# Let's see number of cancelled booking per month for each hotel also make the month order logical
canceled_bookings_per_month = canceled_bookings["arrival_date_month"].value_counts(
)
canceled_bookings_per_month = canceled_bookings_per_month.reindex(
    ordered_months, axis=0
)

# Let's plot a bar chart for cancelled bookings per month for each hotel
plt.figure(figsize=(12, 8))
sns.barplot(
    x=canceled_bookings_per_month.index,
    y=canceled_bookings_per_month.values,
    palette=palette,
)

plt.title("Canceled bookings per month for each hotel", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.ylabel("Number of Bookings", fontsize=16)
plt.show()


# Let's plot a boxplot to visualize the number of days stayed by market segment and hotel type
plt.figure(figsize=(15, 10))
sns.boxplot(
    x="market_segment", y="stays_in_week_nights", data=df, hue="hotel", palette="Set1"
)

plt.figure(figsize=(15, 10))
sns.boxplot(
    x="market_segment",
    y="stays_in_weekend_nights",
    data=df,
    hue="hotel",
    palette="Set1",
)


# Create subsets for different accommodation categories
single_guests = df[(df.adults == 1) & (df.children == 0) & (df.babies == 0)]
couple_guests = df[(df.adults == 2) & (df.children == 0) & (df.babies == 0)]
family_guests = df[df.adults + df.children + df.babies > 2]

# Define category names and calculate their percentages
category_names = ['Single', 'Couple (No Children)', 'Family / Friends']
category_counts = [single_guests.shape[0],
                   couple_guests.shape[0], family_guests.shape[0]]
category_percentages = [count / df.shape[0] * 100 for count in category_counts]

# Plot the data
plt.figure(figsize=(12, 8))
plt.bar(category_names, category_percentages, color=palette)
plt.title('Accommodation Type Distribution', fontsize=16)
plt.xlabel('Accommodation Type', fontsize=16)
plt.ylabel('Percentage', fontsize=16)
plt.show()


# Calculate the number of guests per month for each hotel
guests_per_month_df = df[
    ["hotel", "arrival_date_month", "arrival_date_year", "adults", "children", "babies"]
].copy()

# Calculate the total number of guests for each booking
guests_per_month_df["total_guests"] = (
    guests_per_month_df["adults"]
    + guests_per_month_df["children"]
    + guests_per_month_df["babies"]
)

# Group the data by hotel, arrival year, and month, and sum the total guests
guests_per_month = (
    guests_per_month_df.groupby(["hotel", "arrival_date_year", "arrival_date_month"])[
        "total_guests"
    ]
    .sum()
    .reset_index()
)

# Sort the data to find the busiest months
busiest_months = guests_per_month.sort_values("total_guests", ascending=False)

# Display the top 5 busiest months for hotels
busiest_months.head(5)


# Create a list of month names in chronological order
months_in_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Create a new dataframe with the required columns
guests_df = df[
    ["hotel", "arrival_date_month", "arrival_date_year", "adults", "children", "babies"]
].copy()

# Calculate the total number of guests for each booking
guests_df["total_guests"] = (
    guests_df["adults"] + guests_df["children"] + guests_df["babies"]
)

# Group by hotel, arrival date month, and year, and calculate the total guests
guests_per_month = (
    guests_df.groupby(["hotel", "arrival_date_year", "arrival_date_month"])
    .sum()
    .reset_index()
)

# Arrange the months in chronological order
guests_per_month["arrival_date_month"] = pd.Categorical(
    guests_per_month["arrival_date_month"], categories=months_in_order, ordered=True
)

# Sort the DataFrame by year and month
guests_per_month = guests_per_month.sort_values(
    ["arrival_date_year", "arrival_date_month"]
)

# Plot a chart to visualize the results
plt.figure(figsize=(12, 8))
sns.barplot(
    x="arrival_date_month",
    y="total_guests",
    hue="hotel",
    data=guests_per_month,
    hue_order=["City Hotel", "Resort Hotel"],
)
plt.title("Number of guests per month", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.ylabel("Number of guests", fontsize=16)
plt.xticks(rotation=45)
plt.show()


# Replace '0' with 'No' and '1' with 'Yes' in the 'is_repeated_guest' column
df["is_repeated_guest"] = df["is_repeated_guest"].replace({0: "No", 1: "Yes"})

# Create a countplot to visualize canceled bookings by hotel and repeated guest status
sns.set(style="whitegrid")
plt.title("Booking Cancellation by Repeated Guests", fontdict={"fontsize": 16})
canceled = sns.countplot(x=df.hotel, hue="is_repeated_guest", data=df)


# Set the figure size and style
plt.figure(figsize=(15, 10))
sns.set(style="darkgrid")

# Set the plot title
plt.title("Cancellation Count by Market Segment", fontdict={"fontsize": 20})

# Create the countplot
ax = sns.countplot(x="market_segment", hue="is_canceled", data=df)


# Replace '0' with 'No' and '1' with 'Yes' in the 'is_repeated_guest' column
df["is_repeated_guest"] = df["is_repeated_guest"].replace({0: "No", 1: "Yes"})

# Create a FacetGrid for lead time distribution by cancellation status
grid = sns.FacetGrid(df, hue="is_canceled", height=6, xlim=(0, 500))

# Use 'fill=True' instead of 'shade=True' to fill the area under the KDE plot
grid.map(sns.kdeplot, "lead_time", fill=True)

# Add a legend for visualization clarity
grid.add_legend()


