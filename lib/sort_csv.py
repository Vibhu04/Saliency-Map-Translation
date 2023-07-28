import csv

# File path of the CSV file
csv_file_path = "regions.csv"

# Read the data from the CSV file into a list of lists
with open(csv_file_path, "r", newline="", encoding="utf-8") as csv_file:
    reader = csv.reader(csv_file)
    data = list(reader)

# Sort the data based on the first column (assuming the first column contains numeric values)
sorted_data = sorted(data, key=lambda row: int(row[0]))

# Write the sorted data back to the CSV file
with open('sorted_regions.csv', "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(sorted_data)