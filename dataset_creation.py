import pandas as pd
import numpy as np

# Define the actions and corresponding products
actions = [
    "I want to cook biryani",
    "I want to go fishing",
    "I want to do snorkeling",
    "I want to play tennis",
    "I want to watch a movie",
    "I want to read a book"
]

products = [
    ["Cooking utensils", "Rice", "Spices"],
    ["Fishing gear", "Bait"],
    ["Snorkeling equipment", "Swimsuit", "Goggles"],
    ["Tennis racket", "Tennis balls"],
    ["Movie ticket", "Popcorn", "Beverages"],
    ["Book"]
]

# Flatten the lists of products
products_flat = [product for sublist in products for product in sublist]

# Duplicate actions based on the number of associated products
actions_repeated = np.repeat(actions, [len(prod) for prod in products])

# Generate the dataset
data = pd.DataFrame({"Action": actions_repeated, "Product": products_flat})

# Save the dataset to a CSV file
data.to_csv("dataset.csv", index=False)
