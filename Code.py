from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[10], [20], [30], [40], [50]])
y = np.array([30, 35, 40, 45, 50])

model = LinearRegression()
model.fit(X, y)

future_input = np.array([[60]])   
predicted_load = model.predict(future_input)[0]

nodes = {
    "A": {"capacity": 70},
    "B": {"capacity": 60},
    "C": {"capacity": 80},
    "D": {"capacity": 65}
}

safety_margin = 5   # MW

best_node = None
best_score = -999

print("\n--- Node Analysis ---\n")

for node, data in nodes.items():
    capacity = data["capacity"]

    available = capacity - predicted_load

    score = available - safety_margin

    print(f" Node {node}:")
    print(f" Capacity = {capacity} MW")
    print(f" Predicted Load = {predicted_load:.2f} MW")
    print(f" Available Capacity = {available:.2f} MW")
    print(f" Score = {score:.2f}\n")

    # Select best node
    if available > 0 and score > best_score:
        best_node = node
        best_score = score

print("\n--- FINAL RESULT ---\n")

if best_node:
    print(f"Recommended Node: {best_node}")
    print("Reason: Highest available capacity after ML-based load prediction.")
else:
    print("No suitable node found. Grid upgrade required.")

print("\n--- SYSTEM SUMMARY ---\n")
print(f"Predicted Future Load: {predicted_load:.2f} MW")
print("Model Used: Linear Regression (Machine Learning)")
