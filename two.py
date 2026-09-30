import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("play_tennis.csv")

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(df[features])
y = df['PlayTennis']

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_encoded, y)

print("Decision Tree Rules for Playing Tennis:")
print(export_text(clf, feature_names=features))

test_day = pd.DataFrame({
    'Outlook': ['Rain'],
    'Temperature': ['Mild'],
    'Humidity': ['High'],
    'Wind': ['Weak']
})
prediction = clf.predict(encoder.transform(test_day))[0]
print(f"Prediction for condition [Rain, Mild, High, Weak]: PlayTennis = {prediction}")
