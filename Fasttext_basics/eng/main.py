import fasttext
import pandas as pd
import re

def preprocess(text):
    text = re.sub(r'[^\w\s\']',' ', text)
    text = re.sub(r'[ \n]+', ' ', text)
    return text.strip().lower()

model_en = fasttext.load_model("D:\\Projects\\python_projects\\Fasttext_basics\\eng\\cc.en.300.bin")

print("Nearest Neigbours of Good: ")
print(model_en.get_nearest_neighbors('good'))

print("-----------------------------------------------------")
print("Analogies of Berlin, Germany, France")
print(model_en.get_analogies("berlin","germany","france"))

df = pd.read_csv("Cleaned_Indian_Food_Dataset.csv")

df.TranslatedInstructions = df.TranslatedInstructions.map(preprocess)

df.to_csv("food_receipes.txt", columns=["TranslatedInstructions"], header=None, index=False)
model = fasttext.train_unsupervised("food_receipes.txt")

print("Nearest Neighbours: ")
print("Halwa: ")
print(model.get_nearest_neighbors("halwa"))
print("Paneer")
print(model.get_nearest_neighbors("paneer"))
print("Chutney")
print(model.get_nearest_neighbors("chutney"))
print("Saragva")
print(model.get_nearest_neighbors("saragva"))