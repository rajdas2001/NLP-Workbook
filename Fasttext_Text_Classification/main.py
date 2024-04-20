import pandas as pd
import fasttext
import re
from sklearn.model_selection import train_test_split

def preprocess(text):
    text = re.sub(r'[^\w\s\']',' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()

df = pd.read_csv("ecommerceDataset.csv", names=["category", "description"], header=None)

df.dropna(inplace=True)

df.category.replace("Clothing & Accessories", "Clothing_Accessories", inplace=True)

df['category'] = '__label__' + df['category'].astype(str)

df['category_description'] = df['category'] + ' ' + df['description']

text = "  VIKI's | Bookcase/Bookshelf (3-Shelf/Shelve, White) | ? . hi"
text = re.sub(r'[^\w\s\']',' ', text)
text = re.sub(' +', ' ', text)
text.strip().lower()


df['category_description'] = df['category_description'].map(preprocess)

train, test = train_test_split(df, test_size=0.2)

train.to_csv("ecommerce.train", columns=["category_description"], index=False, header=False)
test.to_csv("ecommerce.test", columns=["category_description"], index=False, header=False)

model = fasttext.train_supervised(input="ecommerce.train")
print(model.test("ecommerce.test"))

print("---------------------------------------------------------------")
txt = "wintech assemble desktop pc cpu 500 gb sata hdd 4 gb ram intel c2d processor 3"
print("Text is", txt)
print("Prediction: ")
print(model.predict(txt))
print("---------------------------------------------------------------")
txt = "hockey men's cotton t shirt fabric details 80 cotton 20 polyester super combed cotton rich fabric"
print("Text is", txt)
print("Prediction: ")
print(model.predict(txt))
print("---------------------------------------------------------------")
print("Nearest Neighbours of Sony: ")
print(model.get_nearest_neighbors("sony"))