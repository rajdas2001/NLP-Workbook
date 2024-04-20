import fasttext
model_hi = fasttext.load_model("D:\\Projects\\python_projects\\Fasttext_basics\hindi\\cc.hi.300.bin")

print("-----------------------------------------------------")
print("Nearest Neighbour of अच्छा: ")
print(model_hi.get_nearest_neighbors("अच्छा"))
