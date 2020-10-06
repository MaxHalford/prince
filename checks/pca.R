library(FactoMineR)

data(iris)

pca <- PCA(iris[c(1:4)], graph=FALSE)

print(pca$eig)
print("---")

print("U")
print(pca$svd$U[1:5,])
print("---")

print("V")
print(pca$svd$V)
print("---")

print("s")
print(pca$svd$vs)
print("---")

print("Row coords")
print(pca$ind$coord[1:5,])
print("---")
