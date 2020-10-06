library(FactoMineR)

data(tea)

#mca <- MCA(tea,quanti.sup=19,quali.sup=20:36)
X <- tea[,1:18]
mca <- MCA(X, graph=FALSE)

print(mca$eig[1:5,])
print("---")

print("U")
print(mca$svd$U[1:5,])
print("---")

print("V")
print(mca$svd$V[1:5,])
print("---")

print("s")
print(mca$svd$vs)
print("---")

print("Row coords")
print(mca$ind$coord[1:5,])
print("---")

