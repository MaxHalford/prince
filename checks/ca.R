library(FactoMineR)

data(children)

X <- children[c(1:5)]

ca <- CA(X, graph=FALSE)

print(ca$eig)
print("---")

print("U")
print(ca$svd$U[1:5,])
print("---")

print("V")
print(ca$svd$V)
print("---")

print("s")
print(ca$svd$vs)
print("---")

print("Row coords")
print(ca$row$coord[1:5,])
print("---")
