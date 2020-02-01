library(FactoMineR)

data(wine)

X <- wine[,c(3:31)]

mfa <- MFA(X, group=c(5,3,10,9,2), type=rep("s",5), ncp=5, name.group=c("olf","vis","olfag","gust","ens"), graph=FALSE)

print(mfa$global.pca$eig[1:5,])
print("---")

print("U")
print(mfa$global.pca$svd$U[1:5,])
print("---")

print("V")
print(mfa$global.pca$svd$V[1:5,])
print("---")

print("s")
print(mfa$global.pca$svd$vs)
print("---")

print("Row coords")
print(mfa$ind$coord[1:5,])
print("---")
