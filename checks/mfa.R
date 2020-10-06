library(FactoMineR)

data(wine)

MFA <- function (base, group, type = rep("s",length(group)), excl = NULL, ind.sup = NULL, ncp = 5, name.group = NULL, num.group.sup = NULL, graph = TRUE, weight.col.mfa = NULL, row.w = NULL, axes=c(1,2),tab.comp=NULL){

    moy.p <- function(V, poids) {
        res <- sum(V * poids,na.rm=TRUE)/sum(poids[!is.na(V)])
    }
    ec <- function(V, poids) {
        res <- sqrt(sum(V^2 * poids,na.rm=TRUE)/sum(poids[!is.na(V)]))
    }
    funcLg <- function (x, y, ponderation.x, ponderation.y, wt = rep(1/nrow(x), nrow(x)), cor = FALSE) {
      if (is.data.frame(x)) x <- as.matrix(x)
      else if (!is.matrix(x)) stop("'x' must be a matrix or a data frame")
      if (is.data.frame(y)) y <- as.matrix(y)
      else if (!is.matrix(y)) stop("'y' must be a matrix or a data frame")
      if (!all(is.finite(x))) stop("'x' must contain finite values only")
      if (!all(is.finite(y))) stop("'y' must contain finite values only")
      s <- sum(wt)
      wt <- wt/s
      center <- colSums(wt * x)
      x <- sqrt(wt) * t(t(sweep(x, 2, center, check.margin = FALSE))*sqrt(ponderation.x))
      center <- colSums(wt * y)
      y <- sqrt(wt) * t(t(sweep(y, 2, center, check.margin = FALSE))*sqrt(ponderation.y))
      Lg <- 0
      for (i in 1:ncol(x)) Lg <- Lg+sum(crossprod(x[,i],y)^2)
      Lg
    }
if (!is.null(tab.comp)){
  if (!is.null(weight.col.mfa)) stop("Weightings on the variables are not allowed with the tab.comp argument")
  if (!is.null(ind.sup)) stop("Supplementary individuals are not allowed with tab.comp")
  if (!is.null(num.group.sup)) stop("Supplementary groups are not allowed with tab.comp")
}

nature.group <- NULL
for (i in 1:length(group)){
  if ((type[i] == "n")&&(!(i%in%num.group.sup))) nature.group <- c(nature.group,"quali")
  if ((type[i] == "n")&&(i%in%num.group.sup)) nature.group <- c(nature.group,"quali.sup")
  if (((type[i] == "s")||(type[i] == "c"))&&(!(i%in%num.group.sup))) nature.group <- c(nature.group,"quanti")
  if (((type[i] == "s")||(type[i] == "c"))&&(i%in%num.group.sup)) nature.group <- c(nature.group,"quanti.sup")
  if (((type[i] == "f")||(type[i] == "f2"))&&(!(i%in%num.group.sup))) nature.group <- c(nature.group,"contingency")
  if (((type[i] == "f")||(type[i] == "f2"))&&(i%in%num.group.sup)) nature.group <- c(nature.group,"contingency.sup")
}
nature.var <- rep(nature.group,times=group)

### Add
type.var <- NULL
for (i in 1:length(group)){
  if ((type[i] == "n")&&(!(i%in%num.group.sup))) type.var <- c(type.var,rep("quali",group[i]))
  if ((type[i] == "n")&&(i%in%num.group.sup)) type.var <- c(type.var,rep("quali.sup",group[i]))
  if (((type[i] == "s")||(type[i] == "c"))&&(!(i%in%num.group.sup))) type.var <- c(type.var,rep("quanti",group[i]))
  if (((type[i] == "s")||(type[i] == "c"))&&(i%in%num.group.sup)) type.var <- c(type.var,rep("quanti.sup",group[i]))
  if (((type[i] == "f")||(type[i] == "f2")||(type[i] == "f3"))&&(!(i%in%num.group.sup))) type.var <- c(type.var,rep(type[i],group[i]))
  if (((type[i] == "f")||(type[i] == "f2")||(type[i] == "f3"))&&(i%in%num.group.sup)) type.var <- c(type.var,rep(paste(type[i],"sup",sep="_"),group[i]))
}
## End add
    if (is.null(rownames(base))) rownames(base) = 1:nrow(base)
    if (is.null(colnames(base))) colnames(base) = paste("V",1:ncol(base),sep="")
#    for (j in 1:ncol(base)) if (colnames(base)[j]=="") colnames(base)[j] = paste("V",j,sep="")
#    for (j in 1:nrow(base)) if (is.null(rownames(base)[j])) rownames(base)[j] = paste("row",j,sep="")
    base <- as.data.frame(base)
    is.quali <- which(!unlist(lapply(base,is.numeric)))
    base[,is.quali] <- lapply(base[,is.quali,drop=FALSE],as.factor)
    base <- droplevels(base)
    if (!is.null(ind.sup)) {
      base <- rbind.data.frame(base[-ind.sup,],base[ind.sup,,drop=FALSE])
      ind.sup <- (nrow(base)-length(ind.sup)+1) : nrow(base)
    }
    nbre.var <- ncol(base)
    nbre.group <- length(group)
    group.actif <- NULL
    if ("n"%in%type){
      niveau = NULL
      for (j in 1:ncol(base)){
        if (!is.numeric(base[,j])) niveau = c(niveau,levels(base[,j]))
      }
      for (j in 1:ncol(base)) {
        if (!is.numeric(base[,j])){
          if (sum(niveau%in%levels(base[,j]))!=nlevels(base[,j])) levels(base[,j]) = paste(colnames(base)[j],levels(base[,j]),sep="_")
        }
      }
    }
    for (i in 1:nbre.group) if (!(i%in%num.group.sup)) group.actif <- c(group.actif,i)
    group.mod <- group
    nbre.ind <- nrow(base)
    nb.actif <- nbre.ind-length(ind.sup)
    if (nbre.var != sum(group)) stop("not convenient group definition")
    if (nbre.group != length(type)) stop("not convenient type definition")
    base <- as.data.frame(base)
    if (!inherits(base, "data.frame")) stop("base should be a data.frame")
    res.separe <- vector(mode = "list", length = nbre.group)
    if (is.null(name.group)) name.group <- paste0("Gr", 1:nbre.group)
    names(res.separe) <- name.group
    ind.grpe <- 0
    if (any(is.na(base))){
      if (!("n"%in%type)) for (j in 1:ncol(base)) base[,j] <- replace(base[,j],is.na(base[,j]),mean(base[,j],na.rm=TRUE))
      else{
        if (type[1]!="n") for (j in 1:group[1]) base[,j] <- replace(base[,j],is.na(base[,j]),mean(base[,j],na.rm=TRUE))
        for (g in 2:nbre.group){
         if (type[g]!="n") for (j in (sum(group[1:(g-1)])+1):sum(group[1:g])) base[,j] <- replace(base[,j],is.na(base[,j]),mean(base[,j],na.rm=TRUE))
        }
        if (is.null(tab.comp)){
          if (type[1]=="n") for (j in 1:group[1]) base[,j] <- as.factor(replace(as.character(base[,j]),is.na(base[,j]),paste(colnames(base)[j],".NA",sep="")))
          for (g in 2:nbre.group){
            if (type[g]=="n") for (j in (sum(group[1:(g-1)])+1):sum(group[1:g])) base[,j] <- as.factor(replace(as.character(base[,j]),is.na(base[,j]),paste(colnames(base)[j],".NA",sep="")))
          }
        }
      }
    }

    if (is.null(row.w)) row.w <- rep(1,nb.actif)

    if (any("f" %in% type)+any("f2" %in% type)+any("f3" %in% type)>1) stop("For the contingency tables, the type must the the same")
    if (("f" %in% type)||("f2" %in% type)||("f3" %in% type)) {
        grfrec<-c(which(type=="f"),which(type=="f2"),which(type=="f3"))

## pour avoir individus actifs, que ind.sup soit NULL ou non
##      ind.actif <- !((1:nrow(base))%in%intersect(ind.sup,(1:nrow(base))))
        for (i in grfrec){
            if ((type[i]=="f2")||(type[i]=="f3")||(i%in%num.group.sup)){
                if (i==1) base[,1:group[1]]<- base[,1:group[1]]/sum(base[1:nb.actif,1:group[1]])
                else base[,(sum(group[1:(i-1)])+1):sum(group[1:i])]<-base[,(sum(group[1:(i-1)])+1):sum(group[1:i])]/sum(base[1:nb.actif,(sum(group[1:(i-1)])+1):sum(group[1:i])])
            }
        }
        type.var=="f"
## Modif november 2011
        if(!any(type.var=="f")) sumT <-1
        else sumT <- sum(base[1:nb.actif,as.logical((type.var=="f")+(type.var=="f2")+(type.var=="f3"))])
## Modif november 2011
#       sumT <- sum(base[1:nb.actif,as.logical((type.var=="f")+(type.var=="f2")+(type.var=="f3"))])
        if (sumT==0) sumT <- 1
        base[,as.logical((type.var=="f")+(type.var=="f_sup")+(type.var=="f2")+(type.var=="f2_sup")+(type.var=="f3")+(type.var=="f3_sup"))]<-base[,as.logical((type.var=="f")+(type.var=="f_sup")+(type.var=="f2")+(type.var=="f2_sup")+(type.var=="f3")+(type.var=="f3_sup"))]/sumT
        F.jt<-list()
        Fi.t<-list()
        for (j in grfrec){
            if (j==1){
                F.jt[[j]]<-apply(base[1:nb.actif,1:group[1]],2,sum)
                Fi.t[[j]]<-apply(base[,1:group[1]],1,sum)
            }else{
                F.jt[[j]]<-apply(base[1:nb.actif,(sum(group[1:(j-1)])+1):(sum(group[1:(j-1)])+group[j])],2,sum)
                Fi.t[[j]]<-apply(base[,(sum(group[1:(j-1)])+1):(sum(group[1:(j-1)])+group[j])],1,sum)
            }
        }
        if ("f"%in%type.var){
            row.w[1:nrow(base)]<-0
            for (j in grfrec){
                if (!(j%in%num.group.sup)) row.w<-row.w+Fi.t[[j]]
            }
        }

        F..t<-numeric()
        for (j in grfrec)   F..t[j]<-sum(Fi.t[[j]][1:nb.actif])

        for (t in grfrec){
            if (t==1) {
                base[,1:group[t]]<-sweep(base[,1:group[t]],2,F.jt[[t]],FUN="/")
                base[,1:group[t]]=sweep(base[,1:group[t]],1,Fi.t[[t]]/F..t[t],FUN="-")
                base[,1:group[t]]=sweep(base[,1:group[t]],1,row.w,FUN="/")
            }
            else {
                base[,(sum(group[1:(t-1)])+1):sum(group[1:t])]<-sweep(base[,(sum(group[1:(t-1)])+1):sum(group[1:t])],2,F.jt[[t]],FUN="/")
                base[,(sum(group[1:(t-1)])+1):sum(group[1:t])]<-sweep(base[,(sum(group[1:(t-1)])+1):sum(group[1:t])],1,Fi.t[[t]]/F..t[t],FUN="-")
                base[,(sum(group[1:(t-1)])+1):sum(group[1:t])]<-sweep(base[,(sum(group[1:(t-1)])+1):sum(group[1:t])],1,row.w,FUN="/")
            }
        }
        row.w <- row.w[1:nb.actif]
    }

    if (!is.null(ind.sup))  row.w.moy.ec <- c(row.w,rep(0,length(ind.sup)))
    else row.w.moy.ec <- row.w

if (is.null(weight.col.mfa)) weight.col.mfa <- rep(1,sum(group.mod))
### Begin handle missing values
if (!is.null(tab.comp)){
  group.mod <- tab.comp$call$group.mod
  ind.var.group <- tab.comp$call$ind.var
  tab.comp <- tab.comp$tab.disj
}
### End  handle missing values
   for (g in 1:nbre.group) {
        aux.base <- as.data.frame(base[, (ind.grpe + 1):(ind.grpe + group[g])])
        dimnames(aux.base) <- list(rownames(base),colnames(base)[(ind.grpe + 1):(ind.grpe + group[g])])
        if (type[g] == "s") res.separe[[g]] <- PCA(aux.base, ind.sup =ind.sup, scale.unit = TRUE, ncp = ncp, row.w=row.w, graph = FALSE, col.w = weight.col.mfa[(ind.grpe + 1):(ind.grpe + group[g])])
        if (type[g] == "c") res.separe[[g]] <- PCA(aux.base, ind.sup =ind.sup, scale.unit = FALSE, ncp = ncp, row.w=row.w,graph = FALSE, col.w = weight.col.mfa[(ind.grpe + 1):(ind.grpe + group[g])])
        if (type[g]=="f"||type[g]=="f2"||type[g]=="f3")  res.separe[[g]] <- PCA(aux.base, ind.sup =ind.sup, scale.unit = FALSE, ncp = ncp, row.w=row.w,graph = FALSE, col.w = F.jt[[g]]*weight.col.mfa[(ind.grpe + 1):(ind.grpe + group[g])])
        if (type[g] == "n") {
          for (v in (ind.grpe + 1):(ind.grpe + group[g])) {
            if (!is.factor(base[, v])) stop("factors are not defined in the qualitative groups")
          }
          res.separe[[g]] <- MCA(aux.base, excl = excl[[g]], ind.sup = ind.sup, ncp=ncp, graph = FALSE, row.w=row.w)
        }
###  Begin handle missing values
if (!is.null(tab.comp)){
 if (type[g] == "s") res.separe[[g]] <- PCA(tab.comp[,ind.var.group[[g]]],scale.unit=TRUE,row.w=row.w,ind.sup=ind.sup,col.w=weight.col.mfa[(ind.grpe + 1):(ind.grpe + group[g])],graph=FALSE)
 if (type[g] == "c") res.separe[[g]] <- PCA(tab.comp[,ind.var.group[[g]]],scale.unit=FALSE,row.w=row.w,ind.sup=ind.sup,col.w=weight.col.mfa[(ind.grpe + 1):(ind.grpe + group[g])],graph=FALSE)
 if (type[g] == "n") res.separe[[g]] <- MCA(aux.base, ind.sup = ind.sup, ncp=ncp, graph = FALSE, row.w=row.w,tab.disj=tab.comp[,ind.var.group[[g]]])
}
###  End handle missing values

        ind.grpe <- ind.grpe + group[g]
    }
    data <- matrix(0,nbre.ind,0)
    ind.grpe <- ind.grpe.mod <- 0
    ponderation <- vector(length = sum(group.mod))
    poids.bary <- NULL
##if (is.null(weight.col.mfa)) weight.col.mfa <- rep(1,length(ponderation))
    for (g in 1:nbre.group) {
        aux.base <- base[, (ind.grpe + 1):(ind.grpe + group[g]),drop=FALSE]
###  Begin handle missing values
        if (!is.null(tab.comp)){
          if (g==1) aux.base <- tab.comp[,1:group.mod[1]]
          else aux.base <- tab.comp[,(cumsum(group.mod)[g-1]+1):cumsum(group.mod)[g],drop=FALSE]
        }
###  End handle missing values
        aux.base <- as.data.frame(aux.base)
        colnames(aux.base) <- colnames(base)[(ind.grpe + 1):(ind.grpe + group[g])]
        if (type[g] == "s") {
          centre.aux.base <- apply(as.data.frame(aux.base), 2, moy.p, row.w.moy.ec)
          aux.base <- t(t(as.matrix(aux.base))-centre.aux.base)
          ecart.type.aux.base <- apply(as.data.frame(aux.base), 2, ec, row.w.moy.ec)
          ecart.type.aux.base[ecart.type.aux.base <= 1e-08] <- 1
          aux.base <- t(t(aux.base)/ecart.type.aux.base)
          type[g]="c"
        }
        if (type[g] == "c") {
          data <- cbind.data.frame(data, aux.base)
          ponderation[(ind.grpe.mod + 1):(ind.grpe.mod + group.mod[g])] <- 1/res.separe[[g]]$eig[1,1]
        }
        if (type[g] == "f"||type[g] == "f2") {
            data <- cbind.data.frame(data, aux.base)
            ponderation[(ind.grpe.mod+1):(ind.grpe.mod+group[g])]<-F.jt[[g]]/res.separe[[g]]$eig[1,1]
#           if (type[g]=="f") ponderation[(ind.grpe.mod+1):(ind.grpe.mod+group[g])]<-F.jt[[g]]/res.separe[[g]]$eig[1,1]
#           else ponderation[(ind.grpe+1):(ind.grpe.mod+group[g])]<-P.jt[[g]]/length(grfrec2)/res.separe[[g]]$eig[1,1]
        }

## modif Avril 2011
        if (type[g] == "n") {
## a remettre si j'enleve yyyyy
#              tmp <- tab.disjonctif(aux.base)
#              group.mod[g] <- ncol(tmp)
## fin  a remettre si j'enleve yyyyy
###  Begin handle missing values
            if (!is.null(tab.comp)){
              if (g==1) tmp <- tab.comp[,1:group.mod[1]]
              else tmp <- tab.comp[,(cumsum(group.mod)[g-1]+1):cumsum(group.mod)[g],drop=FALSE]
            } else {
              tmp <- tab.disjonctif(aux.base)
              group.mod[g] <- ncol(tmp)
            }
###  End handle missing values
# @@@
            centre.tmp <- apply(tmp, 2, moy.p, row.w.moy.ec)
            centre.tmp <- centre.tmp/sum(row.w.moy.ec)
            tmp2 <- tmp*(row.w.moy.ec/sum(row.w.moy.ec))
            poids.bary <- c(poids.bary,colSums(tmp2))
            poids.tmp <- 1-apply(tmp2, 2, sum)

            if(!is.null(excl[[g]])) poids.tmp[excl[[g]]] <- 0
            ponderation[(ind.grpe.mod + 1):(ind.grpe.mod + group.mod[g])] <- poids.tmp/(res.separe[[g]]$eig[1,1] * group[g])
            tmp <- tmp/sum(row.w.moy.ec)
            tmp <- t(t(as.matrix(tmp))-centre.tmp)
            ecart.type.tmp <- apply(tmp, 2, ec, row.w.moy.ec)
### Pb if the disjunctive table doesn't have only 0 and 1
            if (!is.null(tab.comp)) ecart.type.tmp <- sqrt(centre.tmp*sum(row.w.moy.ec) * (1-centre.tmp*sum(row.w.moy.ec) ))/sum(row.w.moy.ec)
### End pb if the disjunctive table doesn't have only 0 and 1
            ecart.type.tmp[ecart.type.tmp <= 1e-08] <- 1
            tmp <- t(t(as.matrix(tmp))/ecart.type.tmp)
            print(ponderation)
            data <- cbind.data.frame(data, as.data.frame(tmp))
        }
## Fin modif
        if(!is.null(excl)) ponderation[ponderation == 0] <- 1e-15
        ind.grpe <- ind.grpe + group[g]
        ind.grpe.mod <- ind.grpe.mod + group.mod[g]
    }
    data.group.sup.indice <- data.group.sup <- NULL
    data.pca <- data
    rownames(data.pca) <- rownames(base)
    if (!is.null(num.group.sup)){
      ponderation.tot <- ponderation
      ponderation.group.sup <- NULL
      nb.of.var <- 0
      supp.quanti <- supp.quali <- NULL
      colnames.data.group.sup <- NULL
      for (i in 1:nbre.group) {
        if (i%in%num.group.sup){
          if ((type[i]=="c")||(type[i]=="f")) supp.quanti <- c(supp.quanti,(1+nb.of.var):(nb.of.var+group.mod[i]))
          if (type[i]=="n") supp.quali <- c(supp.quali,(1+nb.of.var):(nb.of.var+group.mod[i]))
          if (is.null(data.group.sup)) data.group.sup <- as.data.frame(data[,(1+nb.of.var):(nb.of.var+group.mod[i])])
          else data.group.sup <- cbind.data.frame(data.group.sup,data[,(1+nb.of.var):(nb.of.var+group.mod[i])])
          if (ncol(data.group.sup)>1) colnames.data.group.sup <- c(colnames.data.group.sup,colnames(data)[(1+nb.of.var):(nb.of.var+group.mod[i])])
          else colnames.data.group.sup <- colnames(data)[1+nb.of.var]
          ponderation.group.sup <- c(ponderation.group.sup,ponderation[(1+nb.of.var):(nb.of.var+group.mod[i])])
        }
        nb.of.var <- nb.of.var + group.mod[i]
      }
      colnames(data.group.sup) <- colnames.data.group.sup
      ponderation <- ponderation.tot[-c(supp.quanti,supp.quali)]
      data <- data[,-c(supp.quanti,supp.quali)]
      data.group.sup.indice <- (ncol(data)+1):(ncol(data)+ncol(data.group.sup))
      data.pca <- cbind.data.frame(data,data.group.sup)
    }
    ncp.tmp <- min(nb.actif-1, ncol(data))
    ind.var <- 0
    ind.quali <- NULL
    for (g in 1:nbre.group) {
        if (type[g] == "n")  ind.quali <- c(ind.quali, c((ind.var + 1):(ind.var + group[g])))
        ind.var <- ind.var + group[g]
    }
    aux.quali.sup.indice <- aux.quali.sup <- data.sup <- NULL
    if (!is.null(ind.quali)){
      aux.quali.sup <- as.data.frame(base[, ind.quali,drop=FALSE])
      if (is.null(data.group.sup)) aux.quali.sup.indice <- (ncol(data)+1):(ncol(data)+ncol(aux.quali.sup))
      else aux.quali.sup.indice <- (ncol(data)+ncol(data.group.sup)+1):(ncol(data)+ncol(data.group.sup)+ncol(aux.quali.sup))
      data.pca <- cbind.data.frame(data.pca,aux.quali.sup)
    }
row.w = row.w[1:nb.actif]
###  Begin handle missing values
if ((!is.null(tab.comp))&(any("n"%in%type))){
  data.pca <- data.pca[,-aux.quali.sup.indice]
  aux.quali.sup.indice <- NULL
}
###  End handle missing values
 res.globale <- PCA(data.pca, scale.unit = FALSE, col.w = ponderation, row.w=row.w,ncp = ncp, ind.sup = ind.sup, quali.sup = aux.quali.sup.indice, quanti.sup = data.group.sup.indice, graph = FALSE)
# res.globale <- PCA(data.pca, scale.unit = FALSE, col.w = ponderation, row.w=row.w,ncp = ncp.tmp, ind.sup = ind.sup, quali.sup = aux.quali.sup.indice, quanti.sup = data.group.sup.indice, graph = FALSE)
###  Begin handle missing values
if ((!is.null(tab.comp))&(any("n"%in%type))){
  res.globale$quali.var$coord <- res.globale$var$coord[unlist(ind.var.group[type%in%"n"]),]
  res.globale$quali.var$contrib <- res.globale$var$contrib[unlist(ind.var.group[type%in%"n"]),]
  res.globale$quali.var$cos2 <- res.globale$var$cos2[unlist(ind.var.group[type%in%"n"]),]
  res.globale$call$quali.sup$barycentre <- sweep(crossprod(tab.comp[,unlist(ind.var.group[type%in%"n"])],as.matrix(data.pca)),1,apply(tab.comp[,unlist(ind.var.group[type%in%"n"])],2,sum),FUN="/")
  res.globale$quali.sup$coord <- sweep(crossprod(tab.comp[,unlist(ind.var.group[type%in%"n"])],res.globale$ind$coord),1,apply(tab.comp[,unlist(ind.var.group[type%in%"n"])],2,sum),FUN="/")
}
###  End handle missing values
    ncp <- min(ncp, nrow(res.globale$eig))
    call <- res.globale$call
    call$group <- group
    call$type <- type
    call$ncp <- ncp
    call$group.mod <- group.mod
    call$num.group.sup <- num.group.sup
    call$name.group <- name.group
    call$X <- base
    call$XTDC <- data
    call$nature.group <- nature.group
    call$nature.var <- nature.var
    contrib.group <- matrix(NA, length(group.actif), ncp)
    dimnames(contrib.group) <- list(name.group[group.actif], paste("Dim", c(1:ncp), sep = "."))
    dist2.group <- vector(length = length(group.actif))
    ind.var <- ind.var.sup <- 0
    for (g in 1:length(group.actif)) {
      if (group.mod[group.actif[g]]!=1) contrib.group[g, ] <- apply(res.globale$var$contrib[(ind.var + 1):(ind.var + group.mod[group.actif[g]]), 1:ncp]/100, 2, sum)
      else contrib.group[g, ] <- res.globale$var$contrib[ind.var + 1, 1:ncp]/100
      ind.var <- ind.var + group.mod[group.actif[g]]
#      dist2.group[g] <- sum((res.separe[[group.actif[g]]]$eig[1:min(ncol(contrib.group),nrow(res.separe[[group.actif[g]]]$eig)),1]/res.separe[[group.actif[g]]]$eig[1,1])^2)
      dist2.group[g] <- sum((res.separe[[group.actif[g]]]$eig[,1]/res.separe[[group.actif[g]]]$eig[1,1])^2)
    }
    coord.group <- t(t(contrib.group)*res.globale$eig[1:ncol(contrib.group),1])
    cos2.group <- coord.group^2/dist2.group

    if (!is.null(num.group.sup)){
      coord.group.sup <- matrix(NA, length(num.group.sup), ncp)
      dimnames(coord.group.sup) <- list(name.group[num.group.sup], paste("Dim", c(1:ncp), sep = "."))
      ind.gc <- 0
      for (gc in 1:length(num.group.sup)) {
        # if (group.mod[num.group.sup[gc]]!=1) coord.group.sup[gc,] <- apply(tab[1:ncp.tmp,  (ncp.tmp+ind.gc+1):(ncp.tmp+ind.gc+group.mod[num.group.sup[gc]])],1,sum)
        # else coord.group.sup[gc,] <- tab[1:ncp.tmp,  ncp.tmp+ind.gc+1]
        for (k in 1:ncp){
          if (is.null(ind.sup)) coord.group.sup[gc,k] <- funcLg(res.globale$ind$coord[,k,drop=FALSE],data.group.sup[,(ind.gc+1):(ind.gc+group.mod[num.group.sup[gc]]),drop=FALSE],ponderation.x=1/res.globale$eig[k,1],ponderation.y=ponderation.group.sup[(ind.gc+1):(ind.gc+group.mod[num.group.sup[gc]])],wt=row.w/sum(row.w))
          else coord.group.sup[gc,k] <- funcLg(res.globale$ind$coord[-ind.sup,k,drop=FALSE],data.group.sup[-ind.sup,(ind.gc+1):(ind.gc+group.mod[num.group.sup[gc]]),drop=FALSE],ponderation.x=1/res.globale$eig[k,1],ponderation.y=ponderation.group.sup[(ind.gc+1):(ind.gc+group.mod[num.group.sup[gc]])],wt=row.w/sum(row.w))
        }
        ind.gc <- ind.gc + group.mod[num.group.sup[gc]]
      }
    }
    Lg <- matrix(0, nbre.group+1, nbre.group+1)
    ind.gl <- 0
    for (gl in c(group.actif,num.group.sup)) {
        ind.gc <- 0
        for (gc in c(group.actif,num.group.sup)) {
            if (gc>=gl){
#             Lg[gl, gc] <- Lg[gc, gl] <- sum(tab[(ind.gl + 1):(ind.gl + group.mod[gl]),  (ind.gc + 1):(ind.gc + group.mod[gc])])
              if (is.null(num.group.sup)) {
                if (is.null(ind.sup)) Lg[gl, gc] <- Lg[gc, gl] <- funcLg(x=data[,ind.gl + (1:group.mod[gl]),drop=FALSE],y=data[,ind.gc + (1:group.mod[gc]),drop=FALSE],ponderation.x=ponderation[ind.gl + (1:group.mod[gl])],ponderation.y=ponderation[ind.gc + (1:group.mod[gc])],wt=row.w/sum(row.w))
                else Lg[gl, gc] <- Lg[gc, gl] <- funcLg(x=data[-ind.sup,ind.gl + (1:group.mod[gl]),drop=FALSE],y=data[-ind.sup,ind.gc + (1:group.mod[gc]),drop=FALSE],ponderation.x=ponderation[ind.gl + (1:group.mod[gl])],ponderation.y=ponderation[ind.gc + (1:group.mod[gc])],wt=row.w/sum(row.w))
              } else {
                if (is.null(ind.sup)) Lg[gl, gc] <- Lg[gc, gl] <- funcLg(x=cbind.data.frame(data,data.group.sup)[,ind.gl + (1:group.mod[gl]),drop=FALSE],y=cbind.data.frame(data,data.group.sup)[,ind.gc + (1:group.mod[gc]),drop=FALSE],ponderation.x=c(ponderation,ponderation.group.sup)[ind.gl + (1:group.mod[gl])],ponderation.y=c(ponderation,ponderation.group.sup)[ind.gc + (1:group.mod[gc])],wt=row.w/sum(row.w))
                else Lg[gl, gc] <- Lg[gc, gl] <- funcLg(x=cbind.data.frame(data,data.group.sup)[-ind.sup,ind.gl +(1:group.mod[gl]),drop=FALSE],y=cbind.data.frame(data,data.group.sup)[-ind.sup,ind.gc + (1:group.mod[gc]),drop=FALSE],ponderation.x=c(ponderation,ponderation.group.sup)[ind.gl +(1:group.mod[gl])],ponderation.y=c(ponderation,ponderation.group.sup)[ind.gc + (1:group.mod[gc])],wt=row.w/sum(row.w))
              }
            }
            ind.gc <- ind.gc + group.mod[gc]
        }
        ind.gl <- ind.gl + group.mod[gl]
    }
   Lg[nbre.group+1,] <- Lg[,nbre.group+1] <- apply(Lg[group.actif,],2,sum)/res.globale$eig[1,1]
   Lg[nbre.group+1,nbre.group+1] <- sum(Lg[group.actif,nbre.group+1])/res.globale$eig[1,1]
    dist2.group <- diag(Lg)
    if (!is.null(num.group.sup)){
      dist2.group.sup <- dist2.group[num.group.sup]
      dist2.group <- dist2.group[-num.group.sup]
    }
    RV <- sweep(Lg, 2, sqrt(diag(Lg)), "/")
    RV <- sweep(RV, 1, sqrt(diag(Lg)), "/")
    rownames(Lg) <- colnames(Lg) <- rownames(RV) <- colnames(RV) <- c(name.group,"MFA")
    data.partiel <- vector(mode = "list", length = nbre.group)
    names(data.partiel) <- name.group
    ind.col <- 0
    for (g in 1:nbre.group) {
      if (g%in%group.actif){
        data.partiel[[g]] <- as.data.frame(matrix(res.globale$call$centre, nrow(data), ncol(data), byrow = TRUE, dimnames = dimnames(data)))
        data.partiel[[g]][, (ind.col + 1):(ind.col + group.mod[g])] <- data[, (ind.col + 1):(ind.col + group.mod[g])]
        ind.col <- ind.col + group.mod[g]
      }
    }
    res.ind.partiel <- vector(mode = "list", length = nbre.group)
    names(res.ind.partiel) <- name.group

    for (g in group.actif){
      Xis <- t(t(as.matrix(data.partiel[[g]]))-res.globale$call$centre)
      Xis <- t(t(Xis)/res.globale$call$ecart.type)
##      coord.ind.sup <- length(group.actif) * as.matrix(Xis)%*%diag((res.globale$call$col.w))%*%res.globale$svd$V
      coord.ind.sup <- length(group.actif) * as.matrix(Xis)
      coord.ind.sup <- t(t(coord.ind.sup)*res.globale$call$col.w)
      coord.ind.sup <- crossprod(t(coord.ind.sup),res.globale$svd$V)
      res.ind.partiel[[g]]$coord.sup <- coord.ind.sup
    }
    cor.grpe.fact <- as.matrix(matrix(NA, length(group.actif), ncp))
    colnames(cor.grpe.fact) <- paste("Dim", c(1:ncp), sep = ".")
    rownames(cor.grpe.fact) <- name.group[group.actif]
    for (f in 1:ncp) {
## modif April 2011
        for (g in 1:length(group.actif))  cor.grpe.fact[g, f] <- cov.wt(cbind.data.frame(res.ind.partiel[[group.actif[g]]]$coord.sup[1:nb.actif, f], res.globale$ind$coord[, f]),wt=row.w/sum(row.w),method="ML",cor=TRUE)$cor[1,2]
    }
    It <- vector(length = ncp)
    for (g in group.actif)  It <- It + apply(res.ind.partiel[[g]]$coord.sup[1:nb.actif,]^2*row.w,2,sum)
    rap.inertie <- apply(res.globale$ind$coord^2*row.w,2,sum) * length(group.actif) / It

    res.groupes <- list(Lg = Lg, RV = RV, coord = coord.group[, 1:ncp], contrib = contrib.group[, 1:ncp] * 100,  cos2 = cos2.group[, 1:ncp], dist2 = dist2.group[-length(dist2.group)], correlation = cor.grpe.fact[, 1:ncp])
    if (!is.null(num.group.sup)){
      res.groupes$coord.sup <- coord.group.sup[,1:ncp,drop=FALSE]
#      res.groupes$contrib.sup <- sweep(coord.group.sup[,1:ncp,drop=FALSE], 2, res.globale$eig[1:ncp,1], "/")*100
      res.groupes$cos2.sup <- coord.group.sup[,1:ncp,drop=FALSE]^2/dist2.group.sup
      res.groupes$dist2.sup <- dist2.group.sup
    }
  ####CHANGED THIS!!!! ------------------
  # OLD#
  # nom.ligne <- NULL
  # for (i in 1:nb.actif) nom.ligne <- c(nom.ligne, paste(rownames(base)[i], name.group[group.actif], sep = "."))
  # NEW#
  nom.ligne <-
    c(
      sapply(
        rownames(base)[1:nb.actif],
        paste,
        name.group[group.actif],
        sep = ".")
    )

    tmp <- array(0,dim=c(nrow(res.globale$ind$coord),ncp,length(group.actif)))
    for (g in 1:length(group.actif)) tmp[,,g] <- (res.ind.partiel[[group.actif[g]]]$coord.sup[1:nb.actif,1:ncp,drop=FALSE]-res.globale$ind$coord[,1:ncp,drop=FALSE])^2/length(group.actif)
## Ajour Avril 2011
tmp <- tmp*row.w
    variab.auxil <- apply(tmp,2,sum)   ## attention, array
    tmp <- sweep(tmp,2,variab.auxil,FUN="/") * 100  ## attention, array
    inertie.intra.ind <- apply(tmp,c(1,2),sum)
  ####CHANGED THIS!!!! --------------------
  #OLD#
  #  inertie.intra.ind.partiel <- as.data.frame(matrix(NA, (nb.actif * length(group.actif)), ncp.tmp))
#  for (i in 1:nb.actif) inertie.intra.ind.partiel[((i - 1) * length(group.actif)  + 1):(i * length(group.actif)), ] <- t(tmp[i,1:ncp,])
  #NEW#
  inertie.intra.ind.partiel <-
    data.frame(
      do.call(
        rbind,
        lapply(1:dim(tmp)[1], function(x) t(tmp[x, 1:ncp, ]))
      )
    )
  inertie.intra.ind.partiel <- as.matrix(inertie.intra.ind.partiel)
  # xxx <- as.data.frame(matrix(NA, (nb.actif * length(group.actif)), ncp))
  # colnames(inertie.intra.ind.partiel) <- colnames(xxx)


    rownames(inertie.intra.ind) <- rownames(res.globale$ind$coord)
    rownames(inertie.intra.ind.partiel) <- nom.ligne
    colnames(inertie.intra.ind) <- colnames(inertie.intra.ind.partiel) <- paste("Dim", c(1:ncp), sep = ".")
    tab.partial.axes <- matrix(NA, nb.actif, ncp * nbre.group)
    rownames(tab.partial.axes) <- rownames(data)[1:nb.actif]
    nom.axes <- paste("Dim", c(1:ncp), sep = "")
    nom.col <- NULL
    debut <- 0
    for (g in 1:nbre.group) {
      nom.col <- c(nom.col, paste(nom.axes, name.group[g],sep="."))
      nbcol <- min(ncp, ncol(res.separe[[g]]$ind$coord))
      tab.partial.axes[, (debut + 1):(debut + nbcol)] <- res.separe[[g]]$ind$coord[,1:nbcol]
      debut <- debut + ncp
    }
   colnames(tab.partial.axes) <- nom.col
    indice.col.NA <- which(!is.na(tab.partial.axes[1, ]))
    tab.partial.axes <- tab.partial.axes[, indice.col.NA]
    centre <- apply(tab.partial.axes, 2, moy.p, res.globale$call$row.w)
    tab.partial.axes <- t(t(tab.partial.axes)-centre)
    ecart.type <- apply(tab.partial.axes, 2, ec, res.globale$call$row.w)
    ecart.type[ecart.type <= 1e-08] <- 1
    tab.partial.axes <- t(t(tab.partial.axes)/ecart.type)
##    coord.res.partial.axes <- t(tab.partial.axes) %*% diag(res.globale$call$row.w) %*% res.globale$svd$U
    coord.res.partial.axes <- t(tab.partial.axes*res.globale$call$row.w)
    coord.res.partial.axes <- crossprod(t(coord.res.partial.axes),res.globale$svd$U[,1:ncp])
    contrib.res.partial.axes <- coord.res.partial.axes*0
    debut <- 0
    for (g in 1:nbre.group) {
      nbcol <- min(ncp, ncol(res.separe[[g]]$ind$coord))
      if (g %in% group.actif) contrib.res.partial.axes[(debut + 1):(debut + nbcol),] <- coord.res.partial.axes[(debut + 1):(debut + nbcol),]^2*res.separe[[g]]$eig[1:nbcol,1]/res.separe[[g]]$eig[1,1]
      debut <- debut + nbcol
    }
    contrib.res.partial.axes <- t(t(contrib.res.partial.axes)/apply(contrib.res.partial.axes,2,sum)) *100
#   contrib.res.partial.axes <- t(t(coord.res.partial.axes^2)/res.globale$eig[1:ncp,1]) *100

    sigma <- apply(tab.partial.axes, 2, ec, res.globale$call$row.w)
    cor.res.partial.axes <- coord.res.partial.axes/sigma
    colnames(coord.res.partial.axes) <- paste("Dim", c(1:ncol(coord.res.partial.axes)), sep = ".")
    dimnames(contrib.res.partial.axes) <- dimnames(cor.res.partial.axes) <- dimnames (coord.res.partial.axes)
    summary.n <- as.data.frame(matrix(NA, 0, 4))
    colnames(summary.n) <- c("group", "variable", "modalite", "effectif")
    summary.c <- as.data.frame(matrix(NA, 0, 6))
    colnames(summary.c) <- c("group", "variable", "moyenne", "ecart.type", "minimum", "maximum")
   for (g in 1:nbre.group) {
        if ((type[g] == "c")||(type[g]=="f")) {
            statg <- as.data.frame(matrix(NA, ncol(res.separe[[g]]$call$X), 6))
            colnames(statg) <- c("group", "variable", "moyenne", "ecart.type", "minimum", "maximum")
            statg[, "group"] <- rep(g, nrow(statg))
            statg[, "variable"] <- colnames(res.separe[[g]]$call$X)
            statg[, "moyenne"] <- res.separe[[g]]$call$centre
            if (!is.null(res.separe[[g]]$call$ecart.type)) statg[, "ecart.type"] <- res.separe[[g]]$call$ecart.type
            statg[, "minimum"] <- apply(res.separe[[g]]$call$X, 2, min)
            statg[, "maximum"] <- apply(res.separe[[g]]$call$X, 2, max)
            if (!is.null(res.separe[[g]]$call$ecart.type)) statg[, -c(1, 2)] <- round(statg[, -c(1, 2)], digits = 2)
            else statg[, -c(1, 2,4)] <- round(statg[, -c(1, 2,4)], digits = 2)
            summary.c <- rbind(summary.c, statg)
        }
        else {
            if(is.null(excl[[g]])) statg <- as.data.frame(matrix(NA, length(res.separe[[g]]$call$marge.col), 4))
            else statg <- as.data.frame(matrix(NA, length(res.separe[[g]]$call$marge.col[-excl[[g]]]), 4))
            colnames(statg) <- c("group", "variable", "modalite", "effectif")
            statg[, "group"] <- rep(g, nrow(statg))
            res.separe[[g]]$call$X <- as.data.frame(res.separe[[g]]$call$X)
            nb.var <- ncol(res.separe[[g]]$call$X)
            nb.mod <- NULL
            nom.mod <- NULL
            nb.mod.orig <- NULL
            nb.mod.high <- 0
            for (v in 1:nb.var) {
              if(is.null(excl[[g]])) {
                nb.mod <- c(nb.mod, nlevels(res.separe[[g]]$call$X[, v]))
              } else {
                nb.mod.orig <- c(nb.mod.orig, nlevels(res.separe[[g]]$call$X[, v]))
                nb.mod.low <- nb.mod.high
                nb.mod.high <- nb.mod.high + nlevels(res.separe[[g]]$call$X[, v])
                nb.mod.rm.select <- (nb.mod.low < res.separe[[g]]$call$excl & res.separe[[g]]$call$excl <= nb.mod.high)
                nb.mod.rm <- res.separe[[g]]$call$excl[nb.mod.rm.select]
                nb.mod <- c(nb.mod, (nb.mod.orig[v]-length(nb.mod.rm)))
              }
              nom.mod <- c(nom.mod, levels(res.separe[[g]]$call$X[, v]))
            }
            if(!is.null(excl[[g]])) nom.mod <- nom.mod[-excl[[g]]]
            statg[, "variable"] <- rep(colnames(res.separe[[g]]$call$X), nb.mod)
            statg[, "modalite"] <- nom.mod
            if(is.null(excl[[g]])) statg[, "effectif"] <- res.separe[[g]]$call$marge.col * nbre.ind * nb.var
            else statg[, "effectif"] <- res.separe[[g]]$call$marge.col[-excl[[g]]] * nbre.ind * nb.var
            summary.n <- rbind(summary.n, statg)
        }
    }
   eig <- res.globale$eig
  ## CHANGED THIS!!!! ---------------
  # nom.ligne <- NULL
  # for (i in 1:nbre.ind) {
  #   ind.tmp <- rownames(base)[i]
  #   nom.ligne <- c(nom.ligne, paste(ind.tmp, name.group[group.actif], sep = "."))
  # }

  nom.ligne <-
    c(
      sapply(
        rownames(base),
        paste,
        name.group[group.actif],
        sep = ".")
    )
    coord.ind.partiel <- matrix(NA, (nbre.ind * length(group.actif)), ncp)
    rownames(coord.ind.partiel) <- nom.ligne
    colnames(coord.ind.partiel) <- paste("Dim", c(1:ncp), sep = ".")
    coord.ind <- rbind(res.globale$ind$coord[, 1:ncp,drop=FALSE],res.globale$ind.sup$coord[, 1:ncp,drop=FALSE])
    cos2.ind <- rbind(res.globale$ind$cos2[, 1:ncp,drop=FALSE],res.globale$ind.sup$cos2[, 1:ncp,drop=FALSE])
    contrib.ind <- res.globale$ind$contrib[, 1:ncp,drop=FALSE]
    liste.ligne <- seq(1, nbre.ind * length(group.actif), by = length(group.actif))
    for (g in 1:length(group.actif)) coord.ind.partiel[liste.ligne+g-1, ] <- res.ind.partiel[[group.actif[g]]]$coord.sup[, 1:ncp,drop=FALSE]
    if (!is.null(ind.sup)) {
      res.ind.sup <- list(coord = coord.ind[(nb.actif+1):nrow(coord.ind),,drop=FALSE], cos2 = cos2.ind[(nb.actif+1):nrow(coord.ind),,drop=FALSE], coord.partiel = coord.ind.partiel[(length(group.actif)*nb.actif+1):nrow(coord.ind.partiel),,drop=FALSE])
      res.ind <- list(coord = coord.ind[1:nb.actif,,drop=FALSE], contrib = contrib.ind, cos2 = cos2.ind[1:nb.actif,,drop=FALSE], within.inertia = inertie.intra.ind[1:nb.actif,1:ncp,drop=FALSE], coord.partiel = coord.ind.partiel[1:(length(group.actif)*nb.actif),,drop=FALSE], within.partial.inertia = inertie.intra.ind.partiel[1:(length(group.actif)*nb.actif),1:ncp,drop=FALSE] )
    }
    else res.ind <- list(coord = coord.ind, contrib = contrib.ind, cos2 = cos2.ind, within.inertia = inertie.intra.ind[,1:ncp,drop=FALSE], coord.partiel = coord.ind.partiel, within.partial.inertia = inertie.intra.ind.partiel[,1:ncp,drop=FALSE])

    res.quali.var <- res.quali.var.sup <- NULL
    bool.act <- FALSE
    bool.sup <- FALSE
    if (!is.null(ind.quali)) {
      coord.quali <- res.globale$quali.sup$coord[, 1:ncp,drop=FALSE]
      cos2.quali <- res.globale$quali.sup$cos2[, 1:ncp,drop=FALSE]
      val.test.quali <- res.globale$quali.sup$v.test[, 1:ncp,drop=FALSE]
## modif Avril 2011 : attention ligne suivant enlevee et lignes ajoutee au debut du pro pour definir poids.bary
##      poids.bary <- res.globale$call$quali.sup$nombre * res.globale$call$row.w[1]
##      contrib.quali <- 100 * sweep(contrib.quali, 1, poids.bary, "*")[,1:ncp,drop=FALSE]
##      contrib.quali <- sweep(res.globale$quali.sup$coord^2, 2, res.globale$eig[1:ncol(res.globale$quali.sup$coord),1], "/")
      contrib.quali <- coord.quali * 0
      commun <- intersect(rownames(res.globale$var$contrib),rownames(contrib.quali))
      if (!is.null(commun)) contrib.quali[commun,] <- res.globale$var$contrib[commun,1:ncp,drop=FALSE]
      barycentre <- res.globale$call$quali.sup$barycentre
      coord.quali.partiel <- matrix(NA, (nrow(barycentre) * length(group.actif)), ncp)
      nom.ligne.bary <- NULL
      for (q in 1:nrow(barycentre)) {
        ind.tmp <- rownames(barycentre)[q]
        nom.ligne.bary <- c(nom.ligne.bary, paste(ind.tmp, name.group[group.actif], sep = "."))
      }
      rownames(coord.quali.partiel) <- nom.ligne.bary
      liste.ligne <- seq(1, (nrow(barycentre) * length(group.actif) ), by = length(group.actif))
      inertie.intra.cg.partiel <- matrix(NA, (nrow(barycentre) * length(group.actif) ), ncp)
      tmp <- array(0,dim=c(nrow(res.globale$quali.sup$coord),ncp,length(group.actif)))
      ind.col <- 0
      for (g in 1:length(group.actif)) {
        cg.partiel <- as.data.frame(matrix(res.globale$call$centre, nrow(barycentre), ncol(barycentre), byrow = TRUE, dimnames = dimnames(barycentre)))
        cg.partiel[, (ind.col + 1):(ind.col + group.mod[group.actif[g]])] <- barycentre[, (ind.col + 1):(ind.col + group.mod[group.actif[g]])]
        ind.col <- ind.col + group.mod[group.actif[g]]
        Xis <- t((t(cg.partiel)-res.globale$call$centre)/res.globale$call$ecart.type)
##        coord.quali.sup <- length(group.actif) * as.matrix(Xis)%*%diag((res.globale$call$col.w))%*%res.globale$svd$V
        coord.quali.sup <- length(group.actif) * as.matrix(Xis)
        coord.quali.sup <- t(t(coord.quali.sup)*res.globale$call$col.w)
        coord.quali.sup <- crossprod(t(coord.quali.sup),res.globale$svd$V)
        coord.quali.partiel[liste.ligne + g - 1, ] <- coord.quali.sup[,1:ncp]
        tmp[,,g] <- (coord.quali.sup[,1:ncp,drop=FALSE] - res.globale$quali.sup$coord[,1:ncp,drop=FALSE])^2 / length(group.actif)
      }
      colnames(coord.quali.partiel) <-  paste("Dim", 1:ncp, sep = ".")
      tmp <- sweep(tmp,2,variab.auxil,FUN="/") * 100   ### attention array
### modif mais attention, si changement dans PCA, remettre ?
##      tmp <- sweep(tmp,1,res.globale$call$quali.sup$nombre,FUN="*")
      tmp <- sweep(tmp,1,poids.bary*sum(row.w),FUN="*")  ### attention array
      inertie.intra.cg <- apply(tmp,c(1,2),sum)
      for (i in 1:nrow(barycentre)) inertie.intra.cg.partiel[((i - 1) * length(group.actif)  + 1):(i * length(group.actif)), ] <- t(tmp[i,1:ncp,])
      rownames(inertie.intra.cg) <- rownames(res.globale$quali.sup$coord)
      rownames(inertie.intra.cg.partiel) <- nom.ligne.bary
      colnames(inertie.intra.cg) <- colnames(inertie.intra.cg.partiel) <- paste("Dim", c(1:ncp), sep = ".")

      ind.col <- 0
      ind.col.act <- NULL
      ind.col.sup <- NULL
      ind.excl <- NULL
      ind.excl.act <- NULL
      for (g in 1:nbre.group) {
        if (type[g] =="n"){
          if (g%in%num.group.sup) ind.col.sup <- c(ind.col.sup, (ind.col+1):(ind.col+group.mod[g]))
          else ind.col.act <- c(ind.col.act, (ind.col+1):(ind.col+group.mod[g]))
          if(!is.null(excl[[g]])) {
            ind.excl <- ind.col + excl[[g]]
            ind.excl.act <- c(ind.excl.act, ind.excl)
          }
          ind.col = ind.col + group.mod[g]
        }
      }
      if(!is.null(ind.excl.act)) ind.col.act <- ind.col.act[-ind.excl.act]

      if (!is.null(ind.col.sup)) {
            coord.quali.sup <- coord.quali[ind.col.sup,,drop=FALSE]
            cos2.quali.sup <- cos2.quali[ind.col.sup,,drop=FALSE]
            val.test.quali.sup <- val.test.quali[ind.col.sup,,drop=FALSE]
            coord.quali.partiel.sup <- coord.quali.partiel[unlist(lapply(ind.col.sup, function(k) seq(length(group.actif)*(k-1)+1,length=length(group.actif)))),]
            inertie.intra.cg.sup <- inertie.intra.cg[ind.col.sup,1:ncp]
            inertie.intra.cg.partiel.sup <- inertie.intra.cg.partiel[unlist(lapply(ind.col.sup, function(k) seq(length(group.actif)*(k-1)+1,length=length(group.actif)))),1:ncp]
            bool.sup <- TRUE
       }
       if (!is.null(ind.col.act)) {
            coord.quali.act <- coord.quali[ind.col.act,,drop=FALSE]
            contrib.quali.act <- contrib.quali[ind.col.act,,drop=FALSE]
            val.test.quali.act <- NULL
            if (is.null(tab.comp)){
              cos2.quali.act <- cos2.quali[ind.col.act,,drop=FALSE]
              val.test.quali.act <- val.test.quali[ind.col.act,,drop=FALSE]
            } else {
              cos2.quali.act <- res.globale$quali.var$cos2
            }
            coord.quali.partiel.act <- coord.quali.partiel[unlist(lapply(ind.col.act, function(k) seq(length(group.actif)*(k-1)+1,length=length(group.actif)))),]
            inertie.intra.cg.act <- inertie.intra.cg[ind.col.act,1:ncp]
            inertie.intra.cg.partiel.act <- inertie.intra.cg.partiel[unlist(lapply(ind.col.act, function(k) seq(length(group.actif)*(k-1)+1,length=length(group.actif)))),1:ncp]
            bool.act <- TRUE
        }
      # for (g in 1:nbre.group) {
        # if (type[g] =="n"){
          # if (g%in%num.group.sup) {
            # coord.quali.sup <- coord.quali[ind.col.sup,,drop=FALSE]
           # contrib.quali.sup <- contrib.quali[ind.col.sup,,drop=FALSE]
            # cos2.quali.sup <- cos2.quali[ind.col.sup,,drop=FALSE]
            # val.test.quali.sup <- val.test.quali[ind.col.sup,,drop=FALSE]
            # coord.quali.partiel.sup <- coord.quali.partiel[unlist(lapply(ind.col.sup, function(k) seq(length(group.actif)*(k-1)+1,length=length(group.actif)))),]
            # inertie.intra.cg.sup <- inertie.intra.cg[ind.col.sup,1:ncp]
            # inertie.intra.cg.partiel.sup <- inertie.intra.cg.partiel[unlist(lapply(ind.col.sup, function(k) seq(length(group.actif)*(k-1)+1,length=length(group.actif)))),1:ncp]
            # bool.sup <- TRUE
          # }
          # else {
            # coord.quali.act <- coord.quali[ind.col.act,,drop=FALSE]
            # contrib.quali.act <- contrib.quali[ind.col.act,,drop=FALSE]
            # cos2.quali.act <- cos2.quali[ind.col.act,,drop=FALSE]
            # val.test.quali.act <- val.test.quali[ind.col.act,,drop=FALSE]
            # coord.quali.partiel.act <- coord.quali.partiel[unlist(lapply(ind.col.act, function(k) seq(length(group.actif)*(k-1)+1,length=length(group.actif)))),]
            # inertie.intra.cg.act <- inertie.intra.cg[ind.col.act,1:ncp]
            # inertie.intra.cg.partiel.act <- inertie.intra.cg.partiel[unlist(lapply(ind.col.act, function(k) seq(length(group.actif)*(k-1)+1,length=length(group.actif)))),1:ncp]
            # bool.act <- TRUE
          # }
        # }
      # }
      if (bool.act) res.quali.var <- list(coord = coord.quali.act, contrib = contrib.quali.act, cos2 = cos2.quali.act, v.test = val.test.quali.act, coord.partiel = coord.quali.partiel.act, within.inertia = inertie.intra.cg.act, within.partial.inertia = inertie.intra.cg.partiel.act)
      if (bool.sup) res.quali.var.sup <- list(coord = coord.quali.sup, cos2 = cos2.quali.sup, v.test = val.test.quali.sup, coord.partiel = coord.quali.partiel.sup, within.inertia = inertie.intra.cg.sup, within.partial.inertia = inertie.intra.cg.partiel.sup)
    }
    indice.quanti <- NULL
    indice.freq <- NULL
    num.tmp <- 0
    for (g in group.actif) {
        if (type[g] == "c")  indice.quanti <- c(indice.quanti, c((num.tmp + 1):(num.tmp + group.mod[g])))
        if (type[g] == "f")  indice.freq <- c(indice.freq, c((num.tmp + 1):(num.tmp + group.mod[g])))
        num.tmp <- num.tmp + group.mod[g]
    }
    res.quanti.var <- NULL
    if (!is.null(indice.quanti)){
      coord.quanti.var <- res.globale$var$coord[indice.quanti,1:ncp,drop=FALSE]
#      coord.quanti.var <- as.data.frame(res.globale$var$coord[indice.quanti,1:ncp,drop=FALSE])
      cos2.quanti.var <- res.globale$var$cos2[indice.quanti, 1:ncp,drop=FALSE]
      contrib.quanti.var <- res.globale$var$contrib[indice.quanti, 1:ncp,drop=FALSE]
      cor.quanti.var <- res.globale$var$cor[indice.quanti, 1:ncp,drop=FALSE]
      res.quanti.var <- list(coord = coord.quanti.var, contrib = contrib.quanti.var, cos2 = cos2.quanti.var, cor = cor.quanti.var)
    }
    res.freq <- NULL
    if (!is.null(indice.freq)){
      coord.freq <- res.globale$var$coord[indice.freq,1:ncp,drop=FALSE]
#      coord.freq <- as.data.frame(res.globale$var$coord[indice.freq,1:ncp,drop=FALSE])
      cos2.freq <- res.globale$var$cos2[indice.freq, 1:ncp,drop=FALSE]
      contrib.freq <- res.globale$var$contrib[indice.freq, 1:ncp,drop=FALSE]
      res.freq <- list(coord = coord.freq, contrib = contrib.freq, cos2 = cos2.freq)
    }

    res.quanti.var.sup <- NULL
    res.freq.sup <- NULL
    if (!is.null(num.group.sup)){
      num.tmp <- 0
      indice.quanti <- NULL
      indice.freq <- NULL
      for (g in num.group.sup) {
        if (type[g] == "c") indice.quanti <- c(indice.quanti, c((num.tmp + 1):(num.tmp + group.mod[g])))
        if (type[g]=="f") indice.freq <- c(indice.freq, c((num.tmp + 1):(num.tmp + group.mod[g])))
        num.tmp <- num.tmp + group.mod[g]
      }
      if (!is.null(indice.quanti)){
        coord.quanti.var.sup <- res.globale$quanti.sup$coord[indice.quanti,1:ncp,drop=FALSE]
        cos2.quanti.var.sup <- res.globale$quanti.sup$cos2[indice.quanti, 1:ncp,drop=FALSE]
        cor.quanti.var.sup <- res.globale$quanti.sup$cor[indice.quanti, 1:ncp,drop=FALSE]
        res.quanti.var.sup <- list(coord = coord.quanti.var.sup, cos2 = cos2.quanti.var.sup, cor = cor.quanti.var.sup)
      }
      if (!is.null(indice.freq)){
        coord.freq.sup <- res.globale$quanti.sup$coord[indice.freq,1:ncp,drop=FALSE]
        cos2.freq.sup <- res.globale$quanti.sup$cos2[indice.freq, 1:ncp,drop=FALSE]
        res.freq.sup <- list(coord = coord.freq.sup, cos2 = cos2.freq.sup)
      }
    }

    aux <- res.separe[[1]]$ind$coord
    name.aux <- paste(colnames(res.separe[[1]]$ind$coord),name.group[1],sep=".")
    for (g in 2:nbre.group) {
      aux <- cbind(aux,res.separe[[g]]$ind$coord)
      name.aux = c(name.aux,paste(colnames(res.separe[[g]]$ind$coord),name.group[g],sep="."))
    }
## modif Avril 2011
#    cor.partial.axes <- cor(aux)
cor.partial.axes <- cov.wt(aux,wt=row.w/sum(row.w),method="ML",cor=TRUE)$cor
    dimnames(cor.partial.axes) <- list(name.aux,name.aux)
    res.partial.axes <- list(coord = coord.res.partial.axes[, 1:ncp], cor = cor.res.partial.axes[, 1:ncp], contrib = contrib.res.partial.axes[, 1:ncp], cor.between = cor.partial.axes)
    resultats <- list(separate.analyses = res.separe, eig = eig, group = res.groupes,
        inertia.ratio = rap.inertie[1:ncp], ind = res.ind)
    if (!is.null(ind.sup)) resultats$ind.sup <- res.ind.sup
    if (!is.null(c(res.quanti.var,res.quanti.var.sup))) resultats$summary.quanti = summary.c
    if (!is.null(c(bool.act,bool.sup))) resultats$summary.quali = summary.n
    if (!is.null(res.quanti.var)) resultats$quanti.var = res.quanti.var
    if (!is.null(res.quanti.var.sup)) resultats$quanti.var.sup = res.quanti.var.sup
    if (!is.null(res.freq)) resultats$freq = res.freq
    if (!is.null(res.freq.sup)) resultats$freq.sup = res.freq.sup
    if (bool.act) resultats$quali.var = res.quali.var
    if (bool.sup) resultats$quali.var.sup = res.quali.var.sup
    resultats$partial.axes = res.partial.axes
    resultats$call = call
    resultats$call$call <- match.call()
    resultats$global.pca = res.globale
    class(resultats) <- c("MFA", "list")

    if (graph & (ncp>1)){
      if (bool.act | bool.sup){
        cg.plot.partial <- NULL
        if (!is.null(resultats["quali.var"]$quali.var)){
          max.inertia <- order(apply(resultats$quali.var$within.inertia[,1:2],1,sum))
          cg.plot.partial <- rownames(resultats$quali.var$coord)[max.inertia[1:length(max.inertia)]]
        }
        if (!is.null(resultats$quali.var.sup)){
          max.inertia <- order(apply(resultats$quali.var.sup$within.inertia[,1:2],1,sum))
          cg.plot.partial <- c(cg.plot.partial,rownames(resultats$quali.var.sup$coord)[max.inertia[1:length(max.inertia)]])
        }
        print(plot.MFA(resultats,choix="ind",invisible="ind",partial=cg.plot.partial,habillage="group",axes=axes,new.plot=TRUE))
      }
      max.inertia <- order(apply(resultats$ind$within.inertia[,1:2],1,sum))
      print(plot.MFA(resultats,choix="axes",habillage="group",axes=axes,new.plot=TRUE,shadowtext=TRUE))
      print(plot.MFA(resultats,choix="ind",invisible="quali",partial=rownames(resultats$ind$coord)[max.inertia[c(1:2,nrow(resultats$ind$coord)-1,nrow(resultats$ind$coord))]],habillage="group",axes=axes,new.plot=TRUE))
      if ("c"%in%type) print(plot.MFA(resultats,choix="var",habillage="group",axes=axes,new.plot=TRUE,shadowtext=TRUE))
      if ("f"%in%type) print(plot.MFA(resultats,choix="freq",habillage="group",axes=axes,new.plot=TRUE))
      print(plot.MFA(resultats,choix="ind",invisible="quali",habillage = "ind",axes=axes,new.plot=TRUE,col.hab=1+3*(1:nbre.ind)%in%ind.sup))
      print(plot.MFA(resultats,choix="group",axes=axes,new.plot=TRUE))
    }
    return(resultats)
}


mfa <- MFA(wine, group=c(2,5,3,10,9,2), type=c("n",rep("s",5)), ncp=5, name.group=c("orig","olf","vis","olfag","gust","ens"), graph=FALSE)

# print("ORIG")
# print(mfa$separate.analyses$orig$eig)
# print(mfa$separate.analyses$orig$svd$vs)
# print("---")

# print("VIS")
# print(mfa$separate.analyses$vis$eig)
# print(mfa$separate.analyses$vis$svd$vs)
print("---")

print(mfa$global.pca$eig[1:5,])
print("---")

# print("U")
# print(mfa$global.pca$svd$U[1:5,])
# print("---")

# print("V")
# print(mfa$global.pca$svd$V[1:5,])
# print("---")

print("s")
print(mfa$global.pca$svd$vs)
print("---")

print("Row coords")
print(mfa$ind$coord[1:5,])
print("---")
