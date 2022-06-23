library("FactoMineR")

CA <- function (X, ncp = 5, row.sup = NULL, col.sup = NULL, quanti.sup=NULL, quali.sup=NULL, graph = TRUE, axes=c(1,2), row.w=NULL, excl=NULL){

  # fct.eta2 <- function(vec,x,weights) {
  # res <- summary(lm(x~vec,weights=weights,na.action=na.omit))$r.squared
  # }
  fct.eta2 <- function(vec,x,weights) {   ## pb avec les poids
    VB <- function(xx) {
      return(sum((colSums((tt*xx)*weights)^2)/ni))
    }
    tt <- tab.disjonctif(vec)
    ni <- colSums(tt*weights)
    unlist(lapply(as.data.frame(x),VB))/colSums(x*x*weights)
  }

  if (is.table(X)) X <- matrix(as.vector(X),nrow(X),dimnames=dimnames(X))
  if (is.null(rownames(X))) rownames(X) <- 1:nrow(X)
  if (is.null(colnames(X))) colnames(X) <- colnames(X, do.NULL = FALSE,prefix="V")
  X <- as.data.frame(X)
  is.quali <- which(!unlist(lapply(X,is.numeric)))
  X[,is.quali] <- lapply(X[,is.quali,drop=FALSE],as.factor)
  for (i in is.quali) X[,i]=as.factor(X[,i])
  X <- droplevels(X)
  Xtot <- X
  if (any(!sapply(X, is.numeric))) {
    auxi = NULL
    for (j in (1:ncol(X))[!((1:ncol(X))%in%quali.sup)]) if (!is.numeric(X[,j])) auxi = c(auxi,colnames(X)[j])
    if (!is.null(auxi)) stop(paste("\nThe following variables are not quantitative: ", auxi))
  }
  if (!inherits(X, "data.frame")) stop("X is not a data.frame")
  if (!is.null(row.sup)) X <- as.data.frame(X[-row.sup,])
  if ((!is.null(col.sup))||(!is.null(quanti.sup))||(!is.null(quali.sup))) X <- as.data.frame(X[,-c(col.sup,quanti.sup,quali.sup)])
  if (any(apply(X,1,sum)==0)){
    warning(paste0("The rows ",paste(rownames(X)[which(apply(X,1,sum)==0)],collapse=", ")," sum at 0. They were suppressed from the analysis"))
    X <- X[-which(apply(X,1,sum)==0),,drop=FALSE]
  }
  if (any(apply(X,2,sum)==0)){
    warning(paste0("The columns ",paste(colnames(X)[which(apply(X,2,sum)==0)],collapse=", ")," sum at 0. They were suppressed from the analysis"))
    X <- X[,-which(apply(X,2,sum)==0),drop=FALSE]
  }
  ### 3 lignes rajoutees
  if (is.null(row.w)) row.w = rep(1,nrow(X))
  row.w.init <- row.w
  if (length(row.w)!=nrow(X)) stop("length of vector row.w should be the number of active rows")
  total <- sum(X*row.w)
  F <- as.matrix(X)*(row.w/total)
  marge.col <- colSums(F)
  marge.row <- rowSums(F)
  ncp <- min(ncp, (nrow(X) - 1), (ncol(X) - 1))
  Tc <- t(t(F/marge.row)/marge.col) - 1
  print(head(Tc, 5))
  if(!is.null(excl)) marge.col[excl] <- 1e-15
  tmp <- svd.triplet(Tc, row.w = marge.row, col.w = marge.col,ncp=ncp)
  if(!is.null(excl)) marge.col[excl] <- 0
  eig <- tmp$vs^2
  vp <- matrix(NA, length(eig), 3)
  rownames(vp) <- paste("dim", 1:length(eig))
  colnames(vp) <- c("eigenvalue", "percentage of variance", "cumulative percentage of variance")
  vp[, "eigenvalue"] <- eig
  vp[, "percentage of variance"] <- (eig/sum(eig))*100
  vp[, "cumulative percentage of variance"] <- cumsum(vp[, "percentage of variance"])
  V <- tmp$V
  U <- tmp$U
  eig <- eig[1:ncol(U)]
  coord.col <- t(t(V)*sqrt(eig))
  coord.row <- t(t(U)*sqrt(eig))
  dist2.col <- colSums(Tc^2*marge.row)
  contrib.col <- t(t(coord.col^2*marge.col)/eig)
  cos2.col <- coord.col^2/dist2.col
  colnames(coord.col) <- colnames(contrib.col) <- colnames(cos2.col) <- paste("Dim", 1:length(eig))
  rownames(coord.col) <- rownames(contrib.col) <- rownames(cos2.col) <- attributes(X)$names
  dist2.row <- rowSums(t(t(Tc^2)*marge.col))
  contrib.row <- t(t(coord.row^2*marge.row)/eig)
  cos2.row <- coord.row^2/dist2.row
  colnames(coord.row) <- colnames(contrib.row) <- colnames(cos2.row) <- paste("Dim", 1:length(eig))
  rownames(coord.row) <- rownames(contrib.row) <- rownames(cos2.row) <- attributes(X)$row.names
  inertia.row = marge.row*dist2.row
  inertia.col = marge.col*dist2.col
  names(inertia.col) <- attributes(coord.col)$row.names
  names(inertia.row) <- attributes(coord.row)$row.names

  #    res.call <- list(X = X, marge.col = marge.col, marge.row = marge.row, ncp = ncp, row.w=row.w,call=sys.calls()[[1]],Xtot=Xtot,N=sum(row.w*rowSums(X)))
  res.call <- list(X = X, marge.col = marge.col, marge.row = marge.row, ncp = ncp, row.w=row.w,excl=excl,call=match.call(),Xtot=Xtot,N=sum(row.w*rowSums(X)))
  res.col <- list(coord = as.matrix(coord.col[, 1:ncp]), contrib = as.matrix(contrib.col[, 1:ncp] * 100), cos2 = as.matrix(cos2.col[, 1:ncp]), inertia=inertia.col)
  res.row <- list(coord = coord.row[, 1:ncp], contrib = contrib.row[, 1:ncp] * 100, cos2 = cos2.row[, 1:ncp], inertia=inertia.row)
  res <- list(eig = vp[1:min(nrow(X) - 1, ncol(X) - 1),,drop=FALSE], call = res.call, row = res.row, col = res.col, svd = tmp)
  if (!is.null(row.sup)){
    X.row.sup <- as.data.frame(Xtot[row.sup,])
    if ((!is.null(col.sup))||(!is.null(quanti.sup))||(!is.null(quali.sup))) X.row.sup <- as.data.frame(X.row.sup[,-c(col.sup,quanti.sup,quali.sup)])
    somme.row <- rowSums(X.row.sup)
    X.row.sup <- X.row.sup/somme.row
    coord.row.sup <- crossprod(t(as.matrix(X.row.sup)),V)
    # modif
    dist2.row <- rowSums(t((t(X.row.sup)-marge.col)^2/marge.col))
    #dist2.row <- rowSums(sweep(sweep(X.row.sup,2,marge.col,FUN="-")^2,2,1/marge.col,FUN="*"))
    cos2.row.sup <- coord.row.sup^2/dist2.row
    coord.row.sup <- coord.row.sup[, 1:ncp,drop=FALSE]
    cos2.row.sup <- cos2.row.sup[, 1:ncp,drop=FALSE]
    colnames(coord.row.sup) <- colnames(cos2.row.sup) <- paste("Dim", 1:ncp)
    rownames(coord.row.sup) <- rownames(cos2.row.sup) <- rownames(X.row.sup)
    res.row.sup <- list(coord = coord.row.sup, cos2 = cos2.row.sup)
    res$row.sup <- res.row.sup
    res$call$row.sup <- row.sup
  }
  if (!is.null(col.sup)){
    X.col.sup <- as.data.frame(Xtot[,col.sup])
    if (!is.null(row.sup)) X.col.sup <- as.data.frame(X.col.sup[-row.sup,])
    ## 1 ligne rajoutee
    X.col.sup <- X.col.sup*row.w
    colnames(X.col.sup) <- colnames(Xtot)[col.sup]
    somme.col <- colSums(X.col.sup)
    X.col.sup <- t(t(X.col.sup)/somme.col)
    coord.col.sup <- crossprod(as.matrix(X.col.sup),U)

    dist2.col <- colSums((X.col.sup-marge.row)^2/marge.row)
    coord.col.sup <- as.matrix(coord.col.sup[,1:ncp,drop=FALSE])
    cos2.col.sup <- coord.col.sup^2/dist2.col
    cos2.col.sup <- cos2.col.sup[,1:ncp,drop=FALSE]
    colnames(coord.col.sup) <- colnames(cos2.col.sup) <- paste("Dim", 1:ncp)
    rownames(coord.col.sup) <- rownames(cos2.col.sup) <- colnames(X.col.sup)
    res.col.sup <- list(coord = coord.col.sup, cos2 = cos2.col.sup)
    res$col.sup <- res.col.sup
    res$call$col.sup <- col.sup
  }
  ## Ajout variable quanti supp.
  if (!is.null(quanti.sup)) {
    coord.quanti.sup <- matrix(NA, length(quanti.sup), ncp)
    if (is.null(row.sup)) coord.quanti.sup <- cov.wt(cbind.data.frame(res$row$coord,Xtot[,quanti.sup,drop=FALSE]),cor=TRUE,wt=marge.row,method="ML")$cor[-(1:ncp),1:ncp,drop=FALSE]
    else coord.quanti.sup <- cov.wt(cbind.data.frame(res$row$coord,Xtot[-row.sup,quanti.sup,drop=FALSE]),wt=marge.row,cor=TRUE,method="ML")$cor[-(1:ncp),1:ncp,drop=FALSE]
    dimnames(coord.quanti.sup) <- list(colnames(Xtot)[quanti.sup], paste("Dim", 1:ncp, sep = "."))
    res$quanti.sup$coord <- coord.quanti.sup
    res$quanti.sup$cos2 <- coord.quanti.sup^2
    res$call$quanti.sup <- quanti.sup
  }
  ## Ajout variable quali supp.
  if (!is.null(quali.sup)) {
    # res$quali.sup$coord <- NULL
    # if (!is.null(row.sup)) {
    # for (j in 1:length(quali.sup)) res$quali.sup$coord <- rbind.data.frame(res$quali.sup$coord,sweep(sapply(as.data.frame(sweep(res$row$coord,1,marge.row,FUN="*")),tapply,Xtot[-row.sup,quali.sup[j]],sum), 1, tapply(marge.row,Xtot[-row.sup,quali.sup[j]],sum),FUN="/"))
    # } else {
    # for (j in 1:length(quali.sup)) res$quali.sup$coord <- rbind.data.frame(res$quali.sup$coord,sweep(sapply(as.data.frame(sweep(res$row$coord,1,marge.row,FUN="*")),tapply,Xtot[,quali.sup[j]],sum), 1, tapply(marge.row,Xtot[,quali.sup[j]],sum),FUN="/"))
    # }
    # rownames(res$quali.sup$coord) <- paste(rep(colnames(Xtot)[quali.sup],lapply(Xtot[,quali.sup,drop=FALSE],nlevels)) , unlist(lapply(Xtot[,quali.sup,drop=FALSE],levels)),sep=".")

    if (!is.null(row.sup)) X.del <- as.data.frame(Xtot[-row.sup,-c(col.sup,quanti.sup,quali.sup)])
    else X.del <- Xtot[,-c(col.sup,quanti.sup,quali.sup)]
    X.quali.sup <- NULL
    # for (j in 1:length(quali.sup)){
    # if (!is.null(row.sup)) X.quali.sup <- rbind(X.quali.sup,matrix(unlist(by(X.del,Xtot[-row.sup,quali.sup[j]],colSums)),ncol=ncol(X.del),byrow=T))
    # else X.quali.sup <- rbind(X.quali.sup,matrix(unlist(by(X.del,Xtot[,quali.sup[j]],colSums)),ncol=ncol(X.del),byrow=T))
    # }
    Xtot2 <- Xtot
    if (!is.null(row.sup))  Xtot2 <- Xtot[-row.sup,]
    for (j in 1:length(quali.sup)) {
      Xtot2[,quali.sup[j]] <- droplevels(Xtot2[,quali.sup[j]] , reorder=FALSE)
      X.quali.sup <- rbind(X.quali.sup, matrix(unlist(by(X.del,
                                                         Xtot2[, quali.sup[j]], colSums)), ncol = ncol(X.del), byrow = T))
    }
    somme.quali <- rowSums(X.quali.sup)
    X.quali.sup <- X.quali.sup/somme.quali
    coord.quali.sup <- crossprod(t(as.matrix(X.quali.sup)),V)
    dist2.quali <- rowSums(t((t(X.quali.sup)-marge.col)^2/marge.col))
    cos2.quali.sup <- coord.quali.sup^2/dist2.quali
    coord.quali.sup <- coord.quali.sup[, 1:ncp,drop=FALSE]
    cos2.quali.sup <- cos2.quali.sup[, 1:ncp,drop=FALSE]
    rownames(coord.quali.sup) <- rownames(cos2.quali.sup) <- paste(rep(colnames(Xtot2)[quali.sup],lapply(Xtot2[, quali.sup, drop = FALSE], nlevels)), unlist(lapply(Xtot2[, quali.sup, drop = FALSE], levels)),sep = ".")
    colnames(coord.quali.sup) <- colnames(cos2.quali.sup) <- paste("Dim", 1:ncp)
    res$quali.sup <- list(coord = coord.quali.sup, cos2 = cos2.quali.sup)

    Zqs <- tab.disjonctif(Xtot2[, quali.sup])
    Nj <- colSums(Zqs * row.w)
    Nj <- colSums(Zqs * marge.row)*total
    if (total>1) coef <- sqrt(Nj * ((total - 1)/(total - Nj)))
    else coef <- sqrt(Nj)
    res$quali.sup$v.test <- res$quali.sup$coord*coef

    eta2 = matrix(NA, length(quali.sup), ncp)
    eta2 <- sapply(as.data.frame(Xtot2[, quali.sup, drop = FALSE]), fct.eta2, res$row$coord, weights = marge.row)
    eta2 <- t(as.matrix(eta2,ncol=ncp))
    colnames(eta2) = paste("Dim", 1:ncp)
    rownames(eta2) = colnames(Xtot)[quali.sup]

    res$quali.sup$eta2 <- eta2
    res$call$quali.sup <- quali.sup
  }

  class(res) <- c("CA", "list")
  if (graph & (ncp>1)) {
    print(plot(res,axes=axes))
    if (!is.null(quanti.sup)) print(plot(res, choix="quanti.sup",axes=axes,new.plot=TRUE))
  }
  return(res)
}

elections <- read.csv('~/projects/prince/tests/french_elections.csv', row.names=1)
ca = CA(elections, graph=FALSE)
