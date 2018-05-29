支持向量机算法


SMO算法能够通过随机两个点不断地对alphas和b进行调整，直到调整到最佳值
SMO算法的步骤如下：
* 步骤1：计算误差：
	fx = wx + b计算fx的值
	拉格朗日乘子法优化可以得出w的值为(1~n)∑i alphai*yi*xi
	i点误差：Ei  = fXi - labelMat[i]
	j点误差：Ej  = fXj - labelMat[j]

* 步骤2：计算上下界L和H：
	如果labelI != labelJ
		L = max(0, alphaJold - alphaIold)
        	H = min(C, C + alphaJold - alphaIold)
	如果labelI == labelJ
		L = max(0, alphaJold + alphaIold - C)
		H = min(C, alphaJold + alphaIold)

* 步骤3：计算η：
	η = 2 * xi * xj.T - xi * xi.T - xj * xj.T 

* 步骤4：更新αj： 
	alphaJnew = alphaJold - yj(Ei - Ej) / η
	
* 步骤5：根据取值范围修剪αj：
	if aj > H:
        	alphaJnewClipped = H
    	if aj < L:
        	alphaJnewClipped = L
	else:
		alphaJnewClipped = alphaJnew

* 步骤6：更新αi： 
	alphaInew = alphaIold + yi*yj*(alphaJold - alphaJnewClipped)

* 步骤7：更新b1和b2： 
	b1New = bOld - Ei - yj * (alphaInew - alphaIold) * xi.T * xi - yj * (alphaJnew - alphaJold) * xj.T *xi
	b2New = bOld - Ej - yi * (alphaInew - alphaIold) * xi.T * xj - yj * (alphaJnew - alphaJold) * xj.T *xj

* 步骤8：根据b1和b2更新b： 
	if (0 < alphas[i] and alphas[i] < C):
        	b = b1
        elif (0 < alphas[j] and alphas[j] < C):
                b = b2
        else:
                b = (b1 + b2) / 2.0

