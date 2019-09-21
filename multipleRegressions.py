import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def runRegression(y,x,data, cov_type='HC0'):
    print("Covariance type: %s" %cov_type )
    form = '{0} ~ {1}'.format(y,x)
    mod = smf.ols(formula=form, data=data)
    res = mod.fit(cov_type=cov_type)
    return(res)

def runQuantRegression(y,x,data,quant=.5):
    form = '{0} ~ {1}'.format(y,x)
    mod = smf.quantreg(formula=form, data=data)
    res = mod.fit(quant=.5)
    return(res)

def runRegressions(y, xList, key_vars, data, specification_names=None, cov_type='HC0', pvalues=True):
    """
    y - dependent variable (string)
    xList - list of explanatory variable strings (list of Patsy compatible formulas)
    key_vars - list of independent variables you wish to see results for
    df - a Pandas DataFrame that contains the data (list)
    specification_names - list of regression names (list of strings)
    cov_type - type of covariance you wish to estimate the regression with (default is robust)
    pvalues - whether you want to report parameter estimates with p-values or star notation (default p-values)
    
    Description: runRegressions estimates multiple regression specifications
    and returns coefficent and p-values estimates for key_vars and regressions
    stats for each specification in a single table. 
    """
    # if no custom names are given, use numbers
    if specification_names is None:
        specification_names = [str(i) for i in range(len(xList))]
    if len(specification_names) != len(xList):
        print('specification names not the same length as independent variables list')
        return(None)
        
    # run regressions    
    regDict = {}
    for x, name in zip(xList, specification_names):
        print("Estimating specification: %s" %name)
        regDict[name] = runRegression(y,x,data, cov_type='HC0')
    print("Estimation done")   
    
    # extract estimates:
    print("Extracting results")
    outDict = {}    
    for key in regDict.keys():
        # get regression statistics
        outDict[key] = {'nObs':regDict[key].nobs,'R^2 adj':regDict[key].rsquared_adj.round(3), 'cond. num':regDict[key].condition_number}
        
        # loop over variables of interest
        if pvalues:
            for var in key_vars:
                outDict[key][var] = "%.04f (%.03f)" %(regDict[key].params[var], regDict[key].pvalues[var])
        if pvalues == False:
            outDict[key] = regDict[key].params.round(3).astype(str) + pd.cut(regDict[key].pvalues,bins=[0,.01,.05,.1,1], labels=['***','**','*','']).astype(str)
            
    outdf =pd.DataFrame(outDict).transpose()
    outdf.index.name='specification'
    return(outdf)   
