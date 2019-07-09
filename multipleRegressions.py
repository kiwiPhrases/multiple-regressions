def runRegression(y,x,data, cov_type='HC0'):
    form = '{0} ~ {1}'.format(y,x)
    mod = smf.ols(formula=form, data=data)
    res = mod.fit(cov_type=cov_type)
    return(res)

def runQuantRegression(y,x,data,quant=.5):
    form = '{0} ~ {1}'.format(y,x)
    mod = smf.quantreg(formula=form, data=data)
    res = mod.fit(quant=.5)
    return(res)

def runRegressions(y, xList, key_vars, data, specification_names=None):
    """
    y - dependent variable (string)
    xList - list of explanatory variable strings (list of Patsy compatible formulas)
    key_vars - list of independent variables you wish to see results for
    df - a Pandas DataFrame that contains the data (list)
    specification_names - list of regression names (list of strings)
    
    Description: runRegressions estimates multiple regression specifications
    and returns coefficent estimate for key_vars, p-value, and other stats for each
    specification in a single table. 
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
        regDict[name] = runRegression(y,x,data)
    print("Estimation done")   
    
    # extract estimates:
    print("Extracting results")
    outDict = {}    
    for key in regDict.keys():
        cont = regDict[key]
        inc = 'HINCP'
        mills = 'invMills'
        numObs = 'numObs'
        outDict[key] = {'nObs':cont.nobs,'R^2 adj':cont.rsquared_adj.round(3), 'cond. num':cont.condition_number}
        for var in key_vars:
            outDict[key][var] = "%.04f (%.03f)" %(cont.params[var], cont.pvalues[var])
        
    outdf =pd.DataFrame(outDict).transpose()
    outdf.index.name='specification'
    return(outdf)    