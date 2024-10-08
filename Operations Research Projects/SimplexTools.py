from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame

import sympy
import functools

def add_method(obj, func):
    'Bind a function and store it in an object'
    setattr(obj, func.__name__, functools.partial(func, obj))
    getattr(obj, func.__name__).__doc__ = func.__doc__
    return functools.partial(func, obj)

def param(label = "Parameter"):
    """parameter for use in linearExpr"""

    clabel = label
    cnt = 0
    while clabel in LinearExpression.varnames:
        cnt += 1
        clabel = label + str(cnt)
    LinearExpression.varnames.append(clabel)
    symbl = sympy.symbols(clabel)
    globals()[clabel] = symbl
    return symbl

def var(label = "DecisionVariable"):
    clabel = label
    cnt = 0
    while clabel in LinearExpression.varnames:
        cnt += 1
        clabel = label + str(cnt)
    LinearExpression.varnames.append(clabel)
    LinExpr = LinearExpression( clabel, 1 )
    globals()[clabel] = LinExpr
    return LinExpr

def Vars( labels, replace = True ):
    """Deletes current variables and replaces 
    them with those supplied by labels
    if replace is True
       """
    if( isinstance(labels,str)):
        list_of_labels = labels.replace(","," ").split()
    else:
        list_of_labels = labels
    # Delete current variables
    if(replace):
        LinearExpression.varnames = [ "One" ] 
        Simplex._slacks = []
    VarList = []
    for lbl in list_of_labels:
        VarList.append( var(lbl) )
    return VarList

def Params( labels ):
    """Convenience tool for creating variables via
       var from a list of variables

       """
    if( isinstance(labels,str)):
        list_of_labels = labels.replace(","," ").split()
    else:
        list_of_labels = labels
    VarList = []
    for lbl in list_of_labels:
        VarList.append( param(lbl) )
    return VarList

class LinearExpression(dict):
    """ An expression of the form
        param_1 * variable_1 + ... + param_n * variable_n

        where
            param_i = parameter as either numeric or sympy expression
            variable_i = variable created by var
        """
    varnames = [ "One" ] # One is for constants

    def __init__(self, variable, coefficient = 1):
        if( isinstance(variable, dict) ):
            dict.__init__(self, variable )
        else:
            assert variable in LinearExpression.varnames, "Not a valid Variable Name"
            dict.__init__(self)
            self[variable] = coefficient
        # As relation, set equal to zero
        self.lhs = self
        self.rhs =  0
        self.reType = " == "

    def __repr__(self):
        label = ""
        for key in self.keys():
            val = self[key]
            if(key == "One"):
                key = ""
            elif( abs(val) != 1 ):
                key = "*{0}".format(key)
            if( val != 0 ):
                if( isinstance(val, (int, float)) and val < 0 ):
                    if( val == -1 and key != "" ):
                        if( label == ""):  
                            label += "  -{0}".format(key)
                        else:
                            label += " - {0}".format( key )
                    elif( label == "" ):
                        if( val == -1 ):
                            label += "  -{0}{1}".format(val, key)
                        else:
                            label += " - {0}{1}".format(val, key)
                    else:
                        label += " - {0}{1}".format(abs(val), key)
                else:
                    if( val == 1 and key != ""):
                        label += " + {0}".format( key )
                    else:
                        label += " + {0}{1}".format(val, key)
        if( label == "" ): label = "  0"
        return label[2:]

    def __add__(self, other):
        NewExpr = LinearExpression( self )
        if( isinstance(other,LinearExpression)):
            for key in other.keys():
                if( key in NewExpr.keys() ):
                    NewExpr[key] += other[key]
                else:
                    NewExpr[key] = other[key]
        else: # Other is a constant
            if( "One" in NewExpr.keys() ):
                NewExpr["One"] += other
            else:
                NewExpr["One"] = other
        return NewExpr

    def __radd__(self,other):
        return self.__add__(other)

    def __sub__(self,other):
        NewExpr = LinearExpression( self )
        if( isinstance(other,LinearExpression)):
            for key in other.keys():
                if( key in NewExpr.keys() ):
                    NewExpr[key] -= other[key]
                else:
                    NewExpr[key] = -1*other[key]
        else: # Other is a constant
            if( "One" in NewExpr.keys() ):
                NewExpr["One"] -= other
            else:
                NewExpr["One"] = -1*other
        return NewExpr

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        assert not isinstance(other,LinearExpression), "Nonlinear expressions are not supported"
        NewExpr = LinearExpression( self )
        for key in NewExpr.keys():
            NewExpr[key] *= other
        return NewExpr

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)
    
    def __truediv__(self, other):
        return self.__mul__(1/other)

    def __rtruediv__(self, other):
        raise NotImplementedError("Not Implemented")
        return self.__div__(other)

    def __lt__(self,other):
        if( not isinstance(other, LinearExpression) ):
            other = LinearExpression("One", other)
        return LinearRelation( self, other, " < ")

    def __le__(self,other):
        if( not isinstance(other, LinearExpression) ):
            other = LinearExpression("One", other)
        return LinearRelation( self, other, " <= ")

    def __gt__(self,other):
        if( not isinstance(other, LinearExpression) ):
            other = LinearExpression("One", other)
        return LinearRelation( self, other, " > ")

    def __ge__(self,other):
        if( not isinstance(other, LinearExpression) ):
            other = LinearExpression("One", other)
        return LinearRelation( self, other, " >= ")

    def __eq__(self,other):
        if( not isinstance(other, LinearExpression) ):
            other = LinearExpression("One", other)
        return LinearRelation( self, other, " == ")

class LinearRelation(object):
    """ Create simple equation or inequality structure for loading
        into a Linear Program Library """
    def __init__(self, lhs, rhs, reType ):
        self.lhs = lhs
        self.rhs = rhs
        self.reType = reType

    def __repr__(self):
        return self.lhs.__repr__() + self.reType + self.rhs.__repr__()

def GaussPivot(Tb, pivotrow, pivotcol, inplace = True):
    """pivotrow is either int or index name in Tb
       pivotcol is either int or columns name in Tb

       Executes a Gaussian pivot of dataframe at the given
       (pivotrow,pivotcol) location

       Options
       =======

       inplace = if applied in place (default = True)

       """
    if( isinstance(pivotrow,int) ): pivotrow = Tb.index[pivotrow]
    if( isinstance(pivotcol,int) ): pivotcol = Tb.columns[pivotcol]
    assert Tb.loc[pivotrow,pivotcol] != 0, "Cannot pivot on a zero entry"
    ## Ignoring inplace for now
    #if( not inplace):
    #    Tb = Tb.copy(deep = True)
    #    add_method(Tb, GaussPivot)
    Tb.loc[pivotrow] = Tb.loc[pivotrow] / Tb.loc[pivotrow,pivotcol]
    for ind in Tb.index:
        if(ind == pivotrow): continue
        Tb.loc[ind] -= Tb.loc[ind,pivotcol]*Tb.loc[pivotrow]

def Tableau( LP ):
    """ LP is a dictionary in which the keys are constraint or objective names,
        and each value is either LinearRelation (constraint) or LinearExpression (objective)

        A simplex tableau is returned as a pandas DataFrame with an operation

    Examples
    ========

    """
    # Get the Column Headings
    AllKeys = [ "One" ]
    AtEnd = [ "b" ]
    for lbl in LP:
        ConOrObj = LP[lbl]
        assert ConOrObj.reType == " == ", "Only linear equations are currently implemented"
        if( isinstance(ConOrObj, LinearExpression) ):
            AllKeys.extend( [ key for key in ConOrObj.keys() if key not in AllKeys ] )
            AtEnd.append(lbl)
        else:
            AllKeys.extend( [ key for key in ConOrObj.lhs.keys() if key not in AllKeys ]  )
            AllKeys.extend( [ key for key in ConOrObj.rhs.keys() if key not in AllKeys ] )
    AllKeys.extend(AtEnd)
    Lbls = list( LP.keys() )
    for lbl in AtEnd[1:]:
        Lbls.remove(lbl)
    Lbls.extend(["Objective%s"%(i+1) for i in range(len(AtEnd[1:]))])
    nr, nc = len(LP.keys()), len(AllKeys[1:])
    Tb = DataFrame( np.zeros( (nr,nc) ), columns = AllKeys[1:], index = Lbls  )

    # Construct the Rows ( = constraints )
    for lbl in LP:
        ConOrObj = LP[lbl]
        if( isinstance(ConOrObj, LinearExpression) ):
            objName = 'Objective{0}'.format( AtEnd.index(lbl) )
            for key in ConOrObj.keys():
                if(key == 'One'):
                    Tb.loc[objName,'b'] -= ConOrObj[key]
                else:
                    Tb.loc[objName,key] = ConOrObj[key]
            Tb.loc[objName,lbl] = 1
        else:
            for key in ConOrObj.lhs.keys():
                if(key == 'One'):
                    Tb.loc[lbl,'b'] -= ConOrObj.lhs[key]
                else:
                    Tb.loc[lbl,key] += ConOrObj.lhs[key]
            for key in ConOrObj.rhs.keys():
                if(key == 'One'):
                    Tb.loc[lbl,'b'] += ConOrObj.rhs[key]
                else:
                    Tb.loc[lbl,key] -= ConOrObj.rhs[key]

    # Add pivot method
    add_method(Tb,GaussPivot)

    return Tb

from IPython.display import display, HTML

def BasicSolution(Tb, zname = 'z', show = True, msg = '', suppress = 1e-9 ):
    """BasicSolution(Tableau) ->  Basic Solution for that Tableau
    
    options
    =======
    zname = string, name of objective variable (default 'z')
    show = boolean, if True, then print basic solution (default = True)
                    if False, then return basic solution as tuple of tuples
    msg = string, text that precedes printing of solution (default = '') 
    suppress = float, numbers with absolute value below this value
                      are printed as zeros (default = 1e-9)
    """                      
    if( suppress ):
        if( suppress <= 0 ): 
            suppress = False 
    Vw = Tb.drop('Objective1')
    b = Vw['b']
    Vw.drop(['b',zname], axis = 1, inplace = True)
    SolTuples = []
    basis = []
    for var in Vw.columns:
        col = Vw[var]
        pivind = -1
        for i in range(len(col)):
            if( col[i] != 0):
                if( pivind >= 0 ):
                    SolTuples.append( [var, 0, pivind] )
                    break
                else:
                    pivind = i
        else:
            value = b[pivind]/col[pivind]
            if( suppress ):
                if( np.abs(value) < suppress ): 
                    value = 0
            SolTuples.insert(0,[var, value, pivind])
            if( value >= 0):
                basis.append(pivind)
    ## Clean up SolTuples
    for i in range(len(SolTuples)):
        var, val, ind = SolTuples[i]
        if( val >= 0 ):
            SolTuples[i] = (var,val)
        elif( ind in basis):
            SolTuples[i] = (var,0)
        else:
            SolTuples[i] = (var,val)
    if( show ):
        Result = msg + '$'
        Ib = ''
        Ob = ''
        If = ''
        for tup in SolTuples:
            if( tup[1] == 0 ):
                Ob += '{0} = {1:.5g}, \;'.format(*tup)
            elif( tup[1] < 0 ):
                If += '{0} = {1:.5g}, \;'.format(*tup)
            else:
                Ib = '{0} = {1:.5g}, \;'.format(*tup) + Ib
        Result += Ib + "\quad " + Ob + "$"
        if( If != ''):
            Result += "<span style = 'color:red'> $\quad "+ If + "$</span>"
        return HTML(Result)
    else:
        return SolTuples

def OrderBasis(Tb):
    basis = [0]*len(Tb)
    for var in Tb.columns:
        col = Tb[var]
        pivind = -1
        for i in range(len(col)):
            if( col[i] != 0):
                if( pivind >= 0 ):
                    break
                else:
                    pivind = i
        else:
            basis[pivind] = var
    ## basis contains B in correct order
    ReOrdered = basis[:-1]  # not the objective row -- left where it is
    OtherVars = [ var for var in Tb.columns if var not in ReOrdered ]
    ReOrdered.extend(OtherVars)
    return Tb[ ReOrdered ]

class Simplex(object):
    """ Construct and Solve a linear program using the Simplex method"""
    
    _slacks = []
    
    def __init__(self, LP, slacknames = 's' ):
        """Currently only for maximization.  Creates Tableau if LP is a linear program dictionary, and 
        stores LP if it is a DataFrame. If an LP, insures that Tableau is in correct form for Simplex
        Iteration.  
        
        options
        =======
        slacknames = prefix for slacks/surpluses added for inequality constraints (default 's')
        
        
        methods
        =======
        choosepivot = selects pivot row and column for next Simplex iteration (only for maximization) 
        pivot  = chooses pivot row and columns and executes a pivot
        solve = Simplex Iteration until a solution (only maximization is implemented)
        
        """
        if( isinstance(LP,DataFrame) ):
            self.Tableau = LP  # You're on your own
        else:
            self.LP = dict()
            cnt = 0

            for lbl in LP:
                ConOrObj = LP[lbl]
                if( isinstance(ConOrObj, LinearExpression)):
                    self.zname = lbl # only one objective
                    self.LP[lbl] = -1*ConOrObj
                elif( ConOrObj.reType == " == " ):
                    self.LP[lbl] = ConOrObj
                elif( ConOrObj.reType in [ " <= ", " >= "] ):
                    cnt += 1
                    slack = "{0}_{1}".format(slacknames, cnt)
                    if( slack in LinearExpression.varnames):
                        LinearExpression.varnames.remove(slack)
                    if( ConOrObj.reType == " <= " ):
                        ConOrObj.lhs += var( slack )
                    else:
                        ConOrObj.lhs -= var( slack )
                    self.LP[lbl] = LinearRelation(ConOrObj.lhs, ConOrObj.rhs, " == ")
                else:
                    raise NotImplementedError("Relations of type %s are not implemented." % ConOrObj.reType )
            self.Tableau = Tableau(self.LP)
            for indx in self.Tableau.index:
                if( self.Tableau.loc[indx,'b'] < 0 ):  # Implement: What if a parameter
                    self.Tableau.loc[indx] *= -1.0     
 

    def choosepivot(self):
        """Returns pivot row and col, with row = -1 if no further pivoting is possible"""
        col = np.argmin( self.Tableau.loc['Objective1'].values )
        row = -1
        if( self.Tableau.loc['Objective1', col] < 0 ):
            ## pivot is possible
            colmin = 1e50
            for indx in self.Tableau.index: 
                if( self.Tableau.loc[indx, col] > 0 ): 
                    ratio = self.Tableau.loc[indx,'b'] / self.Tableau.loc[indx,col]
                    if(0 <= ratio < colmin ):
                        colmin = ratio
                        row = indx
        return (row,col)
    
    def pivot(self, step = -1):
        row,col = self.choosepivot()
        if( row !=  -1 ):
            GaussPivot(self.Tableau,row,col)
                
    def solve(self, ShowIntermediates = False, BasisOrder = True, minimize = False, MaxIters = 1000 ):
        """ iterates via simplex method until termination condition occurs -- solution, unboundedness, 
        or MaxIters exceeded.  
        
        options
        =======
        ShowIntermediates = displays basic solution for each intermediate tableau (default = False)
        BasisOrder = orders the final tableau into [I,B^{-1}D] order (default = True)
        minimize = currently raises NotImplementedError
        MaxIters = number of iterations before the simplex method stops (prevents infinite loops 
                   due to cycling, e.g.) ( default = 1000)
        """
        raise NotImplementedError('solve method not provided til later in the course')
    
from IPython.display import Markdown

display(Markdown( """The following commands are now available

* __add_method__ = adds a method to an instance of any class
* __var__ = creates a variable for creating Tableaus
* __param__ = creates a parameter for creating Tableaus
* __Vars__ = creates var objects from a list or string
* __Params__ = creates param objects from a list or structure
* __LinearExpression__ = class for expressions for linear programs
* __LinearRelation__ = class for equations and inequalities
* __GaussPivot__ = Gaussian Row Operations pivot for a pandas DataFrame
* __Tableau__ = Maps linear program as dictionary to DataFrame with GaussPivot

Also, the following helper methods are available:
* __BasicSolution__ = prints basic solution for a given tableau
* __OrderBasis__ = Arranges the simplex tableau into [I,M] form, where I is the identity matrix and M is what is left over
* __Simplex__ = Create simplex Tableau for a linear program, adding slack or surplus variables if necessary.  The solve method only implements maximization. 

NOTE: __GaussPivot__ can be applied directly to any DataFrame
"""))
