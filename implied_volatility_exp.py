#!/usr/bin/env python

"""
Example for Implied Volatility using nag4py
Finds a zero of the Black Scholes function using c05ayc and s30aac
Data needs to be downloaded from:
http://www.cboe.com/delayedquote/QuoteTableDownload.aspx
"""

import os, sys
import datetime, re
import pandas
import numpy
import matplotlib.pylab as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from ctypes import POINTER, c_double, c_long, c_int, cast, py_object
from nag4py.a00 import a00acc
from nag4py.s import s30aac
from nag4py.c05 import c05ayc, NAG_C05AYC_FUN
from nag4py.e02 import e02cac, e02cbc
from nag4py.x02 import x02bbc
from nag4py.util import NagError, Nag_Comm, Nag_RowMajor, Nag_Call, Nag_Put, Pointer

__author__ = "John Morrissey and Brian Spector"
__copyright__ = "Copyright 2013, The Numerical Algorithms Group Inc"
__email__ = "support@nag.co.uk"


def callback(x, comm):
    """
    Callback that calculates the Black Scholes Option Price for a given Volatility
    """

    fail = NagError()

    p_userdata = cast(comm[0].p, py_object)
    userdata = p_userdata.value

    time = c_double(userdata[0])
    callput = userdata[1]
    strike = c_double(userdata[2])
    underlying = userdata[3]
    current_price = userdata[4]
    out = c_double(0.0)
  
    # NAG function call 
    s30aac(Nag_RowMajor, callput, 1, 1, strike, underlying, time, x, 0.0, 0.0, out, fail)
    if(fail.code == 0):
        return out.value - current_price
    print fail.message
    return 0.0

def get_nag_int_type():
    naglib_maxint = x02bbc()
    for int_try in [numpy.int32,numpy.int64]:
    	if(naglib_maxint == numpy.iinfo(int_try).max):
    		return int_try

def get_ctype():
    if(get_nag_int_type() == numpy.int32):
    	return c_int
    else:
    	return c_long

def calcvol(exp, strike, todays_date, underlying, current_price, callput):
    """
    Root-finding method that calls NAG Library to calculate Implied Volatility
    """

    fail = NagError()

    volatility = c_double(.5)
    time = exp / 365.0
    userdate = time, callput, strike, underlying, current_price
    comm = Nag_Comm()
    comm.p = cast(id(userdate), Pointer)
    out=c_double(0.0)
    pyfun = NAG_C05AYC_FUN(callback)

    # NAG function call
    c05ayc(0.00000001, 1.0, .00001, 0.0, pyfun, volatility, comm, fail)
    if (fail.code == 0):
        return volatility.value
    else:
        return 0.0

# Set to hold expiration dates
dates = []

cumulative_month = {'Jan': 31, 'Feb': 57, 'Mar': 90,
                    'Apr': 120, 'May': 151, 'Jun': 181,
                    'Jul': 212, 'Aug': 243, 'Sep': 273,
                    'Oct': 304, 'Nov': 334, 'Dec': 365}

cumulative_month_2 = {1 : 0, 2 : 31, 3 : 57,
                    4 : 90, 5 : 120, 6 : 151,
                    7 : 181, 8 : 212, 9 : 243,
                    10 : 273, 11 : 304, 12 : 334}

call_months = "ABCDEFGHIJKL"
put_months = "MNOPQRSTUVWX"
symbol = re.compile("\\([A-Z0-9]{1,5}(1[0-9])([0-9]{2})([A-Z])([0-9]{1,4})")
now = datetime.datetime.now()
year = now.year - 2000
doy = now.timetuple().tm_yday

def getexpiration(x):
    tokens = symbol.split(x)
    try:
        month = call_months.index(tokens[3]) + 1
    except ValueError:
        month = put_months.index(tokens[3]) + 1
    exp = (int(tokens[1]) - year) * 365 + cumulative_month_2[month] - doy + int(tokens[2])
    if exp not in dates:
        dates.append(exp)
    return exp

def getstrike(x):
    monthday = x.split()
    return float(monthday[2])


def main(QuoteData):

    if(a00acc() != 1):
	print "Cannot find a valid NAG license"
        return
    try:
        qd = open(QuoteData, 'r')
        qd_head = []
        qd_head.append(qd.readline())
        qd_head.append(qd.readline())
        qd.close()
    except:
        sys.stderr.write("Couldn't read %s" % QuoteData)

    print "Implied Volatility for %s %s" % (qd_head[0].strip(), qd_head[1])

    # Parse the header information in QuotaData
    first = qd_head[0].split(',')
    second = qd_head[1].split()
    qd_date = qd_head[1].split(',')[0]
    
    company = first[0]
    underlyingprice = float(first[1])
    month, day = second[:2]
    today = cumulative_month[month] + int(day) - 30

    data = pandas.io.parsers.read_csv(QuoteData, sep=',', header=2, na_values=' ')

    # Need to fill the NA values in dataframe
    data = data.fillna(0.0)

    # Let's look at data where there was a recent sale 
    data = data[data.Calls > 0]
    data = data[(data['Last Sale'] > 0) | (data['Last Sale.1'] > 0)]

    # Get the Options Expiration Date
    exp = data.Calls.apply(getexpiration)
    exp.name = 'Expiration'

    # Get the Strike Prices
    strike = data.Calls.apply(getstrike)
    strike.name = 'Strike'

    data = data.join(exp)
    data = data.join(strike)

    print 'Calculating Implied Vol of Calls...'
    impvolcall = pandas.Series(pandas.np.zeros(len(data.index)),
                               index=data.index, name='impvolCall')
    for i in data.index:
        impvolcall[i] = (calcvol(data.Expiration[i],
                                 data.Strike[i],
                                 today,
                                 underlyingprice,
                                 (data.Bid[i] + data.Ask[i]) / 2, Nag_Call))

    print 'Calculated Implied Vol for %d Calls' % len(data.index)
    data = data.join(impvolcall)

    print 'Calculating Implied Vol of Puts...'
    impvolput = pandas.Series(numpy.zeros(len(data.index)),
                              index=data.index, name='impvolPut')

    for i in data.index:
        impvolput[i] = (calcvol(data.Expiration[i],
                                data.Strike[i],
                                today,
                                underlyingprice,
                                (data['Bid.1'][i] + data['Ask.1'][i]) / 2.0, Nag_Put))

    print 'Calculated Implied Vol for %i Puts' % len(data.index)

    data = data.join(impvolput)
    fig = plt.figure(1)
    fig.subplots_adjust(hspace=.4, wspace=.3)

    # Plot the Volatility Curves
    # Encode graph layout: 3 rows, 3 columns, 1 is first graph.
    num = 331
    max_xticks = 4
    
    for date in dates:
        # add each subplot to the figure
        plot_call = data[(data.impvolCall > .01) &
                       (data.impvolCall < 1) &
                       (data.Expiration == date) &
                       (data['Last Sale'] > 0)]
        plot_put = data[(data.impvolPut > .01) &
                        (data.impvolPut < 1) &
                        (data.Expiration == date) &
                        (data['Last Sale.1'] > 0)]

        myfig = fig.add_subplot(num)
        xloc = plt.MaxNLocator(max_xticks)
        myfig.xaxis.set_major_locator(xloc)
        myfig.set_title('Expiry: %d' % date)
        myfig.plot(plot_call.Strike, plot_call.impvolCall, 'pr', label='call')
        myfig.plot(plot_put.Strike, plot_put.impvolPut, 'p', label='put')
        myfig.legend(loc=1, numpoints=1, prop={'size': 10})
        myfig.set_ylim([0,1])
        myfig.set_xlabel('Strike Price')
        myfig.set_ylabel('Implied Volatility')
        num += 1

    plt.suptitle('Implied Volatility for %s Current Price: %s Date: %s' %
                 (company, underlyingprice, qd_date))
    

    print "\nPlotting Volatility Curves/Surface"  
    """
    The code below will plot the Volatility Surface
    It uses e02ca to fit with a polynomial and e02cb to evalute at 
    intermediate points
    """ 

    m = numpy.empty(len(dates), dtype=get_nag_int_type())
    y = numpy.empty(len(dates), dtype=numpy.double)
    xmin = numpy.empty(len(dates), dtype=numpy.double)    
    xmax = numpy.empty(len(dates), dtype=numpy.double)
 
    data = data.sort('Strike') # Need to sort for NAG Algorithm

    k = 3   # this is the degree of polynomial for x-axis (Strike Price)
    l = 3   # this is the degree of polynomial for y-axis (Expiration Date)

    i = 0

    for date in dates:
	call_data = data[(data.Expiration == date) & 
				(data.impvolPut > .01) & 
				(data.impvolPut < 1) &
                                (data['Last Sale.1'] > 0)]
         
	exp_sizes = call_data.Expiration.size
        if(exp_sizes > 0):       
        	m[i] = exp_sizes
        	n = len(dates)

        	if(i == 0):
            		x = call_data.Strike
            		call = call_data.impvolPut
            		xmin[0] = x.min()
            		xmax[0] = x.max()
        	else:
            		x2 = call_data.Strike
            		x = x.append(x2)
            		call2 = call_data.impvolPut
            		call = call.append(call2)
            		xmin[i] = x2.min()
            		xmax[i] = x2.max()
        	y[i] = date
		i+=1 

    nux = numpy.zeros(1,dtype=numpy.double)
    nuy = numpy.zeros(1,dtype=numpy.double)
    inux = 1
    inuy = 1
    
    if(len(dates) != i):
	print "Error with data: the CBOE may not be open for trading or one expiration date has null data"
	return 0
    weight = numpy.ones(call.size, dtype=numpy.double)
    
    output_coef = (c_double * ((k + 1) * (l + 1)))(0.0)

    # To input data into NAG function we convert variables to ctypes
    
    mx = m.ctypes.data_as(POINTER(get_ctype()))
    xx = x.ctypes.data_as(POINTER(c_double))
    yx = y.ctypes.data_as(POINTER(c_double))
    callx = call.ctypes.data_as(POINTER(c_double))
    weightx = weight.ctypes.data_as(POINTER(c_double))
    xminx = xmin.ctypes.data_as(POINTER(c_double))
    xmaxx = xmax.ctypes.data_as(POINTER(c_double))
    nuxx = nux.ctypes.data_as(POINTER(c_double))
    nuyx = nuy.ctypes.data_as(POINTER(c_double))
 
    fail = NagError()    
    
    #Call the NAG Chebyshev fitting function
    e02cac(mx,n,k,l,xx,yx,callx,weightx,output_coef,xminx,xmaxx,nuxx,inux,nuyx,inuy,fail)        
   
    if(fail.code != 0):
        print fail.message
	return 0

    """
    Now that we have fit the function,
    we use e02cb to evaluate at different strikes/expirations 
    """
    nStrikes = 100 # number of Strikes to evaluate    
    spacing = 20 # number of Expirations to evaluate
    
    for i in range(spacing):
        mfirst = 1	
        mlast = nStrikes
        xmin = data.Strike.min()
        xmax = data.Strike.max()
         
	x = numpy.linspace(xmin, xmax, nStrikes)

        ymin = data.Expiration.min() 
        ymax = data.Expiration.max() 
      
        y = (ymin) + i * numpy.floor((ymax - ymin) / spacing) 

        xx = x.ctypes.data_as(POINTER(c_double))
	fx=(c_double * nStrikes)(0.0) 
        fail=NagError()

        e02cbc(mfirst,mlast,k,l,xx,xmin,xmax,y,ymin,ymax,fx,output_coef,fail)
        
        if(fail.code != 0):
            print fail.message 
        
        if 'xaxis' in locals():
            xaxis = numpy.append(xaxis, x)
            temp = numpy.empty(len(x))
            temp.fill(y)
            yaxis = numpy.append(yaxis, temp)    
	    for j in range(len(x)):
		zaxis.append(fx[j])
        else:
            xaxis = x
            yaxis = numpy.empty(len(x), dtype=numpy.double)
            yaxis.fill(y)
            zaxis = []
	    for j in range(len(x)):
		zaxis.append(fx[j])
   
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(xaxis, yaxis, zaxis, cmap=cm.jet)
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Days to Expiration')
    ax.set_zlabel('Implied Volatility for Put Options')
    plt.suptitle('Implied Volatility Surface for %s Current Price: %s Date: %s' %
                 (company, underlyingprice, qd_date))
    
    plt.show()    

if __name__ == "__main__":
    if (len(sys.argv) < 2):
       print "Usage: implied_volatility.py <input file>"
       sys.exit()
    main(sys.argv[1])


