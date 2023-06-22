# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:41:04 2021

@author: paselkin
"""
## Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import scipy.interpolate as spi
import scipy.optimize as spo
import scipy.stats as sps
import altair as alt
from pathlib import Path
import re

## Functions

def read_hyst(filename="",
              path=""):
    """
    s,d = read_hyst(filename,path)
        Read hysteresis file and output PANDAS DataFrame
    """
    if (path != ''):
        path=path.replace("\\","/")
        if (path[-1]!="/"): path=path+"/"
        filename=path+filename
    with open(filename,"r") as hystfile: 
        # One does not simply read_csv a MicroMag file 
        hystlines = hystfile.readlines()
    hystfile.close()
    start_segments=[i for i,l in enumerate(hystlines) if re.search('Segment',l)][0]+3
    end_segments=[i for i,l in enumerate(hystlines) if bool(re.search('^\s$',l))&(i>start_segments)][0]
    segments=pd.DataFrame([l.rstrip().split(",") for l in hystlines[start_segments:end_segments]])
    segments.columns=['segment','averaging_time','initial_field','field_increment','final_field','pause','final_index']
    start_data=[i for i,l in enumerate(hystlines) if re.search('^\s+Field\s+Moment',l)][0]+2
    end_data=[i for i,l in enumerate(hystlines) if re.search('MicroMag 2900/3900 Data File ends',l)][0]
    data=pd.DataFrame([l.rstrip().split(",") for l in hystlines[start_data:end_data] if not(bool(re.search('^\s$',l)))])
    if (len(data.columns)==2):
        data.columns=['field','moment']
    else:
        data.columns=['field','moment','temperature']
    data['field']=data['field'].apply(float)
    data['moment']=data['moment'].apply(float)
    #data['temperature']=data['temperature'].apply(float)
    segments['final_index']=segments['final_index'].apply(int)
    return (segments,data)

def read_dcd(filename="",
              path=""):
    """
    s,d = read_dcd(filename,path)
        Read dc demag file and output PANDAS DataFrame
    """
    if (path != ''):
        path=path.replace("\\","/")
        if (path[-1]!="/"): path=path+"/"
        filename=path+filename
    with open(filename,"r") as dcdfile: 
        # One does not simply read_csv a MicroMag file 
        dcdlines = dcdfile.readlines()
    dcdfile.close()
    start_segments=[i for i,l in enumerate(dcdlines) if re.search('Segment',l)][0]+3
    end_segments=[i for i,l in enumerate(dcdlines) if bool(re.search('^\s$',l))&(i>start_segments)][0]
    segments=pd.DataFrame([l.rstrip().split(",") for l in dcdlines[start_segments:end_segments]])
    segments.columns=['segment','averaging_time','initial_field','field_increment','final_field','pause','final_index']
    start_data=[i for i,l in enumerate(dcdlines) if re.search('^\s+Field\s+Remanence',l)][0]+2
    end_data=[i for i,l in enumerate(dcdlines) if re.search('MicroMag 2900/3900 Data File ends',l)][0]
    data=pd.DataFrame([l.rstrip().split(",") for l in dcdlines[start_data:end_data] if not(bool(re.search('^\s$',l)))])
    if (len(data.columns)==2):
        data.columns=['field','remanence']
    else:
        data.columns=['field','remanence','temperature']
    data['field']=data['field'].apply(float)
    data['remanence']=data['remanence'].apply(float)
    data['file']=filename
    data['id']=data['file'].apply(lambda x: re.sub('-.*','',Path(x).stem))
    #data['temperature']=data['temperature'].apply(float)
    segments['final_index']=segments['final_index'].apply(int)
    segments['file']=filename
    segments['id']=segments['file'].apply(lambda x: re.sub('-.*','',Path(x).stem))
    segments['type']=segments.apply(lambda x: 'irm' if float(x['final_field'])>0. else 'dcd',axis=1)
    segments['start_index']=segments['final_index'].shift(1)+1
    segments.loc[0,'start_index']=0
    segments['end_index']=segments['final_index']
    
    segments.index = pd.IntervalIndex.from_arrays(segments['start_index'],segments['end_index'],closed='both')
    data=data.reset_index()
    data['type'] = data['index'].apply(lambda x : segments.iloc[segments.index.get_loc(x)]['type'])
    segments=segments.reset_index(drop=True)
    data=data.drop('index',axis=1)
    return (segments,data)

def process_dcd(segments,
                data,
                Hgrid=None,
                lam=0.5,
                smooth=None):
    """
    results = process_dcd(segments,
                                    data,
                                    Hgrid=None,
                                    lam=0.5)
        Smooth dc demag curve and extract useful parameters: Hcr, IRM_80mT
    """    
    dcd_curve=data.loc[(data['type']=='dcd')].copy()
    dcd_curve['field']=-1.*dcd_curve['field']
    dcd_curve=dcd_curve.sort_values(by='field').reset_index(drop=True)
    Hmax=np.max(np.abs(dcd_curve['field'].values))
    if (Hgrid is None):
        N=len(dcd_curve)
        Hgrid=make_hgrid2(N,Hmax)
    if (smooth is None):
        smooth = 10**(2.0*np.ceil(np.log10(0.1*np.min(np.abs(dcd_curve.remanence)))))
    dcd_sp=spi.UnivariateSpline(dcd_curve['field'].values,dcd_curve['remanence'].values,s=smooth,ext='extrapolate')
    dcd_interpolated=pd.DataFrame({'field':Hgrid,
                                     'remanence':dcd_sp(Hgrid)})
    #dcd_interpolated=dcd_interpolated.sort_values(by='field').reset_index(drop=True)
    #dcd_interpolated.drop_duplicates(subset='remanence',inplace=True)
    #dcd_interpolated.drop_duplicates(subset='field',inplace=True)
    try:
        Hcr=dcd_sp.roots()[0]
    except:
        Hcr=np.nan
    try:
        IRM_80mT=dcd_sp(-80)
    except:
        IRM_80mT=np.nan
    return {'Hcr':Hcr,
            'IRM_80mT':IRM_80mT,
            'dcd_curve':dcd_curve,
            'dcd_interpolated':dcd_interpolated}
             
def collapse_dcd(data):
    """
    results = collapse_dcd(data)
        Create single IRM and DCD curve from (potentially) multiple segments
    """    
    type_list = segments['type'].unique()
    segments_grouped = segments.groupby('type')
    data_grouped = data.groupby('type')
    new_data=pd.DataFrame([])
    for this_type in type_list:
        data_list = data_grouped.get_group(this_type)
        if (this_type=='irm'):
            data_list = data_list.sort_values('field',inplace=True,ascending=True)
        else:
            data_list = data_list.sort_values('field',inplace=True,ascending=False)
        new_data=pd.concat([new_data,data_list],axis=0)
    return (new_data)

def process_irm(segments,
                data,
                Hgrid=None,
                lam=0.5,
                smooth=None):
    """
    results = process_irm(segments,
                                    data,
                                    Hgrid=None,
                                    lam=0.5)
        Smooth irm curve and extract useful parameters: Hcr, IRM_80mT
    """    
    irm_curve=data.loc[(data['type']=='irm')].copy()
    irm_curve=irm_curve.sort_values(by='field').reset_index(drop=True)
    Hmax=np.max(np.abs(irm_curve['field'].values))
    if (Hgrid is None):
        N=len(irm_curve)
        Hgrid=make_hgrid2(N,Hmax)
    if (smooth is None):
        smooth = 10**(2.0*np.ceil(np.log10(0.1*np.min(np.abs(irm_curve.remanence)))))
    irm_sp=spi.UnivariateSpline(irm_curve['field'].values,irm_curve['remanence'].values,s=smooth,ext='extrapolate')
    irm_interpolated=pd.DataFrame({'field':Hgrid,
                                     'remanence':irm_sp(Hgrid)})
    irm_normalized=irm_interpolated.copy()
    irm_normalized['remanence']=irm_normalized['remanence']/irm_normalized['remanence'].max()
    irm_derivative=irm_normalized.copy()
    irm_derivative['gradient']=pd.Series(np.gradient(irm_derivative['remanence'],irm_derivative['field']))
    #irm_interpolated=irm_interpolated.sort_values(by='field').reset_index(drop=True)
    #irm_interpolated.drop_duplicates(subset='remanence',inplace=True)
    #irm_interpolated.drop_duplicates(subset='field',inplace=True)
    
    return {'irm_curve':irm_curve,
            'irm_interpolated':irm_interpolated,
            'irm_normalized':irm_normalized,
            'irm_derivative':irm_derivative}

def corr_loop(Hoff,upper,lower,return_params=False):
    # Function to minimize in process_hyst to find horizontal and vertical offset
    lower_t=-lower.copy(deep=True)
    lower_t['field']=lower_t['field']-(2*Hoff)
    lower_t=lower_t.sort_values(by='field')
    hu=upper['field']
    sp=spi.splrep(lower_t['field'],lower_t['moment'])
    ml=spi.splev(hu,sp)
    lrs=sps.stats.linregress(upper['moment'].values,ml)
    if return_params:
        return lrs
    else:
        return -(lrs[2]**2)    

def make_hgrid(N,Hmax,lam):
    i=[x for x in np.arange(-N+1,N) if x != 0]
    Hgrid=(np.abs(i)/i)*(Hmax/lam)*(((lam+1)**(np.abs(i)/N))-1)     # From Von Dobeneck 1996
    Hgrid=Hgrid[~np.isnan(Hgrid)]
    return Hgrid

def make_hgrid2(N,Hmax):
    i=np.arange(0,N)
    Hgrid=np.concatenate([np.array([0.]),np.geomspace(Hmax/N, Hmax, num=N)])
    return Hgrid

def process_hyst(segments,
                 data,
                 Hgrid=None,
                 lam=0.5,
                 offset_correct=True,
                 fraction_saturation=0.7,
                 smooth=None):
    """
    results = process_hyst(segments,
                                    data,
                                    lam=0.5,
                                    offsset_correct=True,
                                    fraction_saturation=0.7)
        Smooth hysteresis loop and extract useful parameters: Mrs, Ms, Ki, Khf, etc.
    """
    # Step 1 of approach: split loop into initial, upper, and lower branches
    initial_start=0
    initial_end=segments['final_index'][0]
    upper_start=segments['final_index'][0]
    upper_end=segments['final_index'][1]
    lower_start=segments['final_index'][1]
    lower_end=segments['final_index'][2]
    initial_loop=data.iloc[initial_start:initial_end,0:2]
    upper_loop=data.iloc[upper_start:upper_end,0:2]
    lower_loop=data.iloc[lower_start:lower_end,0:2]
    # Iteratively solve the following problem:
    #   For each Hoff from -Hmax to Hmax
    #   Invert lower loop through Hoff,0
    #   Spline interpolate at fields used in upper loop
    #   Calculate R^2 between lower and upper loops
    #   Maximize R^2 (note: using scipy.optimize.brent, so minimize -R^2)
    Hmax=np.max(np.abs(upper_loop['field'].values))
    try:
        Hoff=spo.brent(corr_loop,args=(upper_loop,lower_loop),brack=(-Hmax,0,Hmax))
    except:
        print('Minimization failed. Hoff=0')
        Hoff=0
    
    # Calculate error estimates R^2 and Q and vertical offset M0
    lrs=corr_loop(Hoff,upper_loop,lower_loop,True)
    r2=(lrs[2]**2)
    M0=lrs[1]/2.
    Q=np.log10(1/(1-r2))
    
    if ((offset_correct==True)|(offset_correct=='H')):
        # Correct loop for horizontal offset
        upper_loop['field']=upper_loop['field']-Hoff
        lower_loop['field']=lower_loop['field']-Hoff
    if ((offset_correct==True)|(offset_correct=='M')):
        # Correct loop for vertical offset
        upper_loop['moment']=upper_loop['moment']-M0
        lower_loop['moment']=lower_loop['moment']-M0
    
    # "Grid" and interpolate both half loops
    if (Hgrid is None):
        N=len(upper_loop)
        Hgrid=make_hgrid(N,Hmax,lam)
    
    upper_loop=upper_loop.sort_values(by='field').reset_index(drop=True)
    upper_loop.drop_duplicates(subset='moment',inplace=True)
    upper_loop.drop_duplicates(subset='field',inplace=True)
    try:
        upper_sp=spi.splrep(upper_loop['field'].values,upper_loop['moment'].values)
        upper_interpolated=pd.DataFrame({'field':Hgrid,
                                     'moment':spi.splev(Hgrid,upper_sp)})
    except TypeError:
        print('Degree too high. rying smoothing spline.\n')
        upper_sp=spi.UnivariateSpline(upper_loop['field'].values,upper_loop['moment'].values)
        upper_interpolated=pd.DataFrame({'field':Hgrid,
                                     'moment':upper_sp(Hgrid)})
    lower_loop=lower_loop.sort_values(by='field').reset_index(drop=True)
    lower_loop.drop_duplicates(subset='moment',inplace=True)
    lower_loop.drop_duplicates(subset='field',inplace=True)
    try:
        lower_sp=spi.splrep(lower_loop['field'].values,lower_loop['moment'].values)
        lower_interpolated=pd.DataFrame({'field':Hgrid,
                                     'moment':spi.splev(Hgrid,lower_sp)})
    except TypeError:
        print('Degree too high. trying smoothing spline.\n')
        lower_sp=spi.UnivariateSpline(lower_loop['field'].values,lower_loop['moment'].values)
        lower_interpolated=pd.DataFrame({'field':Hgrid,
                                     'moment':lower_sp(Hgrid)})
    lower_inverted=(-lower_interpolated).sort_values(by='field').reset_index(drop=True)
    
    # Calculate error estimate err(H)
    err=upper_interpolated['moment']-lower_inverted['moment']
    
    #calculate Mrh and Mih
    reversible=pd.DataFrame({'field':upper_interpolated['field'],
                'moment':(upper_interpolated['moment']-lower_interpolated['moment'])/2.})
    irreversible=pd.DataFrame({'field':upper_interpolated['field'],
                'moment':(upper_interpolated['moment']+lower_interpolated['moment'])/2.})
    
    # Calculate Mrsp (Mrs from reversible moment function), Ms, Xhf
    Mrsp = reversible['moment'].max()
    top_linear=irreversible.loc[irreversible['field']>=(fraction_saturation*Hmax),:]
    bottom_linear=irreversible.loc[irreversible['field']<=(-fraction_saturation*Hmax),:]
    bottom_inverted=(-bottom_linear).sort_values(by='field').reset_index(drop=True)
    all_linear=pd.concat([top_linear,bottom_inverted],axis=0)
    hfs=sps.stats.linregress(all_linear['field'].values,all_linear['moment'].values)
    Ms=hfs[1]
    Xhf=hfs[0]

    # Detrend loop and calculate new Qeff
    upper_detrended=upper_interpolated.copy(deep=True)
    upper_detrended['moment']= upper_detrended['moment']-(upper_detrended['field']*Xhf)
    lower_detrended=lower_interpolated.copy(deep=True)
    lower_detrended['moment']= lower_detrended['moment']-(lower_detrended['field']*Xhf)
    irreversible_detrended=irreversible.copy(deep=True)
    irreversible_detrended['moment']= irreversible_detrended['moment']-(irreversible_detrended['field']*Xhf)
    lrs=corr_loop(0,upper_detrended,lower_detrended,True)
    r2=(lrs[2]**2)
    Qeff=np.log10(1/(1-r2))
    
    if (smooth is None):
        smooth = 0.5*np.sqrt(np.mean(err**2.))  # Smooth within half of MSE
    
    # Calculate Hc, Hcp (Hc from reversible and irreversible loops),Mrs
    #lower_spline=spi.interp1d(lower_detrended['field'],lower_detrended['moment'],fill_value="extrapolate")
    lower_spline=spi.UnivariateSpline(lower_detrended['field'],lower_detrended['moment'],s=smooth)
    print(lower_spline.roots())
    try:
        Hc=lower_spline.roots()[0]
    except:
        Hc=0    
    if (Hc<=0.):
        print ("Warning: Hc<0. Trying upper branch.\n")
        upper_spline=spi.interp1d(upper_detrended['field'],upper_detrended['moment'],fill_value="extrapolate")
        Hc=-spo.fsolve(upper_spline,0.)[0]
        print ("Upper Hc: ",Hc)
    try:
        distance=np.abs(reversible['moment']-irreversible['moment'])
        distance_spline=spi.interp1d(irreversible['field'],distance)
        Hcp=spo.minimize(distance_spline,0,method='CG')['x']
    except ValueError:
        print('An error occurred in finding Hcp.')
        Hcp=None
    upper_search=upper_detrended.loc[abs(upper_detrended['field'])<Hc,:]
    upper_rspline=spi.interp1d(upper_detrended['moment'],upper_detrended['field'],fill_value="extrapolate")
    Mrsu=spo.fsolve(upper_rspline,0,)[0]
    lower_search=lower_detrended.loc[abs(lower_detrended['field'])<Hc,:]
    lower_rspline=spi.interp1d(lower_detrended['moment'],lower_detrended['field'],fill_value="extrapolate")
    Mrsl=spo.fsolve(lower_rspline,0)[0]
    Mrs=0.5*(Mrsu-Mrsl)
    
    return {'upper':upper_loop,
            'upper_interpolated':upper_interpolated,
            'upper_detrended':upper_detrended,
            'lower':lower_loop,
            'lower_interpolated':lower_interpolated,
            'lower_inverted': lower_inverted,
            'lower_detrended':lower_detrended,
            'reversible': reversible,
            'irreversible': irreversible,
            'irreversible_detrended':irreversible_detrended,
            'horizontal_offset':Hoff,
            'vertical_offset':M0,
            'Mrs':Mrs,
            'Ms':Ms,
            'Xhf':Xhf,
            'Mrsp':Mrsp,
            'Hc':Hc,
            'Hcp':Hcp,
            'Q':Q,
            'Qeff':Qeff,
            'err':err}

def plot_hyst(result,ax,plot_detrended=True,title=None,xlim=None,ylim=None):
    ax.plot(result['upper_interpolated']['field'],result['upper_interpolated']['moment'],'-',c='#666666')
    ax.plot(result['lower_interpolated']['field'],result['lower_interpolated']['moment'],'-',c='#666666')

    ax.plot(result['upper']['field'],result['upper']['moment'],'.',c='#000000')
    ax.plot(result['lower']['field'],result['lower']['moment'],'.',c='#000000')

    ax.plot(result['reversible']['field'],result['reversible']['moment'],'-',c='g',linewidth=1)
    ax.plot(result['irreversible']['field'],result['irreversible']['moment'],'-',c='r',linewidth=1)

    if (plot_detrended):
        ax.plot(result['upper_detrended']['field'],result['upper_detrended']['moment'],'-',c='k')
        ax.plot(result['lower_detrended']['field'],result['lower_detrended']['moment'],'-',c='k')

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    if (xlim is not None):
        ax.set_xlim(xlim)
    if (ylim is not None):
        ax.set_ylim(ylim)
    
    if (title is not None):
        bbox_props=dict(boxstyle='square',fc='w',ec='w')
        xb=ax.get_xlim()[0]
        yb=ax.get_ylim()[1]
        ax.text(xb,yb,title,ha='left',va='top',fontsize=12,bbox=bbox_props,fontweight='bold')
    return ax

def plot_err(result,ax,xlim=None,ylim=None):
    ax.plot(result['upper_interpolated']['field'],result['err'],'r-')
    if (xlim is not None):
        ax.set_xlim(xlim)
    if (ylim is not None):
        ax.set_ylim(ylim)
    return ax

def plot_hyst_report(result,fig,id):
    ax_hyst_main=plt.subplot2grid((3,2),(0,0),rowspan=2)
    ax_hyst_inset=plt.subplot2grid((3,2),(0,1),rowspan=2)
    ax_err=plt.subplot2grid((3,2),(2,0),rowspan=1)
    #ax_table=plt.subplot2grid((3,2),(2,1),rowspan=1)
    
    ax_hyst_main=plot_hyst(result,ax_hyst_main,title=id)
    ax_hyst_inset=plot_hyst(result,ax_hyst_inset,xlim=(-10.*result['Hc'],10.*result['Hc']),ylim=(-2.*result['Ms'],2.*result['Ms']))
    ax_err=plot_err(result,ax_err)
    
    #result_table=list(map(lambda x: "{}".format(x),[result['Mrs'],result['Ms'],result['Hc'],result['Xhf'],result['Q']]))
    #result_labels=('Mrs','Ms','Hc','Xhf','Q')
    #ax_table.xaxis.set_visible(False)
    #ax_table.yaxis.set_visible(False)
    #ax_table.table(cellText=result_table,loc='center')
    
    plt.tight_layout()
    
    return (ax_hyst_main,
            ax_hyst_inset,
            ax_err)