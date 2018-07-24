import argparse
from astropy.io import ascii, fits
from astropy.modeling import models, fitting
from astropy.table import Table
from copy import deepcopy as dc
from express import convolve, extrema #, plot2d
import itertools
from lmfit import Parameters, Model
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import chebyshev 
import os.path
from scipy.special import erf
import sys

# Horizontal gaps between the chips
yl = 4616#9232
yu = 4680#9360
ygap = lambda y: np.s_[int(yl/y):int(yu/y)]

# Vertical gaps between the chips
xl = [9896, 8656, 7416, 6176, 4896, 3656, 2416, 1176, 0]
xu = [9920, 8744, 7504, 6264, 5024, 3744, 2504, 1264, 24]
xgap = lambda x: [np.s_[int(xl[0]/x):int(xu[0]/x)],
                  np.s_[int(xl[1]/x):int(xu[1]/x)],
                  np.s_[int(xl[2]/x):int(xu[2]/x)],
                  np.s_[int(xl[3]/x):int(xu[3]/x)],
                  np.s_[int(xl[4]/x):int(xu[4]/x)],
                  np.s_[int(xl[5]/x):int(xu[5]/x)],
                  np.s_[int(xl[6]/x):int(xu[6]/x)],
                  np.s_[int(xl[7]/x):int(xu[7]/x)],
                  np.s_[int(xl[8]/x):int(xu[8]/x)]]

    
class Frame(object):
    """ Class to define the Frame object """

    def __init__(self, file, trans=False, flip=False, axis=0):
        
        self.hdul = fits.open(file)
        self.hdr = self.hdul[0].header
        self.mode = self.hdr['ESO INS MODE']
        self.binx = self.hdr['ESO DET BINX']
        self.biny = self.hdr['ESO DET BINY']
        try:
            self.catg = self.hdr['ESO PRO CATG']
        except:
            self.catg = None
        for hdu in self.hdul[1:]:
            if trans:
                #hdu.data = np.flip(np.transpose(hdu.data), axis=0)
                hdu.data = np.transpose(hdu.data)
            if flip:
                hdu.data = np.flip(hdu.data, axis=axis)
        (self.ny, self.nx) = np.shape(self.hdul[1].data)
    
class Order(object):
    def __init__(self, s, f, ord_num, step_x=50):
        blue = f.hdul[1].data
        red = f.hdul[2].data
        self.min_red = 78
        self.min_blue = 117
        self.num = ord_num #int(np.floor(i/2))
        if self.num < self.min_blue:
            self.arm = red
            self.name = 'red'
            self.min_arm = self.min_red
        else:
            self.arm = blue
            self.name = 'blue'
            self.min_arm = self.min_blue
        #self.i_arm = i-self.min_arm*2#+(s.start_ord-39)*2
            #(nx, ny) = np.shape(prof_arm)
        (self.ny, self.nx) = np.shape(self.arm)
        self.medx = int(self.nx/2)-10

        # Do nothing if lazy and tracing already done
        self.trace = f.mode+str(f.binx)+str(f.biny)+'_trace_'\
                     +str(self.num)+'.dat'
        self.step_x = step_x//f.binx
        self.range_x = list(range(self.medx, 0, -self.step_x))\
                       +list(range(self.medx+1, self.nx, self.step_x))
        if os.path.isfile(self.trace) and s.lazy:
            self.lazy = True
        else:
            self.lazy = False
        #return (ord_num, trace_name, range_x)

class Profile(object):
    def __init__(self, s, f, func, n):
        self.func = func
        if self.func == gaussian:
            self.param_n = 3
        self.n = n
        self.create_model()
        self.ord_len = len(s.range_ord)
        self.nx = f.nx
        
        self.x_step = np.array([])
        self.ord_step = np.array([])
        self.val_step = None
        self.x_full = np.array([])#np.zeros(self.nx*self.ord_len)
        self.ord_full = np.array([])#np.zeros(self.nx*self.ord_len)
        self.val_full = np.zeros((self.nx*self.ord_len, self.param_n*self.n))

        for o in s.range_ord:
            self.x_full = np.append(self.x_full, range(self.nx))
            self.ord_full = np.append(self.ord_full, np.full(self.nx, o))

        self.plots = s.plots
            
    def clean_tab(self):
        if self.func == gaussian:
            where = [True]*len(self.tab)
            for i in range(0, self.n):
                where *= self.tab['f%i_amp' % i]\
                         < 0.2*np.max(self.tab['f%i_amp' % i])
                #print where
            self.tab.remove_rows(where)
        
    def compute(self, x, k):
        values = self.tab_trace[k]
        if len(values)-2 != self.param_n*self.n:
            raise Exception("%i parameters from table are not suitable for a "
                            "model requiring %ix%i=%i parameters!" \
                            % (len(values)-2, self.param_n, self.n,
                               self.param_n*self.n))
        params = Parameters()
        for i, p in enumerate(self.model.param_names):
            params.add(name=p, value=values[p])
        return self.model.eval(params, x=x) 

    #def create_tab(self):
    #    self.tab = Table(names=['ord','x']+self.model.param_names)
    #    self.tab_trace = Table(names=['ord','x']+self.model.param_names)
        
    def fit(self, y, x, hints):
        for i, p in enumerate(self.model.param_names):
            self.model.set_param_hint(p, value=hints[i])
        self.result = self.model.fit(y, x=x)
        if self.func == gaussian:
            ok = True
            for i, c in enumerate(self.cen):
                ok *= np.abs(self.result.params['f%i_cen'%i].value-c) < 2
        else:
            ok = False
        if ok:
            for i, c in enumerate(self.cen):
                self.cen[i] = self.result.params['f%i_cen'%i].value
            values = [self.result.params[p].value for p in self.result.params]
            self.x_step = np.append(self.x_step, self.x)
            self.ord_step = np.append(self.ord_step, self.ord)
            if self.val_step is None:
                self.val_step = np.array([values])
            else:
                self.val_step = np.append(self.val_step, [values], axis=0)
            #self.tab.add_row([self.ord, self.x]+values)
        else:
            values = None
        return ok, values

    def fit2d(self, deg=(6,3)):
        for i, p in enumerate(self.model.param_names):
            if self.ord_len == 1:
                cheb = models.Chebyshev1D(deg[0])
            else:
                cheb = models.Chebyshev2D(deg[0], deg[1])
            fit = fitting.LinearLSQFitter()
            chebfit = fit(cheb, self.x_step, self.ord_step, self.val_step[:, i])
            self.val_full[:, i] = chebfit(self.x_full, self.ord_full)
            #print chebfit.__dict__#fit_info
            #print chebfit.c0_0
#            print chebfit.coeffs
#            print chebfit._parameters
#            prova = chebfit.evaluate(self.x_full, self.ord_full, chebfit._parameters)
            if self.plots:
                plt.scatter(self.x_step, self.val_step[:, i], s=2)
                plt.scatter(self.x_full, self.val_full[:, i], s=2)
#                plt.scatter(self.x_full, prova, s=2)
                plt.show()
        
    def hints(self, cen):
        if np.size(cen) != self.n:
            raise Exception("%i centers are not suitable for %i slices!"
                            % (np.size(cen), self.n))
        if np.size(cen) == 1:
            cen = [cen]
        if self.func == gaussian:
            self.cen = cen
            amp = 10000.0
            wid = 5.0
            hints = []
            for c in cen:
                hints += [amp, c, wid]
        return hints
        
    def create_model(self):
        self.model = Model(self.func, prefix='f0_')
        for i in range(1, self.n):
            self.model += Model(self.func, prefix='f%i_' % i)
        self.model.nan_policy = 'omit'

    def trace(self):
        self.tab.sort('x')
        self.tab_trace = dc(self.tab)
        self.polyc = np.array([])
        self.deg = 10
        for p in self.model.param_names:
            polyc = np.polyfit(self.tab['x'], self.tab[p], self.deg)
            self.tab_trace[p] = np.poly1d(polyc)(self.tab_trace['x'])
            self.polyc = np.append(self.polyc, polyc)
        self.polyc.reshape((self.deg+1, self.param_n*self.n))
            
class Session(object):
    def __init__(self, **kwargs):
        try:
            self.profs = kwargs['order_profile'].split(", ", 1)
        except:
            self.profs = None
        try:
            self.biases = kwargs['master_bias'].split(", ", 1)
        except:
            self.biases = None
        try:
            self.waves = kwargs['wave_matrix'].split(", ", 1)
        except:
            self.waves = None
        try:
            self.scis = kwargs['science'].split(", ", 1)
        except:
            self.scis = None
        self.start_ord = kwargs['start_ord']
        self.end_ord = kwargs['end_ord']
        self.lazy = bool(kwargs['lazy'])
        self.plots = bool(kwargs['plots'])
        self.range_ord = range(self.start_ord, self.end_ord+1)

def execute(func, name, msg, lazy=True):
    if lazy:
        try:
            prod = open(name)
            print msg
        except:
            prod = func()
            save(prod, name)
    else:
        prod = func()
        save(prod, name)
    return prod

def extract(s):
    # Extract science frames

    for (bias, wave, sci) in zip(s.biases, s.waves, s.scis):
        bias_f = Frame(bias, trans=True)
        wave_f = Frame(wave, flip=True)
        sci_f = Frame(sci, trans=True)
        #sci_f = Frame(sci, flip=True)  # For ORDER_PROFILE
        plot2d(s, sci_f)
        plt.show()
        
        mbias_sub(s, sci_f, bias_f)  # For other than ORDER_PROFILE

        p = Profile(s, sci_f, gaussian, 2)
        print "Extracting orders %i-%i:" % (s.start_ord, s.end_ord)
        for ord_num in s.range_ord:
            o = Order(s, sci_f, ord_num)
            
            p.tab_trace = ascii.read(o.trace)
            
            filt = np.zeros(np.shape(o.arm))
            mask = np.zeros(np.shape(o.arm))
            y = np.array(range(o.ny))
            for x, j in enumerate(p.tab_trace['x']):
                prog(x, len(p.tab_trace), " Order %i (%s):" % (o.num, o.name))
                prof = p.compute(y, x)
                if s.plots and x == o.medx:
                #if True:
                    ax = plt.gca()
                    ax.plot(y, o.arm[:, int(j)])
                    ax.plot(y, prof)
                    plt.show()
                filt[:, int(j)] = prof
            mask[filt > 1e-4*np.max(filt)] = 1.0
            plot2d(s, data=o.arm*filt)
            plot2d(s, data=filt)
            #print ""

            # Boxcar extraction
            extr_b = np.sum(o.arm*mask, axis=0)

            # Optimal
            extr_o = np.sum(o.arm*filt, axis=0)/np.sum(filt*filt, axis=0)

            # Adjusting
            if sci_f.mode == 'SINGLEHR':
                wave = wave_f.hdul[1].data[o.num*2-o.min_red*2, :]
                #wave = wave_f.hdul[1].data[o.num*2-o.min_red*2, :]
                extr_b = extr_b[10:-11]
                extr_o = extr_o[10:-11]
                #extr_b = extr_b[42:-43]
                #extr_o = extr_o[42:-43]
            if sci_f.mode == 'MULTIMR':    
                wave = wave_f.hdul[1].data[o.num-o.min_red, :]
                extr_b = extr_b[48/sci_f.binx:-56/sci_f.binx]
                extr_o = extr_o[48/sci_f.binx:-56/sci_f.binx]
            #plt.plot(wave, extr_b, color='black')
            plt.plot(wave, extr_o)
            if s.plots:
                plt.show()
        plt.show()
            
def gaussian(x, amp, cen, wid, mode='norm'):
    if mode == 'norm':
        g = amp * np.exp(-(x-cen)**2 / wid)
    if mode == 'int':
        e = erf((x-cen)/np.sqrt(np.abs(wid)))
        g = np.append(np.append(0, amp * (e[2:]-e[:-2])), 0)
    return g

def gaussian_trace((x, y), ampc, cenc, widc):
    return np.poly1d(ampc)(x) \
        * np.exp(-(y-np.poly1d(cenc)(x))**2 / np.poly1d(widc)(x))

def mbias_sub(s, f, bias_f):
    print("Subtracting master bias...")

    (x, y) = np.shape(f.hdul[1].data)
    (bias_x, bias_y) = np.shape(bias_f.hdul[1].data)
    
    # Remove the horizontal gap between the chips
    if bias_y < y:
        blue = np.delete(f.hdul[1].data, ygap(f.biny), 1)
        red = np.delete(f.hdul[2].data, ygap(f.biny), 1)

    # Subtract the master biases (NB: x and y are swapped in the plot)
    f.hdul[1].data = blue - bias_f.hdul[1].data
    f.hdul[2].data = red - bias_f.hdul[2].data

    for g in xgap(f.binx):
        f.hdul[1].data = np.delete(f.hdul[1].data, g, 0)
        f.hdul[2].data = np.delete(f.hdul[2].data, g, 0)

    plot2d(s, f)#blue=s.blue_sub, red=s.red_sub)#, lt_blue_2, lt_red_2)
    plt.show()
        
def plot2d(s, f=None, data=None, lt_red=None, lt_blue=None):
    if s.plots:
        if f != None:
            blue = f.hdul[1].data
            red = f.hdul[2].data
            fig, ax = plt.subplots(2, 1)
            if lt_red == None and red is not None:
                lt_red = np.median(red)
            if lt_blue == None:
                lt_blue = np.median(blue)
            ax0 = ax[0].imshow(red, norm=col.SymLogNorm(lt_red),
                               cmap='gray')
            fig.colorbar(ax0, ax=ax[0])
            ax1 = ax[1].imshow(blue, norm=col.SymLogNorm(lt_blue),
                               cmap='gray')
            fig.colorbar(ax1, ax=ax[1])
        elif data is not None:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(data, cmap='gray')
            #fig.colorbar(ax, ax=ax)

        return ax

def cheb2d(x, y, c=None, deg=(2,2)):
    a = None
    p = np.zeros(np.shape(x))
    
    if c is None:
        c = np.ones((deg[0]+1, deg[1]+1))
    if len(np.shape(c)) == 1:
        c = np.resize(c, (deg[0]+1, deg[1]+1))
    ij = itertools.product(range(deg[0]+1), range(deg[1]+1))
    for (i, j) in ij:
        if a is None:
            a = np.array([x**i * y**j])
        else:
            a = np.append(a, [x**i * y**j], axis=0)
        p = p + c[i,j] * x**i * y**j
    a = a.T
    return p, a
    
def prog(part, tot, text, step=1):
    pc = (part+1)*100//tot
    if pc % step == 0:
        sys.stdout.write("\r"+text+" %3i%%" % pc)
        sys.stdout.flush()
        if pc == 100:
            print ""
    
def trace(s):
    # Trace orders on ORDER_PROFILE frames

    print "Tracing orders %i-%i:" % (s.start_ord, s.end_ord)
    
    for (bias, prof) in zip(s.biases, s.profs):
        bias_f = Frame(bias, trans=True)
        prof_f = Frame(prof, flip=True)
        #prof_f = Frame(prof, trans=True)  # For raw flats
        plot2d(s, prof_f)
        plt.show()

        #mbias_sub(s, prof_f, bias_f)  # For raw flats

        p = Profile(s, prof_f, gaussian, 2)
        for ord_i, ord_num in enumerate(s.range_ord):
            p.ord = ord_num
            p.ord_i = ord_i
            o = Order(s, prof_f, ord_num)
            #sys.stdout.write(" Order %i (%s)" % (o.num, o.name))
            #sys.stdout.flush()

            y = np.array(range(o.ny))

            if o.lazy == True:
                print " Order %i (%s): using existing trace" % (o.num, o.name)
                if s.plots == True:
                    p.tab_trace = ascii.read(o.trace)
            else:
                zmed = np.transpose(o.arm)[o.medx:o.medx+1, :][0]
                #zmed = o.arm[:, o.medx:o.medx+1][0]
                min_idx, max_idx, extr_idx = extrema(zmed)
                max_idx = max_idx[zmed[max_idx]>1000]
                skip_x = []
                #p.create_tab()
            
                for x_i, x in enumerate(o.range_x):
                    prog(x_i, len(o.range_x),
                         " Order %i (%s):" % (o.num, o.name))
                    p.x = x
                    zsel = np.transpose(o.arm)[x:x+1, :][0]
                    if x == o.medx or x == o.medx+1:
                        cen = [max_idx[2*(o.num-o.min_arm)],
                               max_idx[2*(o.num-o.min_arm)+1]]
                        hints = p.hints(cen)
                        ok = True
                    #if len(max_idx) > o.i_arm and ok:
                    if ok:
                        ok, values = p.fit(zsel, y, hints)
                        hints = values #p.tab[p.model.param_names][-1]
                        if s.plots and x == o.medx:
                            ax = plt.gca()
                            #ax.set_xlim(p.cen[0]-10, p.cen[1]+10)
                            ax.scatter(max_idx, zsel[max_idx])
                            ax.plot(y, zsel)
                            ax.plot(y, p.result.best_fit)
                            plt.show()
                #p.clean_tab()
                #p.trace()
                #print " traced between pixels %i and %i (of %i)"\
                #    % (skip_x[0], skip_x[1], o.nx)
                #ascii.write(p.tab, o.trace, overwrite=True)

                #ok2 = p2.fit(o.arm, (o.range_x, y))
                #print p.polyc, p.deg
                
            """
            if s.plots == True:
                ax = plt.gca()
                if p.tab is not None:
                    for c in p.tab.colnames[2:]:
                        ax.scatter(p.tab['x'], p.tab[c])
                for c in p.tab_trace.colnames[2:]:
                    ax.plot(p.tab_trace['x'], p.tab_trace[c], color='black')
                plt.show()
            """
        p.fit2d()
###
    
def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--order_profile', '-O', help="ORDER_PROFILE frames "
                        "(comma separated list)", type=str)
    parser.add_argument('--master_bias', '-B', help="MASTER_BIAS frames "
                        "(comma separated list)", type=str)
    parser.add_argument('--wave_matrix', '-W', help="WAVE_MATRIX frames "
                        "(comma separated list)", type=str)
    parser.add_argument('--science', '-S', help="Science frames "
                        "(comma separated list)", type=str)
    parser.add_argument('--start-ord', '-s', help="Starting order", type=int,
                        default=78)
    parser.add_argument('--end-ord', '-e', help="Ending order", type=int,
                        default=161)
    parser.add_argument('--lazy', '-l', help="Use existing products", type=int,
                        default=1)
    parser.add_argument('--plots', '-p', help="Show plots", type=int, default=0)
    
    args = vars(parser.parse_args())

    s = Session(**args)
    if s.biases != None and s.profs != None:        
        trace(s)
    if s.biases != None and s.waves != None and s.scis != None:
        extract(s)
        
    
if __name__ == '__main__':
    main()
    
