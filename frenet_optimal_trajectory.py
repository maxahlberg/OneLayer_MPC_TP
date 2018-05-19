#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame](https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import cubic_spline_planner
import cvxpy
from cvxpy import *

# Parameters
#iteration = 0
MAX_Psi = math.pi/2
MAX_KAPPA = 10
MAX_ACCEL = 10.0  # maximum acceleration [m/ss]
MAX_ROAD_WIDTH = 2.0  # maximum road width [m]
N = 10  #Horizon
target_speed = 11.0

dt = 0.1  # [s]

#Vehicle parameters
L = 3.0  # [m] wheel base of vehicle
lr = L*0.5 #[m]
lf = L*0.5 #[m]
Width = 2.0  # [m] Width of the vehicle

show_animation = True

class quinic_polynomial: #Skapar ett 5e grads polynom som beräknar position, velocity och acceleration

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T): # (position_xs, Velocity_xs, Acceleration_xs, P_xe, V_xe, A_xe, Time )

        # calc coefficient of quinic polynomial
        self.xs = xs 
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0 #Varför accelerationen delat med 2? -För att de skall bli rätt dimensioner i slutandan

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b) #Antagligen matris invers som löser a3,a4,a5. En form av jerk, jerk_derivata, jerk dubbelderivata

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]
    def calc_point(self, t): # point on xs at t
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t): #speed in point at t
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t): # acceleration in point at t
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):

        # calc coefficient of quinic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

class Frenet_path:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        self.v = []
        self.psi = []
        self.beta = []

        self.mpc_a = []
        self.mpc_kappa = []

        #self.xx = []
        #self.yy = []

class State:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        self.v = []
        self.psi = []
        self.beta = []


def get_nparray_from_matrix(x): #to getarrays from matrix
    return np.array(x).flatten()

def calc_MPC_frenet_paths(initial, estimated):


    tfp = Frenet_path()

    frenet_paths = []
    for init in initial:
        #y = init.y[-1]
        yaw = init.yaw[-1]+0.01 #The yaw of the track against a global x-axis, eta in my report
        #Kappa =
        #ds = init.ds
        c = init.c[-1] #The curvature of the track
        v = init.v[-1] #The velocity of the car
        psi = init.psi[-1] #The direction of the car

        d = init.d[-1] #The diviation from the center line
        #d_d = init.d_d[-1]
        #d_dd = init.d_dd[-1]

        #s = init.s[-1] #The new point is S derived from x and y - Maybe implement s = s_old + s_prick*dt
        #s_d = init.s_d[-1]
        #s_dd = init.s_dd[-1]

    for est in estimated:
        #x_hat = est.x[-1]
        #y_hat = est.y[-1]
        yaw_hat = est.yaw[1]+0.01
        Kappa_hat = est.mpc_kappa[1]
        #ds_hat = est.ds
        c_hat = est.c[1] + 0.001
        v_hat = est.v[1]
        psi_hat = est.psi[1]

        d_hat = est.d[1]
        #d_d_hat = est.d_d[-1]
        #d_dd_hat = est.d_dd[-1]

        #s_hat = est.s[-1]
        #s_d_hat = est.s_d[-1]
        #s_dd_hat = est.s_dd[-1]

        mpc_a = est.mpc_a[1] - est.mpc_a[0]
        mpc_kappa = est.mpc_kappa[0]

    #Creating the error states.
    e_psi = psi - yaw
    e_psi_hat = psi_hat - yaw_hat
    ey = d
    ey_hat = d_hat
    v = v
    v_hat = v_hat
    #Creating the linearized state vector
    x1 = e_psi - e_psi_hat
    x2 = ey - ey_hat
    x3 = v - v_hat
    #Creating the control vector u
    u1 = mpc_a
    u2 = c - mpc_kappa

    n = 3     #States
    m = 2     #inputs
    dtt = 0.1 #delta T in seconds
    #updatera A och B matriserna
    A = np.matrix([[(1/dtt*c)-math.tan(ey_hat)*v_hat, ((1/c - ey_hat)*v_hat)/(math.tan(e_psi_hat)**2 + 1), math.tan(e_psi_hat)*(1/c - ey_hat)],
                   [-((v_hat*Kappa_hat)/(math.cos(e_psi_hat))),(1/(dtt*c))+(math.sin(e_psi_hat)*Kappa_hat*((1/c)-ey_hat)*v_hat)/(math.cos(e_psi_hat)**2), (Kappa_hat*((1/c)-ey_hat))/(math.cos(e_psi_hat)) - psi/c],
                   [0, 0, 1/(dtt*c)]]) #i DT verkar fungera!
    Ts = dtt*c
    A = np.multiply(Ts,A)
    B = np.matrix([[((1/c)-ey_hat)/(math.cos(e_psi_hat)*(1/c)), 0],
                   [0, 0],
                   [0, dtt]])

    X_0 = [x1, x2, x3] #START VECTOR EACH ITERATION
    U_0 = [u1, u2]

    print('här printas X_0 och U_0: ')
    print(X_0)
    print(U_0)

    Q = np.eye(n, dtype=int)
    Q[0,0] = 1
    Q[1,1] = 1
    Q[2,2] = 1
    R = np.eye(m, dtype=int)
    R[0,0] = 1
    R[1,1] = 1

    #cost_matrix = np.eye(n, dtype=int)
    #cost_matrix[0,0] = -(math.sin(e_psi_hat)*(1/c_hat)*v_hat)/((1/c_hat)-ey_hat)
    #cost_matrix[1,1] = ((math.cos(e_psi_hat)*(1/c_hat)*v_hat)/((1/c_hat)-ey_hat)**2)
    #cost_matrix[2,2] = (((1/c_hat)*math.cos(e_psi_hat))/((1/c_hat)-ey_hat))
    #print "cost matrix :", cost_matrix
    x = cvxpy.Variable(n, N+1)
    u = cvxpy.Variable(m, N)
    cost = 0.0
    constr = []
    
    for t in range(N):
        cost += (v_hat*(1/c_hat)*(math.cos(e_psi_hat)/((1/c_hat)-ey_hat))) #s_dot
        cost += -(math.sin(e_psi_hat)*(1/c_hat)*v_hat)/((1/c_hat)-ey_hat)*x1
        cost += ((math.cos(e_psi_hat)*(1/c_hat)*v_hat)/((1/c_hat)-ey_hat)**2)*x2
        cost += (((1/c_hat)*math.cos(e_psi_hat))/((1/c_hat)-ey_hat))*x3
        constr += [x[:, t+1] == A*x[:, t] + B*u[:, t]]
        #constr += [x[0, t] <= MAX_Psi]                  #upper and lower car angle
        #constr += [x[0, t] >= -MAX_Psi]
        #constr += [x[1, t] <= MAX_ROAD_WIDTH]           #upper and lower road with bound
        #constr += [x[1, t] >= -MAX_ROAD_WIDTH]
        #constr += [x[2, t] <= MAX_v]                   #upper and lower velocity
        #constr += [x[2, t] >= -MAX_v]
        #constr += [u[0,t] <= MAX_KAPPA]                 #upper accel bound
        #constr += [u[0,t] >= -MAX_KAPPA]                #lower accel bound
        #constr += [u[1,t] <= MAX_ACCEL]                 #upper accel bound
        #constr += [u[1,t] >= -MAX_ACCEL]                #lower accel bound
    constr += [x[:, 0] == X_0]
    constr += [u[:, 0] == U_0]
    prob = cvxpy.Problem(cvxpy.Minimize(-cost), constr)
    prob.solve(verbose=False, solver=cvxpy.ECOS)
    print "Status:", prob.status
    print "Optimal value with ECOS:", prob.value

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:

        mpc_e_psi = get_nparray_from_matrix(x.value[0, :])
        mpc_ey = get_nparray_from_matrix(x.value[1, :])
        mpc_v = get_nparray_from_matrix(x.value[2, :]) #DE 3 VÄRDENA I X -STATE VECTORN för hela N
        mpc_kappa = get_nparray_from_matrix(u.value[0, :]) #u värdena
        mpc_a = get_nparray_from_matrix(u.value[1, :])
        #Behöver man räkna ut s,s_d,s_dd osv från detta för att få fram från Frenet till Global
        #tfp.s = s + mpc_v*np.cos(mpc_e_psi)*dtt
        tfp.d = mpc_ey
        tfp.v = mpc_v
        tfp.yaw = mpc_e_psi
        tfp.psi = mpc_e_psi
        tfp.c = mpc_e_psi

        tfp.mpc_a = mpc_a
        tfp.mpc_kappa = mpc_kappa
        frenet_paths.append(tfp) #Skapar en "struct" med all info
        #Control outputs
        #??
    else:
        print('NOT OPTIMAL')

    # Print all
    print('MPC Predictions')
    print(frenet_paths)
    print "d:", tfp.d, "d_d:", tfp.d_d, "d_dd:", tfp.d_dd
    print "s:", tfp.s, "s_d:", tfp.s_d, "s_dd:", tfp.s_dd
    print "x:", tfp.x
    print "y:", tfp.y
    print "yaw:", tfp.yaw
    print "Curvature:", tfp.c
    print "v", tfp.v
    print "Kappa", tfp.mpc_kappa
    print "a:", tfp.mpc_a


    return frenet_paths #interpolera states Detta blir sedan fplist


'''
def calc_global_paths(fplist, csp):  #From S to Global

    for fp in fplist:#for each vector in the list of all vectors of fplist (mystical..)
        for i in range(N):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])                    #Yaw angle derivative wrt curvature
            di = fp.d[i]                                    #Lateral position
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)   #x + closest catheter
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)   #y + furthest catheter
            fp.x.append(fx)                                 #All global x coord in a list
            fp.y.append(fy)                                 #All global y coord in a list



        for i in range(len(fp.x) - 1):#Calc yaw and ds
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append((math.atan2(dy, dx)))       #Yaw angle as derivative of the curve
            fp.ds.append(math.sqrt(dx**2 + dy**2))  #Length of the tangent vector

        fp.yaw.append(fp.yaw[-1])                   #Last yaw one more time, to get equal length
        fp.ds.append(fp.ds[-1])

        for i in range(len(fp.yaw) - 1):            # calc curvature
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / (fp.ds[i]+1)) #Derivative of yaw wrt step lenght ds.
            fp.psi.append(math.atan2(fp.y[i+1]-fp.y[i], fp.x[i+1]-fp.x[i]))


        if iteration == 0: #göra om detta till en class. sparar rader.
            initial_g_x = fp.x[0]
            initial_g_y = fp.y[0]
            initial_g_yaw = fp.yaw[0]
            initial_g_c = fp.c[0]
            initial_psi = math.atan2(fp.y[1]-fp.y[0], fp.x[1]-fp.x[0])
            #initial_x_d = fp.s_d[0] * math.cos(initial_psi) - fp.d_d[0]*math.sin(initial_psi)
            #initial_y_d = fp.s_d[0] * math.sin(initial_psi) + fp.d_d[0]*math.cos(initial_psi)
            initial_v = fp.v[0] #math.sqrt(initial_x_d**2 + initial_y_d**2)
        # Print all
        print('S to GLOBAL')
        print('.d')
        print(fp.d)
        print('.d_d')
        print(fp.d_d)
        print('.d_dd')
        print(fp.d_dd)
        print('.s')
        print(fp.s)
        print('.s_d')
        print(fp.s_d)
        print('.s_dd')
        print(fp.s_dd)
        print('.cd')
        print(fp.cd)
        print('.cv')
        print(fp.cv)
        print('.cf')
        print(fp.cf)

        print('.x')
        print(fp.x)
        print('.y')
        print(fp.y)
        print('.yaw')
        print(fp.yaw)
        print('.ds')
        print(fp.ds)
        print('MPC.c')
        print(fp.c)
        print('.v')
        print(fp.v)
        print('sw')
        print(sw)
        print('mpc_a')
        print(mpc_a)
    return fplist
'''

def global_vehicle_simulator(fplist, LUTs, LUTd, LUTx, LUTy, initial, iteration):
#vart skall inital_g_ komma från?
    print('ITERATION')
    print(iteration)
    if iteration == 0:
        old_state_x = -20
        old_state_y = 0
        old_state_yaw = 0
        old_state_c = 0
        old_state_v = 0
        old_state_psi = 0
    else:
        # Old state
        for init in initial:
            old_state_x = init.x[1]
            old_state_y = init.y[1]
            old_state_yaw = init.yaw[1]
            old_state_c = init.c[1]
            old_state_v = init.v[1]
            old_state_psi = init.psi[1]


    for fplist_for in fplist:
        kappa_stear = fplist_for.mpc_kappa[0]
        sw = math.tan(kappa_stear)/L+0.001
        a = fplist_for.mpc_a[0]

        #Global bicycle model here
        VP = lr/(lf+lr)
        beta = math.atan(VP*math.tan(sw)) #Body Slip angle
        new_state_c = math.sin(beta)/lr
        R = lr / math.sin(beta)
        v_p = a + old_state_v/R**2
        new_state_v = old_state_v + v_p * dt


        psi_d = new_state_v/lr * math.sin(beta)
        new_state_psi = old_state_psi + psi_d*dt

        #x_dd = a * math.cos(new_state_psi)
        #y_dd = a * math.sin(new_state_psi)
        x_d = new_state_v*math.cos(new_state_psi + beta)
        y_d = new_state_v*math.sin(new_state_psi + beta)
        new_state_x = old_state_x + x_d * dt #detta skickas till plotten
        print "Hastigheten: ", new_state_v, "x_d- hastigheten i x", x_d, "nya x: ", new_state_x
        new_state_y = old_state_y + y_d * dt #detta skickas till plotten
        new_state_yaw = math.atan2(new_state_y - old_state_y, new_state_x - old_state_x) #Yaw is the "yaw angle" of the track compare to a global x-axis

        #Transformerar de nya koordinaterna till Frenet

        value = [new_state_x, new_state_y]
        X = np.sqrt(np.square(LUTx - value[0]) + np.square(LUTy - value[1]))
        idx = np.where(X == X.min())
        r_idx = np.asscalar(idx[0])
        c_idx = np.asscalar(idx[1])

        new_state_s = LUTs[r_idx, c_idx] #Position i Frenet
        new_state_d = LUTd[r_idx, c_idx]

        #new_state_s_d = new_state_v * math.sin(((math.pi/2 - new_state_yaw)) + (new_state_psi-beta))
        #new_state_d_d = new_state_v * math.cos(((math.pi/2 - new_state_yaw)) + (new_state_psi-beta))

        #new_state_s_dd = v_p
        #new_state_d_dd = 0

        #new_state_s_dd = v_p * math.sin(((math.pi/2 - new_state_yaw)) + (new_state_psi-beta))
        #new_state_d_dd = v_p * math.cos(((math.pi/2 - new_state_yaw)) + (new_state_psi-beta))

        old_state_s = 0 #, old_state_s_d, old_state_s_dd = 0, 0, 0     VARFÖR 0???
        old_state_d = 0 #, old_state_d_d, old_state_d_dd = 0, 0, 0
        state = State()
        #Update
        state.d = [-old_state_d, -new_state_d]
        #state.d_d = [old_state_d_d, new_state_d_d]
        #state.d_dd = [old_state_d_dd, new_state_d_dd]
        #state.s = [old_state_s, new_state_s]
        #state.s_d = [old_state_s_d, new_state_s_d]
        #state.s_dd = [old_state_s_dd, new_state_s_dd]

        state.x = [old_state_x, new_state_x]
        state.y = [old_state_y, new_state_y]
        state.yaw = [old_state_yaw, new_state_yaw]
        state.c = [old_state_c, new_state_c]
        state.v = [old_state_v, new_state_v]
        state.psi = [old_state_psi, new_state_psi]
        state.beta = beta
        print('OLD STATE X')
        print(old_state_x)
        global_vehicle = []
        global_vehicle.append(state)


    # Print all
    print('The Simulated Vehicle Model State')
    print(state)
    print "d:", state.d, "d_d:", state.d_d, "d_dd:", state.d_dd, "s:", state.s, "s_d:", state.s_d, "s_dd:", state.s_dd
    print "x:", state.x, "y:", state.y, "yaw:", state.yaw
    print "Curvature:", state.c, "v", state.v, "sw", sw, "a", a


    return global_vehicle



def frenet_optimal_planning(csp, LUTs, LUTd, LUTx, LUTy, initial, estimated, iteration):
    #Bräknar en trajektori för position och hastighet
    fplist = calc_MPC_frenet_paths(initial, estimated)

    #Den beräknade trajektorern görs om från Frenet till Globala. Mest för att plotta och ge start position.
    #fplist = calc_global_paths(fplist, csp)

    global_vehicle = global_vehicle_simulator(fplist, LUTs, LUTd, LUTx, LUTy, initial, iteration)

    return fplist[0], global_vehicle[0]



def generate_target_course(x, y):#tar manuelt inmatate coordinater och skapar ett polynom som blir referens!
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)
    d = np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, 0.1) #Skapa 0.1 tät vinkelrät linje
    s_len = s.size
    d_len = d.size
    LUTs = np.zeros((s_len, d_len))
    LUTd = np.zeros((s_len, d_len))
    LUTx = np.zeros((s_len, d_len))
    LUTy = np.zeros((s_len, d_len))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s) #i_s  = incremental s ->här kan vi göra visualitionsconstraintsen
        rx.append(ix)#Ref x
        ry.append(iy)#Ref y
        ryaw.append(csp.calc_yaw(i_s))#Ref yaw
        rk.append(csp.calc_curvature(i_s))#Ref curvature

    LTs, LTd, LTx, LTy, refx, refy, refyaw = [], [], [], [], [], [], []
    s_count, d_count = -1, -1
    for i_ss in s:
        s_count = s_count + 1
        LTs = i_ss
        refx, refy = csp.calc_position(i_ss)
        refyaw = csp.calc_yaw(i_ss)
        for i_dd in d:
            if i_dd == -MAX_ROAD_WIDTH:
                d_count = -1
            d_count = d_count + 1
            LTd = i_dd
            LTx = refx + i_dd*math.sin(refyaw)
            LTy = refy - i_dd*math.cos(refyaw)
            LUTs[s_count, d_count] = LTs
            LUTd[s_count, d_count] = LTd
            LUTx[s_count, d_count] = LTx
            LUTy[s_count, d_count] = LTy

    '''
    print('Matrix Analysis ')
    print(len(LUTs))
    print(len(LUTx))
    print(LUTs)
    print(LUTd)
    print(LUTx)
    print(LUTy)
    '''
    plt.plot(LUTx[:,:], LUTy[:,:])
    plt.grid(True)
    plt.show()



    return rx, ry, ryaw, rk, csp, LUTs, LUTd, LUTx, LUTy


def main():
    print(__file__ + " start!!")
    wx, wy = [], []
    # way points fo the track
    #wxi = [0.0, 0.0, 5.0, 15.0, 17.0, 15.0,  17.0,  12.0,   0.0,   0.0, 0.0]
    #wyi = [0.0, 2.0, 5.0,  5.0,  0.0, -5.0, -10.0, -15.0, -15.0, -10.0, 0.0]

    wxi = [-20.0, -40.0, -70.0, -100.0, -120.0,  -140.0,  -150.0,   -160.0, -180.0,
           -200.0, -180.0, -160.0, -150.0, -140.0, -130.0, -120.0, -90.0, -60.0, -40.0, 0.0, 5.0, 0.0, -15.0, -20]

    wyi = [0.0, 0.0,  5.0,  0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
           20.0, 40.0, 40.0, 40.0, 45.0, 40.0, 35.0, 40.0, 40.0, 40.0, 40.0, 20.0, 0.0, 0.0, 0.0]

    wx += wxi
    wy += wyi


    tx, ty, tyaw, tc, csp, LUTs, LUTd, LUTx, LUTy = generate_target_course(wx, wy) #Get ut target-x (tx) Target-y (ty) target yaw, target Course! csp(the whole function handle)osv.

    x_plus, y_plus, x_minus, y_minus = [], [], [], []#skapar visuella constraints
    for i in range(len(tyaw)):
        x_plus.append(tx[i] + MAX_ROAD_WIDTH * math.sin(tyaw[i]))
        y_plus.append(ty[i] - MAX_ROAD_WIDTH * math.cos(tyaw[i]))
        x_minus.append(tx[i] - MAX_ROAD_WIDTH * math.sin(tyaw[i]))
        y_minus.append(ty[i] + MAX_ROAD_WIDTH * math.cos(tyaw[i]))
    '''
    # initial state ANGE i GLOBAL??
    s0 = 2.0      #current course position
    c_speed = 2   #current speed [m/s]
    c_acc = 0     #CURRENT LATTERAL ACCELERATION [m/s²]
    c_d = 1.5     #current lateral position [m]
    c_d_d = 0     #current lateral speed [m/s]
    c_d_dd = 0.0  #current latral acceleration [m/s²]

    state = State()
    state.s = [s0, s0]
    state.s_d = [c_speed, c_speed]
    state.s_dd = [c_acc, c_acc]
    state.d = [c_d, c_d]
    state.d_d = [c_d_d, c_d_d]
    state.d_dd = [c_d_dd, c_d_dd]
    state.v = [s0, s0]
    state.x = [0.1, 0.1]
    state.y = [0.1, 0.1]
    state.yaw = [1.0, 1.0]
    state.c = [0.1, 0.1]
    state.psi = [0.1, 0.1]
    '''
    state = State()
    state.s = [0, 0]
    #state.s_d = [c_speed, c_speed]
    #state.s_dd = [c_acc, c_acc]
    state.d = [0, 0]
    #state.d_d = [c_d_d, c_d_d]
    #state.d_dd = [c_d_dd, c_d_dd]
    state.v = [5, 5]
    state.x = [tx[0], tx[0]]
    state.y = [ty[0], ty[0]]
    state.yaw = [tyaw[0], tyaw[0]]
    state.c = [1, 1]
    state.psi = [0, 0]

    fpp = Frenet_path()
    fpp.s = [0, 0]
    #fpp.s_d = [c_speed, c_speed]
    #fpp.s_dd = [c_acc, c_acc]
    fpp.d = [1, 1]
    #fpp.d_d = [c_d_d, c_d_d]
    #fpp.d_dd = [c_d_dd, c_d_dd]
    fpp.v = [5, 5]
    fpp.x = [tx[0], tx[0]]
    fpp.y = [ty[0], ty[0]]
    fpp.yaw = [tyaw[0], tyaw[0]]
    fpp.c = [1, 1]
    fpp.psi = [0, 0]
    fpp.mpc_a = [0, 0]
    fpp.mpc_kappa = [0, 0]

    initial = []
    estimated = []
    initial.append(state)
    estimated.append(fpp)


    #initial measured state


    xx, yy = [], []

    area = 20.0  # animation area length [m]
    for i in range(600):  #antalet gånger koden körs. Bryts när målet är nått!! Detta blir Recursive delen i MPC'n
        print('OMGÅNG:')
        print(i)
        iteration = i


        MPC_path, global_vehicle = frenet_optimal_planning(csp, LUTs, LUTd, LUTx, LUTy, initial, estimated, iteration)  #Main function calling others
        estimated.append(MPC_path)
        initial.append(global_vehicle) #Updating the initial states
        '''
        print('DIFF .s[1]')
        print(MPC_path.s[1] - global_vehicle.s[1])
        print('DIFF .d[1]')
        print(MPC_path.d[1] - - global_vehicle.d[1])
        print('DIFF .s_d[1]')
        print(MPC_path.s_d[1] - global_vehicle.s_d[1])
        print('DIFF .d_d[1]')
        print(MPC_path.d_d[1] - global_vehicle.d_d[1])
        print('DIFF .s_dd[1]')
        print(MPC_path.s_dd[1] - global_vehicle.s_dd[1])
        print('DIFF .d_dd[1]')
        print(MPC_path.d_dd[1] - global_vehicle.d_dd[1])
        '''

        '''
        s0 = global_vehicle.s[1]  # CHECK STATE UPDATE!!!! Här någonstans kan allt skickas till funktionen?
        c_d = global_vehicle.d[1]        #Med path. ... [1] har vi nästa MPC steg
        c_d_d = global_vehicle.d_d[1]    #Med state. ... [1] har vi steget efter modellen
        c_d_dd = global_vehicle.d_dd[1]
        c_speed = global_vehicle.s_d[1]
        c_acc = global_vehicle.s_dd[1]
        
        xx.append(MPC_path.x)
        yy.append(MPC_path.y)
        '''
        '''
        if np.hypot(global_vehicle.x[0] - tx[-1], global_vehicle.y[0] - ty[-1]) <= 1.0: #hypot = sqrt(x²+y²) eucleadian norm
            print("Goal")
            break
        '''
        if show_animation:
            plt.cla()
            plt.plot(tx, ty, "k")
            plt.plot(x_plus, y_plus,"y")
            plt.plot(x_minus, y_minus,"b")
            '''
            plt.plot(MPC_path.x[1:], MPC_path.y[1:], "-or")#circle marker - red MPC

            plt.plot(MPC_path.x[0], MPC_path.y[0], 'dy')
            
            plt.xlim(MPC_path.x[1] - area, MPC_path.x[1] + area)
            plt.ylim(MPC_path.y[1] - area, MPC_path.y[1] + area)
            plt.title("SW angle: " + '{:.2f}'.format(sw_angle) +
                      " yaw: " + '{:.2f}'.format(global_vehicle.yaw[1])
                      + " s_d: " + '{:.2f}'.format(10000))
            '''
            plt.plot(global_vehicle.x[1], global_vehicle.y[1], 'gd')

            #Full vehicle coordinates
            FRx = lf*math.cos(global_vehicle.yaw[1]) + (Width/2)*math.sin(global_vehicle.yaw[1])
            FLx = lf*math.cos(global_vehicle.yaw[1]) - (Width/2)*math.sin(global_vehicle.yaw[1])
            RRx = lr*math.cos(global_vehicle.yaw[1]) - (Width/2)*math.sin(global_vehicle.yaw[1])
            RLx = lr*math.cos(global_vehicle.yaw[1]) + (Width/2)*math.sin(global_vehicle.yaw[1])
            FRy = lf*math.sin(global_vehicle.yaw[1]) - (Width/2)*math.cos(global_vehicle.yaw[1])
            FLy = lf*math.sin(global_vehicle.yaw[1]) + (Width/2)*math.cos(global_vehicle.yaw[1])
            RRy = lr*math.sin(global_vehicle.yaw[1]) + (Width/2)*math.cos(global_vehicle.yaw[1])
            RLy = lr*math.sin(global_vehicle.yaw[1]) - (Width/2)*math.cos(global_vehicle.yaw[1])
            carx = [global_vehicle.x[1] - FRx, global_vehicle.x[1] - FLx,
                    global_vehicle.x[1] + RLx, global_vehicle.x[1] + RRx, global_vehicle.x[1] - FRx]
            cary = [global_vehicle.y[1] - FRy, global_vehicle.y[1] - FLy,
                    global_vehicle.y[1] + RLy, global_vehicle.y[1] + RRy, global_vehicle.y[1] - FRy]
            plt.plot(carx, cary, 'r')

            plt.pause(0.01)





    print("Finish")
    print(i)

    if show_animation:
        plt.grid(True)
        plt.plot(xx, yy)

if __name__ == '__main__':
    main()
