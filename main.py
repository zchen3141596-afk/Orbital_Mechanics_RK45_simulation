# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:50:25 2025

@author: Chen Zhouhan
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as si 
import pandas as pd

#Definition of useful functions 
def norm(x,y): 
    """
    Function to compute the norm of a 2D vector
    ----------
    x : x component of the 2D vector
    y : y component of the 2D vector
    --------- 
    Returns: norm of the input 2D vector 
    """ 
    return np.sqrt(x**2 + y**2) 


#Fundamental physical constants 
G = 6.674e-11  #m^3 kg^-1 s^-2, gravitational constant 
Me = 5.972e24  #kg, mass of Earth 
Mm = 7.348e22  #kg, mass of Moon 
Mp = 900.0  #kg, mass of probe 
d_Earth_Moon = 3.844e8  #m, distance between Earth and Moon 
v0_Moon = np.sqrt(G * Me / d_Earth_Moon)  #m s-1, orbital velocity of Moon in circular simulation
d_perigee = 3.564e8  #m, closest distance between Earth and Moon 
d_apogee = 4.067e8  #m, farthest distance between Earth and Moon 
semimajor_axis = (d_perigee + d_apogee) / 2  #m, half length o the longest diameter of an ellipse 



#Definition of the main functions used throughout the programme 
#Derivative function for Moon's orbit 
def moon_orbit_derivatives (t, state, Me, G): 
    """
    Function to compute the derivatives for the orbit motion of the Moon about Earth. 
    ----------
    t : independent variable, floating point, not used. 
    state : tuple of floats containing (x_m, y_m, v_mx, v_my). 
    Me : mass of Earth (kg). 
    G : gravitational constant (m^3 kg^-1 s^-2) 
    -------
    Returns: tuple of derivatives (dx_m/dt, dy_m/dt, dv_mx/dt, d_my/dt) 
    """ 
    x_m, y_m, v_mx, v_my = state 
    
    r_m = norm(x_m, y_m) 
    
    f1 = v_mx 
    f2 = v_my 
    f3 = -Me * G * x_m / r_m**3 
    f4 = -Me * G * y_m / r_m**3 
    
    return (f1,f2,f3,f4) 


#Derivative function for Moon's orbit 
def probe_orbit_derivatives (t, state, Me, Mm, G): 
    """
    Function to compute the derivatives for the orbit motion of the probe about Moon affected by Earth's gravitational field 
    ----------
    t : independent variable, floating point, not used. 
    state : tuple of floats containing (x_m, y_m, v_mx, v_my). 
    Me : mass of Earth (kg). 
    Mm : mass of Moon (kg). 
    G : gravitational constant (m^3 kg^-1 s^-2) 
    -------
    Returns: tuple of derivatives (dx_m/dt, dy_m/dt, dv_mx/dt, d_my/dt) 
    """ 
    x_m, y_m, v_mx, v_my, x_p, y_p, v_px, v_py = state 
    
    r_m = norm(x_m, y_m) 
    r_p = norm(x_p, y_p) 
    x_pm = x_p - x_m 
    y_pm = y_p - y_m 
    r_pm = norm(x_pm, y_pm) 
    
    f1 = v_mx 
    f2 = v_my 
    f3 = - Me * G * x_m / r_m**3 
    f4 = - Me * G * y_m / r_m**3 
    f5 = v_px 
    f6 = v_py 
    f7 = - Me * G * x_p / r_p**3 - Mm * G * x_pm / r_pm**3 
    f8 = - Me * G * y_p / r_p**3 - Mm * G * y_pm / r_pm**3 
    
    return (f1,f2,f3,f4,f5,f6,f7,f8) 


#Moon orbital energy calculation function 
def moon_orbital_energy(state): 
    """ 
    Function to compute the orbital energy of Moon about Earth 
    ---------- 
    state : tuple of floats containing (x_m, y_m, v_mx, v_my). 
    ---------- 
    Returns: tuple of energies (kinetic_energy, potential energy, total_energy) 
    """ 
    x_m, y_m, v_mx, v_my = state 
    r_m = norm(x_m,y_m) 
    v_m = norm(v_mx,v_my) 
    
    #Kinetic energy of Moon 
    E_k = 0.5 * Mm * v_m**2 
    
    #Potential energy of Moon in Earth orbit 
    E_p = -G * Me * Mm / r_m 
    
    #Total energy 
    E_total = E_k + E_p 
    
    return E_k, E_p, E_total 


#Probe orbital energy calculation function 
def probe_orbital_energy(state, Mp, Me, Mm): 
    """ 
    Function to compute the orbital energy of Probe about Moon affected by Earth's gravitational field 
    ---------- 
    state : tuple of floats containing (x_m, y_m, v_mx, v_my, x_p, y_p, v_px, v_py). 
    Mp : mass of probe (kg). 
    Me : mass of Earth (kg). 
    Mm : mass of Moon (kg). 
    ---------- 
    Returns: energies kinetic_energy, potential energy, total_energy of probe 
    """ 
    x_m, y_m, v_mx, v_my, x_p, y_p, v_px, v_py = state 
    r_p = norm(x_p, y_p) 
    v_p = norm(v_px, v_py) 
    
    x_pm = x_p - x_m 
    y_pm = y_p - y_m 
    r_pm = norm(x_pm, y_pm) 
    
    #Kinetic energy of probe
    E_k = 0.5 * Mp * v_p**2 
    
    #Potential energy of probe 
    E_p = - G * Mp * Me / r_p - G * Mm * Mp / r_pm 
    
    #Total energy of probe
    E_total = E_k + E_p 
    
    return E_k, E_p, E_total 


#Earth-Moon-probe system energy calculation function 
def system_orbital_energy(state, Mp, Me, Mm): 
    """ 
    Function to compute the orbital energy of Earth-Moon-probe system 
    ----------
    state : tuple of floats containing (x_m, y_m, v_mx, v_my, x_p, y_p, v_px, v_py). 
    Mp : mass of probe (kg). 
    Me : mass of Earth (kg). 
    Mm : mass of Moon (kg). 
    ---------
    Returns: energies kinetic_energy, potential energy, total_energy of Earth-Moon-probe system 
    """ 
    x_m, y_m, v_mx, v_my, x_p, y_p, v_px, v_py = state 
    r_m = norm(x_m, y_m) 
    r_p = norm(x_p, y_p) 
    v_p = norm(v_px, v_py) 
    v_m = norm(v_mx, v_my) 
    
    x_pm = x_p - x_m 
    y_pm = y_p - y_m 
    r_pm = norm(x_pm, y_pm) 
    
    #Kinetic energy of system 
    E_k = 0.5 * Mp * v_p**2 + 0.5 * Mm * v_m**2 
    
    #Potential energy of system 
    E_p = - G * Me * Mp / r_p - G * Me * Mm / r_m - G * Mm * Mp / r_pm 
    
    #Total energy of system 
    E_total = E_k + E_p 
    
    return E_k, E_p, E_total 


#Plotting function of the energies 
def energy_plot (t, energy, title): 
    """
    Function to plot the graphs of energies for orbital motion 
    ----------
    t: independent variable, floating point. 
    energy: kinetic, potential and total energy of the bodies simulated. 
    title: string variable, title of each graph. 
    --------- 
    Returns: None 
    """ 
    kinetic, potential, total = energy
    plt.figure() 
    plt.plot(t, kinetic, label = "Kinetic energy") 
    plt.plot(t, potential, label = "Potential energy") 
    plt.plot(t, total, label = "Total energy") 
    plt.grid(True) 
    plt.xlabel("Time (s)") 
    plt.ylabel("Energy (J)") 
    plt.title(title) 
    plt.legend(loc = "lower right") 
    plt.show() 

#Initialisation of variables for choice making and simulation setups 
MyInput  = '0' 
circular_or_elliptical = '0' 
t_simulation = '0' 
d_Moon_probe = '0' 


while MyInput != 'q':
    MyInput = input('Enter a choice, "1", "2" or "q" to quit: ')
    print('You entered the choice: ',MyInput) 
    
    
    if MyInput == '1': 
        
        print('You have chosen part (1): simulation of a lunar orbit') 

        while circular_or_elliptical != 'q' : 
            
            
            #List to store and clear list for loops 
            kinetic_energy = [] 
            potential_energy = [] 
            total_energy = [] 
            
            
            circular_or_elliptical = input('Enter a choice,"c" for circular simulation, "e" for elliptical simulation or "q" to quit: ') 
            print('You entered the choice: ', circular_or_elliptical) 
            
            if circular_or_elliptical == 'c': 
                
                print('You have chosen circular lunar orbit simulation') 
                
                t_simulation = float(input("Enter the time you want to simulate [s] (suggested value: 2.5e6): ")) 
                print("You entered the choice: ", t_simulation) 

               
                #Initial conditions for Moon's orbit 
                x0 = d_Earth_Moon 
                y0 = 0.0 
                vx0 = 0.0 
                vy0 = v0_Moon
                
                initial_state = (x0, y0, vx0, vy0)
                
                
                #Timestep generation 
                t_min = 0.0 
                t_max = t_simulation
                numpoints = 5001
                times = np.linspace(t_min, t_max, numpoints)  #creates an array of evenly spread timesteps 
                
                
                #Absolute and relative tolerance setup 
                rtol = 1.0e-9 
                atol = 1.0e-12 
                
                
                #Solve the ODE using solve_ivp() 
                moon_orbit_solution = si.solve_ivp(moon_orbit_derivatives, (t_min, t_max), initial_state, t_eval = times, rtol = rtol, atol=atol, args = (Me, G)) 
                
                
                #Storage of the results obtained 
                xpos = moon_orbit_solution.y[0,:] 
                ypos = moon_orbit_solution.y[1,:] 
                
                
                #Plot of the lunar circular orbit 
                plt.figure() 
                plt.plot(xpos, ypos, label = 'Moon') 
                plt.scatter(0, 0, color = 'red', label = 'Earth') 
                plt.gca().set_aspect('equal', adjustable = 'box') 
                plt.grid(True) 
                plt.xlabel("x position (m)") 
                plt.ylabel("y position (m)") 
                plt.title("Circular orbit of Moon around Earth") 
                plt.legend(loc = "lower right") 
                
                
                #For loop to calculate and store the energy at each timestep 
                for i in range(len(xpos)): 
                    current_state = (moon_orbit_solution.y[0,i], moon_orbit_solution.y[1,i], moon_orbit_solution.y[2,i], moon_orbit_solution.y[3,i]) 
                    KE, PE, TE = moon_orbital_energy(current_state) 
                    kinetic_energy.append(KE) 
                    potential_energy.append(PE) 
                    total_energy.append(TE) 
                

                #Print energy values at every timestep using pandas table plot 
                pd.set_option('display.max_rows', None) 
                moon_energy = pd.DataFrame({'Kinetic energy': kinetic_energy, 'Potential energy': potential_energy, 'Total_energy': total_energy}, index = np.linspace(t_min, t_max, len(total_energy))) 
                print(moon_energy[::50]) 
                
                
                #Plot of energy evolution at every timestep 
                moon_energy = kinetic_energy, potential_energy, total_energy 
                title_energy = "Circular orbit energy evolution at every timestep" 
                energy_plot(times, moon_energy, title_energy) 
                
                
                #Error evaluation 
                abs_error = abs(np.max(total_energy) - total_energy[0])
                rel_error = abs(abs_error / total_energy[0]) 
                print("The absolute error of energy is: ", abs_error) 
                print("The relative error of energy is: ", rel_error) 
                
                
                
            elif circular_or_elliptical == 'e' :
                
                print('You have chosen elliptical lunar orbit simulation') 

                t_simulation = float(input("Enter the time you want to simulate [s] (suggested value: 2.5e6): ")) 
                print("You entered the choice: ", t_simulation) 


                #Initial velocity calculation when motion starts from apogee 
                v0_Moon_apogee = np.sqrt(G * Me * (2 / d_apogee - 1 / semimajor_axis)) 


                #Initial conditions for Moon's orbit 
                x0 = d_apogee 
                y0 = 0.0 
                vx0 = 0.0 
                vy0 = v0_Moon_apogee 

                initial_state = (x0, y0, vx0, vy0) 


                #Initial state of simulation 
                t_min = 0.0 
                t_max = t_simulation
                numpoints = 5001
                times = np.linspace(t_min, t_max, numpoints)  #creates an array of evenly spread timesteps 
                
                
                #Absolute and relative tolerance setup 
                rtol = 1.0e-9 
                atol = 1.0e-12 


                #Solve the ODE using solve_ivp() 
                moon_orbit_solution = si.solve_ivp(moon_orbit_derivatives, (t_min, t_max), initial_state, t_eval = times, rtol = rtol, atol = atol, args = (Me, G))
                print(moon_orbit_solution.y.shape)

                #Storage of the results obtained 
                xpos = moon_orbit_solution.y[0,:] 
                ypos = moon_orbit_solution.y[1,:] 
                
                
                #Plot of the lunar elliptical orbit 
                plt.figure() 
                plt.plot(xpos, ypos, label = 'Moon') 
                plt.scatter(0, 0, color = 'red', label = 'Earth') 
                plt.grid(True) 
                plt.xlabel("x position (m)") 
                plt.ylabel("y position (m)") 
                plt.title("Elliptical orbit of Moon around Earth")  
                plt.legend(loc = "lower right") 
                

                #For loop to calculate and store the energy at each timestep 
                for i in range(len(xpos)): 
                    current_state = (moon_orbit_solution.y[0,i], moon_orbit_solution.y[1,i], moon_orbit_solution.y[2,i], moon_orbit_solution.y[3,i]) 
                    KE, PE, TE = moon_orbital_energy(current_state) 
                    kinetic_energy.append(KE) 
                    potential_energy.append(PE) 
                    total_energy.append(TE) 


                #Print energy values at every timestep using pandas table plot 
                pd.set_option('display.max_rows', None) 
                moon_energy = pd.DataFrame({'Kinetic energy': kinetic_energy, 'Potential energy': potential_energy, 'Total_energy': total_energy}, index = np.linspace(t_min, t_max, len(total_energy))) 
                print(moon_energy[::50]) 
                
                
                #Plot of energy evolution at every timestep 
                moon_energy = kinetic_energy, potential_energy, total_energy 
                title_energy = "Elliptical orbit energy evolution at every timestep" 
                energy_plot(times, moon_energy, title_energy) 
                
                
                #Error evaluation 
                abs_error = abs(np.max(total_energy) - total_energy[0]) 
                rel_error = abs(abs_error / total_energy[0]) 
                print("The absolute error of energy is: ", abs_error) 
                print("The relative error of energy is: ", rel_error) 
                
                
        
    elif MyInput == '2': 
        
        print('You have chosen part (2): earth-moon-probe system') 
        
        while circular_or_elliptical != 'q' : 
            
            
            #List to store and clear list of energies for loops 
            kinetic_energy = [] 
            potential_energy = [] 
            total_energy = [] 
            
            kinetic_energy_system = [] 
            potential_energy_system = [] 
            total_energy_system = [] 
            
            
            circular_or_elliptical = input('Enter a choice, "c" for circular simulation, "e" for elliptical simulation or "q" to quit: ') 
            print('You entered the choice: ', circular_or_elliptical) 

            if circular_or_elliptical == 'c' :
                print('You have chosen circular earth-moon-probe system orbit simulation') 
                
                d_Moon_probe = float(input("Enter the initial distance between Probe and Moon [m] (suggested value: 5.0e6): ")) 
                print("You entered the choice: ", d_Moon_probe) 
                
                t_simulation = float(input("Enter the time you want to simulate [s] (suggested value: 2.5e6): ")) 
                print("You entered the choice: ", t_simulation) 
                
                
                #Initial conditions for Moon's orbit 
                x_m0 = d_Earth_Moon 
                y_m0 = 0.0 
                vx_m0 = 0.0 
                vy_m0 = v0_Moon 
                v0_pm = np.sqrt(G * Mm / d_Moon_probe) 
                x_p0 = x_m0 + d_Moon_probe 
                y_p0 = 0.0 
                vx_p0 = 0.0 
                vy_p0 = vy_m0 + v0_pm 
                
                initial_state = (x_m0, y_m0, vx_m0, vy_m0, x_p0, y_p0, vx_p0, vy_p0) 
                
                
                #Initial state of simulation 
                t_min = 0.0 
                t_max = t_simulation
                numpoints = 5001
                times = np.linspace(t_min, t_max, numpoints)  #creates an array of evenly spread timesteps 
                
                
                #Absolute and relative tolerance setup 
                rtol = 1.0e-8 
                atol = 1.0e-10 
                
                
                #Solve the ODE using solve_ivp() 
                probe_orbit_solution = si.solve_ivp(probe_orbit_derivatives, (t_min, t_max), initial_state, t_eval = times, rtol = rtol, atol = atol, args = (Me, Mm, G))
                
                
                #Storage of the results obtained 
                xpos_m = probe_orbit_solution.y[0,:] 
                ypos_m = probe_orbit_solution.y[1,:] 
                xpos_p = probe_orbit_solution.y[4,:] 
                ypos_p = probe_orbit_solution.y[5,:] 
                
                
                #Plot of the Earth-Moon-probe system circular orbit 
                plt.figure() 
                plt.plot(xpos_m, ypos_m, label = 'Moon') 
                plt.plot(xpos_p, ypos_p, label = 'Probe') 
                plt.scatter(0, 0, color = 'red', label = 'Earth') 
                plt.gca().set_aspect('equal', adjustable = 'box') 
                plt.grid(True) 
                plt.xlabel("x position (m)") 
                plt.ylabel("y position (m)") 
                plt.title("Circular orbit of Earth-Moon-probe system")  
                plt.legend(loc = "lower right") 
                plt.show() 
                
                
                #For loop to calculate and store the energy at each timestep 
                for i in range(len(xpos_m)): 
                    current_state = (probe_orbit_solution.y[0,i], probe_orbit_solution.y[1,i], probe_orbit_solution.y[2,i], probe_orbit_solution.y[3,i], probe_orbit_solution.y[4,i], probe_orbit_solution.y[5,i], probe_orbit_solution.y[6,i], probe_orbit_solution.y[7,i]) 
                    KE, PE, TE = probe_orbital_energy(current_state, Mp, Me, Mm) 
                    kinetic_energy.append(KE) 
                    potential_energy.append(PE) 
                    total_energy.append(TE) 
                
                for i in range(len(xpos_m)): 
                    current_state = (probe_orbit_solution.y[0,i], probe_orbit_solution.y[1,i], probe_orbit_solution.y[2,i], probe_orbit_solution.y[3,i], probe_orbit_solution.y[4,i], probe_orbit_solution.y[5,i], probe_orbit_solution.y[6,i], probe_orbit_solution.y[7,i]) 
                    KE, PE, TE = system_orbital_energy(current_state, Mp, Me, Mm) 
                    kinetic_energy_system.append(KE) 
                    potential_energy_system.append(PE) 
                    total_energy_system.append(TE) 
                
                
                #Print energy values at every timestep using pandas table plot 
                pd.set_option('display.max_rows', None) 
                probe_energy = pd.DataFrame({'Kinetic energy': kinetic_energy, 'Potential energy': potential_energy, 'Total_energy': total_energy}, index = np.linspace(t_min, t_max, len(total_energy)))  
                print(probe_energy[::50]) 
                
                
                #Plot of probe's energy evolution at every timestep 
                probe_energy = kinetic_energy, potential_energy, total_energy 
                title_energy = "Probe's circular orbit energy evolution at every timestep" 
                energy_plot(times, probe_energy, title_energy) 
                
                
                #Plot of system's energy evolution at every timestep 
                system_energy = kinetic_energy_system, potential_energy_system, total_energy_system 
                title_energy_system = "System's circular orbit energy evolution at every timestep" 
                energy_plot(times, system_energy, title_energy_system) 
                
                
                #Error evaluation 
                abs_error = abs(np.max(total_energy) - total_energy[0]) 
                rel_error = abs(abs_error / total_energy[0])  
                print("The absolute error of probe's energy is: ", abs_error) 
                print("The relative error of probe's energy is: ", rel_error) 
                
                abs_error_system = abs(np.max(total_energy_system) - total_energy_system[0]) 
                rel_error_system = abs(abs_error / total_energy_system[0]) 
                print("The absolute error of system's energy is: ", abs_error_system) 
                print("The relative error of system's energy is: ", rel_error_system) 
                
                
                
            elif circular_or_elliptical == 'e': 
                
                print('You have chosen elliptical Earth-Moon-probe system orbit simulation: ') 

                d_Moon_probe = float(input("Enter the initial distance between Probe and Moon [m] (suggested value: 5.0e6): ")) 
                print("You entered the choice: ", d_Moon_probe) 

                t_simulation = float(input("Enter the time you want to simulate [s] (suggested value: 2.5e6): ")) 
                print("You entered the choice: ", t_simulation) 
                
                
                #Initial velocity calculation when motion starts from apogee 
                v0_Moon_apogee = np.sqrt(G * Me * (2 / d_apogee - 1 / semimajor_axis)) 
                
                
                #Initial conditions for Moon's and probe's orbits 
                x_m0 = d_apogee
                y_m0 = 0.0 
                vx_m0 = 0.0 
                vy_m0 = v0_Moon_apogee 
                v0_pm = np.sqrt(G * Mm / d_Moon_probe) 
                x_p0 = x_m0 + d_Moon_probe 
                y_p0 = 0.0 
                vx_p0 = 0.0 
                vy_p0 = vy_m0 + v0_pm 
                
                initial_state = (x_m0, y_m0, vx_m0, vy_m0, x_p0, y_p0, vx_p0, vy_p0) 
                
                
                #Initial state of simulation 
                t_min = 0.0 
                t_max = t_simulation
                numpoints = 5001 
                times = np.linspace(t_min, t_max, numpoints)  #creates an array of evenly spread timesteps 
                
                
                #Absolute and relative tolerance setup 
                rtol = 1.0e-8 
                atol = 1.0e-10 
                
                
                #Solve the ODE using solve_ivp() 
                probe_orbit_solution = si.solve_ivp(probe_orbit_derivatives, (t_min, t_max), initial_state, t_eval = times, rtol = rtol, atol = atol, args = (Me, Mm, G)) 
                
                
                #Storage of results obtained 
                xpos_m = probe_orbit_solution.y[0,:] 
                ypos_m = probe_orbit_solution.y[1,:] 
                xpos_p = probe_orbit_solution.y[4,:] 
                ypos_p = probe_orbit_solution.y[5,:] 
                
                
                #Plot of the Earth-Moon-probe system elliptical orbit 
                plt.figure() 
                plt.plot(xpos_m, ypos_m, label = 'Moon') 
                plt.plot(xpos_p, ypos_p, label = 'Probe') 
                plt.scatter(0, 0, color = 'red', label = 'Earth') 
                plt.grid(True) 
                plt.xlabel("x position (m)") 
                plt.ylabel("y position (m)") 
                plt.title("Elliptical orbit of Earth-Moon-probe system")  
                plt.legend(loc = "lower right") 
                
                
                #For loop to calculate and store the energy at each timestep 
                for i in range(len(xpos_m)): 
                    current_state = (probe_orbit_solution.y[0,i], probe_orbit_solution.y[1,i], probe_orbit_solution.y[2,i], probe_orbit_solution.y[3,i], probe_orbit_solution.y[4,i], probe_orbit_solution.y[5,i], probe_orbit_solution.y[6,i], probe_orbit_solution.y[7,i]) 
                    KE, PE, TE = probe_orbital_energy(current_state, Mp, Me, Mm) 
                    kinetic_energy.append(KE) 
                    potential_energy.append(PE) 
                    total_energy.append(TE) 
                
                for i in range(len(xpos_m)): 
                    current_state = (probe_orbit_solution.y[0,i], probe_orbit_solution.y[1,i], probe_orbit_solution.y[2,i], probe_orbit_solution.y[3,i], probe_orbit_solution.y[4,i], probe_orbit_solution.y[5,i], probe_orbit_solution.y[6,i], probe_orbit_solution.y[7,i]) 
                    KE, PE, TE = system_orbital_energy(current_state, Mp, Me, Mm) 
                    kinetic_energy_system.append(KE) 
                    potential_energy_system.append(PE) 
                    total_energy_system.append(TE) 


                #Print energy values at every timestep using pandas table plot 
                pd.set_option('display.max_rows', None) 
                probe_energy = pd.DataFrame({'Kinetic energy': kinetic_energy, 'Potential energy': potential_energy, 'Total_energy': total_energy}, index = np.linspace(t_min, t_max, len(total_energy)))  
                print(probe_energy[::50]) 
                
                
                #Plot of energy evolution at every timestep 
                probe_energy = kinetic_energy, potential_energy, total_energy 
                title_energy = "Probe's elliptical orbit energy evolution at every timestep" 
                energy_plot(times, probe_energy, title_energy) 
                
                
                #Plot of system's energy evolution at every timestep 
                system_energy = kinetic_energy_system, potential_energy_system, total_energy_system 
                title_energy_system = "System's elliptical orbit energy evolution at every timestep" 
                energy_plot(times, system_energy, title_energy_system) 
                
                
                #Error evaluation 
                abs_error = abs(np.max(total_energy) - total_energy[0]) 
                rel_error = abs(abs_error / total_energy[0]) 
                print("The absolute error of probe's energy is: ", abs_error) 
                print("The relative error of probe's energy is: ", rel_error) 
                
                abs_error_system = abs(np.max(total_energy_system) - total_energy_system[0]) 
                rel_error_system = abs(abs_error / total_energy_system[0]) 
                print("The absolute error of system's energy is: ", abs_error_system) 
                print("The relative error of system's energy is: ", rel_error_system) 
                
                
                
    elif MyInput != 'q':
        print('This is not a valid choice')
print('You have chosen to finish - goodbye.')


