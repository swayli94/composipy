import numpy as np
import pandas as pd
from typing import List, Tuple


__all__ = ['LaminateStrength']


class LaminateStrength():
    '''
    Creates a LaminateStrength object to evaluate strength.
    The class is capable of calculating strain and stress at the mid-plane, as well
    as ply-by-ply at the top and bottom, both in the laminate and material directions.

    Parameters
    ----------
    dproperty : LaminateProperty
        A laminate property object
    Nxx : float, int, optional, default 0
        Membrane load in x direction.
    Nyy : float, int, optional, default 0
        Membrane load in y direction.    
    Nxy : float, int, optional, default 0
        Membrane load in xy direction.
    Mxx : float, int, optional, default 0
        Moment in x direction.
    Myy : float, int, optional, default 0
        Moment in y direction.
    Mxy : float, int, optional, default 0
        Moment in xy direction.
    '''

    def __init__(self, dproperty, Nxx=0, Nyy=0, Nxy=0, Mxx=0, Myy=0, Mxy=0):
        self.dproperty = dproperty
        self.Nxx = Nxx
        self.Nyy = Nyy
        self.Nxy = Nxy
        self.Mxx = Mxx
        self.Myy = Myy
        self.Mxy = Mxy
        self._positions = None #used for plotting.

    @staticmethod
    def get_epsilon0(ABD: np.ndarray, N: np.ndarray) -> np.ndarray:
        '''
        Calculates the mid-plane strains of the laminate
        
        Parameters
        ----------
        ABD : np.ndarray (6x6)
            ABD matrix of the laminate
            
        N : np.ndarray (6,)
            Load vector, [Nxx, Nyy, Nxy, Mxx, Myy, Mxy]
        
        Returns
        -------
        epsilon0 : numpy ndarray (6,)
            mid-plane strains, [epsilon_x0, epsilon_y0, epsilon_xy0, kappa_x0, kappa_y0, kappa_xy0]
        '''
        abd = np.linalg.inv(ABD) #equivalent relations on pg 153 of Daniel
        return abd @ N

    @staticmethod
    def get_epsilon_plies(epsilon0: np.ndarray, n_ply: int, z_positions: List[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
        '''
        Calculates the strains for each ply from the mid-plane strains,
        the strains are in the global coordinate system (x,y).
        
        Parameters
        ----------
        epsilon0 : np.ndarray (6,)
            Strain vector at the mid-plane
            
        n_ply : int
            Number of plies
            
        z_positions : List[float]
            List of z-positions for each ply
            
        Returns
        -------
        epsilon_plies: list of tuples
            For the k-th ply, epsilon_k = (epsilon_top, epsilon_bot), 
            where epsilon_top and epsilon_bot are np.ndarray([epsilon_x, epsilon_y, gamma_xy]).
            The list of tuples is in the order of the plies, from top to bottom.
            For reference refer to page 145 of Daniel equation 5.8
            
        '''
        epsilon_mid = np.array([epsilon0[0], epsilon0[1], epsilon0[2]])
        kappa_mid = np.array([epsilon0[3], epsilon0[4], epsilon0[5]])
        
        # Reverse z_positions so bot is negative and top is positive
        z = z_positions[::-1]
        
        epsilon_plies = []
        
        for i in range(n_ply):
            epsilon_plies.append(
                (epsilon_mid + z[i] * kappa_mid,
                 epsilon_mid + z[i+1] * kappa_mid)
                )
        
        return epsilon_plies

    @staticmethod
    def get_stress_plies(Q_layup: List[np.ndarray], epsilon_plies: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        '''
        Calculates the stresses for each ply from the strains.
        
        Parameters
        ----------
        Q_layup : List[np.ndarray]
            List of Q matrices (stiffness matrices) for each ply
            
        epsilon_plies : List[Tuple[np.ndarray, np.ndarray]]
            List of strain tuples (epsilon_top, epsilon_bot) for each ply
            
        Returns
        -------
        stress_plies: list of tuples
            For the k-th ply, stress_k = (stress_top, stress_bot),
            where stress_top and stress_bot are np.ndarray([sigma_x, sigma_y, tau_xy]).
            The list of tuples is in the order of the plies, from top to bottom.
            For reference refer to page 145 of Daniel equation 5.8
        '''
        stress_plies = []
        for Q_k, epsilon_k in zip(Q_layup, epsilon_plies):
            epsilon_top, epsilon_bot = epsilon_k
            stress_plies.append(
                (Q_k @ epsilon_top,
                 Q_k @ epsilon_bot)
            )
        return stress_plies

    @staticmethod
    def get_epsilon_plies_123(stacking: List[float], epsilon_plies: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        '''
        Transforms strains from global coordinates (x,y) to material coordinates (1,2,3).
        
        Parameters
        ----------
        stacking : List[float]
            List of ply angles in degrees
            
        epsilon_plies : List[Tuple[np.ndarray, np.ndarray]]
            List of strain tuples (epsilon_top, epsilon_bot) for each ply in global coordinates
            
        Returns
        -------
        epsilon_plies_123: List[Tuple[np.ndarray, np.ndarray]]
            For the k-th ply, epsilon_k_123 = (epsilon_top_123, epsilon_bot_123),
            where epsilon_top_123 and epsilon_bot_123 are np.ndarray([epsilon_1, epsilon_2, gamma_12]).
            The list of tuples is in the order of the plies, from top to bottom.
            For reference refer to NASA pg 50
        '''
        epsilon_plies_123 = []
        for theta, epsilon_k in zip(stacking, epsilon_plies):
            epsilon_top, epsilon_bot = epsilon_k
            # Make copies to avoid modifying the originals
            epsilon_top = epsilon_top.copy()
            epsilon_bot = epsilon_bot.copy()
            
            c = np.cos(theta*np.pi/180)
            s = np.sin(theta*np.pi/180)
            epsilon_top[2] /= 2 # engineering shear strain (see nasa pg 50)
            epsilon_bot[2] /= 2

            T = np.array([
                [c**2, s**2, 2*c*s],
                [s**2, c**2, -2*c*s],
                [-c*s, c*s, c**2-s**2]
                ])

            cur_epsilon_top = T @ epsilon_top
            cur_epsilon_bot = T @ epsilon_bot
            cur_epsilon_top[2] *= 2
            cur_epsilon_bot[2] *= 2
            epsilon_plies_123.append((cur_epsilon_top, cur_epsilon_bot)) # engineering shear strain (see nasa pg 50)
        
        return epsilon_plies_123

    @staticmethod
    def get_stress_plies_123(stacking: List[float], stress_plies: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        '''
        Transforms stresses from global coordinates (x,y) to material coordinates (1,2,3).
        
        Parameters
        ----------
        stacking : List[float]
            List of ply angles in degrees
            
        stress_plies : List[Tuple[np.ndarray, np.ndarray]]
            List of stress tuples (stress_top, stress_bot) for each ply in global coordinates
            
        Returns
        -------
        stress_plies_123: List[Tuple[np.ndarray, np.ndarray]]
            For the k-th ply, stress_k_123 = (stress_top_123, stress_bot_123),
            where stress_top_123 and stress_bot_123 are np.ndarray([sigma_1, sigma_2, tau_12]).
            The list of tuples is in the order of the plies, from top to bottom.
            For reference refer to NASA pg 50
        '''
        stress_plies_123 = []
        for theta, stress in zip(stacking, stress_plies):
            stress_top, stress_bot = stress   
            c = np.cos(theta*np.pi/180)
            s = np.sin(theta*np.pi/180)
            #stress_top[2] /= 2 # engineering shear strain (see nasa pg 50)
            #stress_bot[2] /=

            T = np.array([
                [c**2, s**2, 2*c*s],
                [s**2, c**2, -2*c*s],
                [-c*s, c*s, c**2-s**2]
                ])

            cur_stress_top = T @ stress_top
            cur_stress_bot = T @ stress_bot
            #cur_stresstop[2] *= 2
            #cur_stress_bot[2] *= 2
            stress_plies_123.append((cur_stress_top, cur_stress_bot)) 
        
        return stress_plies_123

    @staticmethod
    def get_strain_field(ABD: np.ndarray, N: np.ndarray, stacking: List[float], z_positions: List[float]) -> pd.DataFrame:
        '''
        Calculates the strain field of the laminate
        
        Parameters
        ----------
        ABD: np.ndarray (6x6)
            ABD matrix of the laminate
            
        N: np.ndarray (6,)
            Load vector, [Nxx, Nyy, Nxy, Mxx, Myy, Mxy]
            
        stacking: List[float]
            List of ply angles in degrees
            
        z_positions: List[float]
            List of z-positions for each ply
            
        Returns
        -------
        strain_field: pd.DataFrame
            Strain field of the laminate, ply by ply in plate direction and material direction
        '''
        n_ply = len(stacking)
        epsilon0 = LaminateStrength.get_epsilon0(ABD, N)
        epsilon_plies = LaminateStrength.get_epsilon_plies(epsilon0, n_ply, z_positions)
        epsilon_plies_123 = LaminateStrength.get_epsilon_plies_123(stacking, epsilon_plies)
        
        cur_ply = 1
        data = {}
        data['ply'] = []
        data['position'] = []
        data['angle'] = []       
        data['epsilonx'] = []
        data['epsilony'] = []
        data['gammaxy'] = []
        data['epsilon1'] = []
        data['epsilon2'] = []
        data['gamma12'] = []
        for epsilon, epsilon123, theta in zip(epsilon_plies, epsilon_plies_123, stacking):
            epsilon_top, epsilon_bot = epsilon
            epsilon_top_123, epsilon_bot_123 = epsilon123

            data['ply'].append(cur_ply)
            data['ply'].append(cur_ply)
            data['position'].append('top')
            data['position'].append('bot')
            data['angle'].append(theta)
            data['angle'].append(theta)            
            data['epsilonx'].append(epsilon_top[0]) #plate direction
            data['epsilonx'].append(epsilon_bot[0])
            data['epsilony'].append(epsilon_top[1])
            data['epsilony'].append(epsilon_bot[1])
            data['gammaxy'].append(epsilon_top[2])
            data['gammaxy'].append(epsilon_bot[2])
            data['epsilon1'].append(epsilon_top_123[0]) #material direction
            data['epsilon1'].append(epsilon_bot_123[0])
            data['epsilon2'].append(epsilon_top_123[1])
            data['epsilon2'].append(epsilon_bot_123[1])
            data['gamma12'].append(epsilon_top_123[2])
            data['gamma12'].append(epsilon_bot_123[2])
            cur_ply += 1
        pd.set_option('display.precision', 2)
        return pd.DataFrame(data)
    
    @staticmethod
    def get_stress_field(ABD: np.ndarray, N: np.ndarray, stacking: List[float], z_positions: List[float], Q_layup: List[np.ndarray]) -> pd.DataFrame:
        '''
        Calculates the stress field of the laminate
        
        Parameters
        ----------
        ABD: np.ndarray (6x6)
            ABD matrix of the laminate
            
        N: np.ndarray (6,)
            Load vector, [Nxx, Nyy, Nxy, Mxx, Myy, Mxy]
            
        stacking: List[float]
            List of ply angles in degrees
            
        z_positions: List[float]
            List of z-positions for each ply
            
        Q_layup: List[np.ndarray]
            List of Q matrices (stiffness matrices) for each ply
            
        Returns
        -------
        stress_field: pd.DataFrame
            Stress field of the laminate, ply by ply in plate direction and material direction
        '''
        n_ply = len(stacking)
        epsilon0 = LaminateStrength.get_epsilon0(ABD, N)
        epsilon_plies = LaminateStrength.get_epsilon_plies(epsilon0, n_ply, z_positions)
        stress_plies = LaminateStrength.get_stress_plies(Q_layup, epsilon_plies)
        stress_plies_123 = LaminateStrength.get_stress_plies_123(stacking, stress_plies)
        
        cur_ply = 1
        data = {}
        data['ply'] = []
        data['position'] = []
        data['angle'] = []
        data['sigmax'] = []
        data['sigmay'] = []
        data['tauxy'] = []
        data['sigma1'] = []
        data['sigma2'] = []
        data['tau12'] = []
        for sigma, sigma_123, theta in zip(stress_plies, stress_plies_123, stacking):
            stress_top, stress_bot = sigma
            sigma_top_123, sigma_bot_123 = sigma_123

            data['ply'].append(cur_ply)
            data['ply'].append(cur_ply)
            data['position'].append('top')
            data['position'].append('bot')
            data['angle'].append(theta)
            data['angle'].append(theta)
            data['sigmax'].append(stress_top[0]) #plate direction
            data['sigmax'].append(stress_bot[0])
            data['sigmay'].append(stress_top[1])
            data['sigmay'].append(stress_bot[1])
            data['tauxy'].append(stress_top[2])
            data['tauxy'].append(stress_bot[2])
            data['sigma1'].append(sigma_top_123[0]) #material direction
            data['sigma1'].append(sigma_bot_123[0])
            data['sigma2'].append(sigma_top_123[1])
            data['sigma2'].append(sigma_bot_123[1])
            data['tau12'].append(sigma_top_123[2])
            data['tau12'].append(sigma_bot_123[2])
            cur_ply += 1
        pd.set_option('display.precision', 2)
        return pd.DataFrame(data)
    
    
    def epsilon0(self) -> np.ndarray:
        '''
        Calculates the mid-plane strains of the laminate
        
        Returns
        -------
        epsilon0 : numpy ndarray (6,)
            mid-plane strains, [epsilon_x0, epsilon_y0, epsilon_xy0, kappa_x0, kappa_y0, kappa_xy0]
        '''
        ABD = self.dproperty.ABD
        N = np.array([self.Nxx, self.Nyy, self.Nxy, self.Mxx, self.Myy, self.Mxy])
        return self.get_epsilon0(ABD, N)
    
    def calculate_strain(self) -> pd.DataFrame:
        '''
        Calculates strain ply by at laminate direction and material direction.

        Returns
        -------
        strains : pd.Dataframe
            ply by ply strains in plate direction and material direction       

        Note
        ----
        The sequence of the DataFrame starts from the TOP OF THE LAYUP to the BOTTOM OF THE LAYUP, which is the reverse of the definition order.
        When defining the laminate, the first element of the list corresponds to the bottom-most layer. This is especially important for non-symmetric laminates.
        '''
        stacking = self.dproperty.stacking
        z_positions = self.dproperty.z_position
        ABD = self.dproperty.ABD
        N = np.array([self.Nxx, self.Nyy, self.Nxy, self.Mxx, self.Myy, self.Mxy])
        
        data = self.get_strain_field(ABD, N, stacking, z_positions)
        return data

    def calculate_stress(self) -> pd.DataFrame:
        '''
        Calculates stress ply by at laminate direction and material direction.

        Returns
        -------
        stress : pd.Dataframe
            ply by ply stress in plate direction and material direction       

        Note
        ----
        The sequence of the DataFrame starts from the TOP OF THE LAYUP to the BOTTOM OF THE LAYUP, which is the reverse of the definition order.
        When defining the laminate, the first element of the list corresponds to the bottom-most layer. This is especially important for non-symmetric laminates.
        '''
        Q_layup = self.dproperty.Q_layup
        stacking = self.dproperty.stacking
        z_positions = self.dproperty.z_position
        ABD = self.dproperty.ABD
        N = np.array([self.Nxx, self.Nyy, self.Nxy, self.Mxx, self.Myy, self.Mxy])

        data = self.get_stress_field(ABD, N, stacking, z_positions, Q_layup)
        return data
