'''
Test of example
===============

Test the example
'''

from composipy import OrthotropicMaterial, LaminateProperty, LaminateStrength


def test_LaminateProperty_example():
    
    E1 = 60800
    E2 = 58250
    v12 = 0.07
    G12 = 4550
    t = 0.21

    mat_1 = OrthotropicMaterial(E1, E2, v12, G12, t)
    stacking = [-45, 45, 90, 0, 0, 0, 0, 90, 45, -45]
    laminate1 = LaminateProperty(stacking, mat_1)
    
    print(laminate1.ABD) # prints the ABD matrix as a np.ndarray
    print(laminate1.xiA) # prints lamination parameters of extension as a np.ndarray
    print(laminate1.xiD) # prints lamination parameters of bending as a np.ndarray


def test_LaminateStrength_example():

    E1 = 60800
    E2 = 58250
    v12 = 0.07
    G12 = 4550
    t = 0.21

    mat_1 = OrthotropicMaterial(E1, E2, v12, G12, t)
    stacking = [-45, 45, 90, 0, 0, 0, 0, 90, 45, -45]
    laminate1 = LaminateProperty(stacking, mat_1)

    laminate_strength = LaminateStrength(laminate1, Nxx=100, Mxx=10)
    laminate_strength.epsilon0() #strains at the midplane
    laminate_strength.calculate_strain() #strain ply by ply
    laminate_strength.calculate_stress() #stress ply by ply
    
    
def test_series_run_examples():
    
    E1 = 60800
    E2 = 58250
    v12 = 0.07
    G12 = 4550
    t = 0.21

    mat_1 = OrthotropicMaterial(E1, E2, v12, G12, t)
    stacking = [-45, 45, 90, 0, 0, 0, 0, 90, 45, -45]
    laminate1 = LaminateProperty(stacking, mat_1)
    
    print(laminate1.ABD) # prints the ABD matrix as a np.ndarray
    print(laminate1.xiA) # prints lamination parameters of extension as a np.ndarray
    print(laminate1.xiD) # prints lamination parameters of bending as a np.ndarray

    laminate_strength = LaminateStrength(laminate1, Nxx=100, Mxx=10)
    laminate_strength.epsilon0() #strains at the midplane
    laminate_strength.calculate_strain() #strain ply by ply
    laminate_strength.calculate_stress() #stress ply by ply

if __name__ == "__main__":
    
    test_LaminateProperty_example()
    test_LaminateStrength_example()
    
    test_series_run_examples()

