import numpy as np
from numpy import sin, cos, arctan, arcsin
from PIL import Image
from ast2000tools.space_mission import SpaceMission
import os

def GenPicture(array, filename = None) -> Image:
    '''
    Generates picture from array
    '''
    newImage = Image.fromarray(array)
    if filename:
        newImage.save(filename)
    return newImage
    
def Reverse_Stereographic_Projection(phi, theta, theta_0, phi_0):
    '''
    Converts Spherical to Cartesian coordinates \n
    Uses arrays or float of coordinates as input \n
    Returns X, Y
    '''
    kappa = 2/(1 + cos(theta_0)*cos(theta) + sin(theta_0)*sin(theta)*cos(phi - phi_0))
    X = kappa * sin(theta)*sin(phi - phi_0)
    Y = kappa * (sin(theta_0)*cos(theta) - cos(theta_0)*sin(theta)*cos(phi - phi_0))
    return X,Y 

def Stereographic_Projection(X,Y,phi_0,theta_0):
    '''
    Converts Cartesian to Spherical coordinates \n
    Uses arrays or float of coordinates as input \n
    Return phi, theta 
    '''
    rho = np.sqrt(X**2 + Y**2)
    beta = 2*arctan(rho/2)
    theta = theta_0  - arcsin(cos(beta)*cos(theta_0) + (Y/rho)*sin(beta)*sin(theta_0))
    phi = phi_0 + arctan((X*sin(beta)) / (rho*sin(theta_0)*cos(beta) - Y*cos(theta_0)*sin(beta)))
    return phi, theta

def CoordinateRange(alpha_phi, alpha_theta) -> int:
    '''
    Finds the full range of cartesian coordinates \n
    Returns X_max, X_min, Y_max, Y_min
    '''
    X_max =  (2*sin(alpha_phi/2))/(1 + cos(alpha_phi/2))
    X_min = -(2*sin(alpha_phi/2))/(1 + cos(alpha_phi/2))
    Y_max =  -(2*sin(alpha_theta/2))/(1 + cos(alpha_theta/2)) # Flipping the y-axis to avoid flipped image
    Y_min = (2*sin(alpha_theta/2))/(1 + cos(alpha_theta/2))
    
    return X_max, X_min, Y_max, Y_min

def find_angle(image_filename: str) -> int:
    '''
    Finds the angle (int) the spacecraft is pointing in using the image taken on board.\n
    Return the angle phi_0 in degrees \n
    If None is returned there was no angle found
    '''
    for phi_0 in range(360):
        image = Image.open(image_filename)
        filenumber = f'{phi_0}'.zfill(3)
        filename = f'Sky Images\sky_image_{filenumber}_degrees.png'
        possible_match = Image.open(os.path.join(filename))
        if image == possible_match:
            return phi_0
    


def generate_sky_image(phi: np.ndarray, theta:np.ndarray, height: int, width: int) -> Image:
    '''
    IMPORTANT: phi and theta must be in radians \n
    Converts the phi, theta-grid into pixels then map the corresponding RGB value to the pixels and creating and returning a new image
    '''
    himmelkule = np.load("himmelkule.npy")
    pixels = np.zeros((height, width, 3), dtype="uint8")
    
    for i in range(height):
        for j in range(width):
            pixel_indexes = SpaceMission.get_sky_image_pixel(theta[i,j], phi[i,j])
            pixels[i,j,0] = himmelkule[pixel_indexes][2] 
            pixels[i,j,1] = himmelkule[pixel_indexes][3]
            pixels[i,j,2] = himmelkule[pixel_indexes][4]
    newImage = Image.fromarray(pixels)
    return newImage

if __name__ == "__main__":
    sample = Image.open('sample0000.png')
    pixels = np.array(sample)
    height, width = pixels.shape[0], pixels.shape[1]
    print(f'sample2000.png is {height} X {width} pixels')
    
    alpha_theta = alpha_phi = 70*np.pi/180 # FOV in radians
    phi_0, theta_0 = 0, np.pi/2 # Center of camera position in radians.
    X_max, X_min, Y_max, Y_min = CoordinateRange(alpha_phi, alpha_theta)
    print(f'The full range of coordinates is from {X_min} to {X_max} for X and {Y_min} to {Y_max} for Y')
    
    x, y = np.linspace(X_min, X_max, width), np.linspace(Y_min, Y_max, height)
    X, Y = np.meshgrid(x,y)
    phi, theta = Stereographic_Projection(X,Y,phi_0,theta_0)
    newImage = generate_sky_image(phi, theta, height, width)
    newImage.save('Test_Image.png')
    
    image = Image.open('sample0200.png')
    possible_match = Image.open(os.path.join('Sky Images/sky_image_020_degrees.png'))
    
    print(image == possible_match)
    # print(f'The angle phi_0 of the test image is {find_angle("Test_Image.png")} degrees')
    file_numbers = ['0000', '0200', '1400', '1900']

    
    for n in file_numbers:
        filename = f'sample{n}.png'
        print(f'The angle phi_0 of {filename} is {find_angle(filename)} degrees ')
    
    # Code to generate all possible picture from all values of theta_0
    '''   
    for phi_0 in range(360):
        filenumber = f'{phi_0}'.zfill(3)
        filename = f'sky_image_{filenumber}_degrees.png'
        phi_0 *= np.pi/180 # Converting to radians
        phi, theta = Stereographic_Projection(X,Y,phi_0,theta_0)    
        image = generate_sky_image(phi, theta, height, width)
        print(filename)
        image.save(os.path.join('Sky Images', filename))
    '''