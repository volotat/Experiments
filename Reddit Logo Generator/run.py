from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import math
import random
import numpy as np

from os import listdir
from os.path import isfile, join

# Static parameters (do not change that)
LATENT_SPACE = 10
CHANNELS = 4

out_size = 60
out_half_size = math.floor(out_size / 2)

# Optional parameters
GEN_AMOUNT = 100 #amount of images to generate
BIG_IMAGE = True #if true all images would be collected in one big image 


# All probabilities are in fact weights, so they might be higher than one and would be calculated with respect to others 

# background color
BG_B_PROB = 0.7     #probability of bluish background
BG_W_PROB = 0.25    #probability of whitish background
BG_T_PROB = 0.05    #probability of transparent background

# reddit text color
TXT_B_PROB = 0.8    #probability of blackish text 
TXT_C_PROB = 0.2    #probability of colored text 

# reddit avatar color
AV_W_PROB = 0.9     #probability of whitish avatar
AV_C_PROB = 0.1     #probability of colored avatar 

# avatar position
POS_L_ROPB = 0.95   #probability of appearing on the left side
POS_R_ROPB = 0.05   #probability of appearing on the right side



def create_grid(half_size):
    X,Y = np.mgrid[-half_size:half_size,-half_size:half_size] + 0.5
    grid = np.vstack((X.flatten(), Y.flatten())).T / half_size
    return grid    
    
def get_background_color():
    col = np.ones(CHANNELS) * 255
    
    x = np.random.normal(np.array([206., 227., 248., 255.]),3, CHANNELS)
    y = np.random.normal(np.array([255., 255., 255., 255.]),5, CHANNELS)
    z = np.random.normal(np.array([206., 227., 248., 0.  ]),2, CHANNELS)
    col = random.choices([x,y,z], [BG_B_PROB, BG_W_PROB, BG_T_PROB], k = 1)[0] 
    col = np.uint8(np.clip(col, 0, 255))
    return col

def get_avatar_color():
    col = np.ones(CHANNELS) * 255
    
    x = np.random.normal(np.array([255., 255., 255.]),0.1, 3)
    y = np.random.normal(np.array([200., 255., 200.]),50, 3)
    col[:3] = random.choices([x,y], [AV_W_PROB, AV_C_PROB], k = 1)[0]
    col = np.clip(col, 0, 255)
    return col
    
def get_text_color():
    col = np.zeros(CHANNELS)
    
    x = np.random.normal(np.array([0., 0., 0.]),2,  3)
    y = np.random.normal(np.array([0., 0., 0.]),100, 3)
    col[:3] = random.choices([x,y], [TXT_B_PROB, TXT_C_PROB], k = 1)[0]
    return col
    
# Load avatar generator    
generator = load_model('generator.h5')
grid = create_grid(out_half_size)

# Create a list of reedit text images, names of images suppose to be integer and will be used as weights
DIR = 'fonts/'
font_files = [f for f in listdir(DIR) if isfile(join(DIR, f))]
font_files = [f[:-4] for f in font_files]
font_weights = list(map(float, font_files))
font_weights = [w / 1000 for w in font_weights]

if BIG_IMAGE: 
    depth = GEN_AMOUNT // 10
    big_img = Image.new("RGBA", (125 * 10 + 5 , 45 * depth + 5))
    
for i in range(GEN_AMOUNT):
    # Create new image 
    bg_color = get_background_color()
    final_img = Image.new("RGBA", (120, 40), color = tuple(bg_color))

    # Generate avatar image
    noise = np.random.normal(0,1,(LATENT_SPACE))
    noise = noise.reshape(1, LATENT_SPACE)
    noise = np.repeat(noise, grid.shape[0], axis=0)
    
    predicted = generator.predict([grid, noise])[0] * 255.
    predicted[:,3] = predicted[:,3] * 1.5 - 90 # clean up alpha channel inaccuracy
    
    col = get_avatar_color()
    predicted = predicted * (col / 255.).reshape(1,4) 
    predicted = np.clip(predicted, 0, 255)  
    predicted = (predicted).astype(np.uint8).reshape(out_size, out_size, 4) 

    avatar_img = Image.fromarray(predicted)
    avatar_img = avatar_img.resize((44, 44), resample = Image.BILINEAR)

    # Load text image
    file = random.choices(font_files, font_weights, k = 1)[0] 
    text = Image.open('fonts/' + file+'.png').convert('RGBA')
    text_arr = np.array(text)
    
    col = get_text_color()
    text_arr = text_arr + col
    text_arr = np.clip(text_arr, 0, 255)
    text = Image.fromarray(np.uint8(text_arr))
    
    # Choose avatar position
    x = np.random.normal(0,0.5, 1)
    y = np.random.normal(80,0.5, 1)
    x_off = random.choices([x,y], [POS_L_ROPB, POS_R_ROPB], k = 1)[0]
    x_off = np.clip(x_off,0, 80).astype(np.int)
    
    # Combine images
    if np.random.rand()<0.8:
        final_img.paste(text, (0 - int(x_off / 2.3), 0), text)
        final_img.paste(avatar_img, (-2 + x_off, -1), avatar_img)
    else:
        final_img.paste(avatar_img, (-2 + x_off, -1), avatar_img)
        final_img.paste(text, (0 - int(x_off / 2.3), 0), text)
    
    # Save final image
    final_img.save('samples/%d.png'%(i+1))
    print('Image: ', i+1)
    
    if BIG_IMAGE:
        # Add to big image if told so
        x = 5 + i%10 * (120 + 5)
        y = 5 + i//10 * (40 + 5)
        big_img.paste(final_img, (x, y), final_img)
    
if BIG_IMAGE: big_img.save('big_img.png')