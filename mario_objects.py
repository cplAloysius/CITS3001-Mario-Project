from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np


# other constants (don't change these)
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MARIO_THRESHOLD = 0.8
GOOMBA_THRESHOLD = 0.7
KOOPA_THRESHOLD = 0.9
BLOCK_THRESHOLD = 0.9
PIPE_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

last_ground_block_positions = []
last_ground_block_positions_timer = 0
mario_last_x = 0
mario_last_central_x = 0
mario_last_y = 0
mario_x_timer = 0

#load templates in BGR format
mario_template = [cv.imread('templates/marioA.png', cv.IMREAD_GRAYSCALE), 
                   cv.imread('templates/marioB.png', cv.IMREAD_GRAYSCALE),
                   cv.imread('templates/marioC.png', cv.IMREAD_GRAYSCALE),
                   cv.imread('templates/marioD.png', cv.IMREAD_GRAYSCALE),
                   cv.imread('templates/marioE.png', cv.IMREAD_GRAYSCALE),
                   cv.imread('templates/marioF.png', cv.IMREAD_GRAYSCALE),
                   cv.imread('templates/marioG.png', cv.IMREAD_GRAYSCALE)]

goomba_template = [cv.imread('templates/goomba.png', cv.IMREAD_GRAYSCALE)]

koopa_template = [cv.imread('templates/koopaA.png', cv.IMREAD_GRAYSCALE),
                   cv.imread('templates/koopaB.png', cv.IMREAD_GRAYSCALE),
                   cv.imread('templates/koopaC.png', cv.IMREAD_GRAYSCALE),
                   cv.imread('templates/koopaD.png', cv.IMREAD_GRAYSCALE)]
 
ground_template = [cv.imread('templates/block2.png', cv.IMREAD_GRAYSCALE)]

stair_template = [cv.imread('templates/block4.png', cv.IMREAD_GRAYSCALE)]

pipe_template = [cv.imread('templates/pipe_upper_section.png', cv.IMREAD_GRAYSCALE)]


mario_template2 = [cv.imread('templates/marioA.png', cv.IMREAD_COLOR), 
                  cv.imread('templates/marioB.png', cv.IMREAD_COLOR),
                  cv.imread('templates/marioC.png', cv.IMREAD_COLOR),
                  cv.imread('templates/marioD.png', cv.IMREAD_COLOR),
                  cv.imread('templates/marioE.png', cv.IMREAD_COLOR),
                  cv.imread('templates/marioF.png', cv.IMREAD_COLOR),
                  cv.imread('templates/marioG.png', cv.IMREAD_COLOR)]

goomba_template2 = [cv.imread('templates/goomba.png', cv.IMREAD_COLOR)]

koopa_template2 = [cv.imread('templates/koopaA.png', cv.IMREAD_COLOR),
                  cv.imread('templates/koopaB.png', cv.IMREAD_COLOR),
                  cv.imread('templates/koopaC.png', cv.IMREAD_COLOR),
                  cv.imread('templates/koopaD.png', cv.IMREAD_COLOR)]

ground_template2 = [cv.imread('templates/block2.png', cv.IMREAD_COLOR)]

stair_template2 = [cv.imread('templates/block4.png', cv.IMREAD_COLOR)]

pipe_template2 = [cv.imread('templates/pipe_upper_section.png', cv.IMREAD_COLOR)]

def sky_mask2(templates):
    masks = []
    for template in templates:
        mask = np.uint8(np.where(np.all(template == MASK_COLOUR, axis=2), 0, 1))
        num_pixels = template.shape[0]*template.shape[1]
        if num_pixels - np.sum(mask) < 10:
            mask = None # this is important for avoiding a problem where some things match everything
        masks.append(mask)
    return masks

mario_masks = sky_mask2(mario_template2)
goomba_masks = sky_mask2(goomba_template2)
koopa_masks = sky_mask2(koopa_template2)
ground_masks = sky_mask2(ground_template2)
stair_masks = sky_mask2(stair_template2)
pipe_masks = sky_mask2(pipe_template2)

################################################################################
# LOCATING OBJECTS

def _locate_objects(screen, templates, threshold, subview=None, masks=None):
    locations = []
    i=0

    if subview:
        x_start, x_end, y_start, y_end = subview
        screen = screen[y_start:y_end, x_start:x_end]
    
    for template in templates:
        if screen.shape[0] < template.shape[0] or screen.shape[1] < template.shape[1]:
            continue  # Skip this template
        elif masks[i] is not None and (masks[i].shape[0] != template.shape[0] or masks[i].shape[1] != template.shape[1]):
            continue
        result = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=masks[i])
        i += 1
        loc = np.where(result >= threshold)
    
        if loc[0].size:
            if subview:
                new_loc = [x + x_start for x in loc[1]], [y + y_start for y in loc[0]]
                locations.extend(zip(*new_loc))
            else:
                locations.extend(zip(*loc[::-1]))

    return locations

def locate_objects(screen, mario_x, mario_y):
    screen = cv.cvtColor(screen, cv.COLOR_RGB2GRAY)

    mario_positions = _locate_objects(screen, mario_template, MARIO_THRESHOLD, masks=mario_masks)

    if mario_x != 0:
        x_start, x_end = mario_x - 10, mario_x + 50
    else:
        x_start, x_end = screen.shape[1] // 2, 60 + screen.shape[1] // 2

    if mario_y != 0:
        y_start, y_end = mario_y - 70, screen.shape[0]
    else:
        y_start, y_end = 120, screen.shape[0]

    subview = (x_start, x_end, y_start, y_end)

    return {
        "mario": mario_positions,
        "goomba": _locate_objects(screen, goomba_template, GOOMBA_THRESHOLD, subview, goomba_masks),
        "koopa": _locate_objects(screen, koopa_template, KOOPA_THRESHOLD, subview, koopa_masks),
        "ground": _locate_objects(screen, ground_template, BLOCK_THRESHOLD, subview, ground_masks),
        "stair": _locate_objects(screen, stair_template, BLOCK_THRESHOLD, subview, stair_masks),
        "pipe": _locate_objects(screen, pipe_template, PIPE_THRESHOLD, subview, pipe_masks)
    }

def draw_borders(screen, object_locations):
    colours = {
        "mario": (0, 0, 255),
        "goomba": (0, 255, 0),
        "koopa": (255, 255, 0),
        "ground": (255, 0, 0),
        "stair": (255, 0, 0),
        "pipe": (255, 0, 255),
    }

    for obj_type, positions in object_locations.items():
        for position in positions:
            top_left = position
            template_name = f"{obj_type}_template"
            if template_name == "mario_template":
                template = mario_template[0]
            elif template_name == "goomba_template":
                template = goomba_template[0]
            elif template_name == "koopa_template":
                template = koopa_template[0]
            elif template_name == "ground_template":
                template = ground_template[0]
            elif template_name == "stair_template":
                template = stair_template[0]
            elif template_name == "pipe_template":
                template = pipe_template[0]
            else:
                continue

            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

            cv.rectangle(screen, top_left, bottom_right, colours.get(obj_type, (0, 0, 255)), 2)

    return screen

################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION

def choose_action(screen, info):
    global mario_last_central_x
    global mario_last_x
    global mario_last_y
    global mario_x_timer
    global last_ground_block_positions
    global last_ground_block_positions_timer
    
    object_locations = locate_objects(screen, mario_last_central_x, mario_last_y)
    mario_locations = object_locations["mario"]
    goomba_locations = object_locations["goomba"]
    koopa_locations = object_locations["koopa"]
    ground_locations = object_locations["ground"]
    stair_locations = object_locations["stair"]
    pipe_locations = object_locations["pipe"]

    action = 1

    if mario_locations:
        mario_central_x = mario_locations[0][0] + mario_template[0].shape[1] // 2
        mario_y = mario_locations[0][1]
        if info:
            mario_world_x = info["x_pos"]
        else:
            mario_world_x = 0
    else:
        mario_central_x = 0
        mario_world_x = 0
        mario_y = 0

    if goomba_locations:
        for goomba_location in goomba_locations:
            distance = goomba_location[0] - mario_central_x
            if 0 < distance <= 20:
                action = 4
                break
    
    if koopa_locations:
        for koopa_location in koopa_locations:
            distance = koopa_location[0] - mario_central_x
            if 0 < distance <= 25:
                action = 4
                break
    
    if pipe_locations:
        for pipe_location in pipe_locations:
            distance = pipe_location[0] - mario_central_x
            if 0 < distance <= 45:
                action = 4
                break

    if stair_locations:
        for stair_location in stair_locations:
            distance = stair_location[0] - mario_central_x
            if distance <= 20:
                action = 4
                break
    
    if len(ground_locations) < 4:
        action = 4

    #if mario at max height
    if mario_locations: 
        if mario_locations[0][1] < 126:
            action = 3

        elif mario_locations[0][1] > 126 and stair_locations:
            action = 4

    if mario_last_x == mario_world_x:
        mario_x_timer += 1
    else:
        mario_x_timer = 0
    
    if mario_x_timer in range(20, 25):
        action = 4
    if mario_x_timer in range(25, 35):
        action = 6
    


    #if step_block_positions and not ground_block_positions:
        #if len(step_block_positions) <= 2:
            #action = 4
    
    #if step_block_positions and not ground_block_positions:
        #action = 4
    #if len(step_block_positions) == 6 and len(ground_block_positions) == 0:
        #action = 3 
        

    # Get unstuck
    # if last_ground_block_positions == ground_locations:
    #     last_ground_block_positions_timer += 1
    # else:
    #     last_ground_block_positions_timer = 0

    # if last_ground_block_positions_timer in range(50, 55):
    #     action = 6

    # if last_ground_block_positions_timer in range(55, 65):
    #     action = 3

    # if last_ground_block_positions_timer in range(65, 75):
    #     action = 2

    # if last_ground_block_positions_timer > 75:
    #     last_ground_block_positions_timer = 0

    print(mario_last_x, ":", mario_world_x, ":", mario_x_timer)

    last_ground_block_positions = ground_locations
    mario_last_x = mario_world_x
    mario_last_y = mario_y

    return action, object_locations

################################################################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

screen, info = env.reset()

for step in range(100000):
    action, object_locations = choose_action(screen, info) 

    obs_border = draw_borders(screen.copy(), object_locations)
    cv.imshow("Debug Observation", cv.cvtColor(obs_border, cv.COLOR_RGB2BGR))
    cv.waitKey(1)

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        screen, info = env.reset()
env.close()