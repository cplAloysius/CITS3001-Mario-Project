from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np


# other constants (don't change these)
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256

# stage 1 thresholds
MARIO_THRESHOLD = 0.8
GOOMBA_THRESHOLD = 0.7
KOOPA_THRESHOLD = 0.9
BLOCK_THRESHOLD = 0.9
PIPE_THRESHOLD = 0.9

#stage 2 thresholds
MARIO_THRESHOLD2 = 0.54
GOOMBA_THRESHOLD2 = 0.7
KOOPA_THRESHOLD2 = 0.9
BLOCK_THRESHOLD2 = 0.8
PIPE_THRESHOLD2 = 0.7


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
mario_template = [cv.imread('templates/marioA.png', cv.IMREAD_COLOR), 
                  cv.imread('templates/marioB.png', cv.IMREAD_COLOR),
                  cv.imread('templates/marioC.png', cv.IMREAD_COLOR),
                  cv.imread('templates/marioD.png', cv.IMREAD_COLOR),
                  cv.imread('templates/marioE.png', cv.IMREAD_COLOR),
                  cv.imread('templates/marioF.png', cv.IMREAD_COLOR),
                  cv.imread('templates/marioG.png', cv.IMREAD_COLOR)]

goomba_template = [cv.imread('templates/goomba.png', cv.IMREAD_COLOR)]

koopa_template = [cv.imread('templates/koopaA.png', cv.IMREAD_COLOR),
                  cv.imread('templates/koopaB.png', cv.IMREAD_COLOR),
                  cv.imread('templates/koopaC.png', cv.IMREAD_COLOR),
                  cv.imread('templates/koopaD.png', cv.IMREAD_COLOR)]

ground_template = [cv.imread('templates/block2.png', cv.IMREAD_COLOR)]

stair_template = [cv.imread('templates/block4.png', cv.IMREAD_COLOR)]

pipe_template = [cv.imread('templates/pipe_upper_section.png', cv.IMREAD_COLOR)]

brick_template = [cv.imread('templates/block1.png', cv.IMREAD_COLOR)]

# create masks to ignore sky when matching using the BGR templates
def sky_mask2(templates):
    masks = []
    for template in templates:
        mask = np.uint8(np.where(np.all(template == MASK_COLOUR, axis=2), 0, 1))
        num_pixels = template.shape[0]*template.shape[1]
        if num_pixels - np.sum(mask) < 10:
            mask = None # this is important for avoiding a problem where some things match everything
        masks.append(mask)
    return masks

# flip templates because mario and some enemies sometimes face in opposite directions
def flip_templates(templates):
    flipped_templates = []
    for template in templates:
        flipped_template = cv.flip(template, 1)
        flipped_templates.append(flipped_template)
    return flipped_templates

# convert BGR to gray for easier matching in different stages
def to_gray(templates):
    new_templates = []
    for template in templates:
        new_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        new_templates.append(new_template)
    return new_templates

mario_template += flip_templates(mario_template)

mario_masks = sky_mask2(mario_template)
goomba_masks = sky_mask2(goomba_template)
koopa_masks = sky_mask2(koopa_template)
ground_masks = sky_mask2(ground_template)
stair_masks = sky_mask2(stair_template)
pipe_masks = sky_mask2(pipe_template)
brick_masks = sky_mask2(brick_template)

mario_template = to_gray(mario_template)
goomba_template = to_gray(goomba_template)
koopa_template = to_gray(koopa_template)
ground_template = to_gray(ground_template)
stair_template = to_gray(stair_template)
pipe_template = to_gray(pipe_template)
brick_template = to_gray(brick_template)

################################################################################
# LOCATING OBJECTS

# selecting appropriate thresholds to use for different stages
def get_thresh(info):
    mario_thresh = MARIO_THRESHOLD
    goomba_thresh = GOOMBA_THRESHOLD
    koopa_thresh = KOOPA_THRESHOLD
    block_thresh = BLOCK_THRESHOLD
    pipe_thresh = PIPE_THRESHOLD
    
    if info["world"] == 1 and info["stage"] == 2:
        mario_thresh = MARIO_THRESHOLD2
        goomba_thresh = GOOMBA_THRESHOLD2
        koopa_thresh = KOOPA_THRESHOLD2
        block_thresh = BLOCK_THRESHOLD2
        pipe_thresh = PIPE_THRESHOLD2
    
    return mario_thresh, goomba_thresh, koopa_thresh, block_thresh, pipe_thresh

# converts all black pixels on screen to blue because I found that its hard to match with stage 2's black background
def black_sky(screen):
    black_mask = cv.inRange(screen, (0,0,0), (0,0,0))
    replacement_colour = (252, 136, 104)
    replacement_image = np.full_like(screen, replacement_colour)
    result = cv.bitwise_and(screen, screen, mask=cv.bitwise_not(black_mask))
    result = cv.add(result, replacement_image)
    return result

def _locate_objects(screen, templates, threshold, subview=None, masks=None, stop_early=False):
    locations = []
    i=0

    if subview:
        x_start, x_end, y_start, y_end = subview
        screen = screen[y_start:y_end, x_start:x_end]
    
    for template in templates:
        if screen.shape[0] < template.shape[0] or screen.shape[1] < template.shape[1]:
            continue
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
        
        if stop_early and locations:
            break

    return locations

def locate_objects(screen, mario_x, mario_y):
    mario_thresh, goomba_thresh, koopa_thresh, block_thresh, pipe_thresh = get_thresh(info)
    screen = cv.cvtColor(screen, cv.COLOR_RGB2BGR)

    if info["world"] == 1 and info["stage"] == 2:
        screen = black_sky(screen)

    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    mario_positions = _locate_objects(screen, mario_template, mario_thresh, masks=mario_masks, stop_early=True)

    if mario_x != 0:
        x_start, x_end = mario_x - 10, mario_x + 50
    else:
        x_start, x_end = screen.shape[1] // 2, 60 + screen.shape[1] // 2

    if mario_y != 0:
        y_start, y_end = mario_y - 70, screen.shape[0]
    else:
        y_start, y_end = 120, screen.shape[0]

    subview = (x_start, x_end, y_start, y_end)
    subview2 = (x_start, x_end+20, y_start-20, y_end)

    return {
        "mario": mario_positions,
        "goomba": _locate_objects(screen, goomba_template, goomba_thresh, subview2, goomba_masks),
        "koopa": _locate_objects(screen, koopa_template, koopa_thresh, subview, koopa_masks),
        "ground": _locate_objects(screen, ground_template, block_thresh, subview, ground_masks),
        "stair": _locate_objects(screen, stair_template, block_thresh, subview, stair_masks),
        "pipe": _locate_objects(screen, pipe_template, pipe_thresh, subview, pipe_masks),
        "brick": _locate_objects(screen, brick_template, block_thresh, subview, brick_masks),
    }

# used for debugging, shows bounding box of what the agent detects
def draw_borders(screen, object_locations):
    colours = {
        "mario": (0, 0, 255),
        "goomba": (0, 255, 0),
        "koopa": (255, 255, 0),
        "ground": (255, 0, 0),
        "stair": (255, 100, 0),
        "pipe": (255, 0, 255),
        "brick": (0, 255, 255)
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
            elif template_name == "brick_template":
                template = brick_template[0]
            else:
                continue

            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

            cv.rectangle(screen, top_left, bottom_right, colours.get(obj_type, (0, 0, 255)), 2)

    return screen

################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION

def choose_action(screen):
    global mario_last_central_x # x position of mario on screen taking into account the width of mario's body
    global mario_last_x # previous x position of mario in the world
    global mario_last_y # previous y position of mario
    global mario_x_timer # keeps track of how long mario stays in the same spot in the world

    global last_ground_block_positions
    global last_ground_block_positions_timer
    
    object_locations = locate_objects(screen, mario_last_central_x, mario_last_y)
    mario_locations = object_locations["mario"]
    goomba_locations = object_locations["goomba"]
    koopa_locations = object_locations["koopa"]
    ground_locations = object_locations["ground"]
    stair_locations = object_locations["stair"]
    pipe_locations = object_locations["pipe"]
    brick_locations = object_locations["brick"]

    action = 1 # default action ["right"]

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

    # to jump over an obstacle where there is a gap and a brick wall
    #  on the other side of the gap such as the one in stage 2 
    if len(ground_locations) < 4:
        if brick_locations:
            for brick_x, brick_y in brick_locations:
                if (brick_x - mario_central_x) in range (20, 40):
                    action = 4

    if len(ground_locations) < 2:
        action = 4
    
    # to run left instead of jumping when there is an enemy and a 
    # brick wall above mario (this happens in stage 2)
    elif brick_locations and (koopa_locations or goomba_locations):
        for brick_x, brick_y in brick_locations:
            if (mario_y - brick_y) <= 20 and (brick_x - mario_central_x) < 30:
                if goomba_locations:
                    for goomba_location in goomba_locations:
                        goomba_distance = goomba_location[0] - mario_central_x
                        if goomba_distance in range(0, 100):
                            action = 6
                            break
                if koopa_locations:
                    for koopa_location in koopa_locations:
                        koopa_distance = koopa_location[0] - mario_central_x
                        if koopa_distance in range(0, 100):
                            action = 6
                            break
            else:
                if goomba_locations:
                    for goomba_x, goomba_y in goomba_locations:
                        distance = goomba_x - mario_central_x
                        goomba_higher = mario_y - goomba_y
                        mario_higher = goomba_y - mario_y
                        if goomba_higher in range (60, 80):
                            action = 4
                            break
                        if mario_higher > 50 and distance <= 40:
                            action = 4
                        if 0 < distance <= 25:
                            action = 4
                            break
                if koopa_locations:
                    for koopa_location in koopa_locations:
                        koopa_distance = koopa_location[0] - mario_central_x
                        if 0 < koopa_distance <= 25:
                            action = 4
                            break

    elif goomba_locations:
        for goomba_x, goomba_y in goomba_locations:
            distance = goomba_x - mario_central_x
            goomba_higher = mario_y - goomba_y
            mario_higher = goomba_y - mario_y
            
            if goomba_higher > 50:
                action = 6
                break
            if mario_higher > 50 and distance <= 40:
                action = 4

            if 0 < distance <= 20:
                action = 4
                break
    
    elif koopa_locations:
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
    
    print(mario_last_x, ":", mario_world_x, ":", mario_x_timer)

    last_ground_block_positions = ground_locations
    mario_last_x = mario_world_x
    mario_last_y = mario_y
    mario_last_central_x = mario_central_x

    return action, object_locations

################################################################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

screen = None
done = True

env.reset()

for step in range(100000):
    if screen is None:
        action = env.action_space.sample()
        object_locations = {}
    else:
        action, object_locations = choose_action(screen) 

        obs_border = draw_borders(screen.copy(), object_locations)
        cv.imshow("bounding box obs", cv.cvtColor(obs_border, cv.COLOR_RGB2BGR))
        cv.waitKey(1)

    screen, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        screen, info = env.reset()
env.close()