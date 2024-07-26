import os
import time
import math
import random
import digits
import pip
import pickle
import copy
try:
    __import__('keyboard')
except ImportError:
    pip.main(['install', 'keyboard'])
import keyboard

do_while = True
td = None
print('Per favore massimizza la finestra del terminale.')
while do_while or td.lines < 23 or td.columns < 138:
    do_while = False
    td = os.get_terminal_size()

time_to_display = 0.5
mat = []
depth_mat = []
color_mat = []
prev_mat = []
prev_col_mat = []
polyhedra = []
dice_colors = []
individualized_dice = False
requestedDice = ''
positions = []
try:
    f = open('save.dice', 'rb')
    saved_data = pickle.load(f)
    d = saved_data['dice']
    for i in range(7):
        dice_colors.append(d[i])
    individualized_dice = saved_data['id']
    requestedDice = saved_data['req']
    f.close()
except:
    for i in range(7):
        dice_colors.append({'filling': {'r': 40, 'g': 120, 'b': 170}, 'edge': {'r': 230, 'g': 100, 'b': 100}, 'texture': {'r': 230, 'g': 100, 'b': 100}})

selector = 0
dice_col_ind = 0
min_frame_length = 0.01
td = os.get_terminal_size()
terminal_dims = [td.columns, td.lines*2+2]
pr_center = {'x':terminal_dims[0]/2, 'y':terminal_dims[1]/2}
lightSource = {'x':pr_center['x'], 'y':pr_center['y'], 'z':150}
for i in range(terminal_dims[1]//2):
    mat.append([])
    depth_mat.append([])
    color_mat.append([])
    prev_mat.append([])
    prev_col_mat.append([])
    for j in range(terminal_dims[0]):
        mat[i].append(' ')
        prev_mat[i].append(' ')
        depth_mat[i].append(-1000)
        color_mat[i].append({'fg':{'r':255, 'g': 255, 'b': 255}, 'bg':''})
        prev_col_mat[i].append({'fg':{'r':255, 'g': 255, 'b': 255}, 'bg':''})

def save():
    f = open('save.dice', 'wb')
    saved_data = {'dice':dice_colors, 'id':individualized_dice, 'req': requestedDice}
    pickle.dump(saved_data, f)
    f.close()

def perspectivize(face, verts):
    def calc_factor(z):
        return (z+150)/150
    pr_face = {'face':[]}
    for v in face['face']:
        factor = calc_factor(verts[v]['z'])
        pr_face['face'].append({'x':(verts[v]['x']-pr_center['x'])*factor+pr_center['x'], 'y':(verts[v]['y']-pr_center['y'])*factor+pr_center['y'], 'z':verts[v]['z']})
    if 'texture' in face:
        pr_face['texture'] = []
        for shape in face['texture']:
            pr_face['texture'].append([])
            for v in shape:
                factor = calc_factor(v['z'])
                pr_face['texture'][-1].append({'x':(v['x']-pr_center['x'])*factor+pr_center['x'], 'y':(v['y']-pr_center['y'])*factor+pr_center['y'], 'z':v['z']})
    return pr_face

def setTile(p, val, fg={'r':255, 'g': 255, 'b': 255}, bg=''):
    dot = '·.:'
    block = '█▀▄'
    try:
        z = mat[p['y']//2][p['x']]
    except:
        return
    if p['y']<0 or p['x']<0:
        return
    halfVal = val
    if p['y']%2==0:
        if val == '.':
            halfVal = '·'
        if val == '#':
            halfVal = '▀'
    else:
        if val == '.':
            halfVal = '.'
        if val == '#':
            halfVal = '▄'
    def blockSum(val1, val2, depth):
        r = ' '
        v = val1 if depth>0 else val2
        prev = val2 if depth>0 else val1
        if val1 == val2:
            return val1
        if val1 == '':
            return val2
        if val2 == '':
            return val1
        
        if val1 not in block and val1 not in dot and val1 != ' ' and val1 != '':
            r = val1
        elif val2 not in block and val2 not in dot and val2 != ' ' and val2 != '':
            r = val2
        elif val1 == ' ':
            r =  val2
        elif val2 == ' ':
            r = val1
        elif val1 in dot and val2 in dot:
            r = ':'
        elif val1 in block and val2 in block:
            r = '█'
        elif val1 in dot and val2 in block:
            r = val2
        elif val2 in dot and val1 in block:
            r = val1
        elif val1 == '█' or val2 == '█':
            r = '█'
        elif val1 == ':' or val2 == ':':
            r = ':'
        if (r=='█' and v!= ' ' and prev != '█')  or (bg == color_mat[p['y']//2][p['x']]['bg'] and r==':'):
            return r
        if depth != 0:
            return v
        return r
    new_bg = color_mat[p['y']//2][p['x']]['bg']
    try:
        new_bg = {'r':color_mat[p['y']//2][p['x']]['bg']['r'], 'g':color_mat[p['y']//2][p['x']]['bg']['g'], 'b':color_mat[p['y']//2][p['x']]['bg']['b']}
    except:
        pass
    new_fg = {'r':color_mat[p['y']//2][p['x']]['fg']['r'], 'g':color_mat[p['y']//2][p['x']]['fg']['g'], 'b':color_mat[p['y']//2][p['x']]['fg']['b']}
    mat[p['y']//2][p['x']] = blockSum(mat[p['y']//2][p['x']], halfVal, depth_mat[p['y']//2][p['x']] - p['z'])
    if depth_mat[p['y']//2][p['x']] <= p['z']:
        if bg != '':
            new_bg = bg
        if halfVal != ' ':
            new_fg = fg
        color_mat[p['y']//2][p['x']] = {'fg':new_fg, 'bg':new_bg}
    elif halfVal == ' ' and mat[p['y']//2][p['x']] in block and abs(depth_mat[p['y']//2][p['x']] - p['z'])<10:
        color_mat[p['y']//2][p['x']] = {'fg':new_fg, 'bg':bg}

    depth_mat[p['y']//2][p['x']] = max(p['z'], depth_mat[p['y']//2][p['x']])
    

def help_line(point1, point2):
    p1 = {}
    p2 = {}
    if point1['x']<point2['x']:
        p1=point1
        p2=point2
    else:
        p2=point1
        p1=point2
    if point1['x']==point2['x']:
        if point1['y']<point2['y']:
            p1=point1
            p2=point2
        else:
            p2=point1
            p1=point2
    to_set = []
    step = {'x' : 1, 'y' : 1}
    num_steps = round(max(abs(p2['x']-p1['x']), abs(p2['y']-p1['y'])))
    try:
        step = {'x' : (p2['x']-p1['x'])/num_steps, 'y' : (p2['y']-p1['y'])/num_steps}
    except:
        pass
    point = {'x' : round(p1['x']), 'y' : round(p1['y'])}
    z_factor = 1/(1 if num_steps==0 else num_steps)
    for i in range(num_steps+1):
        to_set.append({'x' : round(point['x']), 'y' : round(point['y']), 'z' : p1['z'] + (p2['z'] - p1['z'])*i*z_factor})
        point['x'] += step['x']
        point['y'] += step['y']
    return to_set

def line(p1, p2, contour='#', fg={'r':255, 'g':255, 'b':255}, bg=''):
    for v in help_line(p1, p2):
        setTile(v, contour, fg=fg, bg=bg)

def tuneBrightness(col, light):
    ret = ''
    if col=='' or col=='default':
        return col
    else:
        ret = {'r': col['r'], 'g': col['g'], 'b': col['b']}
    r = ret['r']/255
    g = ret['g']/255
    b = ret['b']/255  
    Cmax = max(r, g, b)
    Cmin = min(r, g, b)
    d = Cmax - Cmin
    hue = 0
    if d != 0:
        if Cmax == r:
            hue = ((g-b)/d)%6
        if Cmax == g:
            hue = ((b-r)/d)+2
        if Cmax == b:
            hue = ((r-g)/d)+4
    sat = 0 if Cmax == 0 else d/Cmax
    v = Cmax*light
    c = v*sat
    x = c*(1-abs((hue)%2-1))
    m=v-c
    y = []
    if hue<1:
        y = [c,x,0]
    elif hue<2:
        y = [x,c,0]
    elif hue<3:
        y = [0,c,x]
    elif hue<4:
        y = [0,x,c]
    elif hue<5:
        y = [x,0,c]
    else:
        y = [c,0,x]
    ret['r'] = round((y[0]+m)*255)
    ret['g'] = round((y[1]+m)*255)
    ret['b'] = round((y[2]+m)*255)
    if abs(ret['r']-12)<1 and abs(ret['g']-12)<1 and abs(ret['b']-12)<1:
        return 'default'
    return ret

def faceNorm(shape, face):
    
    v1 = shape['vertices'][shape['polyhedron'][face]['face'][0]]
    v2 = shape['vertices'][shape['polyhedron'][face]['face'][1]]
    v3 = shape['vertices'][shape['polyhedron'][face]['face'][2]]
    a = {'x':v2['x']-v1['x'], 'y':v2['y']-v1['y'], 'z':v2['z']-v1['z']}
    b = {'x':v3['x']-v1['x'], 'y':v3['y']-v1['y'], 'z':v3['z']-v1['z']}
    c = cross(a, b)

    return vect_scale(c, (1 if dot(c, vect_sub(v1,center3D(shape)))>0 else -1)/(norm_squared(c)**(0.5)))

def rot2D(face, center, r):
    angle = (r['x']**2+r['y']**2+r['z']**2)**(0.5)
    if angle == 0:
        return
    trasl(face, vect_scale(center, -1))
    u = {'x':r['x']/angle, 'y':r['y']/angle, 'z':r['z']/angle}
    cos = math.cos(angle)
    sin = math.sin(angle)
    inv_cos = 1-cos
    c_00 = u['x']*u['y']*inv_cos
    c_01 = u['z']*sin
    c_10 = u['x']*u['z']*inv_cos
    c_11 = u['y']*sin
    c_20 = u['y']*u['z']*inv_cos
    c_21 = u['x']*sin

    a = cos+u['x']**2*inv_cos
    b = c_00-c_01
    c = c_10+c_11
    d = c_00+c_01
    e = cos+u['y']**2*inv_cos
    f = c_20-c_21
    g = c_10-c_11
    h = c_20+c_21
    i = cos+u['z']**2*inv_cos
    rot(face, a, b, c, d, e, f, g, h, i)
    trasl(face, center)

def scale2D(face, center, s):
    trasl(face, vect_scale(center, -1))
    scale(face, s)
    trasl(face, center)

def mapTexture(texture, shape, f, vertexMode = False):
    w = len(texture[0])+1
    h = len(texture)+1
    border = [{'x':-1, 'y':h, 'z':0}, {'x':w, 'y':h, 'z':0}, {'x':w, 'y':-1, 'z':0}, {'x':-1, 'y':-1, 'z':0}]
    surf = []
    for v in shape['polyhedron'][f]['face']:
        surf.append(shape['vertices'][v])
    txt = vectorize(texture)
    
    t_center = {'x':w/2, 'y':h/2, 'z':0}
    t_norm = {'x':0, 'y':0, 'z':1}
    f_norm = faceNorm(shape, f)
    f_on_yz = {'x':0, 'y':f_norm['y'], 'z':f_norm['z']}
    f_on_yz = vect_scale(f_on_yz, 1/(norm_squared(f_on_yz)**(0.5))) if norm_squared(f_on_yz)**(0.5)!=0 else {'x':0, 'y':1, 'z':0}
    ang = math.acos(dot(f_on_yz, t_norm))
    c = cross(t_norm, f_on_yz)
    modulus_sq = norm_squared(c)
    if modulus_sq > 0.0001:
        c = vect_scale(c, 1/(modulus_sq**(0.5)))
    elif ang > 1:
        c = {'x':1, 'y':0, 'z':0}
    rotator = vect_scale(c, ang)
    for t in txt:
        rot2D(t, t_center, rotator)
    capsule = [t_norm]
    rot2D(capsule, {'x':0, 'y':0, 'z':0}, rotator)
    t_norm = capsule[0]
    rot2D(border, t_center, rotator)

    ang = math.acos(dot(f_norm, t_norm))
    c = cross(t_norm, f_norm)
    modulus_sq = norm_squared(c)
    if modulus_sq > 0.0001:
        c = vect_scale(c, 1/(modulus_sq**(0.5)))
    elif ang > 1:
        c = {'x':0, 'y':0, 'z':-1}
    rotator = vect_scale(c, ang)
    for t in txt:
        rot2D(t, t_center, rotator)
    rot2D(border, t_center, rotator)
    capsule = [t_norm]
    rot2D(capsule, {'x':0, 'y':0, 'z':0}, rotator)
    t_norm = capsule[0]

    border_base = vect_sub(border[1], border[0])
    border_norm = vect_scale(border_base, 1/(norm_squared(border_base)**(0.5)))
    surf_base = vect_sub(surf[2 if vertexMode else 1], surf[0])
    surf_norm = vect_scale(surf_base, 1/(norm_squared(surf_base)**(0.5)))
    ang = math.acos(dot(border_norm, surf_norm))
    orientation = cross(border_norm, surf_norm)
    modulus_sq = norm_squared(orientation)
    if modulus_sq > 0.0001:
        orientation = vect_scale(orientation, 1/(modulus_sq**(0.5)))
    elif ang > 1:
        orientation = t_norm
    rotator = vect_scale(orientation, ang)
    for t in txt:
        rot2D(t, t_center, rotator)
    rot2D(border, t_center, rotator)
    capsule = [t_norm]
    rot2D(capsule, {'x':0, 'y':0, 'z':0}, rotator)
    t_norm = capsule[0]

    s = {'x':1, 'y':1, 'z': 1}
    surf_side = norm_squared(surf_base)**0.5
    border_side = norm_squared(border_base)**0.5
    s = vect_scale(s, surf_side/border_side)
    for t in txt:
        scale2D(t, t_center, s)
    scale2D(border, t_center, s)

    align_base = vect_sub(vect_scale(vect_sum(surf[0], surf[2 if vertexMode else 1]), 0.5), vect_scale(vect_sum(border[0], border[1]), 0.5))
    for t in txt:
        trasl(t, align_base)
    trasl(border, align_base)

    B = vect_sub(border[1], border[0])
    D = vect_sub(border[2], border[0])
    perp = cross(B, D)
    n = cross(D, perp)
    n = vect_scale(n, 1/(norm_squared(n)**(0.5)))
    P = None
    for i in range(1,len(surf)-1):
        den = dot(n, vect_sub(surf[i+1], surf[i]))
        if den == 0:
            continue
        num = dot(n, vect_sub(border[0], surf[i]))
        t = num/den
        if t>=0 and t<=1:
            P = vect_sum(surf[i], vect_scale(vect_sub(surf[i+1], surf[i]),t))
            break
    if P != None:
        s = {'x':1, 'y':1, 'z': 1}
        s = vect_scale(s, (norm_squared(vect_sub(P, border[0]))**(0.5)) / (norm_squared(D)**(0.5)))
        for t in txt:
            scale2D(t, t_center, s)
        scale2D(border, t_center, s)
      
    if len(shape['polyhedron'][f]['face']) < 6 and  not len(shape['polyhedron']) >= 20:
        for t in txt:
            trasl(t, vect_sub(centerFace(shape, shape['polyhedron'][f]['face']), vect_scale(vect_sum(border[0], border[2]), 0.5)))
    else:
        align_base = vect_sub(vect_scale(vect_sum(surf[0], surf[2 if vertexMode else 1]), 0.5), vect_scale(vect_sum(border[0], border[1]), 0.5))
        for t in txt:
            trasl(t, align_base)
    return txt

def face(verts, filling=' ', contour='#', fg={'r':255, 'g': 255, 'b': 255}, bg='default', border_col={'r':255, 'g': 255, 'b': 255}, text_col={'r':255, 'g': 255, 'b': 255}, border_bg='', center={'x':pr_center['x'], 'y':pr_center['y'], 'z':-1000}, borderless=False, bright_offset = 0):

    if borderless:
        border_bg = bg
    aux_mat = []
    min_x = 1000
    min_y = 1000
    max_x = -1000
    max_y = -1000

    for v in verts['face']:
        if v['x'] < min_x:
            min_x = v['x']
        if v['x'] > max_x:
            max_x = v['x']
        if v['y'] < min_y:
            min_y = v['y']
        if v['y'] > max_y:
            max_y = v['y']

    aux_mat = help_face(verts, round(min_x), round(min_y), round(max_x), round(max_y), filling, '@' if borderless else contour)

    v1 = verts['face'][0]
    v2 = verts['face'][1]
    v3 = verts['face'][2]
    a = {'x':v2['x']-v1['x'], 'y':v2['y']-v1['y'], 'z':v2['z']-v1['z']}
    b = {'x':v3['x']-v1['x'], 'y':v3['y']-v1['y'], 'z':v3['z']-v1['z']}
    c = cross(a, b)

    test = dot(c, vect_sub(v1,center))
    if test != 0:
        test /= abs(test)
    poly = {'vertices':[]}
    indeces = []
    ind = 0
    for v in verts['face']:
        poly['vertices'].append(v)
        indeces.append(ind)
        ind += 1
    ray = vect_sub(lightSource, centerFace(poly, indeces))
    brightness = 0
    try:
        brightness = max(0, dot(vect_scale(ray, 1/(norm_squared(ray)**(0.5))), vect_scale(c, test/(norm_squared(c)**(0.5)))))
    except:
        pass
    brightness *= 0.45
    brightness += 0.55
    brightness = max(0, brightness + bright_offset)
    new_fg = tuneBrightness(fg, brightness)
    fg = fg if new_fg == 'default' else new_fg
    bg = tuneBrightness(bg, brightness)
    border_bg = tuneBrightness(border_bg, brightness)
    new_border_col = tuneBrightness(border_col, brightness)
    border_col = border_col if new_border_col == 'default' else new_border_col
    new_text_col = tuneBrightness(text_col, brightness)
    text_col = text_col if new_text_col == 'default' else new_text_col

    try:
        c['x'] /= c['z']
        c['y'] /= c['z']
        c['z'] = 1
    except:
        pass
    k = -(c['x']*v1['x'] + c['y']*v1['y'] + v1['z'])
    norm = c
    try:
        norm = vect_scale(c, 1/norm_squared(c)**(0.5))
    except: pass
    
    if 'texture' in verts and norm['z']>0.15:
        holes = False
        for v in verts['texture']:
            if len(v) == 2:
                points = help_line(v[0], v[1])
                for p in points:
                    try:
                        y = round(p['y']-min_y)
                        x = round(p['x']-min_x)
                        if x>=0 and y>= 0:
                            aux_mat[y][x] = [' ' if holes else '#', 'inside' if holes else 'texture']
                    except:
                        pass
            elif len(v) == 0:
                holes = True
            else:
                fill = ' ' if holes else '#'
                txt_mat = help_face({'face':v}, round(min_x), round(min_y), round(max_x), round(max_y), fill, fill)

                for i in range(len(txt_mat)):
                    for j in range(len(txt_mat[i])):
                        if txt_mat[i][j][0] != '':
                            aux_mat[i][j] = [txt_mat[i][j][0], 'inside' if holes else 'texture']

    zeds = []
    for i in range(len(aux_mat)):
        if round(i+min_y)%2==0:
            zeds = []
        for j in range(len(aux_mat[i])):
            z = -(k+c['x']*(j+min_x)+c['y']*(i+min_y+0.5))
            if round(i+min_y)%2==0:
                zeds.append(z)
            else:
                try:
                    z = zeds[j]
                except:
                    pass
            if c['z'] == 0:
                z = v1['z']
            if aux_mat[i][j][0] != '':
                foreground = fg if aux_mat[i][j][1] == 'inside' else (text_col if aux_mat[i][j][1] == 'texture' else border_col)
                background = bg if aux_mat[i][j][1] == 'inside' else (text_col if aux_mat[i][j][1] == 'texture' else border_bg)
                setTile({'x':round(j+min_x), 'y':round(i+min_y), 'z': z}, aux_mat[i][j][0], foreground, background)


def help_face(verts, min_x, min_y, max_x, max_y, filling, contour):

    verteces = []
    aux_mat = []
    for v in verts['face']:
        verteces.append({'x':round(v['x']), 'y':round(v['y']), 'z':v['z']})

    for i in range(max_y - min_y + 1):
        aux_mat.append([])
        for j in range(max_x - min_x + 1):
            aux_mat[i].append(['', 'outside'])

    draw_min_x = 1000
    draw_min_y = 1000
    draw_max_x = -1000
    draw_max_y = -1000
    for i in range(len(verteces)):
        points = help_line(verteces[i], verteces[(i+1)%len(verteces)])
        for p in points:
            y = p['y']-min_y
            x = p['x']-min_x
            if x < draw_min_x:
                draw_min_x = x
            if x > draw_max_x:
                draw_max_x = x
            if y < draw_min_y:
                draw_min_y = y
            if y > draw_max_y:
                draw_max_y = y
            try:
                aux_mat[y][x] = [contour, 'outside']
            except:
                pass

    for i in range(max(draw_min_y, 0), min(draw_max_y+1, len(aux_mat))):
        state = 'outside'
        drawn = False
        for j in range(max(draw_min_x, 0), min(draw_max_x+1, len(aux_mat[0]))):
            if state == 'done':
                if aux_mat[i][j][0] == '@':
                    aux_mat[i][j][0] = ''
                continue
            if aux_mat[i][j][0] == '':
                if state == 'border':
                    state = 'done' if drawn else 'inside'
                    drawn = True
            if aux_mat[i][j][0] == contour:
                if state != 'border':
                    state = 'border'
            if state == 'inside':
                aux_mat[i][j] = [filling, 'inside']
            if aux_mat[i][j][0] == '@':
                aux_mat[i][j][0] = ''
        if state == 'inside':
            for j in range(max(draw_min_x, 0), min(draw_max_x+1, len(aux_mat[0]))):
                if aux_mat[i][j][1] == 'inside':
                    aux_mat[i][j] = ['', 'outside']
            
    return aux_mat
    
def trasl(verts, t):
    for i in range(len(verts)):
        verts[i] = {'x':verts[i]['x']+t['x'], 'y':verts[i]['y']+t['y'], 'z':verts[i]['z']+t['z']}

def scale(verts, s):
    for i in range(len(verts)):
        verts[i] = {'x':verts[i]['x']*s['x'], 'y':verts[i]['y']*s['y'], 'z':verts[i]['z']*s['z']}

def rot(verts, a,b,c,d,e,f,g,h,j):
    for i in range(len(verts)):
        v = verts[i]
        new_x = v['x']*a + v['y']*b + v['z']*c
        new_y = v['x']*d + v['y']*e + v['z']*f
        new_z = v['x']*g + v['y']*h + v['z']*j
        verts[i] = {'x':new_x, 'y':new_y, 'z':new_z}

def clear():
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            mat[i][j] = ' '
            depth_mat[i][j] = -1000
            color_mat[i][j] = {'fg':{'r':255, 'g': 255, 'b': 255}, 'bg':''}


current_fg = {'r': 255, 'g': 255, 'b': 255}
current_bg = 'default'
def print_mat():
    global current_bg
    global current_fg

    for row in range(len(mat)-1):
        for col in range(len(mat[row])):
            if color_mat[row][col]['bg'] == '':
                color_mat[row][col]['bg'] = 'default'
            if not ((color_mat[row][col]['bg'] == prev_col_mat[row][col]['bg'] or mat[row][col] == '█') and (color_mat[row][col]['fg'] == prev_col_mat[row][col]['fg'] or mat[row][col] == ' ') and mat[row][col] == prev_mat[row][col]):
                prev_mat[row][col] = mat[row][col]
                prev_bg = 'default'
                if color_mat[row][col]['bg'] != '' and color_mat[row][col]['bg'] != 'default':
                    bg_r = color_mat[row][col]['bg']['r']
                    bg_g = color_mat[row][col]['bg']['g']
                    bg_b = color_mat[row][col]['bg']['b']
                    prev_bg = {'r': bg_r, 'g': bg_g, 'b': bg_b}
                fg_r = color_mat[row][col]['fg']['r']
                fg_g = color_mat[row][col]['fg']['g']
                fg_b = color_mat[row][col]['fg']['b']
                prev_fg = {'r': fg_r, 'g': fg_g, 'b': fg_b}
                prev_col_mat[row][col] = {'fg': prev_fg, 'bg': prev_bg}

                resetted = False
                print('\033['+str(row)+';'+str(col)+'f', end='')

                if color_mat[row][col]['bg'] != current_bg:
                    if color_mat[row][col]['bg'] == 'default':
                        print('\033[0m', end='')
                        current_bg = 'default'
                        resetted = True
                    else:
                        r = color_mat[row][col]['bg']['r']
                        g = color_mat[row][col]['bg']['g']
                        b = color_mat[row][col]['bg']['b']
                        print('\033[48;2;'+str(r)+';'+str(g)+';'+str(b)+'m', end='')
                        current_bg = {'r':r, 'g':g, 'b':b}

                if resetted or color_mat[row][col]['fg'] != current_fg:
                    r = color_mat[row][col]['fg']['r']
                    g = color_mat[row][col]['fg']['g']
                    b = color_mat[row][col]['fg']['b']
                    print('\033[38;2;'+str(r)+';'+str(g)+';'+str(b)+'m', end='')
                    current_fg = {'r':r, 'g':g, 'b':b}
                print(mat[row][col], end='')
        if row<len(mat)-2:
            print()
    print('\033[0;0f', end='')
    
def vectorize(texture):
    txt = []
    for i in range(len(texture)):
        txt.append([])
        for j in range(len(texture[i])):
            txt[i].append(texture[i][j])
    shapes = []
    def apply(val):
        for i in range(len(txt)):
            for j in range(len(txt[i])):
                if txt[i][j] == val:
                    y_min = i
                    y_max = i
                    x_min = j
                    x_max = j
                    expandable = True
                    while expandable:
                        expandable = False
                        if y_min-1 >=0:
                            y_min -= 1
                            exp = True
                            for a in range(x_min, x_max+1):
                                if txt[y_min][a] != val:
                                    exp = False
                                    if txt[y_min][a] == ' ':
                                        y_min += 1
                                        break
                            if exp:
                                expandable = True

                        if x_min-1 >=0:
                            x_min -= 1
                            exp = True
                            for a in range(y_min, y_max+1):
                                if txt[a][x_min] != val:
                                    exp = False
                                    if txt[a][x_min] == ' ':
                                        x_min += 1
                                        break
                            if exp:
                                expandable = True
                                    
                        if y_max+1 < len(txt):
                            y_max += 1
                            exp = True
                            for a in range(x_min, x_max+1):
                                if txt[y_max][a] != val:
                                    exp = False
                                    if txt[y_max][a] == ' ':
                                        y_max -= 1
                                        break
                            if exp:
                                expandable = True

                        if x_max+1 < len(txt[0]):
                            x_max += 1
                            exp = True
                            for a in range(y_min, y_max+1):
                                if txt[a][x_max] != val:
                                    exp = False
                                    if txt[a][x_max] == ' ':
                                        x_max -= 1
                                        break
                            if exp:
                                expandable = True

                    if y_min==y_max or x_min==x_max:
                        shapes.append([{'x':x_min, 'y':y_min, 'z':0}, {'x':x_max, 'y':y_max, 'z':0}])
                    else:
                        shapes.append([{'x':x_min, 'y':y_min, 'z':0}, {'x':x_max, 'y':y_min, 'z':0}, {'x':x_max, 'y':y_max, 'z':0}, {'x':x_min, 'y':y_max, 'z':0},])
                    for a in range(y_min, y_max+1):
                        for b in range(x_min, x_max+1):
                            txt[a][b] = 'X'
    apply('#')
    shapes.append([])
    apply('@')
    return shapes

def vect_scale(v, s):
    r = {}
    for key in v:
        r[key] = v[key]*s
    return r

def dot(v1, v2):
    return v1['x']*v2['x'] + v1['y']*v2['y'] + v1['z']*v2['z']

def vect_sum(v1, v2):
    return {'x': v1['x']+v2['x'], 'y': v1['y']+v2['y'], 'z':v1['z']+v2['z']}

def vect_sub(v1, v2):
    return vect_sum(v1, vect_scale(v2, -1))

def cross(v1, v2):
    return {'x':v1['y']*v2['z'] - v1['z']*v2['y'],
              'y':v1['z']*v2['x'] - v1['x']*v2['z'],
                'z':v1['x']*v2['y'] - v1['y']*v2['x']}

def norm_squared(v):
    return v['x']**2+v['y']**2+v['z']**2

def trasl3D(polyhedron, t):
    trasl(polyhedron['vertices'], t)
    for surf in polyhedron['polyhedron']:
        if 'texture' in surf:
            for line in surf['texture']:
                trasl(line, t)

def center3D(shape):
    avg_x = 0
    avg_y = 0
    avg_z = 0
    num_verts = len(shape['vertices'])
    for v in shape['vertices']:
        avg_x += v['x']
        avg_y += v['y']
        avg_z += v['z']
    avg_x /= num_verts
    avg_y /= num_verts
    avg_z /= num_verts
    return {'x':avg_x, 'y':avg_y, 'z':avg_z}

def centerFace(shape, face):
    avg_x = 0
    avg_y = 0
    avg_z = 0
    num_verts = len(face)
    for v in face:
        avg_x += shape['vertices'][v]['x']
        avg_y += shape['vertices'][v]['y']
        avg_z += shape['vertices'][v]['z']
    avg_x /= num_verts
    avg_y /= num_verts
    avg_z /= num_verts
    return {'x':avg_x, 'y':avg_y, 'z':avg_z}

def scale3D(polyhedron, s):
    center = center3D(polyhedron)
    trasl(polyhedron['vertices'], vect_scale(center, -1))
    scale(polyhedron['vertices'], s)
    trasl(polyhedron['vertices'], center)
    calcR(polyhedron)
    if 'genI' in polyhedron:
        polyhedron['genI']()
    for surf in polyhedron['polyhedron']:
        if 'texture' in surf:
            for shape in surf['texture']:
                trasl(shape, vect_scale(center, -1))
                scale(shape, s)
                trasl(shape, center)

def rot3D(polyhedron, r):
    angle = (r['x']**2+r['y']**2+r['z']**2)**(0.5)
    if angle == 0:
        return
    u = {'x':r['x']/angle, 'y':r['y']/angle, 'z':r['z']/angle}
    cos = math.cos(angle)
    sin = math.sin(angle)
    inv_cos = 1-cos
    c_00 = u['x']*u['y']*inv_cos
    c_01 = u['z']*sin
    c_10 = u['x']*u['z']*inv_cos
    c_11 = u['y']*sin
    c_20 = u['y']*u['z']*inv_cos
    c_21 = u['x']*sin

    a = cos+u['x']**2*inv_cos
    b = c_00-c_01
    c = c_10+c_11
    d = c_00+c_01
    e = cos+u['y']**2*inv_cos
    f = c_20-c_21
    g = c_10-c_11
    h = c_20+c_21
    i = cos+u['z']**2*inv_cos
    
    center = center3D(polyhedron)
    trasl(polyhedron['vertices'], vect_scale(center, -1))
    rot(polyhedron['vertices'], a,b,c,d,e,f,g,h,i)
    trasl(polyhedron['vertices'], center)
    for surf in polyhedron['polyhedron']:
        if 'texture' in surf:
            for shape in surf['texture']:
                trasl(shape, vect_scale(center, -1))
                rot(shape, a,b,c,d,e,f,g,h,i)
                trasl(shape, center)

def makePoly(v_text):
    poly = {'vertices':[], 'polyhedron':[]}
    poly['polyhedron'].append({'face':[], 'texture':v_text})
    return poly

def applyTextures(shape, texts, vertexMode = False):
    ind = 0
    for t in texts:
        shape['polyhedron'][ind]['texture'] = mapTexture(t, shape, ind, vertexMode=vertexMode)
        ind += 1

def calcR(shape):
    c = center3D(shape)
    shape['R'] = 0
    for v in shape['vertices']:
        d = norm_squared(vect_sub(c, v))**0.5
        if d > shape['R']:
            shape['R'] = d

def makeDie(v, offsets, m, borderless=[]):
    polyhedron = []
    for o in offsets:
        polyhedron.append({'face':o})
        if o in borderless:
            polyhedron[-1]['borderless'] = None
    state = {'vx':0, 'vy':0, 'vz':0, 'wx':0, 'wy':0, 'wz':0}
    shape = {'polyhedron': polyhedron, 'state': state, 'm': m, 'I':None, 'vertices': v, 'genI': None, 'R':0, 'active': True}
    calcR(shape)
    trasl3D(shape, vect_scale(center3D(shape), -1))
    return shape

def mergeTexts(t1, t2, orientation):
    res = []
    if orientation == 'vertical':
        for t in t1:
            res.append(t)
        for t in t2:
            res.append(t)
    else:
        for i in range(min(len(t1), len(t2))):
            res.append(t1[i] + t2[i])
    return res

def genDigits(sides, step=1, offset=0):
    dig = []
    for s in range(1+offset, sides+1+offset, step):
        if (s==6 or s==9) and sides >=9:
            dig.append(mergeTexts(digits.digits[s], digits.dash, 'vertical'))
        else:
            t = digits.digits[s%10]
            if s>=10:
                t = mergeTexts(digits.digits[(s//10)%10], t, 'horizontal')
            if s!=100:
                dig.append(t)
            else:
                dig = [t] + dig
    return dig
        

def makeD8():
    offsets = []
    vertices = [{'x':60, 'y':20, 'z' : 0},{'x':80, 'y':20, 'z' : 0},{'x':70, 'y':20-14.142, 'z' : 10},
                {'x':60, 'y':20, 'z' : 20}, {'x':80, 'y':20, 'z' : 20}, {'x':70, 'y':20+14.142, 'z' : 10}]
    offsets.append([5, 0, 1])
    offsets.append([0, 3, 2])
    offsets.append([0, 5, 3])
    offsets.append([1, 0, 2])
    offsets.append([4, 3, 5])
    offsets.append([1, 2, 4])
    offsets.append([5, 1, 4])
    offsets.append([2, 3, 4])
    octahedron = makeDie(vertices, offsets, 6)
    def I():
        side_sq = norm_squared(vect_sub(octahedron['vertices'][octahedron['polyhedron'][0]['face'][0]], octahedron['vertices'][octahedron['polyhedron'][0]['face'][1]]))
        octahedron['I'] = octahedron['m']*side_sq/10
    octahedron['genI'] = I
    octahedron['genI']()
    return octahedron

def makeD6():
    offsets = []
    vertices = [{'x':15, 'y':15, 'z' : 0},{'x':15, 'y':35, 'z' : 0},{'x':35, 'y':35, 'z' : 0},{'x':35, 'y':15, 'z' : 0},
            {'x':15, 'y':15, 'z' : 20},{'x':15, 'y':35, 'z' : 20},{'x':35, 'y':35, 'z' : 20},{'x':35, 'y':15, 'z' : 20}]
    
    offsets.append([5, 6, 7, 4])
    offsets.append([2, 6, 5, 1])
    offsets.append([7, 6, 2, 3])
    offsets.append([0, 4, 7, 3])
    offsets.append([0, 1, 5, 4])
    offsets.append([0, 3, 2, 1])
    cube = makeDie(vertices, offsets, 8)
    def I():
        side_sq = norm_squared(vect_sub(cube['vertices'][cube['polyhedron'][0]['face'][0]], cube['vertices'][cube['polyhedron'][0]['face'][1]]))
        cube['I'] = cube['m']*side_sq/6
    cube['genI'] = I
    cube['genI']()
    return cube

def makeD4():
    offsets = []
    angles = 10
    radius = 10
    vertices = [{'x':0, 'y':0, 'z' : 2*radius}, {'x':0, 'y':2*radius, 'z' : 2*radius}, {'x':radius, 'y':2*radius, 'z' : 2*radius}, {'x':radius, 'y':0, 'z' : 2*radius},
                {'x':0, 'y':0, 'z' : 0},        {'x':0, 'y':2*radius, 'z' : 0},        {'x':radius, 'y':2*radius, 'z' : 0},        {'x':radius, 'y':0, 'z' : 0}]
    inc = math.pi/angles
    ang = inc
    loops = 0
    while ang <math.pi:
        p = vect_scale(vect_sum(vertices[0], vertices[1]), 0.5)
        vertices.append({'x':p['x']-radius*math.sin(ang), 'y':p['y']-radius*math.cos(ang), 'z':p['z']})
        ang += inc
        loops += 1
    ang = inc
    while ang <math.pi:
        p = vect_scale(vect_sum(vertices[3], vertices[7]), 0.5)
        vertices.append({'x':p['x']+radius*math.sin(ang), 'y':p['y'], 'z':p['z']+radius*math.cos(ang)})
        ang += inc
    ang = inc
    while ang <math.pi:
        p = vect_scale(vect_sum(vertices[4], vertices[5]), 0.5)
        vertices.append({'x':p['x']-radius*math.sin(ang), 'y':p['y']+radius*math.cos(ang), 'z':p['z']})
        ang += inc
    ang = inc
    while ang <math.pi:
        p = vect_scale(vect_sum(vertices[2], vertices[6]), 0.5)
        vertices.append({'x':p['x']+radius*math.sin(ang), 'y':p['y'], 'z':p['z']-radius*math.cos(ang)})
        ang += inc
    
    offsets.append([2, 3, 0] + [n for n in range(8, 8+loops)] + [1]) #1
    offsets.append([4, 0, 3] + [n for n in range(8+loops, 8+2*loops)] + [7]) #2
    offsets.append([1, 5, 6] + [n for n in range(8+3*loops, 8+4*loops)] + [2]) #3
    offsets.append([7, 6, 5] + [n for n in range(8+2*loops, 8+3*loops)] + [4]) #4
    
    offsets.append([1, 7+loops, 8+2*loops, 5])
    offsets.append([1, 6+loops, 9+2*loops, 5])
    offsets.append([4, 7+3*loops, 8, 0])
    offsets.append([4, 6+3*loops, 9, 0])
    offsets.append([2, 7+4*loops, 8+loops, 3])
    offsets.append([2, 6+4*loops, 9+loops, 3])
    offsets.append([7, 7+2*loops, 8+3*loops, 6])
    offsets.append([7, 6+2*loops, 9+3*loops, 6])
    for i in range(loops-1):
        offsets.append([8+i, 7+3*loops-i, 6+3*loops-i, 9+i])
        offsets.append([7+4*loops-i, 6+4*loops-i, 9+loops+i, 8+loops+i])
        if i < loops-2:
            offsets.append([8+i, 7+3*loops-i, 5+3*loops-i, 10+i])
            offsets.append([7+4*loops-i, 5+4*loops-i, 10+loops+i, 8+loops+i])

    borderless = []
    for o in offsets[4:]:
        borderless.append(o)
    cuboid = makeDie(vertices, offsets, 6, borderless)
    def I():
        side_sq = norm_squared(vect_sub(cuboid['vertices'][cuboid['polyhedron'][0]['face'][0]], cuboid['vertices'][cuboid['polyhedron'][0]['face'][1]]))
        cuboid['I'] = cuboid['m']*side_sq/5
    cuboid['genI'] = I
    scale3D(cuboid, {'x':0.8, 'y':0.8, 'z':0.8})
    return cuboid

def makeD12():
    offsets = []
    ihp = 0.6180339887498948482045868343656381177203091798057628621354486227
    phi = ihp +1
    s1 = phi*10
    s2 = ihp*10
    vertices = [{'x':-10, 'y':-10, 'z' : -10},{'x':-10, 'y':10, 'z' : -10},{'x':10, 'y':10, 'z' : -10},{'x':10, 'y':-10, 'z' : -10},
            {'x':-10, 'y':-10, 'z' : 10},{'x':-10, 'y':10, 'z' : 10},{'x':10, 'y':10, 'z' : 10},{'x':10, 'y':-10, 'z' : 10},
            {'x':0, 'y':s1, 'z':s2}, {'x':0, 'y':-s1, 'z':s2}, {'x':0, 'y':s1, 'z':-s2}, {'x':0, 'y':-s1, 'z':-s2},
            {'x':s2, 'y':0, 'z':s1}, {'x':-s2, 'y':0, 'z':s1}, {'x':s2, 'y':0, 'z':-s1}, {'x':-s2, 'y':0, 'z':-s1},
            {'x':s1, 'y':s2, 'z':0}, {'x':-s1, 'y':s2, 'z':0}, {'x':s1, 'y':-s2, 'z':0}, {'x':-s1, 'y':-s2, 'z':0}]
    
    offsets.append([4, 13, 12, 7, 9])
    offsets.append([6, 12, 7, 18, 16])
    offsets.append([17, 1, 10, 8, 5])
    offsets.append([12, 13, 5, 8, 6])
    offsets.append([4, 9, 11, 0, 19])
    offsets.append([4, 19, 17, 5, 13])
    offsets.append([2, 14, 3, 18, 16])
    offsets.append([6, 8, 10, 2, 16])
    offsets.append([0, 11, 3, 14, 15])
    offsets.append([7, 18, 3, 11, 9])
    offsets.append([0, 15, 1, 17, 19]) 
    offsets.append([15, 14, 2, 10, 1])
    dodecahedron = makeDie(vertices, offsets, 10)
    def I():
        side_sq = norm_squared(vect_sub(dodecahedron['vertices'][dodecahedron['polyhedron'][0]['face'][0]], dodecahedron['vertices'][dodecahedron['polyhedron'][0]['face'][1]]))
        dodecahedron['I'] = dodecahedron['m'] * side_sq * (39*phi + 28)/150
    dodecahedron['genI'] = I
    dodecahedron['genI']()
    return dodecahedron

def makeD10():
    offsets = []
    ihp = 0.6180339887498948482045868343656381177203091798057628621354486227
    phi = ihp +1
    s1 = phi*10
    s2 = ihp*10
    side = 20/phi
    vertices = [{'x':-10, 'y':-10, 'z' : -10}, {'x':10, 'y':-10, 'z' : -10},
            {'x':-10, 'y':10, 'z' : 10}, {'x':10, 'y':10, 'z' : 10},
            {'x':0, 'y':s1, 'z':s2}, {'x':0, 'y':-s1, 'z':-s2},
            {'x':s1, 'y':s2, 'z':0}, {'x':-s1, 'y':s2, 'z':0}, {'x':s1, 'y':-s2, 'z':0}, {'x':-s1, 'y':-s2, 'z':0}]
    
    topFace = [{'x':-10, 'y':-10, 'z' : 10}, {'x':10, 'y':-10, 'z' : 10},{'x':0, 'y':-s1, 'z' : s2},
           {'x':s2, 'y':0, 'z' : s1},{'x':-s2, 'y':0, 'z' : s1}]
    centerTop = {'x':0, 'y':0, 'z' : 0}
    for t in topFace:
        centerTop = vect_sum(centerTop, t)
    centerTop = vect_scale(centerTop, 0.2)
    ct_size = norm_squared(centerTop)**0.5
    top = vect_scale(centerTop, (ct_size+side*((1+2/(5**0.5))**0.5)) / ct_size)

    vertices.append(top)
    vertices.append(vect_scale(top, -1))
    
    offsets.append([9, 7, 2, 10])
    offsets.append([6, 3, 4, 11])
    offsets.append([8, 1, 5, 10])
    offsets.append([7, 9, 0, 11])
    offsets.append([2, 4, 3, 10])
    offsets.append([0, 5, 1, 11])
    offsets.append([3, 6, 8, 10])
    offsets.append([4, 2, 7, 11])
    offsets.append([5, 0, 9, 10])
    offsets.append([1, 8, 6, 11])
    decahedron = makeDie(vertices, offsets, 10)
    def I():
        R_sq = norm_squared(vect_sub(center3D(decahedron), vertices[0]))
        decahedron['I'] = decahedron['m']* R_sq * 3/5
    decahedron['genI'] = I
    decahedron['genI']()
    decahedron['vertexMode'] = True
    height = vect_sub(vertices[11], vertices[10])
    h_norm_sq = norm_squared(height)
    s_factor = ((norm_squared(vect_sub(vertices[4], vertices[5])) / h_norm_sq)**0.5)
    rot3D(decahedron, {'x': math.acos(dot({'x':0, 'y':1, 'z':0}, vect_scale(height, 1/h_norm_sq**0.5))), 'y':0, 'z':0})
    scale3D(decahedron, {'x': 1, 'y':s_factor, 'z':1})
    return decahedron

def makeD20():
    offsets = []
    phi_sq = 2.6180339887498948482045868343656381177203091798057628621354486227
    H = phi_sq-2
    R = 20
    vertices = []
    for i in range(5):
        ang = i*math.pi*2/5
        vertices.append({'x':R*math.cos(ang), 'y':-R*math.sin(ang), 'z':0})
    side = norm_squared(vect_sub(vertices[0], vertices[1]))**0.5
    vertices.append({'x':0, 'y': 0, 'z':R*H})
    c = {'x':0, 'y':0, 'z':vertices[-1]['z'] - (side * math.sqrt(phi_sq+1)/2)}
    for i in range(len(vertices)):
        v = vertices[i]
        vertices.append(vect_sum(c, vect_sub(c, v)))
    for i in range(5):
        offsets.append([i, (i+1)%5, 5])
    for i in range(5):
        offsets.append([(i+1)%5, i, (i+3)%5+6])
    for i in range(5):
        offsets.append([(4-i)%5+6, (5-i)%5+6, (7-i)%5])
    for i in range(5):
        offsets.append([(i+2)%5+6, (i+1)%5+6, 11])
    o = [offsets[0], offsets[15], offsets[2], offsets[17], offsets[12],
         offsets[10], offsets[4], offsets[14], offsets[6], offsets[8],
         offsets[11], offsets[13], offsets[5], offsets[18], offsets[9],
         offsets[7], offsets[3], offsets[16], offsets[1], offsets[19]]
    icosahedron = makeDie(vertices, o, 10)
    def I():
        side_sq = norm_squared(vect_sub(icosahedron['vertices'][icosahedron['polyhedron'][0]['face'][0]], icosahedron['vertices'][icosahedron['polyhedron'][0]['face'][1]]))
        icosahedron['I'] = icosahedron['m']*phi_sq*side_sq/10
    icosahedron['genI'] = I
    scale3D(icosahedron, {'x':0.9, 'y':0.9, 'z':0.9})
    return icosahedron

deltaTime = 0.05
prevTime = time.time()
gravity = 40
elastic_restitution = 0.6
friction = 0.6
initialTime = prevTime
timeLimit = 10

walls = []
walls.append({'perp_dir':'z', 'depth':-40, 'norm':1, 'floor':True})
walls.append({'perp_dir':'x', 'depth':0, 'norm':1})
walls.append({'perp_dir':'x', 'depth':terminal_dims[0], 'norm':-1})
walls.append({'perp_dir':'y', 'depth':0, 'norm':1})
walls.append({'perp_dir':'y', 'depth':terminal_dims[1], 'norm':-1})

def exctractFaces(poly):
    f = []
    for ind in range(len(poly['polyhedron'])):
        f.append(extractFace(poly, ind))
    return f

def extractFace(poly, ind):
    f = []
    for v in poly['polyhedron'][ind]['face']:
        f.append(poly['vertices'][v])
    return f

def collision(A, B, p, n):
    va = {'x': A['state']['vx'], 'y': A['state']['vy'], 'z': A['state']['vz']}
    wa = {'x': A['state']['wx'], 'y': A['state']['wy'], 'z': A['state']['wz']}
    vb = {'x': B['state']['vx'], 'y': B['state']['vy'], 'z': B['state']['vz']}
    wb = {'x': B['state']['wx'], 'y': B['state']['wy'], 'z': B['state']['wz']}

    ca = center3D(A)
    cb = center3D(B)
    ra = vect_sub(p, ca)
    rb = vect_sub(p, cb)
    vap = vect_sum(va, cross(wa, ra))
    vbp = vect_sum(vb, cross(wb, rb))
    vp = vect_sub(vap, vbp)

    fr = vect_sub(vect_scale(n,dot(vp, n)), vp)
    max_fr = norm_squared(fr)**(0.5)
    J = -(1+elastic_restitution) * dot(vp,n)
    J /= 1/A['m'] + 1/B['m'] + norm_squared(cross(ra, n))/A['I'] + norm_squared(cross(rb, n))/B['I']
    if J*friction < max_fr and max_fr != 0:
        fr = vect_scale(fr, J*friction/max_fr)
    impulse = vect_sum(fr, vect_scale(n, J))

    new_va = vect_sum(va, vect_scale(impulse, 1/A['m']))
    new_wa = vect_sum(wa, vect_scale(cross(ra, impulse), 1/A['I']))
    new_vb = vect_sub(vb, vect_scale(impulse, 1/B['m']))
    new_wb = vect_sub(wb, vect_scale(cross(rb, impulse), 1/B['I']))

    A['state'] = {'vx': new_va['x'], 'vy': new_va['y'], 'vz': new_va['z'],
                  'wx': new_wa['x'], 'wy': new_wa['y'], 'wz': new_wa['z']}
    B['state'] = {'vx': new_vb['x'], 'vy': new_vb['y'], 'vz': new_vb['z'],
                  'wx': new_wb['x'], 'wy': new_wb['y'], 'wz': new_wb['z']}
    
    A['active'] = True
    B['active'] = True

def v_to_f_collision(A, B):

    c = center3D(A)
    dir = vect_sub(center3D(B), c)
    dir = vect_scale(dir, 1/norm_squared(dir)**0.5)

    for surf in exctractFaces(B):
        n = cross(vect_sub(surf[0], surf[1]), vect_sub(surf[2], surf[1]))
        if dot(n, dir) > -0.3:
            continue
        n = vect_scale(n, 1/norm_squared(n)**0.5)

        for v in A['vertices']:

            ray = vect_sub(v, c)
            
            r = norm_squared(ray)
            l = vect_scale(ray, r**0.5)
            if dot(dir, l) < 0:
                continue

            den = dot(l, n)
            if abs(den) < 0.00000001:
                continue

            d = dot(vect_sub(surf[0], c), n) / den

            p = vect_sum(c, vect_scale(l, d))

            dist = vect_sub(p, v)
            if not (norm_squared(vect_sub(p, c)) > r or norm_squared(dist) > r):
                last = cross(vect_sub(surf[0], p), vect_sub(surf[1], p))
                collided = True
                for k in range(1, len(surf)):
                    curr = cross(vect_sub(surf[k], p), vect_sub(surf[(k+1)%len(surf)], p))
                    if dot(curr, last) < 0:
                        collided = False
                        break
                    last = curr
                
                if collided:
                    collision(A, B, p, n)
                    trasl3D(A, vect_scale(dist, 0.5))
                    trasl3D(B, vect_scale(dist, -0.5))
                    A['active'] = True
                    B['active'] = True
                    return True
    return False

def edge_collision(A, B):

    c = center3D(A)
    dir = vect_sub(center3D(B), c)
    dir = vect_scale(dir, 1/norm_squared(dir)**0.5)

    edges = {}
    for f in A['polyhedron']:
        for i in range(len(f['face'])):
            key = str(f['face'][i]) + ',' + str(f['face'][(i+1)%len(f['face'])])
            alt_key = str(f['face'][(i+1)%len(f['face'])]) + ',' + str(f['face'][i])
            if key not in edges and alt_key not in edges:
                edges[key] = [A['vertices'][f['face'][i]], A['vertices'][f['face'][(i+1)%len(f['face'])]]]

    for edge in edges:
        intersections = []
        norms = []
        
        e1 = edges[edge][0]
        e2 = edges[edge][1]
        ray = vect_sub(e1, e2)
        
        r = norm_squared(ray)
        l = vect_scale(ray, r**0.5)

        if dot(vect_sub(e1, c), l) < 0 and dot(vect_sub(e2, c), l) < 0:
            continue

        for surf in exctractFaces(B):
            n = cross(vect_sub(surf[0], surf[1]), vect_sub(surf[2], surf[1]))
            if dot(n, dir) > -0.3:
                continue
            n = vect_scale(n, 1/norm_squared(n)**0.5)
            
            den = dot(l, n)
            if abs(den) < 0.00000001:
                continue

            d = dot(vect_sub(surf[0], e2), n) / den

            p = vect_sum(e2, vect_scale(l, d))
            
            collided = False
            if not (norm_squared(vect_sub(p, e1)) > r or norm_squared(vect_sub(p, e2)) > r):
                last = cross(vect_sub(surf[0], p), vect_sub(surf[1], p))
                collided = True
                for k in range(1, len(surf)):
                    curr = cross(vect_sub(surf[k], p), vect_sub(surf[(k+1)%len(surf)], p))
                    if dot(curr, last) < 0:
                        collided = False
                        break
                    last = curr

            if collided:
                intersections.append(p)
                norms.append(n)


        if len(intersections)>1:
            p = vect_sum(intersections[0], intersections[1])
            p = vect_scale(p, 0.5)
            n = vect_sum(norms[0], norms[1])
            n = vect_scale(n, 1/norm_squared(n)**0.5)
            dist = vect_sub(c, p)
            collision(A, B, p, n)
            trasl3D(A, vect_scale(dist, 0.1))
            trasl3D(B, vect_scale(dist, -0.1))
            A['active'] = True
            B['active'] = True
            return True
    return False

still = False
def phisics(polyhedra, deltaTime):
    global still
    if still:
        return False
    
    elapsed = max(0, time.time()-(initialTime+timeLimit))
    dampening = max(0, math.exp(-elapsed/5) - (deltaTime/2))

    still = True
    for i in range(len(polyhedra)):
        polyhedron = polyhedra[i]
        if polyhedron['active']:
            still = False
        grounded = False
        ground = 0
        max_v = 0
        for face in polyhedron['polyhedron']:
            if len(face['face']) > max_v:
                max_v = len(face['face'])
        state = polyhedron['state']
        lie_flat = False
        slowed = False
        
        c = center3D(polyhedron)
        for j in range(i+1, len(polyhedra)):
            B = polyhedra[j]
            if (polyhedron['active'] or B['active']) and norm_squared(vect_sub(c, center3D(polyhedra[j])))**0.5 < polyhedron['R'] + polyhedra[j]['R']:
                
                if v_to_f_collision(polyhedron, B):
                    break
                if v_to_f_collision(B, polyhedron):
                    break
                if edge_collision(polyhedron, B):
                    break
                if edge_collision(B, polyhedron):
                    break
                
        if polyhedron['active']:
            for w in walls:
                w['touching'] = 0
                if 'floor' in w:
                    ground = w['depth']

                for v in polyhedron['vertices']:
                    if w['norm'] * v[w['perp_dir']] < w['norm'] * w['depth']:
                        w['touching'] += 1

            for v in polyhedron['vertices']:
                for w in walls:
                    if w['norm'] * v[w['perp_dir']] < w['norm'] * w['depth']:
                        va = {'x': state['vx'], 'y': state['vy'], 'z': state['vz']}
                        wa = {'x': state['wx'], 'y': state['wy'], 'z': state['wz']}
                        n = {'x': 0, 'y': 0, 'z': 0}
                        n[w['perp_dir']] = w['norm']
                        ra = vect_sub(v, center3D(polyhedron))
                        vp = vect_sum(va, cross(wa, ra))
                        fr = vect_sub(vect_scale(n,dot(vp, n)), vp)
                        max_fr = norm_squared(fr)**(0.5)
                        if w['norm'] * vp[w['perp_dir']] >= 0:
                            polyhedron['state']['v'+w['perp_dir']] = w['norm']*3.5
                            continue
                        e = elastic_restitution
                        if 'floor' in w:
                            grounded = True
                            if va['z'] > -5:
                                e = max(0, min(elastic_restitution, -va['z']/30*elastic_restitution))
                        J = -(1+e) * dot(vp,n)
                        J /= 1/polyhedron['m']+norm_squared(cross(ra, n))/polyhedron['I']
                        if J*friction < max_fr:
                            fr = vect_scale(fr, J*friction/max_fr)
                        impulse = vect_sum(fr, vect_scale(n, J))
                        new_va = vect_sum(va, vect_scale(impulse, 1/polyhedron['m']/w['touching']))
                        new_wa = vect_sum(wa, vect_scale(cross(ra, vect_scale(impulse, 1/w['touching'])), 1/polyhedron['I']))
                        state['vx'] = new_va['x']
                        state['vy'] = new_va['y']
                        state['vz'] = new_va['z']
                        state['wx'] = new_wa['x']
                        state['wy'] = new_wa['y']
                        state['wz'] = new_wa['z']
                    
                    if 'floor' in w:
                        if w['touching'] >= max(3,max_v):
                            lie_flat = True
                            if not slowed:
                                slowed = True
                                state['vx'] *= dampening
                                state['vy'] *= dampening
                                state['vz'] *= dampening
                                state['wx'] *= dampening
                                state['wy'] *= dampening
                                state['wz'] *= dampening
                        elif w['touching'] > 2 and w['touching'] < max(3,max_v/2):
                            va = {'x': state['vx'], 'y': state['vy'], 'z': state['vz']}
                            wa = {'x': state['wx'], 'y': state['wy'], 'z': state['wz']}
                            if norm_squared(va) < 5 and norm_squared(wa) < 0.05:
                                state['wx'] += 0.01*random.random()-0.005
                                state['wy'] += 0.01*random.random()-0.005
                                state['wz'] += 0.01*random.random()-0.005

                va = {'x': state['vx'], 'y': state['vy'], 'z': state['vz']}
                wa = {'x': state['wx'], 'y': state['wy'], 'z': state['wz']}
                stopped = False
                if lie_flat and norm_squared(va) < 15 and norm_squared(wa) < 2:
                        state['vx'] = 0
                        state['vy'] = 0
                        state['vz'] = 0
                        state['wx'] = 0
                        state['wy'] = 0
                        state['wz'] = 0
                        stopped = True
                        polyhedron['active'] = False

            if not stopped:
                trasl3D(polyhedron, {'x': state['vx']*deltaTime, 'y': state['vy']*deltaTime,'z': state['vz']*deltaTime})
                rot3D(polyhedron, {'x': state['wx']*deltaTime, 'y': state['wy']*deltaTime,'z': state['wz']*deltaTime})
                state['vz'] -= gravity*deltaTime

            min_z = 1000
            for v in polyhedron['vertices']:
                if v['z'] < min_z:
                    min_z = v['z']
            if grounded and min_z<ground:
                state['wz'] *= 0.99
                state['vz'] = 0
                trasl3D(polyhedron, {'x': 0, 'y': 0,'z': ground-min_z})
    return True

def setText(text, pos, bg='', fg={'r':255, 'g':255, 'b':255}):
    p = {'x':round(pos['x']), 'y':round(pos['y']), 'z':round(pos['z'])}
    for i in range(len(text)):
        if depth_mat[p['y']//2][p['x']+i] == p['z']:
            depth_mat[p['y']//2][p['x']+i] = -1000
        setTile({'x': round(p['x']) + i, 'y': round(p['y']), 'z': round(p['z'])}, text[i], fg=fg, bg=bg)

wallOutlines = []
max_w = {'x': -1000, 'y': -1000, 'z': 100}
min_w = {'x': 1000, 'y': 1000, 'z': 1000}
for w in walls:
    if w['norm'] == 1:
        if w['depth'] < min_w[w['perp_dir']]:
            min_w[w['perp_dir']] = w['depth']
    else:
        if w['depth'] > max_w[w['perp_dir']]:
            max_w[w['perp_dir']] = w['depth']

order = [[min_w,min_w], [max_w,min_w], [max_w,max_w], [min_w,max_w]]
for w in walls:
    surf = []
    for i in range(4):
        surf.append({'x':0, 'y':0, 'z':0})
        surf[-1][w['perp_dir']] = w['depth']
        selector = 0
        for key in surf[-1]:
            if key != w['perp_dir']:
                surf[-1][key] = order[i][selector][key]
                selector += 1
    wallOutlines.append(perspectivize({'face':[0,1,2,3]}, surf)['face'])

finished = False
somma = 0
def throwDice():
    clear()
    for wo in wallOutlines:
        for i in range(4):
            line(wo[i], wo[(i+1)%4])
        
    if finished:
        msg = '█ Il risultato è: '+str(somma)+' █'
        x = (terminal_dims[0]-len(msg))//2
        y = terminal_dims[1]-4
        y -= 1-(y%2)
        setText('#'*len(msg), {'x': x, 'y': y-2,'z': 100})
        setText(msg, {'x': x, 'y': y,'z': 100})
        setText('#'*len(msg), {'x': x, 'y': y+1,'z': 100})

    polyhedra.sort(key=lambda el: center3D(el)['z'])
    for polyhedron in polyhedra:
        polyhedron['polyhedron'].sort(key=lambda el: centerFace(polyhedron, el['face'])['z'])
        c = center3D(polyhedron)
        for surf in polyhedron['polyhedron']:
            face(perspectivize(surf, polyhedron['vertices']), bg=polyhedron['colors']['filling'], border_col=polyhedron['colors']['edge'], text_col=polyhedron['colors']['texture'], center=c, borderless = 'borderless' in surf, bright_offset=(polyhedron['brightness'] if 'brightness' in polyhedron else 0))
    

pressedKeys = set()
time_since_finished = 0
def handleDice():
    global finished
    global somma
    global pressedKeys
    global time_since_finished
    global polyhedra
    def numUp(p):
        num = 0
        norm = -1
        ind = 0
        for i in range(len(p['polyhedron'])):
            n = dot(faceNorm(p, i), {'x': 0, 'y': 0, 'z':1})
            if n>norm and 'num' in p['polyhedron'][i]:
                ind = i
                norm = n
                num = p['polyhedron'][i]['num']
        return num, ind
    
    if not phisics(polyhedra, deltaTime) and not finished:
        time_since_finished = time.time()
        somma = 0
        oldPs = []
        for p in polyhedra:
            min_d = 100000
            goal = None
            for pos in positions:
                if 'poly' not in pos:
                    p1 = {'x':pos['x'], 'y':pos['y'], 'z':0}
                    p2 = center3D(p)
                    p2['z'] = 0
                    dist = norm_squared(vect_sub(p2, p1))**0.5
                    if dist < min_d:
                        min_d = dist
                        goal = pos
            goal['poly'] = p
            num, ind = numUp(p)
            v0 = p['vertices'][p['polyhedron'][ind]['face'][0]]
            v1 = p['vertices'][p['polyhedron'][ind]['face'][2 if 'vertexMode' in p else 1]] 
            base = vect_sub(v1, v0)
            base = vect_scale(base, 1/norm_squared(base)**0.5)
            angle = math.acos(dot(base, {'x':1, 'y':0, 'z':0}))
            orientation = 1 if dot({'x':0, 'y':0, 'z':1}, cross(base, {'x':1, 'y':0, 'z':0})) > 0 else -1
            angle *= orientation
            goal['angle'] = angle
            goal['z'] = 70
            oldPs.append(copy.deepcopy(p))
            oldPs[-1]['brightness'] = 0
            p['polyhedron'] = [p['polyhedron'][ind]]
            if num == 0 and 'partner' in p and numUp(p['partner'])[0] == 0:
                num = 100
            somma += num
        finished = True
        polyhedra += oldPs

    if finished:
        speed = (1 - math.exp(- 1/(time_to_display)**0.5 * deltaTime))
        for p in positions:
            c = center3D(p['poly'])
            pos = {'x': p['x'], 'y': p['y'], 'z': p['z']}
            trasl3D(p['poly'], vect_scale(vect_sub(pos, c), speed))
            rot3D(p['poly'], {'x': 0, 'y': 0, 'z': p['angle'] * speed})
            if p['angle'] > 0:
                p['angle'] = max(0, p['angle'] * (1-speed))
            elif p['angle'] < 0:
                p['angle'] = min(0, p['angle'] * (1-speed))
        for p in polyhedra:
            if 'brightness' in p:
                p['brightness'] += (-0.3-p['brightness']) * speed

    if keyboard.is_pressed('esc'):
        pressedKeys.add('esc')
        changeScene('menu')
        changeOptions('dice')

def getDie(n):
    if n==4:
        d = makeD4()
        applyTextures(d, genDigits(4))
        return [d]
    if n==6:
        d = makeD6()
        applyTextures(d, genDigits(6))
        return [d]
    if n==8:
        d = makeD8()
        applyTextures(d, genDigits(8))
        return [d]
    if n==10:
        d = makeD10()
        applyTextures(d, genDigits(10, offset=-1), vertexMode=True)
        return [d]
    if n==12:
        d = makeD12()
        applyTextures(d, genDigits(12))
        return [d]
    if n==20:
        d = makeD20()
        applyTextures(d, genDigits(20))
        return [d]
    if n==100:
        d1 = getDie(10)[0]
        d2 = makeD10()
        applyTextures(d2, genDigits(100, step=10, offset=9), vertexMode=True)
        d2['partner'] = d1
        return [d1, d2]

def prepDice():
    global polyhedra
    global still
    global positions
    still = False
    polyhedra = []
    r = requestedDice.replace(" ", "")
    r = r.split('+')
    dice = []
    for die in r:
        s = die.split('d')
        if len(s) == 2:
            q = int(s[0])
            type = int(s[1])
            if type in [4, 6, 8, 10, 12, 20, 100]:
                for i in range(q):
                    toAdd = getDie(type)
                    if len(dice) + len(toAdd) <= 6:
                        dice += toAdd
                        vals = []
                        if type == 10 or type == 100:
                            vals = [j for j in range(10)]
                        else:
                            vals = [j for j in range(1, type+1)]
                        k = 0
                        for d in toAdd:
                            for j in range(len(vals)):
                                d['polyhedron'][j]['num'] = vals[j]
                                if k==1:
                                    d['polyhedron'][j]['num'] *= 10
                            k += 1
                    else:
                        break
    q = len(dice)
    if q==0:
        changeScene('menu')
        return
    positions = []
    w = terminal_dims[0]
    h = terminal_dims[1]
    d = lightSource['z'] * 2/3
    if q==1:
        positions = [{'x': w/2, 'y': h/2, 'z': d}]
    elif q==2:
        positions = [{'x': w/3, 'y': h/2, 'z': d}, {'x': w-w/3, 'y': h/2, 'z': d}]
    elif q==3:
        positions = [{'x': w/4, 'y': h/2, 'z': d}, {'x': w/2, 'y': h/2, 'z': d}, {'x': w-w/4, 'y': h/2, 'z': d}]
    elif q==4:
        positions = [{'x': w/3, 'y': h/3, 'z': d},   {'x': w-w/3, 'y': h/3, 'z': d},
                     {'x': w/3, 'y': h-h/3, 'z': d}, {'x': w-w/3, 'y': h-h/3, 'z': d}]
    elif q==5:
        positions = [{'x': w/2.5, 'y': h/2.5, 'z': d},   {'x': w-w/2.5, 'y': h/2.5, 'z': d},
                     {'x': w/2.5, 'y': h-h/2.5, 'z': d}, {'x': w-w/2.5, 'y': h-h/2.5, 'z': d},
                     {'x': w/2, 'y': h/2, 'z': d}]
    else:
        positions = [{'x': w/4, 'y': h/3, 'z': d}, {'x': w/2, 'y': h/3, 'z': d}, {'x': w-w/4, 'y': h/3, 'z': d},
                     {'x': w/4, 'y': h-h/3, 'z': d}, {'x': w/2, 'y': h-h/3, 'z': d}, {'x': w-w/4, 'y': h-h/3, 'z': d}]
    for i in range(q):
        trasl3D(dice[i], positions[i])
        speed = random.random()*10+1
        dir = random.random()*math.pi*2
        dice[i]['state']['vx'] = math.cos(dir) * speed
        dice[i]['state']['vy'] = math.sin(dir) * speed
        dice[i]['state']['vz'] = 0
        dice[i]['state']['wx'] = (random.random()*3+3) * (random.randint(0,1)*2 - 1)
        dice[i]['state']['wy'] = (random.random()*3+3) * (random.randint(0,1)*2 - 1)
        dice[i]['state']['wz'] = (random.random()*3+3) * (random.randint(0,1)*2 - 1)
        if individualized_dice:
            dice[i]['colors'] = {'filling': dice_colors[i]['filling'], 'edge': dice_colors[i]['edge'], 'texture': dice_colors[i]['texture']}
        else:
            dice[i]['colors'] = {'filling': dice_colors[-1]['filling'], 'edge': dice_colors[-1]['edge'], 'texture': dice_colors[-1]['texture']}
        polyhedra.append(dice[i])

cursor = 0
option = 'main'
def prepMenu():
    global cursor
    cursor = 0
    changeOptions('main')

opzioni = []

def changeOptions(o, reset_cur = True, clear_screen=True):
    global option
    global opzioni
    global cursor
    global selector
    if reset_cur:
        cursor = 0
    if clear_screen:
        clear()
    option = o
    if option == 'main':
        opzioni = ['Tira i dadi', 'Scegli i colori', 'Esci']
    if option == 'dice':
        selector = len(requestedDice)
        opzioni = []
    if option == 'color':
        opzioni = ['Imposta gli stessi colori per tutti i dadi', 'Imposta i colori dei dadi singolarmente', ('☑' if individualized_dice else '☐') + ' Usa colori diversi per i dadi']
    if option == 'picker':
        opzioni = ['facce', 'spigoli', 'numeri']
    if option == 'individual_col':
        opzioni = []
        for i in range(1, 7):
            opzioni.append('Dado '+str(i))
    if option == 'filling' or option == 'edge' or option == 'texture':
        opzioni = ['r', 'g', 'b']

def displayMenu():
    def calcTextBlock(txtList, p):
        hBlock = len(txtList)
        i = 0
        offsets = []
        for t in txtList:
            offsets.append({'x'  :p['x'] - round(len(t)/2), 'y' : p['y']+i-round(hBlock)})
            i += 2
        return offsets
    
    w = terminal_dims[0]
    h = terminal_dims[1]
    o = []
    if option == 'main':
        o = ["Seleziona l'opzione desiderata con le freccette, conferma con invio, esc per tornare indietro:", '']
    if option == 'main' or option == 'color' or option=='individual_col':
        for i in range(len(opzioni)):
            if cursor == i:
                o.append('> '+opzioni[i]+' <')
            else:
                o.append('  '+opzioni[i]+'  ')

    if option == 'dice':
        o = ['Inserisci i dadi nel formato xdy', '(dove x è il numero di dadi e y è il tipo che può essere 4, 6, 8, 10, 12, 20 o 100)', 'puoi inserire più combinazioni intervallandole con dei +, tipo 3d4+1d6, e un massimo cumulativo di 6 dadi:', requestedDice]
    deltaX = 0
    if option == 'color':
        deltaX = w/3
    offsets = calcTextBlock(o, {'x': w/2 - deltaX, 'y': h/2})
    if option == 'dice':
        setText(' '*(w-4), {'x': 0, 'y': offsets[-1]['y'], 'z': 100}, bg='default', fg={'r':255, 'g':255, 'b':255})
    for i in range(len(o)):
        setText(o[i], {'x': offsets[i]['x'], 'y': offsets[i]['y'], 'z': 100}, bg='default', fg={'r':255, 'g':255, 'b':255})
    if option == 'dice':
        blink = round(time.time()*2)%2 == 0
        bg = {'r':255, 'g':255, 'b':255} if blink else 'default'
        fg = {'r':12, 'g':12, 'b':12} if blink else {'r':255, 'g':255, 'b':255}
        setText(o[-1][selector] if selector<len(o[-1]) else ' ', {'x': offsets[-1]['x']+selector, 'y': offsets[-1]['y'], 'z': 100}, bg=bg, fg=fg)
    
    color_intensity = ' ⡀⣀⣠⣤⣦⣶⣾⣿'

    col_el = ''
    if option == 'picker':
        if cursor == 0:
            col_el = 'filling'
        if cursor == 1:
            col_el = 'edge'
        if cursor == 2:
            col_el = 'texture'
        for i in range(len(opzioni)):
            o = opzioni[i]
            if i == cursor:
                o = '> '+o+' <'
            else:
                o = '  '+o+'  '
            setText(o, {'x':round(5/6*w + w/9*(i-1) - len(o)/2), 'y': 5, 'z': 100}, bg='default', fg={'r':255, 'g':255, 'b':255})
    else:
        col_el = option

    if option == 'picker' or option == 'filling' or option == 'edge' or option == 'texture':
        col_pos = [round(13/18*w), round(7/9*w), round(5/6*w)]
        col_ind = ['r', 'g', 'b']
        col = []
        for i in range(3):
            col.append(dice_colors[dice_col_ind][col_el][col_ind[i]])
            for j in range(16):
                val = (15-j)*16
                p = {'x': col_pos[i], 'y': h//2 + (j-8)*2, 'z': 100}
                if col[i] == 255:
                    val = 0
                setText(color_intensity[max(0, min(16, col[i] - val))//2], p, bg='default', fg={'r':255, 'g':255, 'b':255})
            col_label = col_ind[i].upper()
            if option != 'picker' and i == cursor:
                col_label = '> '+col_label+' <'
            else:
                col_label = '  '+col_label+'  '
            setText(col_label, {'x': col_pos[i]-2, 'y': h//2 + 16, 'z': 100}, fg={'r': 255*(1 if col_ind[i]=='r' else 0), 'g': 255*(1 if col_ind[i]=='g' else 0), 'b': 255*(1 if col_ind[i]=='b' else 0)})
        
        border_y = True
        square_half_side = 5
        for i in range(h//2 - square_half_side - ((h//2)%2), h//2 + square_half_side-1 - ((h//2)%2)):
            border_x = True
            for j in range(round(17/18*w) - square_half_side - ((h//2)%2), round(17/18*w) + square_half_side-1 - ((h//2)%2)):
                fg = {'r':col[0], 'g':col[1], 'b':col[2]}
                if border_y or border_x:
                    fg = {'r':255, 'g':255, 'b':255}
                setText('#', {'x': j, 'y': i, 'z': 100}, fg=fg)
                border_x = False
            setText('#', {'x': round(17/18*w) + square_half_side-1 - ((h//2)%2), 'y': i, 'z': 100}, fg={'r':255, 'g':255, 'b':255})
            border_y = False
        for j in range(round(17/18*w) - square_half_side - ((h//2)%2), round(17/18*w) + square_half_side - ((h//2)%2)):
            setText('#', {'x': j, 'y': h//2 + square_half_side-1 - ((h//2)%2), 'z': 100}, fg={'r':255, 'g':255, 'b':255})

    if option == 'filling' or option == 'edge' or option == 'texture':
        inst = ['Usa le frecce su e giù', 'per aumentare/diminuire i colori', 'usa + e - per aumentarli/diminuirli', 'a passi più grandi']
        offsets = calcTextBlock(inst, {'x': round(w*5/6), 'y': h-6})
        for i in range(len(inst)):
            setText(inst[i], {'x': offsets[i]['x'], 'y': offsets[i]['y'], 'z': 100}, bg='default', fg={'r':255, 'g':255, 'b':255})

            
def handleInput():
    global pressedKeys
    global cursor
    global option
    global requestedDice
    global selector
    global dice_col_ind
    global individualized_dice
    if keyboard.is_pressed('freccia su') or keyboard.is_pressed('up arrow'):
        if 'up' not in pressedKeys and len(pressedKeys) == 0 and option!='picker':
            pressedKeys.add('up')
            if option == 'filling' or option == 'edge' or option == 'texture':
                col_keys = ['r', 'g', 'b']
                dice_colors[dice_col_ind][option][col_keys[cursor]] = min(dice_colors[dice_col_ind][option][col_keys[cursor]]+3, 255)
            else:
                cursor = max(cursor-1, 0)
    else:
        pressedKeys.discard('up')
    if keyboard.is_pressed('freccia giù') or keyboard.is_pressed('down arrow'):
        if 'down' not in pressedKeys and len(pressedKeys) == 0 and option!='picker':
            pressedKeys.add('down')
            if option == 'filling' or option == 'edge' or option == 'texture':
                col_keys = ['r', 'g', 'b']
                dice_colors[dice_col_ind][option][col_keys[cursor]] = max(dice_colors[dice_col_ind][option][col_keys[cursor]]-3, 0)
            else:
                cursor = min(cursor+1, len(opzioni)-1)
    else:
        pressedKeys.discard('down')
    if keyboard.is_pressed('+'):
        if '+' not in pressedKeys and len(pressedKeys) == 0:
            if option == 'filling' or option == 'edge' or option == 'texture':
                pressedKeys.add('+')
                col_keys = ['r', 'g', 'b']
                dice_colors[dice_col_ind][option][col_keys[cursor]] = min(dice_colors[dice_col_ind][option][col_keys[cursor]]+16, 255)
    else:
        pressedKeys.discard('+')
    if keyboard.is_pressed('-'):
        if '-' not in pressedKeys and len(pressedKeys) == 0:
            pressedKeys.add('-')
            if option == 'filling' or option == 'edge' or option == 'texture':
                col_keys = ['r', 'g', 'b']
                dice_colors[dice_col_ind][option][col_keys[cursor]] = max(dice_colors[dice_col_ind][option][col_keys[cursor]]-16, 0)
    else:
        pressedKeys.discard('-')
    if keyboard.is_pressed('freccia sinistra') or keyboard.is_pressed('left arrow'):
        if 'left' not in pressedKeys and len(pressedKeys) == 0:
            pressedKeys.add('left')
            if option == 'dice':
                selector = max(selector-1, 0)
            if option == 'picker' or option == 'filling' or option == 'edge' or option == 'texture':
                cursor = max(cursor-1, 0)
    else:
        pressedKeys.discard('left')
    if keyboard.is_pressed('freccia destra') or keyboard.is_pressed('right arrow'):
        if 'right' not in pressedKeys and len(pressedKeys) == 0:
            pressedKeys.add('right')
            if option == 'dice':
                selector = min(selector+1, len(requestedDice))
            if option == 'picker' or option == 'filling' or option == 'edge' or option == 'texture':
                cursor = min(cursor+1, len(opzioni)-1)
    else:
        pressedKeys.discard('right')
    if keyboard.is_pressed('enter'):
        if 'enter' not in pressedKeys and len(pressedKeys) == 0:
            pressedKeys.add('enter')
            if option == 'main':
                if cursor == 0:
                    changeOptions('dice')
                if cursor == 1:
                    changeOptions('color')
                if cursor == 2:
                    cleanScreen()
                    exit()
            elif option == 'dice':
                changeScene('throw')
            elif option == 'color':
                if cursor == 0:
                    dice_col_ind = 6
                    changeOptions('picker', clear_screen=False)
                if cursor == 1:
                    changeOptions('individual_col', clear_screen=False)
                if cursor == 2:
                    individualized_dice = not individualized_dice
                    changeOptions('color', reset_cur=False)
            elif option == 'individual_col':
                dice_col_ind = cursor
                changeOptions('picker', clear_screen=False)
            elif option == 'picker':
                col_el = ''
                if cursor == 0:
                    col_el = 'filling'
                if cursor == 1:
                    col_el = 'edge'
                if cursor == 2:
                    col_el = 'texture'
                changeOptions(col_el, clear_screen=False)
                
    else:
        pressedKeys.discard('enter')
    for k in '0123456789d+':
        if keyboard.is_pressed(k):
            if k not in pressedKeys and len(pressedKeys) == 0:
                pressedKeys.add(k)
                if option == 'dice':
                    requestedDice = requestedDice[:selector] + k + requestedDice[selector:]
                    selector += 1
        else:
            pressedKeys.discard(k)
    if keyboard.is_pressed('backspace'):
        if 'backspace' not in pressedKeys and len(pressedKeys) == 0:
            pressedKeys.add('backspace')
            if option == 'dice':
                requestedDice = requestedDice[:selector-1] + requestedDice[selector:]
                selector = max(0, selector-1)
    else:
        pressedKeys.discard('backspace')
    if keyboard.is_pressed('cancella') or keyboard.is_pressed('delete'):
        if 'delete' not in pressedKeys and len(pressedKeys) == 0:
            pressedKeys.add('delete')
            if option == 'dice':
                requestedDice = requestedDice[:selector] + requestedDice[selector+1:]
    else:
        pressedKeys.discard('delete')
    if keyboard.is_pressed('esc'):
        if 'esc' not in pressedKeys and len(pressedKeys) == 0:
            pressedKeys.add('esc')
            if option == 'color' or option == 'dice':
                changeOptions('main')
            elif option == 'individual_col' or option == 'same_col':
                cursor = 0 if option=='same_col' else 1
                changeOptions('color', reset_cur=False)
            elif option == 'picker':
                for i in range(len(mat)):
                    for j in range(round(terminal_dims[0]*2/3), len(mat[i])):
                        depth_mat[i][j] = 99
                        setTile({'x': j, 'y': i*2, 'z': 100}, ' ')
                if dice_col_ind < 6:
                    cursor = dice_col_ind
                    changeOptions('individual_col', reset_cur=False, clear_screen=False)
                else:
                    cursor = 0
                    changeOptions('color', reset_cur=False)
            elif option == 'filling' or option == 'edge' or option == 'texture':
                for i in range(len(mat)//2, len(mat)):
                    for j in range(round(terminal_dims[0]*2/3), len(mat[i])):
                        depth_mat[i][j] = 99
                        setTile({'x': j, 'y': i*2, 'z': 100}, ' ')
                if option == 'filling':
                    cursor = 0
                if option == 'edge':
                    cursor = 1
                if option == 'texture':
                    cursor = 2
                changeOptions('picker', reset_cur=False, clear_screen=False)
    else:
        pressedKeys.discard('esc')

    save()    

sceneSelector = 'menu'
scenes = {}
scenes['throw'] = [throwDice, handleDice, prepDice]
scenes['menu'] = [displayMenu, handleInput, prepMenu]

def cleanScreen():
    print('\033[0m', end='')
    print('')
    os.system('cls' if os.name == 'nt' else 'clear')

def changeScene(newScene):
    global finished
    finished = False
    global sceneSelector
    sceneSelector = newScene
    cleanScreen()
    scenes[sceneSelector][2]()

changeScene('menu')

while True:
    scenes[sceneSelector][0]()
    print_mat()
    scenes[sceneSelector][1]()
            
    currTime = time.time()
    if currTime - prevTime < min_frame_length:
        time.sleep(min_frame_length - (currTime - prevTime))

    currTime = time.time()
    deltaTime = currTime - prevTime
    prevTime = currTime
