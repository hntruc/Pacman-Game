import tkinter as tk
import tkinter.font as tkFont
from PIL import Image, ImageTk
from random import randint
import datetime
block_size = 25

def load_level(order, number):  # order(1->4), number(1->5). Ex:level1-2
    file = "../pacman/Data/level%s-%s.txt" % (order, number)
    with open(file) as f:
        # nR-number of Rows; nC-number of Cols
        nR, nC = tuple([int(i) for i in f.readline().split(',')])
        maze = [[int(i) for i in f.readline().strip()] for _ in range(nR)]
        pac_pos = tuple([int(i) for i in f.readline().split(',')])
        return (nR, nC), pac_pos, maze, order

def manhattan(x, y):
    return abs(x[0]-y[0])+abs(x[1]-y[1])

def check_step(ex, step):
    index = -1
    for i in range(len(step)):
        if step[i] not in ex:
            index = i
            return index
    times = [ex.count(step[i]) for i in range(len(step))]
    return times.index(min(times))

class pacman(tk.Frame):
    def draw_map(self):  # 0:y, 1:x
        ran_wall = randint(1,2)
        wall_path = '../pacman/Image/wall%s.jpg' %ran_wall
        self.frame_maze = tk.Canvas(
            width=self.size[1]*block_size, height=self.size[0]*block_size, bg='black')
        wall_img = Image.open(wall_path) 
        wall_img = ImageTk.PhotoImage(wall_img.resize(
            (block_size, block_size), Image.ANTIALIAS))
        
        ran_mons = randint(1,5)
        mons_path = '../pacman/Image/monster_%s.png' %ran_mons
        monster_img = Image.open(mons_path)
        monster_img = ImageTk.PhotoImage(monster_img.resize(
(block_size, block_size), Image.ANTIALIAS))
        
        food_img = Image.open('../pacman/Image/food.png')
        food_img = ImageTk.PhotoImage(food_img.resize(
            (block_size, block_size), Image.ANTIALIAS))

        for r in range(self.size[0]):  # get x
            for c in range(self.size[1]):  # get y
                if self.maze[r][c] == 1:
                    self.frame_maze.create_image(
                        c*block_size, r*block_size, anchor=tk.NW, image=wall_img)
                elif self.maze[r][c] == 2:
                    self.food.append([self.frame_maze.create_image(
                        c*block_size, r*block_size, anchor=tk.NW, image=food_img), (c, r)])
                elif self.maze[r][c] == 3:
                    self.mons.append([self.frame_maze.create_image(
                        c*block_size, r*block_size, anchor=tk.NW, image=monster_img), (c, r)])
        self.mons_og = self.mons
        self.frame_maze.image = [wall_img, food_img, monster_img]
        self.m_path = [[] for _ in range(len(self.mons))]
        self.m_pos = [0 for _ in range(len(self.mons))]
        self.frame_maze.pack()

    def draw_pac(self):
        pacman_img = Image.open('../pacman/Image/pac.png')
        pacman_img = pacman_img.resize(
            (block_size, block_size), Image.ANTIALIAS)
        pacman_img = ImageTk.PhotoImage(pacman_img)
        self.pac = self.frame_maze.create_image(
            self.pac_pos[1]*block_size, self.pac_pos[0]*block_size, anchor=tk.NW, image=pacman_img)
        self.frame_maze.image.append(pacman_img)

    def __init__(self, order, number, master=None):
        super().__init__(master)
        self.size, self.pac_pos, self.maze, self.lv = load_level(order, number)
        self.pac, self.food, self.mons, self.mons_og, self.path, self.eaten, self.m_path = None, [], [], [], [], [], []
        self.path_pos, self.score = 0, 0
        self.m_pos = []
        self.draw_map()
        self.draw_pac()
        self.frame_maze.pack()
        # level 3
        self.nR, self.nC = self.size
        #level 4 - list of food pac can see (r,c); list of mons pac can see, maze that pac scan at each move
        self.edible, self.monster, self.scan, self.ex_step = [], [], [[-1 for _ in range(len(self.maze[0]))] for _ in range(len(self.maze))], []

    def countScore(self):
        self.score = 0-len(self.path)+len(self.eaten)*20

    def pac_move(self):
        if self.path_pos >= len(self.path):
            self.frame_maze.update()
            self.frame_maze.after(700)
            return
        move_step = [self.path[self.path_pos][0] - self.pac_pos[0],
                     self.path[self.path_pos][1] - self.pac_pos[1]]
        self.pac_pos = [self.path[self.path_pos][0],
                        self.path[self.path_pos][1]]
        if self.maze[self.path[self.path_pos][0]][self.path[self.path_pos][1]] == 2:
            col, row, f = self.path[self.path_pos][1], self.path[self.path_pos][0], []
            for pos, i in enumerate(self.food):
                if i[1] == (col, row):
                    del_food = i[0]
                    self.frame_maze.delete(del_food)
                    f.append(pos)
            for ele in f:
                self.eaten.append(self.food.pop(ele))
        if self.maze[self.path[self.path_pos][0]][self.path[self.path_pos][1]] == 3:
            col, row = self.path[self.path_pos][1], self.path[self.path_pos][0]
            for i in self.mons:
                if i[1] == (col, row):
                    del_pac = self.pac
                    self.frame_maze.delete(del_pac)
                    self.frame_maze.after(70)
                    self.frame_maze.update()
                    return
        self.frame_maze.move(self.pac, move_step[1]*block_size, move_step[0]*block_size)
        self.frame_maze.after(70)
        self.frame_maze.update()
        self.path_pos += 1
        if self.lv in [3, 4]:
            return self.mon_move()
        if self.lv in [1, 2]:
            return self.pac_move()

    def mon_move(self):
        for i in range(len(self.mons)):
            if self.m_pos[i] >= len(self.m_path[i]):
                self.frame_maze.update()
                self.frame_maze.after(700)
                return
            cur = self.m_path[i][self.m_pos[i]]
            move_step = [cur[0] - self.mons[i][1][0],
                         cur[1] - self.mons[i][1][1]]
            self.frame_maze.move(
                self.mons[i][0], move_step[0]*block_size, move_step[1]*block_size)
            self.mons[i][1] = self.m_path[i][self.m_pos[i]]
            if self.mons[i][1] == (self.pac_pos[1], self.pac_pos[0]):
                self.frame_maze.after(70)
                self.frame_maze.update()
                return
            self.frame_maze.after(70)
            self.frame_maze.update()
            self.m_pos[i] += 1
        return self.pac_move()
# LEVEL 1

    def sln(self, trace):
        i, path = (self.food[0][1][1], self.food[0][1][0]), []
        while i != -1:
            path.append(i)
            i = trace[i[0]][i[1]]
        path.reverse()
        self.path = path
        self.countScore()

    def pathLevel1(self):
        fo = (self.food[0][1][1], self.food[0][1][0])
        h0 = manhattan((self.pac_pos[0], self.pac_pos[1]), fo)
        frontier, explored, trace = [[(self.pac_pos[0], self.pac_pos[1]), h0]], [], [
            [-1 for _ in range(self.size[1])] for _ in range(self.size[0])]
        dist = 0
        while len(frontier) != 0:
            frontier = sorted(frontier, key=lambda tup: (tup[1]))
            t = frontier.pop(0)
            explored.append(t[0])
            tem_front = [i[0] for i in frontier]
            if t[0][0] == fo[0] and t[0][1] == fo[1]:
                self.sln(trace)
                return
            dist += 1
            if t[0][0]-1 >= 0:  # up
                if self.maze[t[0][0]-1][t[0][1]] in [0, 2]:
                    if (t[0][0]-1, t[0][1]) not in tem_front and (t[0][0]-1, t[0][1]) not in explored:
                        trace[t[0][0]-1][t[0][1]] = t[0]
                        h1 = manhattan((t[0][0]-1, t[0][1]), fo)
                        f1 = dist + h1
                        frontier.append([(t[0][0]-1, t[0][1]), f1])
                    elif (t[0][0]-1, t[0][1]) in tem_front and dist + manhattan((t[0][0]-1, t[0][1]), fo) < frontier[tem_front.index((t[0][0]-1, t[0][1]))][1]:
                        trace[t[0][0]-1][t[0][1]] = t[0]
                        del frontier[tem_front.index((t[0][0]-1, t[0][1]))]
                        f1 = dist + manhattan((t[0][0]-1, t[0][1]), fo)
                        frontier.insert((t[0][0]-1, t[0][1]), f1)
            if t[0][0]+1 < self.size[0]:  # down
                if self.maze[t[0][0]+1][t[0][1]] in [0, 2]:
                    if (t[0][0]+1, t[0][1]) not in tem_front and (t[0][0]+1, t[0][1]) not in explored:
                        trace[t[0][0]+1][t[0][1]] = t[0]
                        h1 = manhattan((t[0][0]+1, t[0][1]), fo)
                        f1 = dist + h1
                        frontier.append([(t[0][0]+1, t[0][1]), f1])
                    elif (t[0][0]+1, t[0][1]) in tem_front and dist + manhattan((t[0][0]+1, t[0][1]), fo) < frontier[tem_front.index((t[0][0]+1, t[0][1]))][1]:
                        trace[t[0][0]+1][t[0][1]] = t[0]
                        del frontier[tem_front.index((t[0][0]+1, t[0][1]))]
                        f1 = dist + manhattan((t[0][0]+1, t[0][1]), fo)
                        frontier.insert((t[0][0]+1, t[0][1]), f1)
            if t[0][1]+1 < self.size[1]:  # right
                if self.maze[t[0][0]][t[0][1]+1] in [0, 2]:
                    if (t[0][0], t[0][1]+1) not in tem_front and (t[0][0], t[0][1]+1) not in explored:
                        trace[t[0][0]][t[0][1]+1] = t[0]
                        h1 = manhattan((t[0][0], t[0][1]+1), fo)
                        f1 = dist + h1
                        frontier.append([(t[0][0], t[0][1]+1), f1])
                    elif (t[0][0], t[0][1]+1) in tem_front and dist + manhattan((t[0][0], t[0][1]+1), fo) < frontier[tem_front.index((t[0][0], t[0][1]+1))][1]:
                        trace[t[0][0]][t[0][1]+1] = t[0]
                        del frontier[tem_front.index((t[0][0], t[0][1]+1))]
                        f1 = dist + manhattan((t[0][0], t[0][1]+1), fo)
                        frontier.insert((t[0][0], t[0][1]+1), f1)
            if t[0][1]-1 >= 0:  # left
                if self.maze[t[0][0]][t[0][1]-1] in [0, 2]:
                    if (t[0][0], t[0][1]-1) not in tem_front and (t[0][0], t[0][1]-1) not in explored:
                        trace[t[0][0]][t[0][1]-1] = t[0]
                        h1 = manhattan((t[0][0], t[0][1]-1), fo)
                        f1 = dist + h1
                        frontier.append([(t[0][0], t[0][1]-1), f1])
                    elif (t[0][0], t[0][1]-1) in tem_front and dist + manhattan((t[0][0], t[0][1]-1), fo) < frontier[tem_front.index((t[0][0], t[0][1]-1))][1]:
                        trace[t[0][0]][t[0][1]-1] = t[0]
                        del frontier[tem_front.index((t[0][0], t[0][1]-1))]
                        f1 = dist + manhattan((t[0][0], t[0][1]-1), fo)
                        frontier.insert((t[0][0], t[0][1]-1), f1)
        print("Cannot find the path.")
        return

    def BFS(self):
        fo = (self.food[0][1][1], self.food[0][1][0])
        frontier, explored, trace = [(self.pac_pos[0], self.pac_pos[1])], [], [
            [-1 for _ in range(self.size[1])] for _ in range(self.size[0])]
        while len(frontier) != 0:
            t = frontier.pop(0)
            explored.append(t)
            if t[0] == fo[0] and t[1] == fo[1]:
                self.sln(trace)
                return
            if t[0]-1 >= 0:  # up
                if self.maze[t[0]-1][t[1]] in [0, 2]:
                    if (t[0]-1, t[1]) not in frontier and (t[0]-1, t[1]) not in explored:
                        trace[t[0]-1][t[1]] = t
                        frontier.append((t[0]-1, t[1]))
                    if t[0]-1 == fo[0] and t[1] == fo[1]:
                        self.sln(trace)
                        return
            if t[0] + 1 < self.size[0]:  # down
                if self.maze[t[0]+1][t[1]] in [0, 2]:
                    if (t[0]+1, t[1]) not in frontier and (t[0]+1, t[1]) not in explored:
                        trace[t[0]+1][t[1]] = t
                        frontier.append((t[0]+1, t[1]))
                    if t[0] + 1 == fo[0] and t[1] == fo[1]:
                        self.sln(trace)
                        return
            if t[1]+1 < self.size[1]:  # right
                if self.maze[t[0]][t[1]+1] in [0, 2]:
                    if (t[0], t[1]+1) not in frontier and (t[0], t[1]+1) not in explored:
                        trace[t[0]][t[1]+1] = t
                        frontier.append((t[0], t[1]+1))
                    if t[0] == fo[0] and t[1] + 1 == fo[1]:
                        self.sln(trace)
                        return
            if t[1]-1 >= 0:  # left
                if self.maze[t[0]][t[1]-1] in [0, 2]:
                    if (t[0], t[1]-1) not in frontier and (t[0], t[1]-1) not in explored:
                        trace[t[0]][t[1]-1] = t
                        frontier.append((t[0], t[1]-1))
                    if t[0] == fo[0] and t[1]-1 == fo[1]:
                        self.sln(trace)
                        return
        print("Cannot find the path.")
        return

# LEVEL 2
    def find_min(self, frontier):
        return min([f[0] for f in frontier])

    def is_exist(self, coor, frontier, explored):
        for ex in explored:
            if coor == ex:
                return True

        for fr in frontier:
            if coor == fr[1]:
                return True
        return False

    def available_move(self, node):
        a_rows = [node[0] - i for i in [-1, 0, 1]
                  if -1 < node[0] - i < self.size[0]]
        a_cols = [node[1] - i for i in [-1, 0, 1]
                  if -1 < node[1] - i < self.size[1]]
        move = [(r, c)
                for r in a_rows for c in a_cols if manhattan((r, c), node) == 1 and self.maze[r][c] != 1 and self.maze[r][c] != 3]

        return move

    def parent_node(self, explored, node):
        for ex in explored:
            for m in self.available_move(ex):
                if node == m:  # if a node in explored can reach the present node
                    return ex  # return that node
        return (-1, -1)

    def find_path(self, explored, node):
        if node == self.pac_pos:
            return [node]
        # manhattan(m, (0, 0)) < manhattan(node, (0, 0)) (choose the shorter path)
        par = self.parent_node(explored, node)
        if par != (-1, -1):
            return self.find_path(explored, par) + [node]

    def path_level2(self):
        # (x, y)
        goal = (self.food[0][1][1], self.food[0][1][0])
        # cost, coordinate
        frontier = [[manhattan(self.pac_pos, goal), self.pac_pos]]
        explored = []
        while frontier:
            for i, n in enumerate(frontier):
                if n[0] == self.find_min(frontier):
                    cost, node = frontier.pop(i)
                    explored += [node]
                    if node == goal:
                        self.path = self.find_path(explored, goal)
                        self.countScore()
                        return
                    break
            move = self.available_move(node)
            frontier += [[cost - manhattan(node, goal) + 1 + manhattan(m, goal), m]
                         for m in move if not self.is_exist(m, frontier, explored)]
        return [-1]

# LEVEL 3
    def a_move(self, r, c):  # available area of movement
        a_rows = [r - i for i in [-1, 0, 1]
                  if -1 < r - i < self.nR]
        a_cols = [c - i for i in [-1, 0, 1]
                  if -1 < c - i < self.nC]
        return a_rows, a_cols

    def mons_dir(self, m):
        a_rows, a_cols = self.a_move(m[1][1], m[1][0])
        return [(c, r) for r in a_rows for c in a_cols
                if manhattan(m[1], (c, r)) == 1 and self.maze[r][c] != 1]  # available move
    
    def mons_path(self, m_path):  # (col, row)
        if len(self.mons) == 0:
            return
        new_mons = []
        for m, m_og in zip(self.mons, self.mons_og):
            temp = self.mons_dir(m)
            self.maze[m[1][1]][m[1][0]] = 0  # release tile

            a_m = [move for move in temp if abs(
                m_og[1][0] - move[0]) < 2 and abs(m_og[1][1] - move[1]) < 2]
            # update maze when move
            choice = a_m[randint(0, len(a_m) - 1)]
            new_mons += [[m[0], choice]]
            self.maze[choice[1]][choice[0]] = 3
        # change monster position
        self.mons = new_mons
        # path for monster
        for i, step in enumerate(new_mons):
            m_path[i] += [step[1]]

    def dir(self, pos):  # decide which direction pacman
        a_rows, a_cols = self.a_move(pos[0], pos[1])
        return [(r, c)
                for r in a_rows for c in a_cols if manhattan((r, c), pos) == 1
                and self.maze[r][c] != 1 and self.maze[r][c] != 3]

    def predict(self, m):  # predict monster move
        a_rows, a_cols = self.a_move(m[0], m[1])
        return [(r, c) for r in a_rows for c in a_cols
                if manhattan(m, (r, c)) == 1 and self.maze[r][c] != 1]  # available move

    def prio_food(self, pos, m_path):
        stack = [pos]
        path = []
        explored = [pos]
        step_back = 0
        count = 0
        while stack:
            temp = self.dir(stack[-1])

            s_rows = [stack[-1][0] - i for i in [-3, -2, -1, 0, 1, 2, 3]
                      if -1 < stack[-1][0] - i < self.nR]
            s_cols = [stack[-1][1] - i for i in [-3, -2, -1, 0, 1, 2, 3]
                      if -1 < stack[-1][1] - i < self.nC]

            area = [(r, c) for r in s_rows for c in s_cols]

            mons = [m for m in area if self.maze[m[0]][m[1]] == 3]
            food = [f for f in area if self.maze[f[0]][f[1]] == 2]

            dirc = [d for d in temp if not self.is_exist(d, [], explored)
                    and all(not self.is_exist(d, [], self.predict(m)) for m in mons)]

            if food and dirc:
                distance = [min([manhattan(f, d) for f in food])
                            for d in dirc]

                min_dis = distance.index(min(distance))
                dirc = [dirc[min_dis]]
                if min(distance) == 0:
                    count += 1
                    self.maze[dirc[0][0]][dirc[0][1]] = 0
            if dirc:
                step_back = 0
                stack += [dirc[0]]
                path += [dirc[0]]
                self.mons_path(m_path)
                explored += [dirc[0]]
            else:
                step_back += 1
                if step_back > 30:
                    break

                if len(stack) > 2:
                    if all(not self.is_exist(stack[-2], [], self.predict(m)) for m in mons):
                        stack.pop(-1)
                    else:
                        for m in mons:
                            if self.is_exist(stack[-1], [], self.predict(m)):
                                stack.pop(-1)
                                break
                if stack:
                    path += [stack[-1]]
                    self.mons_path(m_path)
                else:
                    break
        return path, count

    def path_level3(self):
        pos = self.pac_pos
        exp, path = [], []
        m_path = [[] for _ in range(len(self.mons))]
        # self.DFS(pos, exp, path, m_path, food)
        path, food = self.prio_food(pos, m_path)
        self.score = food * 20 - len(path)
        self.mons = [m for m in self.mons_og]
        for f in self.food:
            self.maze[f[1][1]][f[1][0]] = 2
        self.m_path = m_path
        self.path = path

# LEVEL 4
    def pac_scan(self):
        p_r, p_c = self.pac_pos
        if p_r > 2 and p_r < self.size[0]-3:
            if p_c > 2 and p_c < self.size[1]-3:
                for i in range(-3,4,1):
                    for j in range(-3,4,1):
                        self.scan[p_r+i][p_c+j] = self.maze[p_r+i][p_c+j]
            elif p_c >= self.size[1]-3:
                for i in range(-3,4,1):
                    for j in range(-3,self.size[1]-p_c,1):
                        self.scan[p_r+i][p_c+j] = self.maze[p_r+i][p_c+j]
            else:
                for i in range(-3,4,1):
                    for j in range(p_c+4):
                        self.scan[p_r+i][j] = self.maze[p_r+i][j]
        elif p_r >= self.size[0]-3:
            if p_c > 2 and p_c < self.size[1]-3:
                for i in range(-3,self.size[0]-p_r,1):
                    for j in range(-3,4,1):
                        self.scan[p_r+i][p_c+j] = self.maze[p_r+i][p_c+j]
            elif p_c >= self.size[1]-3:
                for i in range(-3,self.size[0]-p_r,1):
                    for j in range(-3,self.size[0]-p_c,1):
                        self.scan[p_r+i][p_c+j] = self.maze[p_r+i][p_c+j]
            else:
                for i in range(-3,self.size[0]-p_r,1):
                    for j in range(p_c+4):
                        self.scan[p_r+i][j] = self.maze[p_r+i][j]
        else:
            if p_c > 2 and p_c < self.size[1]-3:
                for i in range(p_r+4):
                    for j in range(-3,4,1):
                        self.scan[i][p_c+j] = self.maze[i][p_c+j]
            elif p_c >= self.size[1]-3:
                for i in range(p_r+4):
                    for j in range(-3,self.size[1]-p_c,1):
                        self.scan[i][p_c+j] = self.maze[i][p_c+j]
            else:
                for i in range(p_r+4):
                    for j in range(p_c+4):
                        self.scan[i][j] = self.maze[i][j]
        mon, f = [],[]
        for r in range(len(self.scan)):
            for c in range(len(self.scan[0])):
                if self.scan[r][c] == 2:
                    f.append((r,c))
                if self.scan[r][c] == 3:
                    mon.append((r,c))
        self.edible, self.monster = f, mon
        
        #return self.scan
    
    def moveable(self):
        self.pac_scan()
        p_r, p_c = self.pac_pos
        avai_move = []
        if p_c-1 >= 0:
            if self.scan[p_r][p_c-1] in [0, 2]:
                if len(self.monster) > 0:
                    dist_m = [manhattan((p_r,p_c-1),(self.monster[i])) for i in range(len(self.monster))]
                    if min(dist_m) >= 2: 
                        avai_move.append((p_r,p_c-1))
                elif len(self.monster) == 0:
                    avai_move.append((p_r,p_c-1))
        if p_c+1 <= self.size[1]:
            if self.scan[p_r][p_c+1] in [0, 2]:
                if len(self.monster) > 0:
                    dist_m = [manhattan((p_r,p_c+1),(self.monster[i])) for i in range(len(self.monster))]
                    if min(dist_m) >= 2: 
                        avai_move.append((p_r,p_c+1))
                elif len(self.monster) == 0:
                    avai_move.append((p_r,p_c+1))
        if p_r-1 >= 0:
            if self.scan[p_r-1][p_c] in [0, 2]:
                if len(self.monster) > 0:
                    dist_m = [manhattan((p_r-1,p_c),(self.monster[i])) for i in range(len(self.monster))]
                    if min(dist_m) >= 2: 
                        avai_move.append((p_r-1,p_c))
                elif len(self.monster) == 0:
                    avai_move.append((p_r-1,p_c))
        if p_r+1 <= self.size[0]:
            if self.scan[p_r+1][p_c] in [0, 2]:
                if len(self.monster) > 0:
                    dist_m = [manhattan((p_r+1,p_c),(self.monster[i])) for i in range(len(self.monster))]
                    if min(dist_m) >= 2: 
                        avai_move.append((p_r+1,p_c))
                elif len(self.monster) == 0:
                    avai_move.append((p_r+1,p_c))
        self.path = avai_move

    def pac_lv4(self):
        self.moveable()
        dist_m = [manhattan((self.pac_pos[0]+1,self.pac_pos[1]),(self.monster[i])) for i in range(len(self.monster))]
        if len(self.food) == 0 or len(self.path) == 0:
            return
        if len(dist_m) != 0:
            if len(self.path) == 1 and min(dist_m)<=2:
                return
        if len(self.edible)!=0:
            dist_f = [manhattan((self.edible[0]),(self.path[i])) for i in range(len(self.path))]
            min_d = min(dist_f)
            temp = []
            for pos, i in enumerate(dist_f):
                if i == min_d:
                    temp.append(pos)
            times = [self.ex_step.count(self.path[i]) for i in temp]
            if min(times) <=2:
                self.path_pos = temp[times.index(min(times))]
            else:
                self.path_pos = check_step(self.ex_step,self.path)
        else:
            self.path_pos = check_step(self.ex_step,self.path)

        move_step = [self.path[self.path_pos][0] - self.pac_pos[0],
                     self.path[self.path_pos][1] - self.pac_pos[1]]
        self.pac_pos = [self.path[self.path_pos][0],
                        self.path[self.path_pos][1]]
        if self.scan[self.path[self.path_pos][0]][self.path[self.path_pos][1]] == 2:
            col, row, f = self.path[self.path_pos][1], self.path[self.path_pos][0], []
            for pos, i in enumerate(self.food):
                if i[1] == (col, row):
                    del_food = i[0]
                    self.frame_maze.delete(del_food)
                    f.append(pos)
                    self.maze[row][col] = 0
            for ele in f:
                self.eaten.append(self.food.pop(ele))
        if self.scan[self.path[self.path_pos][0]][self.path[self.path_pos][1]] == 3:
            col, row = self.path[self.path_pos][1], self.path[self.path_pos][0]
            for i in self.mons:
                if i[1] == (col, row):
                    del_pac = self.pac
                    self.frame_maze.delete(del_pac)
                    self.frame_maze.after(70)
                    self.frame_maze.update()
                    return 3
        self.frame_maze.move(self.pac, move_step[1]*block_size, move_step[0]*block_size)
        self.ex_step.append(tuple(self.pac_pos))
        self.frame_maze.after(70)
        self.frame_maze.update()
        self.mon_lv4()
        # self.pac_move()
    
    def mon_step(self):
        avai_move = [[] for _ in range(len(self.mons))]
        for m in range(len(self.mons)):
            p_r, p_c = self.mons[m][1][1], self.mons[m][1][0]
            dist_p, coor = [], []
            if p_c-1 >= 0:
                if self.maze[p_r][p_c-1] in [0, 2, 3]:
                    dist_p.append(manhattan((p_r,p_c-1),(self.pac_pos)))
                    coor.append((p_r,p_c-1))
            if p_c+1 <= self.size[1]:
                if self.maze[p_r][p_c+1] in [0, 2, 3]:
                    dist_p.append(manhattan((p_r,p_c+1),(self.pac_pos)))
                    coor.append((p_r,p_c+1))
            if p_r-1 >= 0:
                if self.maze[p_r-1][p_c] in [0, 2, 3]:
                    dist_p.append(manhattan((p_r-1,p_c),(self.pac_pos)))
                    coor.append((p_r-1,p_c))
            if p_r+1 <= self.size[0]:
                if self.maze[p_r+1][p_c] in [0, 2, 3]:
                    dist_p.append(manhattan((p_r+1,p_c),(self.pac_pos)))
                    coor.append((p_r+1,p_c))
            avai_move[m].append(coor[dist_p.index(min(dist_p))])
        self.m_path = avai_move
    
    def mon_lv4(self):
        self.mon_step()
        for i in range(len(self.mons)):
            if len(self.m_path[i]) != 0:
                cur = self.m_path[i][0]
                move_step = [cur[1] - self.mons[i][1][0],
                             cur[0] - self.mons[i][1][1]]
                self.frame_maze.move(
                    self.mons[i][0], move_step[0]*block_size, move_step[1]*block_size)
                self.maze[self.mons[i][1][1]][self.mons[i][1][0]] = 0
                self.mons[i][1] = (self.m_path[i][0][1], self.m_path[i][0][0])
                if self.mons[i][1] == (self.pac_pos[1], self.pac_pos[0]):
                    del_pac = self.pac
                    self.frame_maze.delete(del_pac)
                    self.frame_maze.after(70)
                    self.frame_maze.update()
                    return
                self.maze[self.m_path[i][0][0]][self.m_path[i][0][1]] = 3
                self.frame_maze.after(70)
                self.frame_maze.update()
        self.pac_lv4()

    def path_level4(self):
        a = self.pac_lv4()
        self.score = 0 - len(self.ex_step) + 20*len(self.eaten)
        if a==3:
            self.score -=1000

######################## MENU ##########################

# main_menu


def start():
    start_frame.destroy()
    maze_menu()


def init_menu():
    #frame = tk.Frame(master)
    # frame.pack()
    global start_frame
    start_frame = tk.Tk()
    start_frame.geometry('700x400+300+100')
    start_frame.resizable(width=False, height=False)
    start_frame.title('PACMAN')

    background = tk.Canvas(start_frame, width=700, height=400, bg='black')
    background_img = Image.open('../pacman/Image/bg.jpg')
    background_img = ImageTk.PhotoImage(
        background_img.resize((700, 400), Image.ANTIALIAS))
    background.create_image(0, 0, anchor=tk.NW, image=background_img)

    start_img = Image.open('../pacman/Image/start_btn.png')
    start_img = ImageTk.PhotoImage(start_img.resize((90, 70), Image.ANTIALIAS))

    start_btn = tk.Button(start_frame, text="START", command=start)
    start_btn.pack(padx=10, pady=10)
    start_btn.place(x=311, y=180)
    start_btn.config(height=70, width=90, activebackground='black',
                     image=start_img, bg='black', borderwidth=0)

    background.pack()
    start_frame.mainloop()


def back_menu():
    end_frame.destroy()
    global start_frame
    start_frame = tk.Tk()
    start_frame.geometry('700x400+300+100')
    start_frame.resizable(width=False, height=False)
    start_frame.title('PACMAN')

    background = tk.Canvas(start_frame, width=700, height=400, bg='black')
    background_img = Image.open('../pacman/Image/bg.jpg')
    background_img = ImageTk.PhotoImage(
        background_img.resize((700, 400), Image.ANTIALIAS))
    background.create_image(0, 0, anchor=tk.NW, image=background_img)

    start_img = Image.open('../pacman/Image/start_btn.png')
    start_img = ImageTk.PhotoImage(start_img.resize((90, 70), Image.ANTIALIAS))

    start_btn = tk.Button(start_frame, text="START", command=start)
    start_btn.pack(padx=10, pady=10)
    start_btn.place(x=311, y=180)
    start_btn.config(height=70, width=90, activebackground='black',
                     image=start_img, bg='black', borderwidth=0)

    background.pack()
    start_frame.mainloop()

# mazemenu


def level1():
    ini_frame.destroy()
    global draw_level
    draw_level = tk.Tk()
    number = randint(1, 5)
    (nR, nC), maze, pos, lv = load_level(1, number)
    w, h = (draw_level.winfo_screenwidth()-nC *
            block_size)//2, (draw_level.winfo_screenheight()-nR*block_size)//2
    draw_level.geometry('+'+str(w)+'+'+str(h-30))
    del (nR, nC), maze, pos, lv
    draw_level.resizable(width=False, height=False)
    draw_level.title('LEVEL 1')
    game = pacman(1, number, draw_level)
    game.pathLevel1()
    game.pac_move()
    end_dlg(game.score)
    draw_level.mainloop()


def level2():
    ini_frame.destroy()
    global draw_level
    draw_level = tk.Tk()
    number = randint(1, 5)
    (nR, nC), maze, pos, lv = load_level(2, number)
    w, h = (draw_level.winfo_screenwidth()-nC *
            block_size)//2, (draw_level.winfo_screenheight()-nR*block_size)//2
    draw_level.geometry('+'+str(w)+'+'+str(h-30))
    del (nR, nC), maze, pos, lv
    draw_level.resizable(width=False, height=False)
    draw_level.title('LEVEL 2')
    game = pacman(2, number, draw_level)
    game.pathLevel1()
    game.pac_move()
    end_dlg(game.score)
    draw_level.mainloop()


def level3():
    ini_frame.destroy()
    global draw_level
    draw_level = tk.Tk()
    number = randint(1, 5)
    (nR, nC), maze, pos, lv = load_level(3, number)
    w, h = (draw_level.winfo_screenwidth()-nC *
            block_size)//2, (draw_level.winfo_screenheight()-nR*block_size)//2
    draw_level.geometry('+'+str(w)+'+'+str(h-30))
    del (nR, nC), maze, pos, lv
    draw_level.resizable(width=False, height=False)
    draw_level.title('LEVEL 3')
    game = pacman(3, number, draw_level)
    game.path_level3()
    game.pac_move()

    end_dlg(game.score)
    draw_level.mainloop()


def level4():
    ini_frame.destroy()
    global draw_level
    draw_level = tk.Tk()
    number = randint(1, 5)
    (nR, nC), maze, pos, lv = load_level(4, number)
    w, h = (draw_level.winfo_screenwidth()-nC *
            block_size)//2, (draw_level.winfo_screenheight()-nR*block_size)//2
    draw_level.geometry('+'+str(w)+'+'+str(h-30))
    del (nR, nC), maze, pos, lv
    draw_level.resizable(width=False, height=False)
    draw_level.title('LEVEL 4')
    game = pacman(4, number, draw_level)
    game.path_level4()
    end_dlg(game.score)
    draw_level.mainloop()


def maze_menu():
    #frame = tk.Frame(master)
    # frame.pack()
    global ini_frame
    ini_frame = tk.Tk()
    ini_frame.geometry('871x520+220+60')
    ini_frame.resizable(width=False, height=False)
    ini_frame.title('LEVEL MENU')

    background = tk.Canvas(ini_frame, width=871, height=520, bg='black')

    button_img = Image.open('../pacman/Image/green_btn.png')
    button_img = ImageTk.PhotoImage(
        button_img.resize((30, 30), Image.ANTIALIAS))

    lvl1_img = Image.open('../pacman/Image/m1_lv1.png')
    lvl1_img = ImageTk.PhotoImage(lvl1_img.resize((370, 200), Image.ANTIALIAS))
    label1 = tk.Label(background, image=lvl1_img,
                      highlightthickness=2, borderwidth=1)
    label1.image = lvl1_img
    label1.place(x=20, y=20)

    lv1_btn = tk.Button(ini_frame, text="LEVEL 1", command=level1)
    lv1_btn.pack(padx=50, pady=20)
    lv1_btn.place(x=175, y=227)
    lv1_btn.config(height=25, width=50, activebackground='black',
                   image=button_img, bg='#90B666', borderwidth=1, relief="groove")

    lvl2_img = Image.open('../pacman/Image/m1_lv2.png')
    lvl2_img = ImageTk.PhotoImage(lvl2_img.resize((370, 200), Image.ANTIALIAS))
    label2 = tk.Label(background, image=lvl2_img,
                      highlightthickness=2, borderwidth=1)
    label2.image = lvl2_img
    label2.place(x=480, y=20)

    lv2_btn = tk.Button(ini_frame, text="LEVEL 2",
                        command=level2, borderwidth=0)
    lv2_btn.pack(padx=50, pady=20)
    lv2_btn.place(x=655, y=227)
    lv2_btn.config(height=25, width=50, activebackground='black',
                   image=button_img, bg='#90B666', borderwidth=1, relief="groove")

    lvl3_img = Image.open('../pacman/Image/m1_lv3.png')
    lvl3_img = ImageTk.PhotoImage(lvl3_img.resize((370, 200), Image.ANTIALIAS))
    label3 = tk.Label(background, image=lvl3_img,
                      highlightthickness=2, borderwidth=1)
    label3.image = lvl3_img
    label3.place(x=20, y=270)

    lv3_btn = tk.Button(ini_frame, text="LEVEL 3", command=level3)
    lv3_btn.pack(padx=50, pady=20)
    lv3_btn.place(x=175, y=477)
    lv3_btn.config(height=25, width=50, activebackground='black',
                   image=button_img, bg='#90B666', borderwidth=1, relief="groove")

    lvl4_img = Image.open('../pacman/Image/m1_lv4.png')
    lvl4_img = ImageTk.PhotoImage(lvl4_img.resize((370, 200), Image.ANTIALIAS))
    label4 = tk.Label(background, image=lvl4_img,
                      highlightthickness=2, borderwidth=1)
    label4.image = lvl4_img
    label4.place(x=480, y=270)

    lv4_btn = tk.Button(ini_frame, text="LEVEL 4", command=level4)
    lv4_btn.pack(padx=50, pady=20)
    lv4_btn.place(x=655, y=477)
    lv4_btn.config(height=25, width=50, activebackground='black',
                   image=button_img, bg='#90B666', borderwidth=1, relief="groove")

    background.pack()
    ini_frame.mainloop()


def close_program():
    end_frame.destroy()
    # draw_level.destroy()


def end_dlg(score):
    draw_level.destroy()
    global end_frame
    end_frame = tk.Tk()
    end_frame.geometry('300x300+500+150')
    end_frame.resizable(width=False, height=False)
    end_frame.title('GAME OPTION')

    end_game = tk.Canvas(end_frame, width=300, height=300, bg='black')

    # resize font
    fontStyleNoti = tkFont.Font(family="Lucida Grande", size=20)
    fontStyleScore = tkFont.Font(family="Lucida Grande", size=18)

    # Placing the Label at
    # the middle of the Tk() window
    # relx and rely should be properly
    # set to position the label on
    # Tk() window

    #score = 10
    Label_noti = tk.Label(end_frame, text='GAME SCORE', width=20,
                          height=4, bg='black', fg='red', font=fontStyleNoti)
    Label_noti.place(relx=0.5, rely=0.25, anchor='center')

    Label_score = tk.Label(end_frame, text=score, width=20,
                           height=4, fg='#00ffff', bg='black', font=fontStyleScore)
    Label_score.place(relx=0.5, rely=0.5, anchor='center')

    # Buttons
    quit_btn = tk.Button(end_game, text="QUIT", command=close_program)
    quit_btn.pack(padx=50, pady=20)
    quit_btn.place(x=40, y=260)

    menu_btn = tk.Button(end_game, text="MENU", command=back_menu)
    menu_btn.pack(padx=50, pady=20)
    menu_btn.place(x=220, y=260)

    end_game.pack()
    end_frame.mainloop()

init_menu()