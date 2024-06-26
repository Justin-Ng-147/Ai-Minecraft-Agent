from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #6: Discrete movement, rewards, and learning

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import malmo.MalmoPython as MalmoPython
import json
import logging
import os
import random
import sys
import time
import tkinter as tk

# if sys.version_info[0] == 2:
#     print(True)
#     # Workaround for https://github.com/PythonCharmers/python-future/issues/262
#     import tkinter as tk
# else:
#     import tkinter as tk


class TabQAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self):
        self.epsilon = 0.00 # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.reward = 0 #haley change


        self._MOVE_COST = -5
        self._JUMP_COST = -5

        # possible actions migh need to change it to move turnleft turn right and jump

        # add 4 more actions jump up 1 move 2 for nesw
        self.actions = ["moves", "movew", "movee","jumps", "jumpw", "jumpe"]

        self.q_table = {}
        self.canvas = None
        self.root = None
        self.frame = None
        self.scrollbar = None
    def move(self, move: str):


        # check if we have already moved to said spot if yes add -20 penalty
        # we can do this by checking if state already exists in q_table


        # time.sleep(0.3)
        if move == "moves":
            # north
            agent_host.sendCommand("move 1"  )
        elif move =="movee":
            # east
            agent_host.sendCommand("moveeast 1" )
        elif move =="moves":
            # south
            agent_host.sendCommand("move -1")
        elif move =="movew":
            # west
            agent_host.sendCommand("movewest 1")

        elif move == "jumpn":
            # j npi
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("move 1")
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("move 1")
        elif move =="jumpe":
            # j e
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("moveeast 1"  )
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("moveeast 1"  )
        elif move =="jumps":
            # j s
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("move 1"  )
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("move 1"  )
        elif move =="jumpw":
            # j w
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("movewest 1"  )
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("movewest 1"  )
        

    # create json files
    def save_q_table(self,q_table, map = 1):
        # change file name if needed
        filename = ''
        if map == 1:
            filename = "q_table1.json"
        elif map == 2:
            filename = "q_table2.json"
        elif map == 3:
            filename = "q_table3.json"
        with open(filename, 'w') as f:
            f.write(json.dumps(q_table))

    # load json files
    def load_q_table(self, map = 1):
        if map == 1:
            print("Loading current q t:")
            q_file = open(r"q_table1.json")
            self.q_table= json.load(q_file)
        elif map == 2:
            print("Loading current q t:")
            q_file = open(r"q_table2.json")
            self.q_table= json.load(q_file)
        elif map == 3:
            print("Loading current q t:")
            q_file = open(r"q_table3.json")
            self.q_table= json.load(q_file)

    def updateQTable( self, reward, current_state ):
        """Change q_table to reflect what we have learnt."""
    
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        
        # TODO: what should the new action value be?
        new_q = reward
        
        # assign the new action value to the Q-table
        if old_q == 0 or new_q > old_q:
            self.q_table[self.prev_s][self.prev_a] = new_q
        
    def updateQTableFromTerminatingState( self, reward ):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        
        # TODO: what should the new action value be?
        new_q = reward
        
        # assign the new action value to the Q-table
        if old_q == 0 or new_q > old_q:
            self.q_table[self.prev_s][self.prev_a] = new_q
        
    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        self.logger.debug(obs)
        
         
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        
        #changes::::: changed ints to float so we can move right :::::::
        current_s = "%f:%f" % (obs[u'XPos'], obs[u'ZPos'])
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))

        if current_s not in self.q_table:
            self.q_table[current_s] = ([0] * len(self.actions))

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable( current_r, current_s )

        # drawing the q table
        self.drawQ( curr_x = float(obs[u'XPos']), curr_y = float(obs[u'ZPos']) )

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
            self.logger.info("Random action: %s" % self.actions[a])
        else:
            # checking for max val on q table
            m = max(self.q_table[current_s])

             #changes::::: hard codes for more punishment for reaching same block:::::::
            if m == self._MOVE_COST: #moving to a block it already has been to 
                for x in range(0,len(self.actions)):
                    if self.q_table[current_s][x] == m:
                        self.q_table[current_s][x] -= 100
                m = max(self.q_table[current_s])
                
            elif m == self._JUMP_COST: #jumping to a block it already has been to
                for x in range(0,len(self.actions)):
                    if self.q_table[current_s][x] == m:
                        self.q_table[current_s][x] -= 100
                m = max(self.q_table[current_s])
            
            # grab list of actions and if that action already appeared during this, add penalty
            self.logger.info("Current values(%s): %s" % (current_s, ",".join(str(x) for x in self.q_table[current_s])))
            self.logger.info("OBS observations ---------------")
            self.logger.info(obs)
            self.logger.info("--------------------------------")
            self.logger.info(self.actions)
            l = list()
            # for all possible acttions
            for x in range(0, len(self.actions)):
                # if that action have max val that is the selected move
                if self.q_table[current_s][x] == m:
                    l.append(x)
            y = random.randint(0, len(l)-1)
            a = l[y]
            self.logger.info("Taking q action: %s" % self.actions[a])

        # try to send the selected action, only update prev_s if this succeeds
        try:
            # agent_host.sendCommand(self.actions[a])

            self.move(self.actions[a])
            # self.move("moven")
            # self.move("jumpn")
            # self.move("movew")
            # self.move("jumpn")
            # self.move("jumpn")

            self.prev_s = current_s
            self.prev_a = a

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        return current_r

    def run(self, agent_host):
        """run the agent on the world"""
        # rewards for agent_host
        total_reward = 0
        
        self.prev_s = None
        self.prev_a = None

        

        is_first_action = True
        
        # main loop:
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:
            
            # start reward being 0
            #current_r = 0
            self.reward = 0
            
            #if first move
            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(1)
                    world_state = agent_host.getWorldState()

                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        self.reward += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, self.reward)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and self.reward > 1:
                    time.sleep(1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        self.reward += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        self.reward += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, self.reward)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % self.reward)
        total_reward += self.reward

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState( self.reward )
            
        self.drawQ()
    
        return total_reward
        
    def drawQ( self, curr_x=None, curr_y=None ):
        # how big the q table is
        if(map == 1 or map ==2):
            scale = 40
            world_x = 13
            world_y = 26
            x_offset = 4
            y_offset = 0
            start_x = -4
            start_y = -2
        elif(map == 3):
            scale = 30
            world_x = 11
            world_y = 62
            x_offset = 3
            y_offset = 0
            start_x = -3
            start_y = -2

        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root,
                                    width=world_x*scale,
                                    height=world_y*scale if world_y <30 else 30*scale,
                                    borderwidth=0,
                                    highlightthickness=0,
                                    bg="black",
                                    # scrollregion=(0,0,0,62*scale)
                                    )
            self.canvas.grid()
            self.frame=tk.Frame(self.canvas,width=world_x*scale,height=world_y*scale, bg="black")
            self.canvas.create_window(0,0,window=self.frame,anchor='nw')
            self.scrollbar = tk.Scrollbar(self.root, orient='vertical',command=self.canvas.yview)
            self.scrollbar.grid(row=0, column=1, sticky=tk.NS)
            self.canvas.config(yscrollcommand=self.scrollbar.set)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        action_offset = .4
        curr_radius = 0.2
        # (NSWE to match action order)
        min_value = -125
        max_value = 125
        for x in range(start_x,start_x+world_x+1):
            ax = x + x_offset
            for y in range(start_y,start_y+world_y+1):
                ay = y +y_offset
                s = "%f:%f" % (x+0.5,y+0.5)
                self.canvas.create_rectangle( ax*scale, ay*scale, (ax+1)*scale, (ay+1)*scale, outline="#fff", fill="")
                if self.actions == ["moves", "movew", "movee","jumps", "jumpw", "jumpe"]:
                    flat_action_positions = [ ( action_offset, 1-action_inset ), ( action_inset, action_offset ), ( 1-action_inset, action_offset ) ]
                    jump_action_postitions = [ (1-action_offset, 1-action_inset), (action_inset, 1-action_offset), (1-action_inset, 1-action_offset) ]
                    for action in range(6):
                        if not s in self.q_table:
                            continue
                        value = self.q_table[s][action]
                        color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
                        color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                        color_string = '#%02x%02x%02x' % (255-color, color, 0)
                        if action < 3:
                            self.canvas.create_oval((ax + flat_action_positions[action][0] - action_radius) *scale,
                                                    (ay + flat_action_positions[action][1] - action_radius) *scale,
                                                    (ax + flat_action_positions[action][0] + action_radius) *scale,
                                                    (ay + flat_action_positions[action][1] + action_radius) *scale, 
                                                    outline=color_string, fill=color_string )
                        elif action == 3:
                            self.canvas.create_line((ax+jump_action_postitions[action-3][0])*scale,
                                                    (ay+jump_action_postitions[action-3][1]+action_radius)*scale,
                                                    (ax+jump_action_postitions[action-3][0])*scale,
                                                    (ay+jump_action_postitions[action-3][1]-action_radius)*scale,
                                                    fill=color_string, arrow=tk.FIRST)
                        elif action == 4:
                            self.canvas.create_line((ax+jump_action_postitions[action-3][0]+action_radius)*scale,
                                                    (ay+jump_action_postitions[action-3][1])*scale,
                                                    (ax+jump_action_postitions[action-3][0]-action_radius)*scale,
                                                    (ay+jump_action_postitions[action-3][1])*scale,
                                                    fill=color_string, arrow=tk.LAST)
                        elif action == 5:
                            self.canvas.create_line((ax+jump_action_postitions[action-3][0]+action_radius)*scale,
                                                    (ay+jump_action_postitions[action-3][1])*scale,
                                                    (ax+jump_action_postitions[action-3][0]-action_radius)*scale,
                                                    (ay+jump_action_postitions[action-3][1])*scale,
                                                    fill=color_string, arrow=tk.FIRST)  
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (curr_x - curr_radius +x_offset) * scale, 
                                     (curr_y - curr_radius +y_offset) * scale, 
                                     (curr_x + curr_radius +x_offset) * scale, 
                                     (curr_y + curr_radius +y_offset) * scale, 
                                     outline="#fff", fill="#fff" )
        self.root.update()

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

agent = TabQAgent()
agent_host = MalmoPython.AgentHost()

# asking user which map to load
map = int(input("What map to load? (1: easy) (2: medium) (3: hard): "))
# loading q table
if input("load q table?") == "Yes":

    agent.load_q_table(map)
    print("loaded q table")
    print(agent.q_table)

try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

# -- set up the mission -- #
mission_file= ""

if map == 1:
    mission_file = './175easy.xml'
elif map == 2:
    mission_file = './175medium.xml'
elif map == 3:
    mission_file = './175hard.xml'


with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

# add 20% holes for interest
# for x in range(1,4):
#     for z in range(1,13):
#         if random.random()<0.1:
#             my_mission.drawBlock( x,45,z,"lava")

max_retries = 3

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 80

cumulative_rewards = []
for i in range(num_repeats):
    print()
    print('Repeat %d of %d' % ( i+1, num_repeats ))
    
    my_mission_record = MalmoPython.MissionRecordSpec()

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2.5)

    print("Waiting for the mission to start", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    print()

    # -- run the agent in the world -- #
    cumulative_reward = agent.run(agent_host)
    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [ cumulative_reward ]

    # -- clean up -- #
    time.sleep(0.5) # (let the Mod reset)

agent.save_q_table(agent.q_table, map)
print("Done.")
print()
print("Cumulative rewards for all %d runs:" % num_repeats)
print(cumulative_rewards)