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
        self.epsilon = 0.01 # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        # possible actions migh need to change it to move turnleft turn right and jump

        # add 4 more actions jump up 1 move 2 for nesw
        self.actions = ["moves", "movew", "movee","jumps", "jumpw", "jumpe"]

        self.q_table = {}
        self.canvas = None
        self.root = None
    def move(self, move: str):
        time.sleep(0.3)
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
            # j n
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("move 1"  )
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("move 1"  )
        elif move =="jumpe":
            # j e
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("moveeast 1"  )
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("moveeast 1"  )
        elif move =="jumps":
            # j s
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("move 1"  )
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("move 1"  )
        elif move =="jumpw":
            # j w
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("movewest 1"  )
            agent_host.sendCommand("jump 1")
            agent_host.sendCommand("movewest 1"  )
        

    # create json files
    def save_q_table(self,q_table):
        # change file name if needed
        filename = "q_table.json"
        with open(filename, 'w') as f:
            f.write(json.dumps(q_table))

    # load json files
    def load_q_table(self):
        print("Loading current q t:")
        q_file = open(r"q_table.json")
        self.q_table= json.load(q_file)

    def updateQTable( self, reward, current_state ):
        """Change q_table to reflect what we have learnt."""
    
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        
        # TODO: what should the new action value be?
        new_q = reward
        
        # assign the new action value to the Q-table
        # if old_q == 0 or new_q> old_q:
        self.q_table[self.prev_s][self.prev_a] = new_q
        
    def updateQTableFromTerminatingState( self, reward ):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        
        # TODO: what should the new action value be?
        new_q = reward
        
        # assign the new action value to the Q-table
        if old_q == 0 or new_q> old_q:
            self.q_table[self.prev_s][self.prev_a] = new_q
        
    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        if current_s not in self.q_table:
            self.q_table[current_s] = ([0] * len(self.actions))

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable( current_r, current_s )

        # drawing the q table
        self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
            self.logger.info("Random action: %s" % self.actions[a])
        else:
            m = max(self.q_table[current_s])
            self.logger.info("Current values(%s): %s" % (current_s, ",".join(str(x) for x in self.q_table[current_s])))
            self.logger.info(self.actions)
            l = list()
            for x in range(0, len(self.actions)):
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
            current_r = 0
            
            #if first move
            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.3)
                    world_state = agent_host.getWorldState()

                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r > 1:
                    time.sleep(0.3)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.3)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState( current_r )
            
        self.drawQ()
    
        return total_reward
        
    def drawQ( self, curr_x=None, curr_y=None ):
        scale = 40
        # how big the q table is
        world_x = 6
        world_y = 14

        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")

        action_inset = 0.1
        action_radius = 0.1
        action_offset = .4
        curr_radius = 0.2

        

        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x,y)
                self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")

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
                            self.canvas.create_oval( (x + flat_action_positions[action][0] - action_radius ) *scale,
                                                    (y + flat_action_positions[action][1] - action_radius ) *scale,
                                                    (x + flat_action_positions[action][0] + action_radius ) *scale,
                                                    (y + flat_action_positions[action][1] + action_radius ) *scale, 
                                                    outline=color_string, fill=color_string )
                        elif action == 3:
                            self.canvas.create_line((x+jump_action_postitions[action-3][0])*scale,
                                                    (y+jump_action_postitions[action-3][1]+action_radius)*scale,
                                                    (x+jump_action_postitions[action-3][0])*scale,
                                                    (y+jump_action_postitions[action-3][1]-action_radius)*scale,
                                                    fill=color_string, arrow=tk.FIRST)
                        elif action == 4:
                            self.canvas.create_line((x+jump_action_postitions[action-3][0]+action_radius)*scale,
                                                    (y+jump_action_postitions[action-3][1])*scale,
                                                    (x+jump_action_postitions[action-3][0]-action_radius)*scale,
                                                    (y+jump_action_postitions[action-3][1])*scale,
                                                    fill=color_string, arrow=tk.LAST)
                        elif action == 5:
                            self.canvas.create_line((x+jump_action_postitions[action-3][0]+action_radius)*scale,
                                                    (y+jump_action_postitions[action-3][1])*scale,
                                                    (x+jump_action_postitions[action-3][0]-action_radius)*scale,
                                                    (y+jump_action_postitions[action-3][1])*scale,
                                                    fill=color_string, arrow=tk.FIRST)

        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale, 
                                     (curr_y + 0.5 - curr_radius ) * scale, 
                                     (curr_x + 0.5 + curr_radius ) * scale, 
                                     (curr_y + 0.5 + curr_radius ) * scale, 
                                     outline="#fff", fill="#fff" )
        self.root.update()

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

agent = TabQAgent()
agent_host = MalmoPython.AgentHost()

# loading q table
if input("load q table?") == "Yes":
    agent.load_q_table()
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
mission_file = './tutorial_6.xml'

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
    num_repeats = 250

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

agent.save_q_table(agent.q_table)
print("Done.")
print()
print("Cumulative rewards for all %d runs:" % num_repeats)
print(cumulative_rewards)
